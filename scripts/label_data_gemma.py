import os
import io
import time
import json
import vllm
import torch
import duckdb
import contextlib
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from vllm import LLM, SamplingParams
from duckdb import ConstraintException

def get_hf_api_key() -> str:
    """Gets the user's Huggingface API key from a config file."""
    home_dir = Path.home()
    path = home_dir / "patents_name_matching" / "secrets.json"
    try:
        with open(path, "r") as config_file:
            config = json.load(config_file)
        return config.get("huggingface_api_key")
    except FileNotFoundError:
        print(f"Error: secrets file not found at {path}")
        return None

def connect_db():
    home_dir = Path.home()
    path = home_dir / "patents_name_matching" / "data" / "database" / "patent_database"
    return duckdb.connect(database=path, read_only=False)

def insert_label(han_id: int, firm_id: int, label: int):
    """Inserts a label into the label table

    Args:
        han_id: The id of the han record
        firm_id: The id of the firm record
        label: The label of the record
    Returns:
        None
    """
    con = connect_db()
    try:
        con.execute(f"""
            INSERT INTO labels_gemma
            VALUES ({han_id}, {firm_id}, {label})
        """)
    except ConstraintException as e:
        # Entry already in Database
        pass
    con.close()

def create_label_table(reset: bool = False):
    """Creates a table in the database to store the labels

    Args:
        reset: Whether to reset the table if it already exists
    Returns:
        None
    """
    con = connect_db()
    if reset:
        con.execute("DROP TABLE IF EXISTS labels_gemma")

    con.execute("""
        CREATE TABLE IF NOT EXISTS labels_gemma (
            han_id INTEGER,
            firm_id INTEGER REFERENCES firm_names(firm_id),
            label BOOLEAN,
            PRIMARY KEY (han_id, firm_id)
        )
    """)
    con.close()

    
def get_prompt_batch():
    con = connect_db()
    sql = """
    SELECT DISTINCT
        pfm.firm_id,
        pfm.han_id,
        pfm.similarity,
        fn.name,
        pd.han_name,
        pd.person_name,
        pd.psn_name
    FROM
        patstat_firm_match AS pfm
    JOIN
        firm_names AS fn USING(firm_id)
    JOIN
        patstat_data AS pd USING(han_id)
    WHERE
        pfm.similarity >= 0.9
    AND NOT EXISTS (
        SELECT 1
        FROM labels_gemma AS lg
        WHERE lg.firm_id = pfm.firm_id AND lg.han_id = pfm.han_id
    )
    LIMIT 200000;
    """
    data = con.execute(sql).fetchdf()
    
    user_prompt_template = """
    Input Data (JSON):
    {input_data}

    Output:
    Your response must be a single word: true if a valid match is found, and false otherwise. Do not include any explanations, punctuation, or other text.
    """

    system_prompt = """Your task is to determine if a given company name ('name') matches any of the provided company names from the PATSTAT database ('han_name', 'person_name', 'psn_name'). You must be very thorough in your analysis. 
    
    - Assume that the provided names are accurate and free of spelling errors.
    - Focus on identifying exact or near-exact matches, considering only common and accepted abbreviations. 
    - Do NOT consider minor variations or potential spelling mistakes as valid matches.\n
    """

    prompt_batch = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Building prompt batch..."):
        han_id = row["han_id"]
        firm_id = row["firm_id"]
        
        # If the name jaro-winkler similarity is >= .99, we assume it is a match
        if row['similarity'] >= 0.99:
            # We assume it is a match
            insert_label(han_id, firm_id, True)
        else:
            # 1. Convert the row to JSON to be used as input
            input_data_json = row.to_json()
            
            # 2. Format the template with the current row's data
            user_prompt = user_prompt_template.format(input_data=input_data_json)
            
            # append formatted prompt to batch
            message = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
            
            helper_dict = {
                "han_id":han_id,
                "firm_id":firm_id,
                "message":message
            }
            prompt_batch.append(helper_dict)

    con.close()
    return prompt_batch


def gemma_batch_inference(
    prompt_batch: list[dict],
    llm:vllm.entrypoints.llm.LLM,
):

    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
    
    # Get the tokenizer from the llm object
    tokenizer = llm.get_tokenizer()

    # Prepare all formatted prompt strings in a list
    print("Applying chat templates to all prompts...")
    prompts_to_generate = [
        tokenizer.apply_chat_template(
            prompt_dict.get("message"),
            tokenize=False,
            add_generation_prompt=True
        )
        for prompt_dict in prompt_batch
    ]
    # Make a SINGLE call to llm.generate with the entire batch
    print(f"Generating responses for {len(prompts_to_generate)} prompts in one batch...")
    outputs = llm.generate(prompts_to_generate, sampling_params)

    # Process the results and insert into the database
    print("Processing results and inserting into database...")
    
    data_to_insert = []
    for prompt_dict, output in tqdm(
        zip(prompt_batch, outputs),
        total=len(prompt_batch),
        desc="Processing model outputs"
    ):
        # Get model answer and IDs
        generated_answer = output.outputs[0].text.strip()
        boolean_answer = generated_answer.lower() == 'true'
        han_id = prompt_dict.get("han_id")
        firm_id = prompt_dict.get("firm_id")
        
        # Append a dictionary or tuple to our list
        data_to_insert.append({
            "han_id": han_id,
            "firm_id": firm_id,
            "label": int(boolean_answer)
        })
    
    if data_to_insert:
        # --- Step 1: Create the DataFrame ---
        insert_df = pd.DataFrame(data_to_insert)
    
        con = connect_db()
        try:
            # Register the DataFrame as a temporary table ---
            con.register('new_labels', insert_df)
    
            # Execute the INSERT with ON CONFLICT ---
            # This SQL query selects all data from our temporary table and inserts it,
            # ignoring any rows that violate the primary key constraint.
            con.execute("""
                INSERT INTO labels_gemma
                SELECT * FROM new_labels
                ON CONFLICT (han_id, firm_id) DO NOTHING
            """)
            
            print(f"Processed {len(insert_df)} records (new records inserted, duplicates ignored).")
    
        except Exception as e:
            print(f"An error occurred during bulk insert: {e}")
        finally:
            con.close()

        

def main():
    """
    Main function to load the LLM and start the predictions.
    """
    # Set token as an environment variable for vLLM
    print("Retrieving Huggingface API key...")
    hf_token = get_hf_api_key()
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    else:
        print("Hugging Face API key not found. Exiting.")
        return

    # vLLM Initialization
    start_time = time.time()
    model_name = "google/gemma-3-27b-it"
    
    print(f"Initializing LLM '{model_name}' with vLLM...")
    print("This may take several minutes...")
    try:
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs")
        llm = LLM(
            model=model_name,
            tensor_parallel_size=num_gpus,  # Use all GPUs
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error during model initialization: {e}")
        return

    elapsed_time = (time.time() - start_time) / 60
    print(f"--- LLM setup complete in {elapsed_time:.2f} min. ---")

    create_label_table(False)
    prompt_batch = get_prompt_batch()
    gemma_batch_inference(prompt_batch, llm)



if __name__ == "__main__":
    main()