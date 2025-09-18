import os
import io
import time
import json
import vllm
import duckdb
import contextlib
from pathlib import Path
from vllm import LLM, SamplingParams
import inference_helper as ih

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
    path = home_dir / "patents_name_matching" / "secrets.json" / "data" / "database" / "patent_database"
    return duckdb.connect(database=path, read_only=False)

def create_label_table(reset: bool = False):
    """Creates a table in the database to store the labels

    Args:
        reset: Whether to reset the table if it already exists
    Returns:
        None
    """
    con = connect_db()
    if reset:
        con.execute("DROP TABLE IF EXISTS labels")

    con.execute("""
        CREATE TABLE IF NOT EXISTS labels_gemma (
            han_id INTEGER,
            firm_id INTEGER REFERENCES firm_names(firm_id),
            label INTEGER
        )
    """)
    con.close()

def insert_label(han_id: int, firm_id: int, label: int):
    """Inserts a label into the label table

    Args:
        han_id: The id of the han record
        firm_id: The id of the firm record
        label: The label of the record
    Returns:
        None
    """
    try:
        con.execute(f"""
            INSERT INTO labels_gemma
            VALUES ({han_id}, {firm_id}, {label})
        """)
    except ConstraintException as e:
        # Entry already in Database
        pass

    
def get_prompt_batch():
    con = connect_db()
    sql = """
    SELECT DISTINCT firm_id, han_id, similarity, name, han_name, person_name, psn_name FROM patstat_firm_match
    JOIN firm_names USING(firm_id)
    JOIN patstat_data USING(han_id)
    WHERE similarity >= 0.9
    LIMIT 1
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
            insert_label(han_id, firm_id, 1)
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
    records_to_insert = []
    # We zip the original prompt_batch with the outputs to maintain metadata
    for prompt_dict, output in zip(prompt_batch, outputs):
        print(prompt_dict, output)
    #     paragraph_id = prompt_dict.get("paragraph_id")
    #     prompt_type = prompt_dict.get("prompt_type")
    #     generated_text = output.outputs[0].text.strip()

    #     # If technique is chain-of-thought we need to parse our prediction and insert the CoT as thinking_process
    #     if technique == "CoT":
    #         records_to_insert.append({
    #             "id": paragraph_id, 
    #             "model": 'gemma-3-27b-it', 
    #             "prompt_type": prompt_type, 
    #             "technique": technique, 
    #             "prediction": ih.extract_stance_cot(generated_text), 
    #             "thinking_process": generated_text,
    #             "thoughts": None # Placeholder
    #         })
    #     else:
    #         records_to_insert.append({
    #             "id": paragraph_id, 
    #             "model": 'gemma-3-27b-it', 
    #             "prompt_type": prompt_type, 
    #             "technique": technique, 
    #             "prediction": generated_text, 
    #             "thinking_process": None, # Placeholder
    #             "thoughts": None # Placeholder
    #         })
            
    # print("Inserting predictions into db")
    # ih.insert_batch(records_to_insert, engineering)  

        

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
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,  # Use 1 GPUs
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