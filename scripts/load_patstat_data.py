import duckdb
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def connect_db():
    home_dir = Path.home()
    path = home_dir / "patents_name_matching" / "data" / "database" / "patent_database"
    return duckdb.connect(database=path, read_only=False)

def insert_patstat_data(patstat_data: pd.DataFrame):
    """Inserts the patstat data into the database.

    Args:
        patstat_data: The data containing the patstat data.

    Returns:
        None
    """
    con = connect_db()
    con.register('patstat_data_tmp', patstat_data)
    sql = """
        INSERT INTO patstat_data
        SELECT DISTINCT pat_publn_id, publn_date, han_id, 
                        UPPER(han_name) AS han_name, 
                        UPPER(person_name) AS person_name, 
                        UPPER(psn_name) AS psn_name
        FROM patstat_data_tmp;
    """
    con.execute(sql)
    con.execute("DROP VIEW IF EXISTS patstat_data_tmp")
    con.close()



def load_patstat_data():
    """
    Loads all patstat .csv files from all subdirectories into the database.
    """
    # 1. Define the main data directory using Path
    base_dir = Path('../data/patstat_data/')
    
    # 2. Use rglob to recursively find all files ending in .csv
    all_csv_files = list(base_dir.rglob('*.csv'))
    
    print(f"Found {len(all_csv_files)} CSV files to process.")

    # 3. Iterate through the list of file paths with a progress bar
    for file_path in tqdm(all_csv_files, desc="Processing PATSTAT files"):
        try:
            patstat_data = pd.read_csv(file_path, sep=';', low_memory=False)
            insert_patstat_data(patstat_data)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

def main():
    load_patstat_data()

if __name__ == "__main__":
    main()