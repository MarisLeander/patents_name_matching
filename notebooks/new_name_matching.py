import os
import duckdb
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from duckdb import CatalogException
from duckdb import ConstraintException


# We're using the google style guide for python (https://google.github.io/styleguide/pyguide.html)

def connect_db():
    home_dir = Path.home()
    path = home_dir / "patents_name_matching" / "data" / "database" / "patent_database"
    return duckdb.connect(database=path, read_only=False)

def process_patstat_files():
    """
    Processes the patstat_data and finds matching firms using a single,
    efficient set-based SQL query.
    """

    con = connect_db()
    # Enable progress bar
    con.execute("PRAGMA enable_progress_bar = true;")
    
    # This single query replaces the entire Python loop and its sub-functions.
    sql = """
    -- Use an INSERT statement with ON CONFLICT to update existing matches if a better one is found.
    INSERT INTO patstat_firm_match (han_id, firm_id, similarity)
    WITH
    -- Step 1: Create a unified list of PATSTAT names to match, giving priority to han_name.
    patstat_names_to_match AS (
        SELECT DISTINCT
            han_id,
            -- Clean the names by extracting text before the first comma
            regexp_extract(name, '^([^,]+)') AS patstat_name,
            priority
        FROM (
            SELECT han_id, han_name AS name, 1 AS priority FROM patstat_data WHERE han_name IS NOT NULL
            UNION ALL
            SELECT han_id, psn_name AS name, 2 AS priority FROM patstat_data WHERE psn_name IS NOT NULL
            UNION ALL
            SELECT han_id, person_name AS name, 3 AS priority FROM patstat_data WHERE person_name IS NOT NULL
        )
    ),

    -- Step 2: Calculate similarity scores for all potential pairs above a threshold.
    -- This cross-join is the heavy lifting part.
    candidate_matches AS (
        SELECT
            p.han_id,
            f.firm_id,
            jaro_winkler_similarity(f.name, p.patstat_name) AS similarity,
            p.priority
        FROM patstat_names_to_match AS p
        CROSS JOIN firm_names AS f
        WHERE jaro_winkler_similarity(f.name, p.patstat_name) > 0.8 -- Pre-filter to reduce work
    ),

    -- Step 3: Rank the matches for each han_id to find the best one.
    -- We rank by similarity (desc) and then by name priority (asc).
    ranked_matches AS (
        SELECT
            han_id,
            firm_id,
            similarity,
            ROW_NUMBER() OVER (PARTITION BY han_id ORDER BY similarity DESC, priority ASC) as rn
        FROM candidate_matches
    )

    -- Step 4: Select only the top-ranked match for each han_id.
    SELECT
        han_id,
        firm_id,
        similarity
    FROM ranked_matches
    WHERE rn = 1
    
    -- This clause handles updates: if a han_id already exists, it will only
    -- be updated if the new match has a higher similarity score.
    ON CONFLICT (han_id) DO UPDATE
    SET
        firm_id = excluded.firm_id,
        similarity = excluded.similarity
    WHERE
        excluded.similarity > patstat_firm_match.similarity;
    """
    
    con.execute(sql)
    con.close()
    print("Finished processing and updating all matches.")

if __name__ == "__main__":
    process_patstat_files()