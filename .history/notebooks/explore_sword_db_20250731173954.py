import duckdb
import pandas as pd

def explore_sword_db(db_path):
    """
    Connects to the SWORD DuckDB database, performs a sample query,
    and prints the results to demonstrate its usability.

    Args:
        db_path (str): The file path for the input DuckDB database file.
    """
    print(f"Connecting to DuckDB at: {db_path}")
    con = duckdb.connect(database=db_path, read_only=True)

    print("\nTables in the database:")
    tables = con.execute("SHOW TABLES").fetchdf()
    print(tables)

    print("\nSchema of the 'reaches' table:")
    schema = con.execute("DESCRIBE reaches").fetchdf()
    print(schema)

    print("\nQuerying for the 10 widest river reaches:")
    query = """
    SELECT 
        reach_id, 
        river_name, 
        width,
        reach_length,
        slope
    FROM reaches
    WHERE width IS NOT NULL
    ORDER BY width DESC
    LIMIT 10;
    """
    widest_reaches = con.execute(query).fetchdf()
    
    print("Results:")
    print(widest_reaches)

    con.close()
    print(f"\nConnection to {db_path} closed.")

if __name__ == '__main__':
    DUCKDB_FILE = 'data/duckdb/sword_eu.duckdb'
    
    import os
    if not os.path.exists(DUCKDB_FILE):
        print(f"Error: DuckDB file not found at '{DUCKDB_FILE}'")
        print("Please run 'create_duckdb_from_sword.py' first.")
    else:
        explore_sword_db(db_path=DUCKDB_FILE) 