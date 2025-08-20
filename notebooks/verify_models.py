import duckdb
import dataclasses
from src.updates.sword_models import ReachData, NodeData, CenterlineData
import os

def verify_models(db_path):
    """
    Compares the schema of the DuckDB tables with the fields of the
    corresponding dataclasses to ensure they match.
    """
    print("--- Verifying Data Models against Database Schema ---")
    all_ok = True

    try:
        con = duckdb.connect(database=db_path, read_only=True)

        checks = {
            'reaches': ReachData,
            'nodes': NodeData,
            'centerlines': CenterlineData
        }

        for table_name, model in checks.items():
            print(f"\nVerifying table '{table_name}' against model '{model.__name__}'...")

            db_columns = set(con.execute(f"DESCRIBE {table_name}").fetchdf()['column_name'])
            model_fields = set(f.name for f in dataclasses.fields(model))
            
            db_only = db_columns - model_fields
            model_only = model_fields - db_columns
            
            if not db_only and not model_only:
                print(f"  [SUCCESS] Schema for '{table_name}' and model '{model.__name__}' match perfectly.")
            else:
                all_ok = False
                if db_only:
                    print(f"  [ERROR] Columns in DB table '{table_name}' but NOT in model '{model.__name__}': {db_only}")
                if model_only:
                    print(f"  [ERROR] Fields in model '{model.__name__}' but NOT in DB table '{table_name}': {model_only}")

    finally:
        if 'con' in locals() and con:
            con.close()
    
    print("\n--- Verification Complete ---")
    if all_ok:
        print("All data models are perfectly synchronized with the database schema. ✅")
    else:
        print("Discrepancies found between data models and database schema. ❌")

    return all_ok

if __name__ == "__main__":
    db_path = os.path.join("data", "duckdb", "sword_global.duckdb")
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        print("Please run the script to create the global database first.")
    else:
        verify_models(db_path) 