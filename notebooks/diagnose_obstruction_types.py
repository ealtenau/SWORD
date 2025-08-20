import duckdb
import pandas as pd

def diagnose_obstruction_types(db_path, pairs_csv):
    """
    Connects to the SWORD DuckDB and runs a diagnostic query to categorize
    obstructions based on the lake flags of their upstream/downstream nodes.

    Args:
        db_path (str): Path to the SWORD global DuckDB database.
        pairs_csv (str): Path to the CSV file containing obstruction pairs.
    """
    sql_query = """
    -- This query categorizes our obstructions based on the water type of their upstream/downstream nodes.
    WITH obstruction_geography AS (
        SELECT
            obs.obstruction_node_id,
            -- Get the lakeflag for the immediate upstream and downstream nodes
            up.lakeflag AS upstream_flag,
            down.lakeflag AS downstream_flag
        FROM
            read_csv_auto(?) AS obs -- Use placeholder for parameterized query
        LEFT JOIN
            nodes AS up ON obs.upstream_node_id = up.node_id
        LEFT JOIN
            nodes AS down ON obs.downstream_node_id = down.node_id
    )
    SELECT
        -- Create a descriptive category for each pair
        CASE
            WHEN upstream_flag = 1 AND downstream_flag = 0 THEN 'Lake-to-River (Classic Dam)'
            WHEN upstream_flag = 1 AND downstream_flag = 1 THEN 'Lake-to-Lake (Inter-reservoir)'
            WHEN upstream_flag = 0 AND downstream_flag = 0 THEN 'River-to-River (Weir/Obstruction)'
            ELSE 'Other/Unknown'
        END AS obstruction_type,
        COUNT(*) AS number_of_obstructions
    FROM
        obstruction_geography
    GROUP BY
        obstruction_type
    ORDER BY
        number_of_obstructions DESC;
    """
    
    print("--- Running Obstruction Type Diagnosis ---")
    with duckdb.connect(database=db_path, read_only=True) as con:
        print(f"Analyzing {pairs_csv} against the database...")
        results_df = con.execute(sql_query, [pairs_csv]).df()
    
    print("\n--- Diagnostic Results ---")
    print("Categorization of global obstructions based on adjacent water body type:")
    print(results_df.to_string(index=False))

if __name__ == '__main__':
    DUCKDB_FILE = 'data/duckdb/sword_global.duckdb'
    PAIRS_CSV = 'data/analysis/global_obstruction_hydro_pairs.csv'
    diagnose_obstruction_types(db_path=DUCKDB_FILE, pairs_csv=PAIRS_CSV) 