"""
Diagnoses Hydrocron API failures by correlating them with the `lakeflag` 
attribute in the SWORD database, using only existing cached data.

This script will:
1. Load the most recent cached Hydrocron API results.
2. Identify which nodes were successfully queried and which were not.
3. Use DuckDB to look up the `lakeflag` for both successful and failed nodes.
4. Compare the `lakeflag` distributions to test the hypothesis that failures
   are concentrated in nodes located within lakes/reservoirs.
"""
import os
import pickle
from pathlib import Path

import duckdb
import pandas as pd

# --- Configuration ---
CACHE_DIR = Path("cache_wse_drop")
INPUT_CSV = "data/analysis/global_obstruction_hydro_pairs.csv"
DUCKDB_FILE = 'data/duckdb/sword_global.duckdb'

LAKEFLAG_MAP = {
    0: "River",
    1: "Lake/Reservoir",
    2: "Canal",
    3: "Tidally Influenced River"
}

def get_lakeflags_for_nodes(node_ids: list) -> pd.Series:
    """Queries DuckDB to get the lakeflag for a list of node IDs."""
    if not node_ids:
        return pd.Series(dtype=int)
    
    with duckdb.connect(database=DUCKDB_FILE, read_only=True) as con:
        # Using a parameterized query with a list of values
        query = "SELECT node_id, lakeflag FROM nodes WHERE node_id IN ({})".format(','.join(['?'] * len(node_ids)))
        # DuckDB requires parameters to be passed as a list of strings for this kind of IN clause
        params = [str(n) for n in node_ids]
        
        flags_df = con.execute(query, params).df()
        return flags_df.set_index('node_id')['lakeflag']

def format_distribution(series: pd.Series) -> str:
    """Formats a value_counts series into a readable string with labels."""
    dist = series.value_counts(normalize=True).sort_index()
    output = []
    for flag, pct in dist.items():
        int_flag = int(flag)
        label = LAKEFLAG_MAP.get(int_flag, f"Unknown ({int_flag})")
        output.append(f"{label:<25}: {pct:>6.1%}")
    return "\n".join(output) if output else "No data found."


def main():
    """Main diagnostic function."""
    print("--- Hydrocron API Failure Diagnosis ---")

    # 1. Load Cached API Results to find the "Success Set"
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    if not cache_files:
        print("Error: No cache files found. Please run `calculate_global_wse_drop.py` first.")
        return
    
    latest_cache_file = max(cache_files, key=os.path.getmtime)
    print(f"Loading latest cache file: {latest_cache_file}")
    with open(latest_cache_file, 'rb') as f:
        cached_df = pickle.load(f)
    
    success_set = set(cached_df['node_id'].dropna().astype(str).unique())
    print(f"Found {len(success_set)} unique nodes with data in cache (Success Set).")

    # 2. Re-create the "Attempted Set" from the swarm logic
    print("Re-creating the list of all nodes attempted in the last run...")
    pairs_df = pd.read_csv(INPUT_CSV)
    pairs_df.dropna(
        subset=['upstream_node_id', 'downstream_node_id', 'upstream_reach_id', 'downstream_reach_id',
                'upstream_node_distance_from_outlet', 'downstream_node_distance_from_outlet'],
        inplace=True
    )

    with duckdb.connect(database=DUCKDB_FILE, read_only=True) as con:
        con.register('pairs_df', pairs_df)
        swarm_query = """
        SELECT n.node_id FROM pairs_df p JOIN nodes n ON p.upstream_reach_id = n.reach_id WHERE n.dist_out >= p.upstream_node_distance_from_outlet AND n.dist_out <= (p.upstream_node_distance_from_outlet + 2000)
        UNION
        SELECT n.node_id FROM pairs_df p JOIN nodes n ON p.downstream_reach_id = n.reach_id WHERE n.dist_out <= p.downstream_node_distance_from_outlet AND n.dist_out >= (p.downstream_node_distance_from_outlet - 1000)
        """
        attempted_df = con.execute(swarm_query).df()
    
    attempted_set = set(attempted_df['node_id'].dropna().astype('int64').astype(str).unique())
    print(f"Found {len(attempted_set)} unique nodes that were attempted (Attempted Set).")

    # 3. Calculate the "Failed Set"
    failed_set = attempted_set - success_set
    print(f"Identified {len(failed_set)} nodes with no data (Failed Set).")

    # 4. Characterize Node Populations
    print("\nQuerying DuckDB to get lakeflag characteristics...")
    # Convert sets to lists for querying
    success_flags = get_lakeflags_for_nodes(list(success_set))
    failed_flags = get_lakeflags_for_nodes(list(failed_set))

    # 5. Analyze and Compare Distributions
    print("\n--- Lakeflag Distribution for SUCCESSFUL Nodes ---")
    print(format_distribution(success_flags))

    print("\n--- Lakeflag Distribution for FAILED Nodes ---")
    print(format_distribution(failed_flags))

    # 6. Conclusion
    print("\n--- Conclusion ---")
    try:
        failed_lake_pct = failed_flags.value_counts(normalize=True).get(1, 0)
        success_lake_pct = success_flags.value_counts(normalize=True).get(1, 0)
        if failed_lake_pct > success_lake_pct * 5 and failed_lake_pct > 0.5:
            conclusion = f"The hypothesis is strongly supported. Nodes flagged as Lake/Reservoir make up {failed_lake_pct:.1%} of failures, a dramatically higher proportion than in successful queries. This indicates that API calls are failing because we are using the 'Node' endpoint for features that are classified as 'Lakes'."
        else:
            conclusion = "The hypothesis is not strongly supported by the data. While some failed nodes are in lakes, the distribution is not dramatically different from successful nodes. The root cause of failures may be more complex."
    except Exception:
        conclusion = "Could not draw a firm conclusion from the data."
        
    print(conclusion)

if __name__ == "__main__":
    main() 