#!/usr/bin/env python3
"""
Create a directed graph from v17b's ORIGINAL topology.

This script bypasses the phi algorithm and creates a directed graph directly
from the reach_topology table in v17b. This allows validating/comparing the
original SWORD topology without phi modifications.

Input: SWORD v17b database
Output: {continent}_v17b_directed.pkl - MultiDiGraph with original flow directions

Usage:
    python create_v17b_graph.py --continent NA --workdir /path/to/workdir

Note: v17c_pipeline.py builds the graph internally - this script is for standalone use.
"""

import argparse
import os
import pickle
import sys
from datetime import datetime as dt

import duckdb
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd


def log(msg):
    """Print timestamped log message."""
    print(f"[{dt.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_v17b_topology_from_db(db_path: str, continent: str) -> pd.DataFrame:
    """
    Load reach_topology directly from DuckDB.

    Returns DataFrame with: reach_id, direction, neighbor_rank, neighbor_reach_id
    """
    log(f"Connecting to {db_path}...")
    conn = duckdb.connect(db_path, read_only=True)

    query = f"""
        SELECT reach_id, direction, neighbor_rank, neighbor_reach_id
        FROM reach_topology
        WHERE region = '{continent.upper()}'
    """
    log(f"Querying reach_topology for {continent.upper()}...")
    df = conn.execute(query).fetchdf()
    log(f"Loaded {len(df):,} topology rows")

    conn.close()
    return df


def load_v17b_reaches_from_db(db_path: str, continent: str) -> pd.DataFrame:
    """
    Load reaches with attributes from DuckDB.
    """
    log(f"Loading reaches from {db_path}...")
    conn = duckdb.connect(db_path, read_only=True)

    # Get key attributes needed for assign_attribute
    # Note: v17b uses 'reach_length', 'n_rch_down' (not reach_len, n_rch_dn)
    query = f"""
        SELECT
            reach_id, region,
            reach_length, width, slope, facc,
            n_rch_up, n_rch_down, dist_out
        FROM reaches
        WHERE region = '{continent.upper()}'
    """
    df = conn.execute(query).fetchdf()
    # Rename to match expected names
    df = df.rename(columns={'reach_length': 'reach_len', 'n_rch_down': 'n_rch_dn'})
    log(f"Loaded {len(df):,} reaches")

    conn.close()
    return df


def load_v17b_data_from_gpkg(data_dir: str, continent: str) -> gpd.GeoDataFrame:
    """
    Load reaches from GeoPackage (for geometry and full attributes).
    """
    reaches_file = f"{data_dir}/{continent.upper()}_sword_reaches_v17b.gpkg"
    log(f"Loading reaches from {reaches_file}...")
    gdf = gpd.read_file(reaches_file)
    log(f"Loaded {len(gdf):,} reaches from GPKG")
    return gdf


def create_directed_graph_from_topology(
    topology_df: pd.DataFrame,
    reaches_df: pd.DataFrame
) -> nx.MultiDiGraph:
    """
    Create a directed MultiDiGraph from v17b topology.

    Flow direction is defined by:
    - direction='up' means neighbor_reach_id is UPSTREAM of reach_id
    - direction='down' means neighbor_reach_id is DOWNSTREAM of reach_id

    So edge direction is: upstream_reach -> downstream_reach
    For 'up' rows: neighbor_reach_id -> reach_id
    For 'down' rows: reach_id -> neighbor_reach_id
    """
    log("Creating directed graph from v17b topology...")

    G = nx.MultiDiGraph()

    # Create reach_id -> attributes mapping
    reach_attrs = {}
    for _, row in reaches_df.iterrows():
        rid = row['reach_id']
        reach_attrs[rid] = {
            'reach_len': row['reach_len'],
            'width': row['width'],
            'slope': row['slope'],
            'facc': row.get('facc', 0),
            'n_rch_up': row.get('n_rch_up', 0),
            'n_rch_dn': row.get('n_rch_dn', 0),
            'dist_out': row.get('dist_out', 0),
        }

    # Add nodes (all reaches)
    for rid in reaches_df['reach_id']:
        G.add_node(rid, **reach_attrs.get(rid, {}))

    # Add edges from topology
    # Group by reach_id
    edges_added = set()

    for _, row in topology_df.iterrows():
        reach_id = row['reach_id']
        neighbor_id = row['neighbor_reach_id']
        direction = row['direction']

        if direction == 'up':
            # neighbor is upstream of reach: neighbor -> reach
            u, v = neighbor_id, reach_id
        else:  # direction == 'down'
            # neighbor is downstream of reach: reach -> neighbor
            u, v = reach_id, neighbor_id

        # Avoid duplicate edges
        edge_key = (u, v)
        if edge_key in edges_added:
            continue
        edges_added.add(edge_key)

        # Get attributes from downstream reach (v)
        attrs = reach_attrs.get(v, {}).copy()
        attrs['reach_id'] = v

        G.add_edge(u, v, **attrs)

    log(f"Graph created: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    return G


def create_network_node_ids(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Create network node IDs by combining connected reach endpoints.

    This mimics what SWORD_graph.py does to create junction nodes.
    """
    log("Creating network node IDs...")

    # For simplicity, use reach_id strings as node IDs
    # The full implementation would compute geometric junctions

    # Relabel nodes to string format expected by assign_attribute
    mapping = {node: str(node) for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    # Also update reach_id references in edge data
    for u, v, k, d in G.edges(keys=True, data=True):
        if 'reach_id' in d and isinstance(d['reach_id'], (int, np.integer)):
            d['reach_id'] = int(d['reach_id'])

    log(f"Network nodes created: {G.number_of_nodes():,} nodes")
    return G


def validate_dag(G: nx.MultiDiGraph) -> dict:
    """
    Validate that the graph is a DAG (Directed Acyclic Graph).
    """
    log("Validating DAG property...")

    results = {
        'is_dag': nx.is_directed_acyclic_graph(G),
        'cycles': [],
        'n_components': 0,
        'orphan_nodes': 0,
    }

    if not results['is_dag']:
        # Find cycles
        try:
            cycles = list(nx.simple_cycles(G))[:10]  # Limit to first 10
            results['cycles'] = cycles
            log(f"WARNING: Found {len(cycles)}+ cycles!")
        except:
            log("WARNING: Could not enumerate cycles")
    else:
        log("Graph is a valid DAG")

    # Count weakly connected components
    components = list(nx.weakly_connected_components(G))
    results['n_components'] = len(components)
    log(f"Weakly connected components: {len(components):,}")

    # Count orphan nodes (no in or out edges)
    orphans = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
    results['orphan_nodes'] = len(orphans)
    if orphans:
        log(f"WARNING: Found {len(orphans):,} orphan nodes")

    return results


def main():
    ap = argparse.ArgumentParser(description="Create directed graph from v17b topology")
    ap.add_argument("--continent", required=True, help="Continent code (NA, SA, EU, AF, AS, OC)")
    ap.add_argument("--workdir", required=True, help="Working directory")
    ap.add_argument("--db-path", default=None, help="Path to v17b database (default: data/duckdb/sword_v17b.duckdb)")
    ap.add_argument("--output", default=None, help="Output pickle path (default: output/{cont}/{cont}_v17b_directed.pkl)")
    ap.add_argument("--validate-only", action="store_true", help="Only validate, don't save")

    args = ap.parse_args()

    continent = args.continent.upper()
    workdir = args.workdir

    # Default paths
    if args.db_path is None:
        # Look for database in standard location
        db_path = os.path.join(workdir, "data", "duckdb", "sword_v17b.duckdb")
        if not os.path.exists(db_path):
            # Try project root
            db_path = "data/duckdb/sword_v17b.duckdb"
    else:
        db_path = args.db_path

    if args.output is None:
        output_dir = os.path.join(workdir, "output", continent.lower())
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{continent.lower()}_v17b_directed.pkl")
    else:
        output_path = args.output

    log(f"Configuration:")
    log(f"  Continent: {continent}")
    log(f"  Database: {db_path}")
    log(f"  Output: {output_path}")

    # Load data
    topology_df = load_v17b_topology_from_db(db_path, continent)
    reaches_df = load_v17b_reaches_from_db(db_path, continent)

    # Create directed graph
    G = create_directed_graph_from_topology(topology_df, reaches_df)

    # Create network node IDs
    G = create_network_node_ids(G)

    # Validate
    validation = validate_dag(G)

    if not validation['is_dag']:
        log("ERROR: Graph contains cycles! Cannot proceed with assign_attribute.")
        log(f"Sample cycles: {validation['cycles'][:3]}")
        sys.exit(1)

    if args.validate_only:
        log("Validation only mode - not saving")
        return

    # Save
    log(f"Saving graph to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(G, f)
    log(f"Saved: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    log("Done!")

    # Print summary
    print("\n=== Summary ===")
    print(f"Nodes: {G.number_of_nodes():,}")
    print(f"Edges: {G.number_of_edges():,}")
    print(f"Is DAG: {validation['is_dag']}")
    print(f"Components: {validation['n_components']:,}")
    print(f"Orphan nodes: {validation['orphan_nodes']:,}")


if __name__ == "__main__":
    main()
