#!/usr/bin/env python3

"""
Merge WSE drop EDA CSV with SWORD DuckDB tables for richer context.

- Joins CSV to DuckDB `nodes` on obstruction_node_id -> node_id (auto-detected)
- Joins CSV to DuckDB `reaches` on reach_id (auto-detected)
- Inspects and reports available columns per table
- Outputs merged Parquet and a metadata JSON with relevant variables grouped by patterns

Usage:
  python scripts/merge_wse_drop_with_duckdb.py \
    --csv /Users/jakegearon/projects/SWORD/data/analysis/master_wse_drop_crossreach_5k.csv \
    --duckdb /Users/jakegearon/projects/SWORD/data/duckdb/sword_global.duckdb \
    --out /Users/jakegearon/projects/SWORD/data/analysis/master_wse_drop_crossreach_5k_merged.parquet

Requirements: duckdb>=0.8, pandas
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Set

import duckdb
import pandas as pd


def get_table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> List[str]:
    """Return a list of column names for a given table, or empty list if missing."""
    try:
        df = con.execute(f"PRAGMA table_info('{table_name}')").df()
        if 'name' in df.columns:
            return df['name'].tolist()
        # Fallback if schema differs
        return df.iloc[:, 1].tolist() if df.shape[1] > 1 else []
    except Exception:
        return []


def detect_join_key(candidate_base_key: str, table_columns: List[str], candidates_table_keys: List[str]) -> Optional[str]:
    """Pick the first matching join key in table_columns from candidates_table_keys."""
    normalized = {c.lower(): c for c in table_columns}
    for key in candidates_table_keys:
        if key.lower() in normalized:
            return normalized[key.lower()]
    # Last resort: exact base key
    if candidate_base_key.lower() in normalized:
        return normalized[candidate_base_key.lower()]
    return None


def build_prefixed_select_list(table: str, columns: List[str], prefix: str, exclude: Optional[List[str]] = None) -> List[str]:
    """Build SELECT expressions aliasing each column to a prefixed name to avoid collisions."""
    excluded_set: Set[str] = set(exclude or [])
    select_exprs: List[str] = []
    for col in columns:
        if col in excluded_set:
            continue
        alias = f"{prefix}{col}"
        select_exprs.append(f"{table}.\"{col}\" AS \"{alias}\"")
    return select_exprs


def group_relevant_columns(columns: List[str]) -> Dict[str, List[str]]:
    """Heuristically group relevant variables by substring patterns."""
    patterns = {
        'ids': ['id', 'reach', 'node', 'continent', 'cl_ids'],
        'geometry': ['x', 'y', 'lon', 'lat', 'geom'],
        'hydraulics': ['wse', 'width', 'depth', 'slope', 'vel', 'q', 'area', 'stage'],
        'topology': ['order', 'up', 'dn', 'trib', 'junction', 'main', 'side'],
        'quality': ['flag', 'qa', 'qc', 'valid', 'conf'],
        'time': ['time', 'date', 'epoch']
    }
    lowered_to_orig = {c.lower(): c for c in columns}
    grouped: Dict[str, List[str]] = {k: [] for k in patterns}
    for key, subs in patterns.items():
        for c in columns:
            cl = c.lower()
            if any(sub in cl for sub in subs):
                grouped[key].append(lowered_to_orig[cl])
    # Deduplicate within groups while preserving order
    for k, vals in grouped.items():
        seen = set()
        deduped: List[str] = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        grouped[k] = deduped
    return grouped


def merge_csv_with_duckdb(csv_path: str, duckdb_path: str, out_path: str, out_meta_path: Optional[str] = None) -> Tuple[int, Dict[str, List[str]]]:
    """Perform the merge in DuckDB and write outputs. Returns row count and metadata."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(duckdb_path):
        raise FileNotFoundError(f"DuckDB not found: {duckdb_path}")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    con = duckdb.connect(duckdb_path, read_only=True)

    # Discover tables and columns
    tables_df = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").df()
    table_names = set(tables_df['table_name'].tolist())

    nodes_cols: List[str] = []
    reaches_cols: List[str] = []

    if 'nodes' in table_names:
        nodes_cols = get_table_columns(con, 'nodes')
    if 'reaches' in table_names:
        reaches_cols = get_table_columns(con, 'reaches')

    # Choose join keys
    nodes_key = detect_join_key(
        candidate_base_key='obstruction_node_id',
        table_columns=nodes_cols,
        candidates_table_keys=['node_id', 'nodes_node_id', 'id']
    ) if nodes_cols else None

    reaches_key = detect_join_key(
        candidate_base_key='reach_id',
        table_columns=reaches_cols,
        candidates_table_keys=['reach_id', 'rch_id']
    ) if reaches_cols else None

    print("Detected tables:", sorted(table_names))
    print("nodes columns:", len(nodes_cols))
    print("reaches columns:", len(reaches_cols))
    print("Join key for nodes:", nodes_key)
    print("Join key for reaches:", reaches_key)

    # Read CSV into temp view
    con.execute(f"CREATE OR REPLACE TEMP VIEW base AS SELECT * FROM read_csv_auto('{csv_path}', sample_size=-1, header=TRUE)")

    # Build SELECT list
    base_cols_df = con.execute("PRAGMA table_info('base')").df()
    base_columns: List[str] = base_cols_df['name'].tolist()

    select_parts: List[str] = ["base.*"]

    if nodes_key:
        select_parts += build_prefixed_select_list('nodes', nodes_cols, prefix='nodes__')
    if reaches_key:
        select_parts += build_prefixed_select_list('reaches', reaches_cols, prefix='reaches__')

    select_sql = ",\n       ".join(select_parts)

    # Build JOINs
    join_sql = "FROM base\n"
    if nodes_key:
        join_sql += f"LEFT JOIN nodes ON base.obstruction_node_id = nodes.\"{nodes_key}\"\n"
    if reaches_key:
        join_sql += f"LEFT JOIN reaches ON base.reach_id = reaches.\"{reaches_key}\"\n"

    final_sql = f"SELECT {select_sql}\n{join_sql}"

    # Execute and write Parquet
    print("Executing merge and writing Parquet →", out_path)
    con.execute(f"COPY ({final_sql}) TO '{out_path}' (FORMAT 'parquet')")

    # For metadata, collect column names without loading full data
    merged_cols_df = con.execute(f"SELECT * FROM ({final_sql}) LIMIT 0").df()
    merged_columns = merged_cols_df.columns.tolist()

    # Relevant columns by pattern
    relevant = group_relevant_columns(merged_columns)

    base_rows: int = int(con.execute("SELECT COUNT(*) AS n FROM base").fetchone()[0])

    meta: Dict[str, object] = {
        'inputs': {
            'csv_path': csv_path,
            'duckdb_path': duckdb_path
        },
        'rows': base_rows,
        'tables': sorted(table_names),
        'base_columns': base_columns,
        'nodes_columns': nodes_cols,
        'reaches_columns': reaches_cols,
        'join_keys': {
            'nodes': nodes_key,
            'reaches': reaches_key
        },
        'merged_columns_count': len(merged_columns),
        'relevant_columns': relevant,
        'output_path': out_path
    }

    if out_meta_path:
        meta_dir = os.path.dirname(out_meta_path)
        if meta_dir:
            os.makedirs(meta_dir, exist_ok=True)
        with open(out_meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print("Wrote metadata JSON →", out_meta_path)

    con.close()
    return base_rows, {'base': base_columns, 'nodes': nodes_cols, 'reaches': reaches_cols}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge WSE drop CSV with SWORD DuckDB tables")
    parser.add_argument('--csv', required=False, default='/Users/jakegearon/projects/SWORD/data/analysis/master_wse_drop_crossreach_5k.csv', help='Path to input EDA CSV')
    parser.add_argument('--duckdb', required=False, default='/Users/jakegearon/projects/SWORD/data/duckdb/sword_global.duckdb', help='Path to SWORD DuckDB database')
    parser.add_argument('--out', required=False, default='/Users/jakegearon/projects/SWORD/data/analysis/master_wse_drop_crossreach_5k_merged.parquet', help='Path to output merged Parquet')
    parser.add_argument('--meta', required=False, default='/Users/jakegearon/projects/SWORD/data/analysis/master_wse_drop_crossreach_5k_merged.meta.json', help='Path to output metadata JSON')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    try:
        rows, cols = merge_csv_with_duckdb(args.csv, args.duckdb, args.out, args.meta)
        print(f"Done. Base rows: {rows}. Columns (base/nodes/reaches): {len(cols['base'])}/{len(cols['nodes'])}/{len(cols['reaches'])}")
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)


if __name__ == '__main__':
    main() 