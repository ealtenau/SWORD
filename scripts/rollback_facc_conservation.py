#!/usr/bin/env python3
"""
Rollback facc conservation corrections (P1, P2, P3) in DuckDB.

Restores original_facc from rollback CSVs in REVERSE order (P3 → P2 → P1).
Each pass's CSV records the original_facc AT THE TIME THAT PASS RAN,
so rolling back in reverse restores the correct pre-pass values.

Usage:
    # Dry run — show what would be rolled back
    python scripts/rollback_facc_conservation.py \
        --db data/duckdb/sword_v17c.duckdb

    # Roll back all 3 passes
    python scripts/rollback_facc_conservation.py \
        --db data/duckdb/sword_v17c.duckdb --apply

    # Roll back only P3
    python scripts/rollback_facc_conservation.py \
        --db data/duckdb/sword_v17c.duckdb --apply --passes 3

    # Roll back P3 and P2 (keep P1)
    python scripts/rollback_facc_conservation.py \
        --db data/duckdb/sword_v17c.duckdb --apply --passes 3 2

    # Single region
    python scripts/rollback_facc_conservation.py \
        --db data/duckdb/sword_v17c.duckdb --apply --region SA
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

REGIONS = ["NA", "SA", "EU", "AF", "AS", "OC"]


def rollback_pass(
    conn: duckdb.DuckDBPyConnection,
    pass_num: int,
    region: str,
    csv_dir: Path,
    dry_run: bool,
) -> int:
    csv_path = csv_dir / f"facc_conservation_p{pass_num}_{region}.csv"
    if not csv_path.exists():
        print(f"  P{pass_num} {region}: no CSV found at {csv_path}, skipping")
        return 0

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print(f"  P{pass_num} {region}: empty CSV, skipping")
        return 0

    if "original_facc" not in df.columns:
        print(f"  P{pass_num} {region}: CSV missing original_facc column!")
        return 0

    restore = df[["reach_id", "original_facc"]].copy()
    n = len(restore)

    if dry_run:
        print(f"  P{pass_num} {region}: would restore {n:,} reaches")
        return n

    # Load spatial extension and drop RTREE indexes
    conn.execute("INSTALL spatial; LOAD spatial;")
    indexes = conn.execute(
        "SELECT index_name, table_name, sql FROM duckdb_indexes() "
        "WHERE sql LIKE '%RTREE%'"
    ).fetchall()
    for idx_name, tbl, sql in indexes:
        conn.execute(f'DROP INDEX IF EXISTS "{idx_name}"')

    # Batch update via temp table
    conn.execute("DROP TABLE IF EXISTS _rollback_facc")
    conn.execute(
        "CREATE TEMP TABLE _rollback_facc ("
        "  reach_id BIGINT PRIMARY KEY,"
        "  original_facc DOUBLE"
        ")"
    )
    conn.executemany(
        "INSERT INTO _rollback_facc VALUES (?, ?)",
        list(zip(restore["reach_id"].astype(int), restore["original_facc"].astype(float))),
    )

    conn.execute(
        "UPDATE reaches SET facc = t.original_facc "
        "FROM _rollback_facc t WHERE reaches.reach_id = t.reach_id"
    )
    conn.execute("DROP TABLE IF EXISTS _rollback_facc")

    # Recreate RTREE indexes
    for idx_name, tbl, sql in indexes:
        conn.execute(sql)

    print(f"  P{pass_num} {region}: restored {n:,} reaches")
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Rollback facc conservation corrections"
    )
    parser.add_argument("--db", required=True, help="Path to DuckDB")
    parser.add_argument("--apply", action="store_true", help="Actually write (default: dry run)")
    parser.add_argument(
        "--passes", nargs="+", type=int, default=[3, 2, 1],
        help="Which passes to roll back, in order (default: 3 2 1)",
    )
    parser.add_argument("--region", help="Single region (default: all)")
    parser.add_argument(
        "--csv-dir", default="output/facc_detection",
        help="Directory with rollback CSVs",
    )
    args = parser.parse_args()

    dry_run = not args.apply
    mode = "DRY RUN" if dry_run else "APPLYING"
    regions = [args.region.upper()] if args.region else REGIONS
    csv_dir = Path(args.csv_dir)
    passes = sorted(args.passes, reverse=True)  # always roll back newest first

    print(f"\nFacc Conservation Rollback [{mode}]")
    print(f"  Passes to roll back: {passes}")
    print(f"  Regions: {regions}")
    print(f"  CSV dir: {csv_dir}")
    print()

    conn = duckdb.connect(args.db, read_only=dry_run)
    total = 0
    try:
        for p in passes:
            for reg in regions:
                total += rollback_pass(conn, p, reg, csv_dir, dry_run)
    finally:
        conn.close()

    print(f"\nTotal: {total:,} reaches {'would be' if dry_run else ''} restored")
    if dry_run:
        print("Run with --apply to execute.")


if __name__ == "__main__":
    main()
