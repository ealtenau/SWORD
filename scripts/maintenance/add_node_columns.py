#!/usr/bin/env python
"""
Add dn_node_id, up_node_id (reaches) and node_order (nodes) columns.

Computes boundary nodes and node ordering from dist_out, not node_id,
so results are correct even when flow direction changes reorder node IDs.

Usage:
    uv run python scripts/maintenance/add_node_columns.py --db data/duckdb/sword_v17c.duckdb
    uv run python scripts/maintenance/add_node_columns.py --db data/duckdb/sword_v17c.duckdb --verify-only
"""

from __future__ import annotations

import argparse
import sys

import duckdb


def add_columns(con: duckdb.DuckDBPyConnection) -> None:
    """Add new columns if they don't exist."""
    existing_reach_cols = {
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'reaches'"
        ).fetchall()
    }
    existing_node_cols = {
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'nodes'"
        ).fetchall()
    }

    if "dn_node_id" not in existing_reach_cols:
        con.execute("ALTER TABLE reaches ADD COLUMN dn_node_id BIGINT")
        print("Added reaches.dn_node_id")
    if "up_node_id" not in existing_reach_cols:
        con.execute("ALTER TABLE reaches ADD COLUMN up_node_id BIGINT")
        print("Added reaches.up_node_id")
    if "node_order" not in existing_node_cols:
        con.execute("ALTER TABLE nodes ADD COLUMN node_order INTEGER")
        print("Added nodes.node_order")


def drop_rtree_indexes(
    con: duckdb.DuckDBPyConnection,
) -> list[tuple[str, str, str]]:
    """Drop RTREE indexes and return their definitions for recreation."""
    con.execute("INSTALL spatial; LOAD spatial;")
    indexes = con.execute(
        "SELECT index_name, table_name, sql FROM duckdb_indexes() "
        "WHERE sql LIKE '%RTREE%'"
    ).fetchall()
    for idx_name, _tbl, _sql in indexes:
        con.execute(f'DROP INDEX "{idx_name}"')
        print(f"Dropped RTREE index: {idx_name}")
    return indexes


def recreate_rtree_indexes(
    con: duckdb.DuckDBPyConnection, indexes: list[tuple[str, str, str]]
) -> None:
    """Recreate previously dropped RTREE indexes."""
    for idx_name, _tbl, sql in indexes:
        con.execute(sql)
        print(f"Recreated RTREE index: {idx_name}")


def populate_reach_boundary_nodes(con: duckdb.DuckDBPyConnection) -> int:
    """Set dn_node_id/up_node_id from dist_out ordering."""
    result = con.execute("""
        WITH boundary AS (
            SELECT reach_id,
                FIRST(node_id ORDER BY dist_out ASC) AS dn_nid,
                FIRST(node_id ORDER BY dist_out DESC) AS up_nid
            FROM nodes
            GROUP BY reach_id
        )
        UPDATE reaches
        SET dn_node_id = boundary.dn_nid,
            up_node_id = boundary.up_nid
        FROM boundary
        WHERE reaches.reach_id = boundary.reach_id
    """)
    count = result.fetchone()[0]
    print(f"Updated {count} reaches with boundary node IDs")
    return count


def populate_node_order(con: duckdb.DuckDBPyConnection) -> int:
    """Set node_order = 1..n per reach, ordered by dist_out ascending."""
    result = con.execute("""
        WITH ordered AS (
            SELECT node_id, region,
                ROW_NUMBER() OVER (
                    PARTITION BY reach_id ORDER BY dist_out ASC
                ) AS rn
            FROM nodes
        )
        UPDATE nodes
        SET node_order = ordered.rn
        FROM ordered
        WHERE nodes.node_id = ordered.node_id
          AND nodes.region = ordered.region
    """)
    count = result.fetchone()[0]
    print(f"Updated {count} nodes with node_order")
    return count


def verify(con: duckdb.DuckDBPyConnection) -> bool:
    """Validate the new columns. Returns True if all checks pass."""
    ok = True

    # 1. No NULL dn_node_id/up_node_id on reaches that have nodes
    nulls = con.execute("""
        SELECT COUNT(*) FROM reaches r
        WHERE r.n_nodes > 0
          AND (r.dn_node_id IS NULL OR r.up_node_id IS NULL)
    """).fetchone()[0]
    if nulls > 0:
        print(f"FAIL: {nulls} reaches with nodes have NULL boundary IDs")
        ok = False
    else:
        print("PASS: all reaches with nodes have boundary node IDs")

    # 2. Boundary nodes exist in nodes table
    bad_refs = con.execute("""
        SELECT COUNT(*) FROM reaches r
        WHERE r.dn_node_id IS NOT NULL
          AND (
            r.dn_node_id NOT IN (SELECT node_id FROM nodes)
            OR r.up_node_id NOT IN (SELECT node_id FROM nodes)
          )
    """).fetchone()[0]
    if bad_refs > 0:
        print(f"FAIL: {bad_refs} reaches reference non-existent nodes")
        ok = False
    else:
        print("PASS: all boundary node IDs exist in nodes table")

    # 3. dn_node_id dist_out <= up_node_id dist_out
    inverted = con.execute("""
        SELECT COUNT(*) FROM reaches r
        JOIN nodes n_dn ON n_dn.node_id = r.dn_node_id
        JOIN nodes n_up ON n_up.node_id = r.up_node_id
        WHERE n_dn.dist_out > n_up.dist_out
    """).fetchone()[0]
    if inverted > 0:
        print(f"FAIL: {inverted} reaches have dn dist_out > up dist_out")
        ok = False
    else:
        print("PASS: dn dist_out <= up dist_out for all reaches")

    # 4. No NULL node_order
    null_order = con.execute(
        "SELECT COUNT(*) FROM nodes WHERE node_order IS NULL"
    ).fetchone()[0]
    if null_order > 0:
        print(f"FAIL: {null_order} nodes have NULL node_order")
        ok = False
    else:
        print("PASS: all nodes have node_order")

    # 5. max(node_order) per reach == n_nodes
    mismatch = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT n.reach_id, MAX(n.node_order) AS max_order, r.n_nodes
            FROM nodes n
            JOIN reaches r ON r.reach_id = n.reach_id
            GROUP BY n.reach_id, r.n_nodes
            HAVING MAX(n.node_order) != r.n_nodes
        )
    """).fetchone()[0]
    if mismatch > 0:
        print(f"FAIL: {mismatch} reaches have max(node_order) != n_nodes")
        ok = False
    else:
        print("PASS: max(node_order) == n_nodes for all reaches")

    # 6. node_order starts at 1 for every reach
    bad_start = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT reach_id, MIN(node_order) AS min_order
            FROM nodes GROUP BY reach_id
            HAVING MIN(node_order) != 1
        )
    """).fetchone()[0]
    if bad_start > 0:
        print(f"FAIL: {bad_start} reaches don't start node_order at 1")
        ok = False
    else:
        print("PASS: node_order starts at 1 for all reaches")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add dn_node_id, up_node_id, node_order columns"
    )
    parser.add_argument("--db", required=True, help="Path to DuckDB file")
    parser.add_argument(
        "--verify-only", action="store_true", help="Only run verification"
    )
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    try:
        if args.verify_only:
            ok = verify(con)
            sys.exit(0 if ok else 1)

        add_columns(con)
        indexes = drop_rtree_indexes(con)
        populate_reach_boundary_nodes(con)
        populate_node_order(con)
        recreate_rtree_indexes(con, indexes)

        print("\n--- Verification ---")
        ok = verify(con)
        if not ok:
            print("\nVerification failed!")
            sys.exit(1)
        print("\nDone.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
