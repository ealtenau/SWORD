#!/usr/bin/env python3
"""Fix dist_out monotonicity violations at bifurcations.

243 bifurcation reaches in v17c have dist_out computed with MIN instead of
MAX of downstream neighbors. This is a v17b bug — UNC's dist_out_from_topo.py
intends max(dn_dist) but its retry mechanism settles for the only computed
downstream value when the larger branch hasn't been processed yet.

Algorithm:
    1. Find violations: reaches where ANY downstream neighbor has higher dist_out
    2. Fix: dist_out = reach_length + MAX(downstream dist_outs)
    3. Cascade upstream via BFS, recomputing each ancestor
    4. Shift node dist_out by the same delta as the parent reach
    5. Bulk UPDATE with RTREE drop/recreate pattern

Usage:
    python scripts/topology/fix_dist_out_bifurcations.py --db data/duckdb/sword_v17c.duckdb
    python scripts/topology/fix_dist_out_bifurcations.py --db data/duckdb/sword_v17c.duckdb --dry-run
"""

from __future__ import annotations

import argparse
import sys
from collections import deque

import duckdb
import pandas as pd


def find_violations(con: duckdb.DuckDBPyConnection) -> list[int]:
    """Find reaches where ANY downstream neighbor has higher dist_out."""
    query = """
    SELECT DISTINCT r1.reach_id
    FROM reaches r1
    JOIN reach_topology rt
        ON r1.reach_id = rt.reach_id AND r1.region = rt.region
    JOIN reaches r2
        ON rt.neighbor_reach_id = r2.reach_id AND rt.region = r2.region
    WHERE rt.direction = 'down'
        AND r1.dist_out > 0 AND r1.dist_out != -9999
        AND r2.dist_out > 0 AND r2.dist_out != -9999
        AND r2.dist_out > r1.dist_out
    ORDER BY r1.reach_id
    """
    return [row[0] for row in con.execute(query).fetchall()]


def build_topology_maps(
    con: duckdb.DuckDBPyConnection,
) -> tuple[
    dict[int, list[int]], dict[int, list[int]], dict[int, float], dict[int, float]
]:
    """Build upstream/downstream maps and reach attribute lookups.

    Returns:
        upstream_map: reach_id -> list of upstream reach_ids
        downstream_map: reach_id -> list of downstream reach_ids
        reach_lengths: reach_id -> reach_length
        current_dist_out: reach_id -> dist_out
    """
    # Topology
    topo = con.execute(
        "SELECT reach_id, neighbor_reach_id, direction FROM reach_topology"
    ).fetchall()

    upstream_map: dict[int, list[int]] = {}
    downstream_map: dict[int, list[int]] = {}
    for reach_id, neighbor_id, direction in topo:
        if direction == "up":
            upstream_map.setdefault(reach_id, []).append(neighbor_id)
        else:
            downstream_map.setdefault(reach_id, []).append(neighbor_id)

    # Reach attributes
    attrs = con.execute(
        "SELECT reach_id, reach_length, dist_out FROM reaches"
    ).fetchall()
    reach_lengths = {r[0]: r[1] for r in attrs}
    current_dist_out = {r[0]: r[2] for r in attrs}

    return upstream_map, downstream_map, reach_lengths, current_dist_out


def compute_fixes(
    violations: list[int],
    upstream_map: dict[int, list[int]],
    downstream_map: dict[int, list[int]],
    reach_lengths: dict[int, float],
    current_dist_out: dict[int, float],
) -> dict[int, float]:
    """Fix violations and cascade upstream via BFS.

    Returns:
        updated: reach_id -> new_dist_out for all affected reaches
    """
    updated: dict[int, float] = {}
    queue: deque[int] = deque(violations)
    visited: set[int] = set()

    while queue:
        rid = queue.popleft()
        if rid in visited:
            continue
        visited.add(rid)

        dn_ids = downstream_map.get(rid, [])
        if not dn_ids:
            # Outlet — dist_out = reach_length (shouldn't be a violation, but safe)
            new_do = reach_lengths.get(rid, 0.0)
        else:
            # Use already-updated values where available
            dn_dists = []
            for d in dn_ids:
                do = updated.get(d, current_dist_out.get(d, -9999))
                if do > 0 and do != -9999:
                    dn_dists.append(do)
            if not dn_dists:
                continue
            new_do = reach_lengths.get(rid, 0.0) + max(dn_dists)

        old_do = current_dist_out.get(rid, -9999)
        if old_do <= 0 or old_do == -9999:
            continue

        # Only update if value actually changes (within 0.001m tolerance)
        if abs(new_do - old_do) < 0.001:
            continue

        updated[rid] = new_do

        # Enqueue upstream neighbors for cascade
        for up_id in upstream_map.get(rid, []):
            if up_id not in visited:
                queue.append(up_id)

    return updated


def apply_updates(
    con: duckdb.DuckDBPyConnection,
    reach_updates: dict[int, float],
    current_dist_out: dict[int, float],
) -> None:
    """Write reach and node dist_out updates to DuckDB.

    Uses RTREE drop/recreate pattern and delta shift for nodes.
    """
    if not reach_updates:
        print("No updates to apply.")
        return

    # Build DataFrames for bulk update
    reach_df = pd.DataFrame(
        [
            {"reach_id": rid, "new_dist_out": new_do}
            for rid, new_do in reach_updates.items()
        ]
    )

    # Node updates: shift by delta = new_dist_out - old_dist_out
    node_delta_df = pd.DataFrame(
        [
            {
                "reach_id": rid,
                "delta": new_do - current_dist_out[rid],
            }
            for rid, new_do in reach_updates.items()
        ]
    )

    # Load spatial extension and drop RTREE indexes
    con.execute("INSTALL spatial; LOAD spatial;")
    rtree_indexes = con.execute(
        "SELECT index_name, table_name, sql FROM duckdb_indexes() "
        "WHERE sql LIKE '%RTREE%'"
    ).fetchall()
    for idx_name, _tbl, _sql in rtree_indexes:
        con.execute(f'DROP INDEX "{idx_name}"')

    try:
        # Update reaches
        con.register("reach_fix", reach_df)
        try:
            con.execute("""
                UPDATE reaches SET dist_out = rf.new_dist_out
                FROM reach_fix rf
                WHERE reaches.reach_id = rf.reach_id
            """)
        finally:
            con.unregister("reach_fix")

        # Update nodes via delta shift
        con.register("node_delta", node_delta_df)
        try:
            con.execute("""
                UPDATE nodes SET dist_out = nodes.dist_out + nd.delta
                FROM node_delta nd
                WHERE nodes.reach_id = nd.reach_id
                    AND nodes.dist_out > 0
                    AND nodes.dist_out != -9999
            """)
        finally:
            con.unregister("node_delta")

    finally:
        # Always recreate RTREE indexes
        for _idx_name, _tbl, sql in rtree_indexes:
            con.execute(sql)

    print(f"Updated {len(reach_updates):,} reaches and their nodes.")


def verify(con: duckdb.DuckDBPyConnection) -> int:
    """Verify no remaining violations. Returns count."""
    remaining = con.execute("""
        SELECT COUNT(DISTINCT r1.reach_id)
        FROM reaches r1
        JOIN reach_topology rt
            ON r1.reach_id = rt.reach_id AND r1.region = rt.region
        JOIN reaches r2
            ON rt.neighbor_reach_id = r2.reach_id AND rt.region = r2.region
        WHERE rt.direction = 'down'
            AND r1.dist_out > 0 AND r1.dist_out != -9999
            AND r2.dist_out > 0 AND r2.dist_out != -9999
            AND r2.dist_out > r1.dist_out
    """).fetchone()[0]
    return remaining


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix dist_out monotonicity violations at bifurcations"
    )
    parser.add_argument("--db", required=True, help="Path to sword_v17c.duckdb")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report violations without modifying the database",
    )
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    try:
        # Step 1: Find violations
        violations = find_violations(con)
        print(f"Found {len(violations)} violation reaches.")
        if not violations:
            print("No violations to fix.")
            return

        # Step 2: Build topology
        print("Building topology maps...")
        upstream_map, downstream_map, reach_lengths, current_dist_out = (
            build_topology_maps(con)
        )

        # Step 3: Compute fixes + cascade
        print("Computing fixes and cascading upstream...")
        reach_updates = compute_fixes(
            violations, upstream_map, downstream_map, reach_lengths, current_dist_out
        )
        print(
            f"  {len(violations)} direct violations -> "
            f"{len(reach_updates):,} total reaches affected (incl. upstream cascade)."
        )

        if args.dry_run:
            print("\n[DRY RUN] No changes written.")
            # Show sample corrections
            for rid in sorted(reach_updates)[:10]:
                old = current_dist_out[rid]
                new = reach_updates[rid]
                print(f"  reach {rid}: {old:.1f} -> {new:.1f} (delta={new - old:+.1f})")
            if len(reach_updates) > 10:
                print(f"  ... and {len(reach_updates) - 10:,} more")
            return

        # Step 4: Apply updates
        apply_updates(con, reach_updates, current_dist_out)

        # Step 5: Verify
        remaining = verify(con)
        if remaining == 0:
            print("Verification passed: 0 remaining violations.")
        else:
            print(f"WARNING: {remaining} violations remain after fix!", file=sys.stderr)
            sys.exit(1)

    finally:
        con.close()


if __name__ == "__main__":
    main()
