# -*- coding: utf-8 -*-
"""
Facc Conservation Correction — Pass 2 (Junction Floor)
=======================================================

Eliminates remaining F006 violations left after Pass 1 by enforcing
``facc >= sum(upstream_facc)`` at every junction, cascading downstream.

Pass 1 propagated original headwater values with equal 1/n splits —
conservative but leaves asymmetric junctions under-counted.  Pass 2
reads the current (post-P1) facc values and enforces the hard
conservation floor in topological order.

**Key properties:**
- Only raises facc, never lowers.
- Cascading: if a raise propagates, downstream reaches are also fixed.
- Should eliminate 100% of remaining F006 violations.
- Deterministic, no ML.

**Rollback:** The output CSV records original_facc (pre-P2) for every
corrected reach.  To roll back, restore facc from the original_facc
column.  To roll back both passes, use the Pass 1 CSV's original_facc
for P1-corrected reaches and this CSV for P2-only reaches.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Tuple

import duckdb
import networkx as nx
import pandas as pd

from .correct_topological import apply_corrections_to_db

REGIONS = ["NA", "SA", "EU", "AF", "AS", "OC"]


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------


def _load_topology(conn: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT reach_id, direction, neighbor_rank, neighbor_reach_id
        FROM reach_topology
        WHERE region = ?
        """,
        [region.upper()],
    ).fetchdf()


def _load_reaches(conn: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT reach_id, region, facc, width, n_rch_up, n_rch_down,
               end_reach, lakeflag
        FROM reaches
        WHERE region = ?
        """,
        [region.upper()],
    ).fetchdf()


# ------------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------------


def _build_graph(
    topology_df: pd.DataFrame,
    reaches_df: pd.DataFrame,
) -> nx.DiGraph:
    """Build DiGraph where edges follow flow (u → v = upstream → downstream)."""
    G = nx.DiGraph()

    for _, row in reaches_df.iterrows():
        rid = int(row["reach_id"])
        G.add_node(
            rid,
            facc=float(row["facc"]) if pd.notna(row["facc"]) else 0.0,
            width=float(row["width"]) if pd.notna(row["width"]) else 0.0,
        )

    edges_added: Set[Tuple[int, int]] = set()
    for _, row in topology_df.iterrows():
        reach_id = int(row["reach_id"])
        neighbor_id = int(row["neighbor_reach_id"])
        direction = row["direction"]

        if direction == "up":
            u, v = neighbor_id, reach_id
        else:
            u, v = reach_id, neighbor_id

        if (u, v) not in edges_added:
            edges_added.add((u, v))
            G.add_edge(u, v)

    return G


# ------------------------------------------------------------------
# Core algorithm — cascading conservation floor
# ------------------------------------------------------------------


def _run_conservation_pass2(
    G: nx.DiGraph,
) -> Dict[int, Tuple[float, float]]:
    """
    Fix remaining F006 violations at junctions only (no cascade).

    At each junction (≥2 upstream), enforces facc >= sum(upstream_facc)
    using current (post-P1) DB values.  Single-upstream reaches are
    NOT cascaded — this avoids the double-counting inflation that
    occurs when bifurcation children carrying full parent facc rejoin.

    Returns dict  reach_id → (original_facc, corrected_facc)
    for reaches where original < sum(upstream).
    """
    changes: Dict[int, Tuple[float, float]] = {}

    for node in G.nodes():
        original = G.nodes[node].get("facc", 0.0)
        if original < 0 or original == -9999:
            original = 0.0

        predecessors = list(G.predecessors(node))
        if len(predecessors) < 2:
            continue

        # Junction: enforce facc >= sum(upstream_facc)
        upstream_sum = sum(max(G.nodes[p].get("facc", 0.0), 0.0) for p in predecessors)

        if upstream_sum > original and abs(upstream_sum - original) > 0.01:
            changes[node] = (original, upstream_sum)

    return changes


# ------------------------------------------------------------------
# Apply to DB (RTREE-safe)
# ------------------------------------------------------------------


def _apply_to_db(
    conn: duckdb.DuckDBPyConnection,
    corrections_df: pd.DataFrame,
) -> int:
    """Write corrected facc + tag edit_flag and facc_quality."""
    if len(corrections_df) == 0:
        return 0

    facc_df = corrections_df[["reach_id", "corrected_facc"]].copy()
    facc_df = facc_df.rename(columns={"corrected_facc": "predicted_facc"})
    n = apply_corrections_to_db(conn, facc_df, "facc_conservation_pass2")

    cols = {
        r[0].lower()
        for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'reaches'"
        ).fetchall()
    }

    ids_list = corrections_df["reach_id"].tolist()
    if not ids_list:
        return n

    conn.execute("DROP TABLE IF EXISTS _temp_tag_ids")
    conn.execute("CREATE TEMP TABLE _temp_tag_ids (reach_id BIGINT PRIMARY KEY)")
    for rid in ids_list:
        conn.execute("INSERT INTO _temp_tag_ids VALUES (?)", [int(rid)])

    if "edit_flag" in cols:
        conn.execute("""
            UPDATE reaches SET edit_flag = 'facc_conservation_p2'
            FROM _temp_tag_ids t WHERE reaches.reach_id = t.reach_id
        """)

    if "facc_quality" in cols:
        conn.execute("""
            UPDATE reaches SET facc_quality = 'conservation_corrected_p2'
            FROM _temp_tag_ids t WHERE reaches.reach_id = t.reach_id
        """)

    conn.execute("DROP TABLE IF EXISTS _temp_tag_ids")
    return n


# ------------------------------------------------------------------
# Output files
# ------------------------------------------------------------------


def _save_outputs(
    corrections_df: pd.DataFrame,
    region: str,
    output_dir: Path,
    summary_stats: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"facc_conservation_p2_{region}.csv"
    corrections_df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}  ({len(corrections_df)} rows)")

    summary_path = output_dir / f"facc_conservation_p2_summary_{region}.json"
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"  Saved summary: {summary_path}")


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def correct_facc_conservation_p2(
    db_path: str,
    region: str,
    dry_run: bool = True,
    output_dir: str = "output/facc_detection",
) -> pd.DataFrame:
    """
    Enforce facc conservation for a single region (Pass 2).

    Reads current (post-P1) facc values and enforces
    facc >= sum(upstream_facc) at every junction, cascading.

    Parameters
    ----------
    db_path : str
        Path to DuckDB database (v17c, post-Pass-1).
    region : str
        Region code (NA, SA, EU, AF, AS, OC).
    dry_run : bool
        If True (default), don't write to DB.
    output_dir : str
        Directory for output files.

    Returns
    -------
    pd.DataFrame
        Corrections with reach_id, original_facc, corrected_facc, delta, etc.
    """
    region = region.upper()
    out_path = Path(output_dir)
    mode_str = "DRY RUN" if dry_run else "APPLYING TO DB"

    print(f"\n{'=' * 60}")
    print(f"Facc Conservation Pass 2 — {region} [{mode_str}]")
    print(f"{'=' * 60}")

    conn = duckdb.connect(db_path, read_only=dry_run)
    try:
        print("  Loading topology...")
        topo_df = _load_topology(conn, region)
        print(f"    {len(topo_df)} topology rows")

        print("  Loading reaches...")
        reaches_df = _load_reaches(conn, region)
        print(f"    {len(reaches_df)} reaches")

        if len(reaches_df) == 0:
            print("  No reaches found — skipping")
            return pd.DataFrame()

        print("  Building graph...")
        G = _build_graph(topo_df, reaches_df)
        print(f"    {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        print("  Running conservation pass 2 (junction floor, cascading)...")
        changes = _run_conservation_pass2(G)
        print(f"    {len(changes)} reaches corrected")

        if len(changes) == 0:
            print("  No corrections needed")
            summary = {
                "timestamp": datetime.now().isoformat(),
                "region": region,
                "dry_run": dry_run,
                "pass": 2,
                "total_reaches": len(reaches_df),
                "corrections": 0,
                "db_path": str(db_path),
            }
            _save_outputs(pd.DataFrame(), region, out_path, summary)
            return pd.DataFrame()

        # Build corrections DataFrame
        rows = []
        for rid, (orig, corr) in changes.items():
            delta = corr - orig
            delta_pct = 100.0 * delta / orig if orig > 0 else float("inf")
            n_up = G.in_degree(rid)
            rows.append(
                {
                    "reach_id": rid,
                    "region": region,
                    "original_facc": round(orig, 4),
                    "corrected_facc": round(corr, 4),
                    "delta": round(delta, 4),
                    "delta_pct": round(delta_pct, 2),
                    "n_rch_up": n_up,
                    "correction_type": "junction_floor" if n_up > 1 else "cascade",
                }
            )
        corrections_df = pd.DataFrame(rows)

        # Stats
        total_facc_before = float(reaches_df["facc"].clip(lower=0).sum())
        total_facc_delta = float(corrections_df["delta"].sum())
        pct_increase = (
            100.0 * total_facc_delta / total_facc_before if total_facc_before > 0 else 0
        )

        by_type = corrections_df.groupby("correction_type").agg(
            count=("reach_id", "size"),
            median_delta=("delta", "median"),
            median_delta_pct=("delta_pct", "median"),
        )

        print("\n  Summary:")
        print(f"    Total corrections:        {len(corrections_df)}")
        print(f"    Total facc before:        {total_facc_before:,.0f} km²")
        print(
            f"    Total facc increase:      {total_facc_delta:,.0f} km² ({pct_increase:.3f}%)"
        )
        print(
            f"    Median delta:             {corrections_df['delta'].median():,.1f} km²"
        )
        print(
            f"    Median delta pct:         {corrections_df['delta_pct'].median():.1f}%"
        )
        print("\n    By correction type:")
        for ctype, row in by_type.iterrows():
            print(
                f"      {ctype:25s}  n={int(row['count']):>6,}  "
                f"med_delta={row['median_delta']:>10,.1f} km²  "
                f"med_pct={row['median_delta_pct']:>8.1f}%"
            )

        if not dry_run:
            print("\n  Applying corrections to DB...")
            _apply_to_db(conn, corrections_df)
            print("  Done.")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "region": region,
            "dry_run": dry_run,
            "pass": 2,
            "total_reaches": len(reaches_df),
            "corrections": len(corrections_df),
            "total_facc_before": total_facc_before,
            "total_facc_increase": total_facc_delta,
            "pct_increase": round(pct_increase, 4),
            "by_correction_type": {
                ctype: {
                    "count": int(row["count"]),
                    "median_delta": round(float(row["median_delta"]), 2),
                    "median_delta_pct": round(float(row["median_delta_pct"]), 2),
                }
                for ctype, row in by_type.iterrows()
            },
            "db_path": str(db_path),
        }
        _save_outputs(corrections_df, region, out_path, summary)

        return corrections_df

    finally:
        conn.close()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Facc conservation correction — Pass 2 (junction floor, cascading)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Dry run for NA
  python -m src.sword_duckdb.facc_detection.correct_conservation_p2 \\
      --db data/duckdb/sword_v17c.duckdb --region NA

  # Apply to DB
  python -m src.sword_duckdb.facc_detection.correct_conservation_p2 \\
      --db data/duckdb/sword_v17c.duckdb --region NA --apply

  # All regions
  python -m src.sword_duckdb.facc_detection.correct_conservation_p2 \\
      --db data/duckdb/sword_v17c.duckdb --all --apply
        """,
    )
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--region", help="Region code (NA, SA, EU, AF, AS, OC)")
    parser.add_argument("--all", action="store_true", help="Process all regions")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write corrections to DB (default: dry run)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/facc_detection",
        help="Output directory",
    )

    args = parser.parse_args()

    if not args.region and not args.all:
        parser.error("Specify --region or --all")

    regions = REGIONS if args.all else [args.region.upper()]
    all_corrections = []

    for region in regions:
        df = correct_facc_conservation_p2(
            db_path=args.db,
            region=region,
            dry_run=not args.apply,
            output_dir=args.output_dir,
        )
        if len(df) > 0:
            all_corrections.append(df)

    if all_corrections:
        combined = pd.concat(all_corrections, ignore_index=True)
        print(f"\n{'=' * 60}")
        print(
            f"GRAND TOTAL: {len(combined)} corrections across {len(regions)} region(s)"
        )
        print(f"{'=' * 60}")
    else:
        print("\nNo corrections needed.")


if __name__ == "__main__":
    main()
