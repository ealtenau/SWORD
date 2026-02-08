# -*- coding: utf-8 -*-
"""
Facc Conservation Correction — Pass 3 (Bifurcation Surplus Fix)
================================================================

Fixes the **opposite** problem from Passes 1–2: D8 sends full parent
facc to ALL children at bifurcations instead of splitting.  This creates
"surplus" reaches with inflated facc.

Algorithm
---------
1. Walk topological order (headwater → outlet).
2. At each bifurcation (out_degree ≥ 2), check each child:
   - Expected share = ``parent_corrected * (child_width / sum_sibling_widths)``
   - If ``child_facc > expected * inflate_threshold`` (default 1.5x):
     the child got the full parent facc via D8.  Replace with expected.
3. Cascade: once a child is lowered, use the lowered value when computing
   downstream expectations.  Stop cascading at the first reach where
   ``original_facc <= expected`` (original is trustworthy again).
4. After all surplus fixes, re-apply the junction floor (Pass 2 logic)
   to fix any new deficits created by the lowering.

Properties
----------
- Lowers inflated bifurcation children to width-proportional shares.
- Cascading stops when original values become trustworthy.
- Junction floor re-applied to maintain conservation.
- Targeted: only modifies reaches downstream of D8-inflated bifurcations.

Rollback
--------
Output CSV records original_facc for every modified reach.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Tuple

import duckdb
import networkx as nx
import numpy as np
import pandas as pd

from .correct_topological import apply_corrections_to_db

REGIONS = ["NA", "SA", "EU", "AF", "AS", "OC"]

# If child_facc > expected_share * this threshold, it's D8-inflated
INFLATE_THRESHOLD = 1.5


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
    G = nx.DiGraph()

    for _, row in reaches_df.iterrows():
        rid = int(row["reach_id"])
        facc = float(row["facc"]) if pd.notna(row["facc"]) else 0.0
        width = float(row["width"]) if pd.notna(row["width"]) else 0.0
        G.add_node(rid, facc=facc, width=max(width, 1.0))

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
# Core: bifurcation surplus detection + cascade
# ------------------------------------------------------------------

def _run_pass3(
    G: nx.DiGraph,
    inflate_threshold: float = INFLATE_THRESHOLD,
) -> Dict[int, Tuple[float, float, str]]:
    """
    Single topological-order walk that:

    1. At bifurcations: lower inflated children to width-proportional share.
    2. At single-upstream: cascade lowering if child inherited inflated value.
    3. At junctions: raise to sum(upstream) if deficit.

    Because we process in topo order and update ``corrected`` as we go,
    bifurcation fixes cascade naturally through the entire downstream
    distributary network.

    Returns dict  reach_id → (original_facc, corrected_facc, correction_type)
    for all modified reaches.
    """
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print("    WARNING: graph has cycles — using node list")
        topo_order = list(G.nodes())

    # Precompute width shares at bifurcations
    bifurc_share: Dict[Tuple[int, int], float] = {}
    for node in G.nodes():
        succs = list(G.successors(node))
        if len(succs) < 2:
            continue
        widths = [G.nodes[c]["width"] for c in succs]
        total_w = sum(widths)
        if total_w <= 0:
            for c in succs:
                bifurc_share[(node, c)] = 1.0 / len(succs)
        else:
            for c, w in zip(succs, widths):
                bifurc_share[(node, c)] = w / total_w

    # Working facc — starts as original
    corrected: Dict[int, float] = {}
    for node in G.nodes():
        orig = G.nodes[node].get("facc", 0.0)
        corrected[node] = max(orig, 0.0)

    changes: Dict[int, Tuple[float, float, str]] = {}
    n_bifurc = 0
    n_cascade = 0
    n_junction = 0

    for node in topo_order:
        original = G.nodes[node].get("facc", 0.0)
        if original < 0:
            original = 0.0

        predecessors = list(G.predecessors(node))

        if not predecessors:
            # Headwater — keep original
            continue

        if len(predecessors) == 1:
            parent = predecessors[0]
            parent_facc = corrected[parent]

            # Single upstream: cascade lowering if child looks inflated
            # relative to (already-corrected) parent
            if parent_facc > 0 and corrected[node] > parent_facc * inflate_threshold:
                # Child's facc is suspiciously high vs corrected parent.
                # Cap at parent value (local catchment is negligible at
                # these scales — a 10km reach adds ~50-100 km² max).
                corrected[node] = parent_facc
                if abs(corrected[node] - original) > 0.01:
                    changes[node] = (original, parent_facc, "cascade_surplus")
                    n_cascade += 1

        elif len(predecessors) >= 2:
            # Junction: enforce facc >= sum(upstream corrected)
            upstream_sum = sum(corrected.get(p, 0.0) for p in predecessors)
            if upstream_sum > corrected[node] and abs(upstream_sum - corrected[node]) > 0.01:
                corrected[node] = upstream_sum
                if abs(corrected[node] - original) > 0.01:
                    changes[node] = (original, upstream_sum, "junction_floor_p3")
                    n_junction += 1

        # Now check this node's children at bifurcations
        successors = list(G.successors(node))
        if len(successors) >= 2:
            parent_facc = corrected[node]
            if parent_facc <= 0:
                continue

            for child in successors:
                share = bifurc_share.get((node, child), 1.0 / len(successors))
                expected = parent_facc * share
                child_facc = corrected[child]

                if child_facc > expected * inflate_threshold and child_facc > expected + 100:
                    child_orig = G.nodes[child].get("facc", 0.0)
                    if child_orig < 0:
                        child_orig = 0.0
                    corrected[child] = expected
                    if abs(expected - child_orig) > 0.01:
                        changes[child] = (child_orig, expected, "bifurc_surplus")
                        n_bifurc += 1

    print(f"    Bifurcation surplus: {n_bifurc}")
    print(f"    Cascade surplus:     {n_cascade}")
    print(f"    Junction floor:      {n_junction}")

    return changes


# ------------------------------------------------------------------
# Apply to DB
# ------------------------------------------------------------------

def _apply_to_db(
    conn: duckdb.DuckDBPyConnection,
    corrections_df: pd.DataFrame,
) -> int:
    if len(corrections_df) == 0:
        return 0

    facc_df = corrections_df[["reach_id", "corrected_facc"]].copy()
    facc_df = facc_df.rename(columns={"corrected_facc": "predicted_facc"})
    n = apply_corrections_to_db(conn, facc_df, "facc_conservation_pass3")

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
            UPDATE reaches SET edit_flag = 'facc_conservation_p3'
            FROM _temp_tag_ids t WHERE reaches.reach_id = t.reach_id
        """)

    if "facc_quality" in cols:
        conn.execute("""
            UPDATE reaches SET facc_quality = 'topology_derived'
            FROM _temp_tag_ids t WHERE reaches.reach_id = t.reach_id
        """)

    conn.execute("DROP TABLE IF EXISTS _temp_tag_ids")
    return n


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

def _save_outputs(
    corrections_df: pd.DataFrame,
    region: str,
    output_dir: Path,
    summary_stats: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"facc_conservation_p3_{region}.csv"
    corrections_df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}  ({len(corrections_df)} rows)")

    summary_path = output_dir / f"facc_conservation_p3_summary_{region}.json"
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"  Saved summary: {summary_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def correct_facc_conservation_p3(
    db_path: str,
    region: str,
    dry_run: bool = True,
    output_dir: str = "output/facc_detection",
    inflate_threshold: float = INFLATE_THRESHOLD,
) -> pd.DataFrame:
    region = region.upper()
    out_path = Path(output_dir)
    mode_str = "DRY RUN" if dry_run else "APPLYING TO DB"

    print(f"\n{'='*60}")
    print(f"Facc Conservation Pass 3 — {region} [{mode_str}]")
    print(f"  inflate_threshold = {inflate_threshold}")
    print(f"{'='*60}")

    conn = duckdb.connect(db_path, read_only=dry_run)
    try:
        print("  Loading topology...")
        topo_df = _load_topology(conn, region)

        print("  Loading reaches...")
        reaches_df = _load_reaches(conn, region)
        print(f"    {len(reaches_df)} reaches")

        if len(reaches_df) == 0:
            return pd.DataFrame()

        print("  Building graph...")
        G = _build_graph(topo_df, reaches_df)
        n_bifurc = sum(1 for n in G.nodes() if G.out_degree(n) >= 2)
        print(f"    {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {n_bifurc} bifurcations")

        print("  Running Pass 3...")
        changes = _run_pass3(G, inflate_threshold)
        print(f"    Total changes: {len(changes)}")

        if len(changes) == 0:
            print("  No changes needed")
            _save_outputs(pd.DataFrame(), region, out_path, {
                "timestamp": datetime.now().isoformat(),
                "region": region, "dry_run": dry_run, "pass": 3,
                "total_reaches": len(reaches_df), "corrections": 0,
                "db_path": str(db_path),
            })
            return pd.DataFrame()

        rows = []
        for rid, (orig, corr, ctype) in changes.items():
            delta = corr - orig
            delta_pct = 100.0 * delta / orig if orig > 0 else float("inf")
            rows.append({
                "reach_id": rid,
                "region": region,
                "original_facc": round(orig, 4),
                "corrected_facc": round(corr, 4),
                "delta": round(delta, 4),
                "delta_pct": round(delta_pct, 2),
                "correction_type": ctype,
            })
        corrections_df = pd.DataFrame(rows)

        n_raised = (corrections_df["delta"] > 0).sum()
        n_lowered = (corrections_df["delta"] < 0).sum()

        total_before = float(reaches_df["facc"].clip(lower=0).sum())
        total_delta = float(corrections_df["delta"].sum())
        pct = 100.0 * total_delta / total_before if total_before > 0 else 0

        by_type = corrections_df.groupby("correction_type").agg(
            count=("reach_id", "size"),
            median_delta=("delta", "median"),
        )

        print(f"\n  Summary:")
        print(f"    Raised:  {n_raised}")
        print(f"    Lowered: {n_lowered}")
        print(f"    Net facc change: {total_delta:>+,.0f} km² ({pct:+.3f}%)")
        for ctype, row in by_type.iterrows():
            print(f"      {ctype:30s}  n={int(row['count']):>5,}  med_delta={row['median_delta']:>+12,.1f} km²")

        if not dry_run:
            print("\n  Applying to DB...")
            _apply_to_db(conn, corrections_df)
            print("  Done.")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "region": region, "dry_run": dry_run, "pass": 3,
            "inflate_threshold": inflate_threshold,
            "total_reaches": len(reaches_df), "bifurcations": n_bifurc,
            "corrections": len(corrections_df),
            "raised": int(n_raised), "lowered": int(n_lowered),
            "total_facc_before": total_before,
            "net_facc_change": total_delta,
            "pct_change": round(pct, 4),
            "by_type": {
                ctype: {"count": int(row["count"]), "median_delta": round(float(row["median_delta"]), 2)}
                for ctype, row in by_type.iterrows()
            },
            "db_path": str(db_path),
        }
        _save_outputs(corrections_df, region, out_path, summary)
        return corrections_df

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Facc conservation — Pass 3 (bifurcation surplus fix)",
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--region")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output-dir", default="output/facc_detection")
    parser.add_argument("--threshold", type=float, default=INFLATE_THRESHOLD,
                        help=f"Inflate threshold (default {INFLATE_THRESHOLD})")

    args = parser.parse_args()
    if not args.region and not args.all:
        parser.error("Specify --region or --all")

    regions = REGIONS if args.all else [args.region.upper()]
    all_corrections = []
    for region in regions:
        df = correct_facc_conservation_p3(
            db_path=args.db, region=region,
            dry_run=not args.apply, output_dir=args.output_dir,
            inflate_threshold=args.threshold,
        )
        if len(df) > 0:
            all_corrections.append(df)

    if all_corrections:
        combined = pd.concat(all_corrections, ignore_index=True)
        n_up = (combined["delta"] > 0).sum()
        n_dn = (combined["delta"] < 0).sum()
        print(f"\n{'='*60}")
        print(f"GRAND TOTAL: {len(combined)} modifications ({n_up} raised, {n_dn} lowered)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
