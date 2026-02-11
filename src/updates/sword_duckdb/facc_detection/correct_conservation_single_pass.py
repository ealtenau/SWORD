# -*- coding: utf-8 -*-
"""
Facc Conservation — Single-Pass Correction (v2: Asymmetric Propagation)
=======================================================================

Replaces the sequential P1/P2/P3 passes with a single topological-order walk
from headwaters to outlets, starting from **v17b original facc**.

Algorithm (headwater → outlet, in topological order):

    1. Headwater (no predecessors): keep v17b value
    2. Junction (2+ predecessors):
            sum(corrected upstream) + max(base - sum(baseline upstream), 0)
       The lateral term isolates real local drainage from D8 clone inflation.
    3. Bifurcation child (1 pred, parent out_degree >= 2):
            corrected_parent * (width_child / sum_sibling_widths)
    4. 1:1 link (1 pred, parent out_degree == 1):
            - Parent lowered (bifurc split):  corrected_parent + max(base - base_parent, 0)
            - Parent raised or unchanged:     base  (keep original D8 value)

Key design choice: **lowering propagates, raising does not.** Junction raises are
confined to the junction reach.  This prevents the exponential inflation that occurs
in multi-bifurcation deltas (e.g. Lena: 674x under v1 additive → 2.95x under v2).

See docs/facc_conservation_algorithm.md for full details.

Usage:
    # Dry run on NA
    python -m src.updates.sword_duckdb.facc_detection.correct_conservation_single_pass \
        --db data/duckdb/sword_v17c.duckdb --v17b data/duckdb/sword_v17b.duckdb --region NA

    # Apply to all regions
    python -m src.updates.sword_duckdb.facc_detection.correct_conservation_single_pass \
        --db data/duckdb/sword_v17c.duckdb --v17b data/duckdb/sword_v17b.duckdb --all --apply
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

REGIONS = ["NA", "SA", "EU", "AF", "AS", "OC"]


# ---------------------------------------------------------------------------
# Data loading (same pattern as existing passes)
# ---------------------------------------------------------------------------

def _load_topology(conn: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    return conn.execute(
        "SELECT reach_id, direction, neighbor_rank, neighbor_reach_id "
        "FROM reach_topology WHERE region = ?",
        [region.upper()],
    ).fetchdf()


def _load_reaches(conn: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    return conn.execute(
        "SELECT reach_id, region, facc, width, n_rch_up, n_rch_down, "
        "end_reach, lakeflag FROM reaches WHERE region = ?",
        [region.upper()],
    ).fetchdf()


def _load_v17b_facc(
    v17b_path: str, region: str
) -> Dict[int, float]:
    """Load v17b original facc as {reach_id: facc} dict."""
    conn = duckdb.connect(v17b_path, read_only=True)
    try:
        df = conn.execute(
            "SELECT reach_id, facc FROM reaches WHERE region = ?",
            [region.upper()],
        ).fetchdf()
    finally:
        conn.close()
    return dict(zip(df["reach_id"].astype(int), df["facc"].astype(float)))


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def _build_graph(
    topology_df: pd.DataFrame, reaches_df: pd.DataFrame
) -> nx.DiGraph:
    """Build DiGraph where edges follow flow (upstream → downstream)."""
    G = nx.DiGraph()
    for _, row in reaches_df.iterrows():
        rid = int(row["reach_id"])
        facc = float(row["facc"]) if pd.notna(row["facc"]) else 0.0
        width = float(row["width"]) if pd.notna(row["width"]) else 0.0
        G.add_node(rid, facc=facc, width=max(width, 0.0))

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


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _run_single_pass(
    G: nx.DiGraph,
    baseline: Dict[int, float],
) -> Dict[int, Tuple[float, float, str]]:
    """
    Single topological-order pass from headwaters to outlets.

    Parameters
    ----------
    G : DiGraph with node attrs (width)
    baseline : v17b original facc values

    Returns
    -------
    dict: reach_id → (baseline_facc, corrected_facc, correction_type)
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
        widths = []
        for c in succs:
            w = G.nodes[c].get("width", 0.0)
            widths.append(max(w, 0.0))
        total_w = sum(widths)
        if total_w <= 0:
            # Fallback: equal split
            for c in succs:
                bifurc_share[(node, c)] = 1.0 / len(succs)
        else:
            for c, w in zip(succs, widths):
                bifurc_share[(node, c)] = w / total_w

    # Working corrected values — start from baseline
    corrected: Dict[int, float] = {}
    for node in G.nodes():
        corrected[node] = max(baseline.get(node, 0.0), 0.0)

    changes: Dict[int, Tuple[float, float, str]] = {}
    counts: Dict[str, int] = {}

    for node in topo_order:
        base = max(baseline.get(node, 0.0), 0.0)
        preds = list(G.predecessors(node))

        if not preds:
            # Headwater: keep baseline
            corrected[node] = base
            continue

        if len(preds) >= 2:
            # JUNCTION: lateral-increment approach (generalised 1:1 rule).
            # corrected = sum(corrected upstream) + local lateral drainage
            # where lateral = max(base - sum(baseline upstream), 0).
            # This avoids recovering the inflated D8 clone at inner junctions
            # of nested bifurcation-junction pairs.
            floor = sum(corrected.get(p, 0.0) for p in preds)
            sum_base_up = sum(max(baseline.get(p, 0.0), 0.0) for p in preds)
            lateral = max(base - sum_base_up, 0.0)
            new_val = floor + lateral
            corrected[node] = new_val
            if abs(new_val - base) > 0.01:
                _record(changes, counts, node, base, new_val,
                        "junction_floor")
            continue

        # Single predecessor
        parent = preds[0]
        parent_out = G.out_degree(parent)

        if parent_out >= 2:
            # BIFURCATION CHILD: width-proportional share of corrected parent
            share = bifurc_share.get((parent, node), 1.0 / parent_out)
            new_val = corrected[parent] * share
            corrected[node] = new_val
            if abs(new_val - base) > 0.01:
                _record(changes, counts, node, base, new_val,
                        "bifurc_share")
        else:
            # 1:1 LINK — only propagate *lowering* (from bifurcation
            # splits).  Junction raises must NOT cascade downstream or
            # they compound exponentially in multi-bifurcation deltas.
            parent_base = max(baseline.get(parent, 0.0), 0.0)
            if parent_base == 0 and corrected[parent] == 0:
                corrected[node] = 0.0
                if base > 0.01:
                    _record(changes, counts, node, base, 0.0,
                            "cascade_zero")
            elif corrected[parent] < parent_base:
                # Parent was LOWERED (bifurcation split) → additive lateral
                # Take the parent's corrected value and add only the local
                # drainage difference.  The lateral is real watershed area
                # entering between parent and child — not scaled by the
                # upstream split ratio.
                lateral = max(base - parent_base, 0.0)
                new_val = corrected[parent] + lateral
                corrected[node] = new_val
                if abs(new_val - base) > 0.01:
                    _record(changes, counts, node, base, new_val,
                            "lateral_lower")
            else:
                # Parent raised or unchanged → keep original D8 value.
                # Equivalent to: base_parent + (base - base_parent) = base.
                # The lateral difference is added on the parent's baseline,
                # not on the inflated corrected value.
                corrected[node] = base

    for ctype, n in sorted(counts.items()):
        if n > 0:
            print(f"    {ctype:25s} {n:>6,}")

    return changes


def _record(
    changes: Dict[int, Tuple[float, float, str]],
    counts: Dict[str, int],
    node: int,
    orig: float,
    new_val: float,
    ctype: str,
) -> None:
    changes[node] = (orig, new_val, ctype)
    counts[ctype] = counts.get(ctype, 0) + 1


# ---------------------------------------------------------------------------
# DB application (RTREE-safe pattern)
# ---------------------------------------------------------------------------

def _apply_to_db(
    conn: duckdb.DuckDBPyConnection,
    corrections_df: pd.DataFrame,
) -> int:
    """Write corrected facc to DB using RTREE-safe pattern."""
    if len(corrections_df) == 0:
        return 0

    conn.execute("INSTALL spatial; LOAD spatial;")
    indexes = conn.execute(
        "SELECT index_name, table_name, sql FROM duckdb_indexes() "
        "WHERE sql LIKE '%RTREE%'"
    ).fetchall()
    for idx_name, tbl, sql in indexes:
        conn.execute(f'DROP INDEX IF EXISTS "{idx_name}"')

    conn.execute("DROP TABLE IF EXISTS _sp_facc")
    conn.execute(
        "CREATE TEMP TABLE _sp_facc ("
        "  reach_id BIGINT PRIMARY KEY, new_facc DOUBLE)"
    )
    data = list(zip(
        corrections_df["reach_id"].astype(int),
        corrections_df["corrected_facc"].astype(float),
    ))
    conn.executemany("INSERT INTO _sp_facc VALUES (?, ?)", data)
    conn.execute(
        "UPDATE reaches SET facc = t.new_facc "
        "FROM _sp_facc t WHERE reaches.reach_id = t.reach_id"
    )
    n = len(data)
    conn.execute("DROP TABLE IF EXISTS _sp_facc")

    for idx_name, tbl, sql in indexes:
        conn.execute(sql)

    # Tag modified reaches
    cols = {
        r[0].lower()
        for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'reaches'"
        ).fetchall()
    }

    conn.execute("DROP TABLE IF EXISTS _sp_tag_ids")
    conn.execute("CREATE TEMP TABLE _sp_tag_ids (reach_id BIGINT PRIMARY KEY)")
    for rid in corrections_df["reach_id"].astype(int):
        conn.execute("INSERT INTO _sp_tag_ids VALUES (?)", [int(rid)])

    if "edit_flag" in cols:
        conn.execute(
            "UPDATE reaches SET edit_flag = 'facc_conservation_single' "
            "FROM _sp_tag_ids t WHERE reaches.reach_id = t.reach_id"
        )
    if "facc_quality" in cols:
        conn.execute(
            "UPDATE reaches SET facc_quality = 'conservation_single_pass' "
            "FROM _sp_tag_ids t WHERE reaches.reach_id = t.reach_id"
        )
    conn.execute("DROP TABLE IF EXISTS _sp_tag_ids")

    return n


def _restore_v17b(
    conn: duckdb.DuckDBPyConnection,
    v17b_facc: Dict[int, float],
    region: str,
) -> None:
    """Restore v17b facc values for region before applying corrections."""
    conn.execute("INSTALL spatial; LOAD spatial;")
    indexes = conn.execute(
        "SELECT index_name, table_name, sql FROM duckdb_indexes() "
        "WHERE sql LIKE '%RTREE%'"
    ).fetchall()
    for idx_name, tbl, sql in indexes:
        conn.execute(f'DROP INDEX IF EXISTS "{idx_name}"')

    conn.execute("DROP TABLE IF EXISTS _v17b_restore")
    conn.execute(
        "CREATE TEMP TABLE _v17b_restore ("
        "  reach_id BIGINT PRIMARY KEY, orig_facc DOUBLE)"
    )
    data = [(int(rid), float(facc)) for rid, facc in v17b_facc.items()]
    conn.executemany("INSERT INTO _v17b_restore VALUES (?, ?)", data)
    conn.execute(
        "UPDATE reaches SET facc = t.orig_facc "
        "FROM _v17b_restore t WHERE reaches.reach_id = t.reach_id"
    )
    conn.execute("DROP TABLE IF EXISTS _v17b_restore")

    for idx_name, tbl, sql in indexes:
        conn.execute(sql)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def correct_facc_single_pass(
    db_path: str,
    v17b_path: str,
    region: str,
    dry_run: bool = True,
    output_dir: str = "output/facc_detection",
) -> pd.DataFrame:
    """
    Run single-pass facc conservation correction for one region.

    Parameters
    ----------
    db_path : path to v17c DuckDB
    v17b_path : path to v17b DuckDB (read-only baseline)
    region : region code (NA, SA, EU, AF, AS, OC)
    dry_run : if True, don't modify DB
    output_dir : where to write CSV + JSON
    """
    region = region.upper()
    out_path = Path(output_dir)
    mode_str = "DRY RUN" if dry_run else "APPLYING TO DB"

    print(f"\n{'='*60}")
    print(f"Facc Conservation — Single Pass — {region} [{mode_str}]")
    print(f"{'='*60}")

    # Step 1: Load v17b baseline
    print("  Loading v17b baseline...")
    v17b_facc = _load_v17b_facc(v17b_path, region)
    baseline = v17b_facc
    print(f"    {len(baseline)} v17b reaches")

    # Step 2: Load topology + reaches from v17c
    conn = duckdb.connect(db_path, read_only=dry_run)
    try:
        print("  Loading topology...")
        topo_df = _load_topology(conn, region)
        print("  Loading reaches...")
        reaches_df = _load_reaches(conn, region)
        print(f"    {len(reaches_df)} reaches")

        if len(reaches_df) == 0:
            return pd.DataFrame()

        # Step 4: Build graph
        print("  Building graph...")
        G = _build_graph(topo_df, reaches_df)
        n_bifurc = sum(1 for n in G.nodes() if G.out_degree(n) >= 2)
        print(f"    {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"{n_bifurc} bifurcations")

        # Step 5: Run single pass
        print("  Running single-pass conservation...")
        changes = _run_single_pass(G, baseline)
        print(f"    Total changes: {len(changes)}")

        if len(changes) == 0:
            print("  No changes needed")
            return pd.DataFrame()

        # Build corrections DataFrame
        rows = []
        for rid, (orig, corr, ctype) in changes.items():
            delta = corr - orig
            delta_pct = 100.0 * delta / orig if orig > 0 else (
                float("inf") if delta > 0 else 0.0
            )
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

        # Summary stats
        n_raised = int((corrections_df["delta"] > 0).sum())
        n_lowered = int((corrections_df["delta"] < 0).sum())

        total_before = sum(max(v, 0.0) for v in v17b_facc.values())
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
            print(f"      {ctype:25s}  n={int(row['count']):>6,}  "
                  f"med_delta={row['median_delta']:>+12,.1f} km²")

        # Apply to DB: restore v17b → apply conservation
        if not dry_run:
            print("\n  Restoring v17b baseline...")
            _restore_v17b(conn, v17b_facc, region)
            print("  Clearing old tags...")
            _clear_old_tags(conn, region)
            print("  Applying conservation corrections...")
            _apply_to_db(conn, corrections_df)
            print("  Done.")

        # Save outputs
        out_path.mkdir(parents=True, exist_ok=True)
        csv_path = out_path / f"facc_conservation_single_pass_{region}.csv"
        corrections_df.to_csv(csv_path, index=False)
        print(f"  Saved CSV: {csv_path} ({len(corrections_df)} rows)")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "region": region,
            "dry_run": dry_run,
            "pass": "single_pass",
            "total_reaches": len(reaches_df),
            "bifurcations": n_bifurc,
            "corrections": len(corrections_df),
            "raised": n_raised,
            "lowered": n_lowered,
            "total_facc_before": total_before,
            "net_facc_change": total_delta,
            "pct_change": round(pct, 4),
            "by_type": {
                ctype: {
                    "count": int(row["count"]),
                    "median_delta": round(float(row["median_delta"]), 2),
                }
                for ctype, row in by_type.iterrows()
            },
            "db_path": str(db_path),
            "v17b_path": str(v17b_path),
        }
        summary_path = out_path / f"facc_conservation_single_pass_summary_{region}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved summary: {summary_path}")

        return corrections_df

    finally:
        conn.close()


def _clear_old_tags(
    conn: duckdb.DuckDBPyConnection,
    region: str,
) -> None:
    """Clear P1/P2/P3 facc_quality and edit_flag tags for a region."""
    cols = {
        r[0].lower()
        for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'reaches'"
        ).fetchall()
    }
    old_quality = (
        "'conservation_corrected_p1','conservation_corrected_p2',"
        "'topology_derived','conservation_single_pass',"
        "'conservation_corrected_p3'"
    )
    old_edit = (
        "'facc_conservation_p1','facc_conservation_p2',"
        "'facc_conservation_p3','facc_conservation_single'"
    )
    if "facc_quality" in cols:
        n = conn.execute(
            f"UPDATE reaches SET facc_quality = NULL "
            f"WHERE region = ? AND facc_quality IN ({old_quality})",
            [region],
        ).fetchone()
        print(f"    Cleared facc_quality tags")
    if "edit_flag" in cols:
        n = conn.execute(
            f"UPDATE reaches SET edit_flag = NULL "
            f"WHERE region = ? AND edit_flag IN ({old_edit})",
            [region],
        ).fetchone()
        print(f"    Cleared edit_flag tags")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Facc conservation — single-pass correction"
    )
    parser.add_argument("--db", required=True,
                        help="v17c DuckDB path")
    parser.add_argument("--v17b", default="data/duckdb/sword_v17b.duckdb",
                        help="v17b DuckDB path (baseline)")
    parser.add_argument("--region", help="Single region")
    parser.add_argument("--all", action="store_true",
                        help="Process all regions")
    parser.add_argument("--apply", action="store_true",
                        help="Write corrections to DB (default: dry run)")
    parser.add_argument("--output-dir", default="output/facc_detection",
                        help="Output directory")

    args = parser.parse_args()
    if not args.region and not args.all:
        parser.error("Specify --region or --all")

    regions = REGIONS if args.all else [args.region.upper()]
    all_corrections = []
    for region in regions:
        df = correct_facc_single_pass(
            db_path=args.db,
            v17b_path=args.v17b,
            region=region,
            dry_run=not args.apply,
            output_dir=args.output_dir,
        )
        if len(df) > 0:
            all_corrections.append(df)

    if all_corrections:
        combined = pd.concat(all_corrections, ignore_index=True)
        n_up = int((combined["delta"] > 0).sum())
        n_dn = int((combined["delta"] < 0).sum())
        print(f"\n{'='*60}")
        print(f"GRAND TOTAL: {len(combined)} modifications "
              f"({n_up} raised, {n_dn} lowered)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
