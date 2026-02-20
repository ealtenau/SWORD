# -*- coding: utf-8 -*-
"""
Facc Conservation — Unified Single Pass
========================================

Replaces P1/P2/P3 with one clean topological-order walk that enforces
all conservation rules simultaneously, starting from **v17b original**
facc values (not corrected values from previous passes).

Rules (applied in topo order, headwater → outlet):
1. **Headwater**: keep original MERIT facc.
2. **Single-upstream, parent has n_dn=1**: child >= parent (monotonicity).
3. **Bifurcation child** (parent has n_dn>=2): width-proportional share.
   - If original > share * threshold: lower to share (D8 surplus).
   - Else: raise to share (D8 deficit).
4. **Junction** (n_up>=2): facc >= sum(upstream corrected).

Properties:
- Consistent: no junction deficits, no single-dn drops.
- Mass-conservative at bifurcations: children split parent by width.
- Single pass, no fighting between correction stages.
- Rollback CSV records v17b original for every modified reach.
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
INFLATE_THRESHOLD = 1.5


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


def _build_graph(
    topology_df: pd.DataFrame, reaches_df: pd.DataFrame
) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in reaches_df.iterrows():
        rid = int(row["reach_id"])
        facc = float(row["facc"]) if pd.notna(row["facc"]) else 0.0
        width = float(row["width"]) if pd.notna(row["width"]) else 0.0
        n_dn = int(row["n_rch_down"]) if pd.notna(row["n_rch_down"]) else 0
        G.add_node(rid, facc=facc, width=max(width, 1.0), n_rch_down=n_dn)

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


def _run_unified(
    G: nx.DiGraph,
    original_facc: Dict[int, float],
    inflate_threshold: float = INFLATE_THRESHOLD,
) -> Dict[int, Tuple[float, float, str]]:
    """
    Single topo-order pass enforcing all conservation rules.

    Parameters
    ----------
    G : DiGraph with node attrs (width, n_rch_down)
    original_facc : v17b facc values (the baseline)
    inflate_threshold : ratio above which bifurc child is considered surplus

    Returns
    -------
    dict: reach_id → (original_facc, corrected_facc, correction_type)
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

    # Working facc
    corrected: Dict[int, float] = {}
    for node in G.nodes():
        corrected[node] = max(original_facc.get(node, 0.0), 0.0)

    changes: Dict[int, Tuple[float, float, str]] = {}
    counts = {"headwater": 0, "single_dn_floor": 0, "bifurc_share": 0,
              "junction_floor": 0}

    for node in topo_order:
        orig = max(original_facc.get(node, 0.0), 0.0)
        predecessors = list(G.predecessors(node))

        if not predecessors:
            # Headwater: keep original
            corrected[node] = orig
            continue

        # Compute expected facc from upstream
        contributions = []
        correction_type = None

        for pred in predecessors:
            pred_facc = corrected[pred]
            pred_out_degree = G.out_degree(pred)

            if pred_out_degree >= 2:
                # This node is a bifurcation child — get width share
                share = bifurc_share.get((pred, node), 1.0 / pred_out_degree)
                contributions.append(pred_facc * share)
            else:
                # Single downstream — full inheritance
                contributions.append(pred_facc)

        expected = sum(contributions)

        if len(predecessors) == 1:
            pred = predecessors[0]
            pred_out_degree = G.out_degree(pred)

            if pred_out_degree >= 2:
                # Bifurcation child
                if orig > expected * inflate_threshold and orig > expected + 100:
                    # D8 surplus — lower to share
                    corrected[node] = expected
                    correction_type = "bifurc_surplus"
                else:
                    # Raise to at least the share
                    corrected[node] = max(orig, expected)
                    if corrected[node] > orig:
                        correction_type = "bifurc_share"
            else:
                # Single-downstream parent: monotonicity floor
                corrected[node] = max(orig, expected)
                if corrected[node] > orig:
                    correction_type = "single_dn_floor"
        else:
            # Junction: facc >= sum(upstream)
            corrected[node] = max(orig, expected)
            if corrected[node] > orig:
                correction_type = "junction_floor"

        # Record if changed
        if correction_type and abs(corrected[node] - orig) > 0.01:
            changes[node] = (orig, corrected[node], correction_type)
            counts[correction_type] = counts.get(correction_type, 0) + 1

    for ctype, n in sorted(counts.items()):
        if n > 0:
            print(f"    {ctype:25s} {n:>6,}")

    return changes


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

    conn.execute("DROP TABLE IF EXISTS _unified_facc")
    conn.execute(
        "CREATE TEMP TABLE _unified_facc ("
        "  reach_id BIGINT PRIMARY KEY, new_facc DOUBLE)"
    )
    data = list(zip(
        corrections_df["reach_id"].astype(int),
        corrections_df["corrected_facc"].astype(float),
    ))
    conn.executemany(
        "INSERT INTO _unified_facc VALUES (?, ?)", data
    )
    conn.execute(
        "UPDATE reaches SET facc = t.new_facc "
        "FROM _unified_facc t WHERE reaches.reach_id = t.reach_id"
    )
    n = len(data)
    conn.execute("DROP TABLE IF EXISTS _unified_facc")

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

    conn.execute("DROP TABLE IF EXISTS _tag_ids")
    conn.execute("CREATE TEMP TABLE _tag_ids (reach_id BIGINT PRIMARY KEY)")
    for rid in corrections_df["reach_id"].astype(int):
        conn.execute("INSERT INTO _tag_ids VALUES (?)", [int(rid)])

    if "edit_flag" in cols:
        conn.execute(
            "UPDATE reaches SET edit_flag = 'facc_conservation_unified' "
            "FROM _tag_ids t WHERE reaches.reach_id = t.reach_id"
        )
    if "facc_quality" in cols:
        conn.execute(
            "UPDATE reaches SET facc_quality = 'topology_derived' "
            "FROM _tag_ids t WHERE reaches.reach_id = t.reach_id"
        )
    conn.execute("DROP TABLE IF EXISTS _tag_ids")

    return n


def correct_facc_unified(
    db_path: str,
    v17b_path: str,
    region: str,
    dry_run: bool = True,
    output_dir: str = "output/facc_detection",
    inflate_threshold: float = INFLATE_THRESHOLD,
) -> pd.DataFrame:
    region = region.upper()
    out_path = Path(output_dir)
    mode_str = "DRY RUN" if dry_run else "APPLYING TO DB"

    print(f"\n{'='*60}")
    print(f"Facc Conservation — Unified Pass — {region} [{mode_str}]")
    print(f"  inflate_threshold = {inflate_threshold}")
    print(f"{'='*60}")

    # Read v17b original facc as baseline
    print("  Loading v17b original facc...")
    v17b_conn = duckdb.connect(v17b_path, read_only=True)
    v17b_df = v17b_conn.execute(
        "SELECT reach_id, facc FROM reaches WHERE region = ?",
        [region],
    ).fetchdf()
    v17b_conn.close()
    original_facc = dict(zip(
        v17b_df["reach_id"].astype(int),
        v17b_df["facc"].astype(float),
    ))
    print(f"    {len(original_facc)} v17b reaches")

    # Read topology + reach attrs from v17c (topology is same, but width etc.)
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
        print(f"    {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"{n_bifurc} bifurcations")

        print("  Running unified pass...")
        changes = _run_unified(G, original_facc, inflate_threshold)
        print(f"    Total changes: {len(changes)}")

        if len(changes) == 0:
            print("  No changes needed")
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

        total_before = float(v17b_df["facc"].clip(lower=0).sum())
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

        if not dry_run:
            # First restore v17b facc for ALL reaches in this region,
            # then apply our corrections
            print("\n  Restoring v17b baseline...")
            _restore_v17b(conn, v17b_df)
            print("  Applying unified corrections...")
            _apply_to_db(conn, corrections_df)
            print("  Done.")

        # Save outputs
        out_path.mkdir(parents=True, exist_ok=True)
        csv_path = out_path / f"facc_conservation_unified_{region}.csv"
        corrections_df.to_csv(csv_path, index=False)
        print(f"  Saved CSV: {csv_path} ({len(corrections_df)} rows)")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "region": region,
            "dry_run": dry_run,
            "pass": "unified",
            "inflate_threshold": inflate_threshold,
            "total_reaches": len(reaches_df),
            "bifurcations": n_bifurc,
            "corrections": len(corrections_df),
            "raised": int(n_raised),
            "lowered": int(n_lowered),
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
        summary_path = out_path / f"facc_conservation_unified_summary_{region}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved summary: {summary_path}")

        return corrections_df

    finally:
        conn.close()


def _restore_v17b(
    conn: duckdb.DuckDBPyConnection,
    v17b_df: pd.DataFrame,
) -> None:
    """Restore all v17b facc values for a region before applying unified."""
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
    data = list(zip(
        v17b_df["reach_id"].astype(int),
        v17b_df["facc"].astype(float),
    ))
    conn.executemany("INSERT INTO _v17b_restore VALUES (?, ?)", data)
    conn.execute(
        "UPDATE reaches SET facc = t.orig_facc "
        "FROM _v17b_restore t WHERE reaches.reach_id = t.reach_id"
    )
    conn.execute("DROP TABLE IF EXISTS _v17b_restore")

    for idx_name, tbl, sql in indexes:
        conn.execute(sql)


def main():
    parser = argparse.ArgumentParser(
        description="Facc conservation — unified single pass"
    )
    parser.add_argument("--db", required=True,
                        help="v17c DuckDB path")
    parser.add_argument("--v17b", default="data/duckdb/sword_v17b.duckdb",
                        help="v17b DuckDB path (baseline)")
    parser.add_argument("--region", help="Single region")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output-dir", default="output/facc_detection")
    parser.add_argument("--threshold", type=float, default=INFLATE_THRESHOLD)

    args = parser.parse_args()
    if not args.region and not args.all:
        parser.error("Specify --region or --all")

    regions = REGIONS if args.all else [args.region.upper()]
    all_corrections = []
    for region in regions:
        df = correct_facc_unified(
            db_path=args.db,
            v17b_path=args.v17b,
            region=region,
            dry_run=not args.apply,
            output_dir=args.output_dir,
            inflate_threshold=args.threshold,
        )
        if len(df) > 0:
            all_corrections.append(df)

    if all_corrections:
        combined = pd.concat(all_corrections, ignore_index=True)
        n_up = (combined["delta"] > 0).sum()
        n_dn = (combined["delta"] < 0).sum()
        print(f"\n{'='*60}")
        print(f"GRAND TOTAL: {len(combined)} modifications "
              f"({n_up} raised, {n_dn} lowered)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
