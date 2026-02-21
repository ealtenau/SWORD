# -*- coding: utf-8 -*-
"""
Facc Conservation Correction — Pass 1 (Equal-Split Propagation)
================================================================

Enforces topological consistency of flow accumulation (facc) by
propagating headwater facc values through the network topology:

- At **bifurcations** (≥2 downstream children): parent's facc is
  split equally (1/n) among children.
- At **junctions** (≥2 upstream parents): contributions are summed.
- Propagation uses ONLY original headwater values — no cascade.

Reaches whose original facc falls below the topology-derived
expectation are corrected upward.  Reaches already at or above
expectation keep their original value (surplus = local catchment).

**Key properties:**
- Only raises facc, never lowers.
- No cascade inflation (expected is computed from originals only).
- Mass-conservative by construction (splits recombine at junctions).
- Deterministic, no ML.

**Rollback:** The output CSV records original_facc for every corrected
reach.  To roll back, restore facc from the original_facc column.

Pass 1 fixes ~59% of F006 violations (conservation deficits).
Remaining violations at asymmetric junctions require Pass 2.
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
# Graph construction (upstream → downstream edges)
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
# Bifurcation share
# ------------------------------------------------------------------


def _get_share(G: nx.DiGraph, parent: int, child: int) -> float:
    """Share of parent's facc flowing into child (equal split at bifurcations)."""
    succs = list(G.successors(parent))
    if len(succs) <= 1:
        return 1.0
    return 1.0 / len(succs)


# ------------------------------------------------------------------
# Core algorithm
# ------------------------------------------------------------------


def _run_conservation_pass(
    G: nx.DiGraph,
) -> Dict[int, Tuple[float, float, str]]:
    """
    Single topological-order pass propagating headwater facc values.

    ``expected`` is computed from original headwater values only —
    no corrected values feed back in, so there is no cascade.

    Returns dict  reach_id → (original_facc, corrected_facc, correction_type)
    for reaches where original < expected.
    """
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print("    WARNING: graph has cycles — using node list")
        topo_order = list(G.nodes())

    expected: Dict[int, float] = {}
    changes: Dict[int, Tuple[float, float, str]] = {}

    for node in topo_order:
        original = G.nodes[node].get("facc", 0.0)
        if original < 0 or original == -9999:
            original = 0.0

        predecessors = list(G.predecessors(node))
        if not predecessors:
            expected[node] = original
            continue

        exp = sum(expected.get(p, 0.0) * _get_share(G, p, node) for p in predecessors)
        expected[node] = exp

        if exp > original and abs(exp - original) > 0.01:
            had_junction = len(predecessors) > 1
            ctype = "conservation_floor" if had_junction else "single_upstream"
            changes[node] = (original, exp, ctype)

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
    n = apply_corrections_to_db(conn, facc_df, "facc_conservation_pass1")

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
            UPDATE reaches SET edit_flag = 'facc_conservation_p1'
            FROM _temp_tag_ids t WHERE reaches.reach_id = t.reach_id
        """)

    if "facc_quality" in cols:
        conn.execute("""
            UPDATE reaches SET facc_quality = 'conservation_corrected_p1'
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

    csv_path = output_dir / f"facc_conservation_p1_{region}.csv"
    corrections_df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}  ({len(corrections_df)} rows)")

    try:
        import geopandas as gpd

        db_path = summary_stats.get("db_path")
        if db_path and len(corrections_df) > 0:
            conn = duckdb.connect(str(db_path), read_only=True)
            conn.execute("INSTALL spatial; LOAD spatial;")
            ids_str = ", ".join(str(int(r)) for r in corrections_df["reach_id"])
            geom_df = conn.execute(
                f"""
                SELECT reach_id, ST_AsText(geom) as wkt
                FROM reaches WHERE reach_id IN ({ids_str})
                """
            ).fetchdf()
            conn.close()

            merged = corrections_df.merge(geom_df, on="reach_id", how="left")
            from shapely import wkt as shapely_wkt

            merged["geometry"] = merged["wkt"].apply(
                lambda w: shapely_wkt.loads(w) if pd.notna(w) else None
            )
            merged = merged.drop(columns=["wkt"])
            gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
            geojson_path = output_dir / f"facc_conservation_p1_{region}.geojson"
            gdf.to_file(geojson_path, driver="GeoJSON")
            print(f"  Saved GeoJSON: {geojson_path}")
    except Exception as e:
        print(f"  Skipped GeoJSON: {e}")

    summary_path = output_dir / f"facc_conservation_p1_summary_{region}.json"
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"  Saved summary: {summary_path}")


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def correct_facc_conservation(
    db_path: str,
    region: str,
    dry_run: bool = True,
    output_dir: str = "output/facc_detection",
) -> pd.DataFrame:
    """
    Enforce facc conservation for a single region (Pass 1).

    Parameters
    ----------
    db_path : str
        Path to DuckDB database (v17c).
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
    print(f"Facc Conservation Pass 1 — {region} [{mode_str}]")
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

        print("  Running conservation pass (equal-split at bifurcations)...")
        changes = _run_conservation_pass(G)
        print(f"    {len(changes)} reaches corrected")

        if len(changes) == 0:
            print("  No corrections needed")
            summary = {
                "timestamp": datetime.now().isoformat(),
                "region": region,
                "dry_run": dry_run,
                "pass": 1,
                "total_reaches": len(reaches_df),
                "corrections": 0,
                "db_path": str(db_path),
            }
            _save_outputs(pd.DataFrame(), region, out_path, summary)
            return pd.DataFrame()

        # Build corrections DataFrame
        rows = []
        for rid, (orig, corr, ctype) in changes.items():
            delta = corr - orig
            delta_pct = 100.0 * delta / orig if orig > 0 else float("inf")
            rows.append(
                {
                    "reach_id": rid,
                    "region": region,
                    "original_facc": round(orig, 4),
                    "corrected_facc": round(corr, 4),
                    "delta": round(delta, 4),
                    "delta_pct": round(delta_pct, 2),
                    "correction_type": ctype,
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
            "pass": 1,
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
        description="Facc conservation correction — Pass 1 (equal-split propagation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Dry run for NA
  python -m src.sword_duckdb.facc_detection.correct_conservation \\
      --db data/duckdb/sword_v17c.duckdb --region NA

  # Apply to DB
  python -m src.sword_duckdb.facc_detection.correct_conservation \\
      --db data/duckdb/sword_v17c.duckdb --region NA --apply

  # All regions dry run
  python -m src.sword_duckdb.facc_detection.correct_conservation \\
      --db data/duckdb/sword_v17c.duckdb --all
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
        df = correct_facc_conservation(
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
