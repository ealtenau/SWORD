# -*- coding: utf-8 -*-
"""
Facc Denoising — Topology-Aware v3 Pipeline
============================================

Three-stage architecture that cleans baseline accuracy FIRST, then propagates:

  Stage A — Baseline Cleanup (internal data only, no propagation):
    A1: Node denoising (dn-node for noisy reaches)
    A2: Node validation on baseline (cap extreme lows)
    A3: Outlier detection on baseline (diagnostics only)
    A4: Isotonic regression on baseline chains (smooth 1:1 chains)

  Stage B — Propagation + Refinement:
    B1: Full topology propagation
    B2: Isotonic smoothing (with bifurc anchors)
    B3: Junction floor
    B4: Node validation
    B5: Final consistency pass

  Stage C — Optional MERIT refinement (only if --merit provided):
    C1: MERIT resample remaining T003 violations
    C2: Re-run Stage B on updated baseline (if C1 changed anything)

The key improvement: cleaning MERIT D8 noise from the baseline before
propagation prevents error amplification through the network.

Relationship to Yushan's Integrator:
    Both enforce conservation (sum upstream <= downstream) and non-negativity.
    Our junction rule ``corrected = sum(corrected_upstream) + max(base - sum(base_upstream), 0)``
    is equivalent to enforcing local_area >= 0 at each junction — the max(..., 0)
    clamps the lateral term to non-negative, same as Yushan's x >= 0 constraint.

    Key difference: O(n) topological-order pass vs O(n^3) CVXPY per basin.

Usage::

    # Dry run on NA (no DB changes, no UPA re-sampling)
    python -m src.updates.sword_duckdb.facc_detection.correct_facc_denoise \\
        --db data/duckdb/sword_v17c.duckdb --v17b data/duckdb/sword_v17b.duckdb --region NA

    # Full run with UPA re-sampling
    python -m src.updates.sword_duckdb.facc_detection.correct_facc_denoise \\
        --db data/duckdb/sword_v17c.duckdb --v17b data/duckdb/sword_v17b.duckdb \\
        --region NA --merit /Volumes/SWORD_DATA/data/MERIT_Hydro

    # Apply to all regions
    python -m src.updates.sword_duckdb.facc_detection.correct_facc_denoise \\
        --db data/duckdb/sword_v17c.duckdb --v17b data/duckdb/sword_v17b.duckdb \\
        --all --apply --merit /Volumes/SWORD_DATA/data/MERIT_Hydro
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import duckdb
import networkx as nx
import numpy as np
import pandas as pd

from .merit_search import MeritGuidedSearch, create_merit_search

logger = logging.getLogger(__name__)

REGIONS = ["NA", "SA", "EU", "AF", "AS", "OC"]


# ---------------------------------------------------------------------------
# Data loading
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


def _load_v17b_facc(v17b_path: str, region: str) -> Dict[int, float]:
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


def _load_node_facc_stats(v17b_path: str, region: str) -> pd.DataFrame:
    """
    Load per-reach node facc statistics from v17b.

    Returns DataFrame with reach_id, max_facc, dn_facc, variability_ratio.
    variability_ratio = max_facc / min_facc within each reach.
    Reaches with high variability_ratio (>1.5) are noisy — MAX grabbed stray
    D8 values from adjacent flow paths.

    Uses v17b (not v17c) because v17c node facc values may have been modified.
    """
    conn = duckdb.connect(v17b_path, read_only=True)
    try:
        df = conn.execute(
            """
            WITH stats AS (
                SELECT reach_id,
                       MAX(facc) as max_facc,
                       MIN(facc) as min_facc,
                       FIRST(facc ORDER BY dist_out ASC) as dn_facc,
                       COUNT(*) as n_nodes
                FROM nodes
                WHERE region = ? AND facc > 0 AND dist_out >= 0
                GROUP BY reach_id
            )
            SELECT reach_id, max_facc, dn_facc, n_nodes,
                   CASE WHEN min_facc > 0 THEN max_facc / min_facc ELSE 1.0 END as variability_ratio
            FROM stats
            """,
            [region.upper()],
        ).fetchdf()
    finally:
        conn.close()
    return df


def _load_reach_geometries(
    conn: duckdb.DuckDBPyConnection,
    reach_ids: List[int],
    region: str,
) -> Dict[int, Tuple[str, float]]:
    """
    Load geometries and widths for specific reaches.

    Returns {reach_id: (wkt, width)}.
    """
    if not reach_ids:
        return {}

    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("DROP TABLE IF EXISTS _tmp_resample_ids")
    conn.execute("CREATE TEMP TABLE _tmp_resample_ids (reach_id BIGINT PRIMARY KEY)")
    conn.executemany(
        "INSERT INTO _tmp_resample_ids VALUES (?)",
        [(int(rid),) for rid in reach_ids],
    )

    df = conn.execute(
        """
        SELECT r.reach_id, ST_AsText(r.geom) as wkt, r.width
        FROM reaches r
        JOIN _tmp_resample_ids t ON r.reach_id = t.reach_id
        WHERE r.region = ?
        """,
        [region.upper()],
    ).fetchdf()

    conn.execute("DROP TABLE IF EXISTS _tmp_resample_ids")

    result = {}
    for _, row in df.iterrows():
        rid = int(row["reach_id"])
        wkt = row["wkt"]
        w = float(row["width"]) if pd.notna(row["width"]) else 100.0
        result[rid] = (wkt, w)
    return result


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------


def _build_graph(topology_df: pd.DataFrame, reaches_df: pd.DataFrame) -> nx.DiGraph:
    """Build DiGraph where edges follow flow (upstream -> downstream)."""
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
# Phase 1: Node-based initialization
# ---------------------------------------------------------------------------


def _phase1_node_init(
    G: nx.DiGraph,
    v17b_facc: Dict[int, float],
    node_stats: pd.DataFrame,
    variability_threshold: float = 1.5,
) -> Tuple[Dict[int, float], Set[int]]:
    """
    Initialize facc estimates. Use v17b MAX(node facc) as default baseline.
    Only switch to downstream-node facc for reaches with high within-reach
    variability (MAX/MIN > threshold), where MAX likely grabbed stray D8 values.

    Returns (baseline, denoised_ids) where:
        baseline: {reach_id: initial_facc_estimate}
        denoised_ids: set of reach_ids where dn-node was used instead of MAX
    """
    # Build lookup from node stats
    dn_lookup = {}
    var_lookup = {}
    for _, row in node_stats.iterrows():
        rid = int(row["reach_id"])
        dn_lookup[rid] = float(row["dn_facc"])
        var_lookup[rid] = float(row["variability_ratio"])

    baseline: Dict[int, float] = {}
    denoised_ids: Set[int] = set()
    n_denoised = 0
    n_default = 0

    for node in G.nodes():
        v17b_val = max(v17b_facc.get(node, 0.0), 0.0)
        var_ratio = var_lookup.get(node, 1.0)
        dn_val = dn_lookup.get(node, 0.0)

        if var_ratio > variability_threshold and dn_val > 0:
            # Noisy reach: use downstream-node facc (avoids stray MAX)
            baseline[node] = dn_val
            denoised_ids.add(node)
            n_denoised += 1
        else:
            # Clean reach: keep v17b MAX(node facc)
            baseline[node] = v17b_val
            n_default += 1

    print(
        f"    Phase 1: {n_denoised} denoised (var>{variability_threshold}x), "
        f"{n_default} kept v17b MAX"
    )
    return baseline, denoised_ids


# ---------------------------------------------------------------------------
# Phase 2: Topology-aware correction (single pass)
# ---------------------------------------------------------------------------


def _phase2_topo_correction(
    G: nx.DiGraph,
    dn_node_facc: Dict[int, float],
) -> Tuple[Dict[int, float], Dict[int, Tuple[float, float, str]]]:
    """
    Single topological-order pass from headwaters to outlets.

    Uses dn_node_facc as the baseline (not v17b directly).

    Returns (corrected, changes) where:
        corrected: {reach_id: corrected_facc}
        changes: {reach_id: (baseline, corrected, correction_type)}
    """
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print("    WARNING: graph has cycles - using node list")
        topo_order = list(G.nodes())

    # Precompute width shares at bifurcations
    bifurc_share: Dict[Tuple[int, int], float] = {}
    for node in G.nodes():
        succs = list(G.successors(node))
        if len(succs) < 2:
            continue
        widths = [max(G.nodes[c].get("width", 0.0), 0.0) for c in succs]
        total_w = sum(widths)
        if total_w <= 0:
            for c in succs:
                bifurc_share[(node, c)] = 1.0 / len(succs)
        else:
            for c, w in zip(succs, widths):
                bifurc_share[(node, c)] = w / total_w

    corrected: Dict[int, float] = {}
    for node in G.nodes():
        corrected[node] = max(dn_node_facc.get(node, 0.0), 0.0)

    changes: Dict[int, Tuple[float, float, str]] = {}
    counts: Dict[str, int] = {}

    for node in topo_order:
        base = max(dn_node_facc.get(node, 0.0), 0.0)
        preds = list(G.predecessors(node))

        if not preds:
            corrected[node] = base
            continue

        if len(preds) >= 2:
            floor = sum(corrected.get(p, 0.0) for p in preds)
            sum_base_up = sum(max(dn_node_facc.get(p, 0.0), 0.0) for p in preds)
            lateral = max(base - sum_base_up, 0.0)
            new_val = floor + lateral
            corrected[node] = new_val
            if abs(new_val - base) > 0.01:
                _record(changes, counts, node, base, new_val, "junction_floor")
            continue

        parent = preds[0]
        parent_out = G.out_degree(parent)

        if parent_out >= 2:
            share = bifurc_share.get((parent, node), 1.0 / parent_out)
            new_val = corrected[parent] * share
            corrected[node] = new_val
            if abs(new_val - base) > 0.01:
                _record(changes, counts, node, base, new_val, "bifurc_share")
        else:
            parent_base = max(dn_node_facc.get(parent, 0.0), 0.0)
            if parent_base == 0 and corrected[parent] == 0:
                corrected[node] = 0.0
                if base > 0.01:
                    _record(changes, counts, node, base, 0.0, "cascade_zero")
            else:
                lateral = max(base - parent_base, 0.0)
                new_val = corrected[parent] + lateral
                corrected[node] = new_val
                if abs(new_val - base) > 0.01:
                    _record(changes, counts, node, base, new_val, "lateral_propagate")

    print("    Phase 2 corrections:")
    for ctype, n in sorted(counts.items()):
        if n > 0:
            print(f"      {ctype:25s} {n:>6,}")
    print(f"      {'TOTAL':25s} {len(changes):>6,}")

    return corrected, changes


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
# Phase 3: Outlier detection (log-space, Tukey IQR)
# ---------------------------------------------------------------------------


def _phase3_outlier_detection(
    G: nx.DiGraph,
    corrected: Dict[int, float],
    dn_node_facc: Dict[int, float],
    log_threshold: float = 1.0,
    junction_raise_threshold: float = 2.0,
    link_drop_threshold: float = 2.0,
) -> Set[int]:
    """
    Detect remaining outliers after topology correction.

    Flags reaches that are:
    1. >exp(log_threshold) (~2.7x) off from their neighborhood median in log-space
    2. Junctions where corrected > base * junction_raise_threshold
    3. 1:1 links with upstream/downstream ratio > link_drop_threshold

    Returns set of flagged reach_ids.
    """
    flagged: Set[int] = set()

    # Method 1: Neighborhood log-deviation (Tukey IQR)
    log_devs = []
    for node in G.nodes():
        val = corrected.get(node, 0.0)
        if val <= 0:
            continue
        neighbors = list(G.predecessors(node)) + list(G.successors(node))
        neighbor_vals = [
            corrected.get(n, 0.0) for n in neighbors if corrected.get(n, 0.0) > 0
        ]
        if not neighbor_vals:
            continue
        all_vals = neighbor_vals + [val]
        ref = float(np.median(all_vals))
        if ref > 0:
            log_dev = abs(math.log(val) - math.log(ref))
            log_devs.append((node, log_dev))

    if log_devs:
        devs_array = np.array([d for _, d in log_devs])
        q1, q3 = np.percentile(devs_array, [25, 75])
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        effective_threshold = max(log_threshold, upper_fence)

        for node, dev in log_devs:
            if dev > effective_threshold:
                flagged.add(node)

        n_log = sum(1 for _, d in log_devs if d > effective_threshold)
        print(
            f"    Phase 3 — log-deviation outliers: {n_log} "
            f"(threshold={effective_threshold:.2f}, IQR fence={upper_fence:.2f})"
        )

    # Method 2: Junction raises > threshold
    n_junc_raise = 0
    for node in G.nodes():
        preds = list(G.predecessors(node))
        if len(preds) < 2:
            continue
        base = max(dn_node_facc.get(node, 0.0), 0.0)
        if base > 0 and corrected.get(node, 0.0) > base * junction_raise_threshold:
            flagged.add(node)
            n_junc_raise += 1
    print(f"    Phase 3 — junction raises >2x: {n_junc_raise}")

    # Method 3: 1:1 link drops > threshold
    n_link_drop = 0
    for node in G.nodes():
        preds = list(G.predecessors(node))
        if len(preds) != 1:
            continue
        parent = preds[0]
        if G.out_degree(parent) != 1:
            continue
        val_up = corrected.get(parent, 0.0)
        val_dn = corrected.get(node, 0.0)
        if val_dn > 0 and val_up / val_dn > link_drop_threshold:
            flagged.add(node)
            flagged.add(parent)
            n_link_drop += 1
    print(f"    Phase 3 — 1:1 link drops >2x: {n_link_drop}")

    print(f"    Phase 3 — total flagged: {len(flagged)}")
    return flagged


# ---------------------------------------------------------------------------
# Phase 4: Re-sample flagged reaches from UPA rasters
# ---------------------------------------------------------------------------


def _phase4_resample_t003(
    conn: duckdb.DuckDBPyConnection,
    G: nx.DiGraph,
    corrected: Dict[int, float],
    dn_node_facc: Dict[int, float],
    region: str,
    merit_searcher: Optional[MeritGuidedSearch],
    node_max_lookup: Optional[Dict[int, float]] = None,
) -> Tuple[Dict[int, float], int, Dict[str, int], Dict[int, Dict]]:
    """
    T003-targeted MERIT re-sampling with dual-endpoint D8 flow-walk.

    Instead of generic radial buffering, uses three strategies:
    1. **D8 walk A** (primary): From the downstream reach's upstream endpoint,
       walk downstream along MERIT's D8 flow direction for up to 150 cells
       (~13.5km). Snaps to MERIT's actual thalweg.
    2. **D8 walk B** (secondary): From the upstream reach's downstream endpoint,
       walk D8 downstream. Covers the other side of the junction.
    3. **Radial buffer** (fallback): If neither D8 walk fixes monotonicity,
       use wide buffer around the junction point.

    For each downstream reach in a 1:1 T003 violation:
    - Pick the *minimum* UPA >= corrected[upstream] (fixes monotonicity
      with minimal distortion)
    - If none, pick the *maximum* candidate (reduces the gap)

    Returns (updated_dn_node_facc, n_resampled, stats_dict, diagnostics).
    diagnostics: {reach_id: {reason, target_min, best_candidate, delta_to_fix, method, ...}}
    """
    stats = {
        "t003_violations_total": 0,
        "t003_on_1to1_links": 0,
        "t003_on_bifurc_edges": 0,
        "t003_on_junction_edges": 0,
        "downstream_resampled": 0,
        "downstream_fixed_mono": 0,
        "downstream_reduced_gap": 0,
        "downstream_no_geom": 0,
        "downstream_no_candidates": 0,
        "fixed_via_d8_walk": 0,
        "fixed_via_d8_walk_b": 0,
        "fixed_via_radial": 0,
        "gap_reduced_via_d8_walk": 0,
        "gap_reduced_via_d8_walk_b": 0,
        "gap_reduced_via_radial": 0,
    }
    diagnostics: Dict[int, Dict] = {}

    if merit_searcher is None:
        print("    Phase 4 — SKIPPED (no MERIT path provided)")
        return dn_node_facc, 0, stats, diagnostics

    from shapely import wkt as shapely_wkt
    from shapely.geometry import Point

    # ---- Step 1: Find T003-violating edges ----
    t003_downstream: Dict[int, float] = {}
    for u, v in G.edges():
        corr_u = corrected.get(u, 0.0)
        corr_v = corrected.get(v, 0.0)
        if corr_u <= corr_v + 1.0:
            continue

        stats["t003_violations_total"] += 1

        if G.out_degree(u) >= 2:
            stats["t003_on_bifurc_edges"] += 1
            continue

        if G.in_degree(v) >= 2:
            stats["t003_on_junction_edges"] += 1
            continue

        stats["t003_on_1to1_links"] += 1
        if v not in t003_downstream or corr_u > t003_downstream[v]:
            t003_downstream[v] = corr_u

    n_targets = len(t003_downstream)
    if n_targets == 0:
        print("    Phase 4 — no 1:1-link T003 violations to fix")
        return dn_node_facc, 0, stats, diagnostics

    print(f"    Phase 4 — {stats['t003_violations_total']} T003 violations total:")
    print(f"      1:1 links (targetable): {stats['t003_on_1to1_links']}")
    print(f"      bifurcation edges (expected): {stats['t003_on_bifurc_edges']}")
    print(f"      junction edges: {stats['t003_on_junction_edges']}")
    print(f"    Targeting {n_targets} downstream reaches...")

    # ---- Step 2: Load geometries ----
    target_ids = sorted(t003_downstream.keys())
    geom_data = _load_reach_geometries(conn, target_ids, region)

    upstream_ids = set()
    for v_rid in target_ids:
        for u_rid in G.predecessors(v_rid):
            upstream_ids.add(u_rid)
    upstream_geom = _load_reach_geometries(conn, sorted(upstream_ids), region)

    # ---- Step 3: Re-sample using D8 walk + radial fallback ----
    n_resampled = 0

    for v_rid in target_ids:
        target_min = t003_downstream[v_rid]
        diag: Dict = {
            "target_min": round(target_min, 2),
            "original_facc": round(dn_node_facc.get(v_rid, 0.0), 2),
            "method": None,
            "reason": None,
            "best_candidate": None,
            "delta_to_fix": None,
            "d8_steps": None,
        }

        if v_rid not in geom_data:
            stats["downstream_no_geom"] += 1
            diag["reason"] = "no_geometry"
            diagnostics[v_rid] = diag
            continue

        wkt_v, width_v = geom_data[v_rid]
        if wkt_v is None:
            stats["downstream_no_geom"] += 1
            diag["reason"] = "no_geometry"
            diagnostics[v_rid] = diag
            continue

        try:
            geom_v = shapely_wkt.loads(wkt_v)
        except Exception:
            stats["downstream_no_geom"] += 1
            diag["reason"] = "invalid_geometry"
            diagnostics[v_rid] = diag
            continue

        # Find upstream endpoint of downstream reach (walk A start)
        upstream_point = None
        upstream_dn_point = None  # downstream endpoint of upstream reach (walk B)
        for u_rid in G.predecessors(v_rid):
            if u_rid in upstream_geom:
                wkt_u, _ = upstream_geom[u_rid]
                if wkt_u is None:
                    continue
                try:
                    geom_u = shapely_wkt.loads(wkt_u)
                    # Walk A: downstream reach's upstream end
                    start_pt = Point(geom_v.coords[0])
                    end_pt = Point(geom_v.coords[-1])
                    d_start = start_pt.distance(geom_u)
                    d_end = end_pt.distance(geom_u)
                    upstream_point = start_pt if d_start < d_end else end_pt
                    # Walk B: upstream reach's downstream end (closest to v)
                    u_start = Point(geom_u.coords[0])
                    u_end = Point(geom_u.coords[-1])
                    d_us = u_start.distance(geom_v)
                    d_ue = u_end.distance(geom_v)
                    upstream_dn_point = u_start if d_us < d_ue else u_end
                except Exception:
                    pass
                break

        if upstream_point is None:
            upstream_point = Point(geom_v.coords[0])

        junction_lon = upstream_point.x
        junction_lat = upstream_point.y

        # ---- Strategy A: D8 flow-walk from downstream reach's upstream end ----
        best = None
        method_used = None

        d8_val, d8_meta = merit_searcher.walk_d8_downstream(
            lon=junction_lon,
            lat=junction_lat,
            region=region,
            target_min=target_min,
            max_steps=150,
        )

        diag["d8_steps"] = d8_meta.get("steps_walked", 0)

        if d8_val is not None and d8_val > 0:
            best = d8_val
            if d8_val >= target_min:
                method_used = "d8_walk_fixed"
                stats["fixed_via_d8_walk"] += 1
            else:
                method_used = "d8_walk_gap"
                stats["gap_reduced_via_d8_walk"] += 1

        # ---- Strategy A2: D8 flow-walk from upstream reach's downstream end ----
        if (best is None or best < target_min) and upstream_dn_point is not None:
            d8_val_b, d8_meta_b = merit_searcher.walk_d8_downstream(
                lon=upstream_dn_point.x,
                lat=upstream_dn_point.y,
                region=region,
                target_min=target_min,
                max_steps=150,
            )
            diag["d8_steps_b"] = d8_meta_b.get("steps_walked", 0)

            if d8_val_b is not None and d8_val_b > 0:
                # Pick best of walk A + walk B
                if d8_val_b >= target_min and (best is None or best < target_min):
                    best = d8_val_b
                    method_used = "d8_walk_b_fixed"
                    stats["fixed_via_d8_walk_b"] += 1
                elif d8_val_b >= target_min and best is not None and best >= target_min:
                    # Both above target — take the minimum (least distortion)
                    if d8_val_b < best:
                        best = d8_val_b
                        method_used = "d8_walk_b_fixed"
                        stats["fixed_via_d8_walk_b"] += 1
                elif best is None or d8_val_b > best:
                    best = d8_val_b
                    method_used = "d8_walk_b_gap"
                    stats["gap_reduced_via_d8_walk_b"] += 1

        # ---- Strategy B: Radial buffer (fallback if D8 didn't fix) ----
        if best is None or best < target_min:
            buffer_m = max(500, min(5 * max(width_v, 100), 5000))
            meters_per_deg = 111320 * np.cos(np.radians(junction_lat))
            if meters_per_deg <= 0:
                meters_per_deg = 111320
            buffer_deg = buffer_m / meters_per_deg
            buffered = upstream_point.buffer(buffer_deg)

            merit_values = merit_searcher._sample_in_polygon(buffered, region)
            candidates = [
                val
                for val in merit_values
                if val > 0 and target_min / 100 <= val <= target_min * 100
            ]

            if candidates:
                above = [c for c in candidates if c >= target_min]
                if above:
                    radial_best = min(above)
                    if best is None or radial_best < best or best < target_min:
                        best = radial_best
                        method_used = "radial_fixed"
                        stats["fixed_via_radial"] += 1
                elif best is None or max(candidates) > best:
                    best = max(candidates)
                    if method_used is None:
                        method_used = "radial_gap"
                        stats["gap_reduced_via_radial"] += 1

        # ---- Apply result (with sanity guards) ----
        if best is not None and best > 0:
            original_val = dn_node_facc.get(v_rid, 0.0)

            # Guard 1: never lower the baseline — if D8 walk returned a
            # value worse than what we already have, it followed a wrong
            # tributary.  Skip entirely.
            if best < original_val - 1.0:
                diag["resample_rejected_lower"] = round(best, 2)
                diag["kept_original"] = round(original_val, 2)
                best = None
                stats["downstream_no_candidates"] += 1
                diag["reason"] = "resample_would_lower"
                diagnostics[v_rid] = diag
                continue

            # Guard 2: if MERIT re-sample is wildly below node evidence,
            # the D8 walk followed the wrong tributary. Use node MAX instead.
            if node_max_lookup:
                node_max = node_max_lookup.get(v_rid, 0.0)
                if node_max > 0 and best < node_max * 0.1:
                    diag["resample_before_guard"] = round(best, 2)
                    diag["node_max_override"] = round(node_max, 2)
                    best = node_max

            dn_node_facc[v_rid] = best
            n_resampled += 1

            if best >= target_min:
                stats["downstream_fixed_mono"] += 1
                diag["reason"] = "fixed_monotonicity"
            else:
                stats["downstream_reduced_gap"] += 1
                diag["reason"] = "merit_shift"
                diag["delta_to_fix"] = round(target_min - best, 2)

            diag["method"] = method_used
            diag["best_candidate"] = round(best, 2)
        else:
            stats["downstream_no_candidates"] += 1
            diag["reason"] = "no_candidates"

        diagnostics[v_rid] = diag

    stats["downstream_resampled"] = n_resampled
    print("    Phase 4 results:")
    print(f"      Resampled: {n_resampled} / {n_targets}")
    print(f"      Fixed monotonicity: {stats['downstream_fixed_mono']}")
    print(f"        via D8 walk A: {stats['fixed_via_d8_walk']}")
    print(f"        via D8 walk B: {stats['fixed_via_d8_walk_b']}")
    print(f"        via radial:    {stats['fixed_via_radial']}")
    print(f"      Reduced gap: {stats['downstream_reduced_gap']}")
    print(f"        via D8 walk A: {stats['gap_reduced_via_d8_walk']}")
    print(f"        via D8 walk B: {stats['gap_reduced_via_d8_walk_b']}")
    print(f"        via radial:    {stats['gap_reduced_via_radial']}")
    print(f"      No geometry: {stats['downstream_no_geom']}")
    print(f"      No candidates: {stats['downstream_no_candidates']}")

    return dn_node_facc, n_resampled, stats, diagnostics


# ---------------------------------------------------------------------------
# Phase 4b: Chain-wise isotonic regression (PAVA)
# ---------------------------------------------------------------------------


def _pava_nondecreasing(
    values: List[float],
    weights: Optional[List[float]] = None,
) -> List[float]:
    """
    Pool Adjacent Violators Algorithm for non-decreasing constraint.

    Finds y* = argmin sum_i w_i*(y_i - v_i)^2  s.t. y_1 <= y_2 <= ... <= y_n.
    O(n) time.
    """
    n = len(values)
    if n <= 1:
        return list(values)
    if weights is None:
        weights = [1.0] * n

    # Blocks: list of [sum_wy, sum_w, start, end]
    blocks: List[List] = []
    for i in range(n):
        blocks.append([values[i] * weights[i], weights[i], i, i])
        # Pool backwards while violation exists
        while len(blocks) > 1:
            curr = blocks[-1]
            prev = blocks[-2]
            mean_prev = prev[0] / prev[1]
            mean_curr = curr[0] / curr[1]
            if mean_prev <= mean_curr:
                break
            # Merge
            prev[0] += curr[0]
            prev[1] += curr[1]
            prev[3] = curr[3]
            blocks.pop()

    result = [0.0] * n
    for sw, w_sum, start, end in blocks:
        mean = sw / w_sum
        for j in range(start, end + 1):
            result[j] = mean
    return result


def _extract_1to1_chains(G: nx.DiGraph) -> List[List[int]]:
    """
    Extract maximal 1:1 chains (no junctions or bifurcations internal).

    A chain is a maximal path where every internal node has in_degree=1
    and its predecessor has out_degree=1. Chain endpoints are
    junction/bifurcation/headwater/outlet nodes.

    Returns list of chains, each a list of reach_ids in topological
    (upstream-to-downstream) order.
    """
    # Find chain start nodes: headwaters, junction children, bifurcation children
    visited: Set[int] = set()
    chains: List[List[int]] = []

    for node in nx.topological_sort(G):
        if node in visited:
            continue
        preds = list(G.predecessors(node))
        # Start a chain if: headwater, junction child, or bifurcation child
        is_start = (
            len(preds) == 0  # headwater
            or len(preds) >= 2  # junction
            or any(G.out_degree(p) >= 2 for p in preds)  # bifurcation child
        )
        if not is_start:
            continue

        # Walk downstream along 1:1 links
        chain = [node]
        visited.add(node)
        curr = node
        while True:
            succs = list(G.successors(curr))
            if len(succs) != 1:
                break  # bifurcation or outlet
            nxt = succs[0]
            if G.in_degree(nxt) >= 2:
                break  # next node is a junction — end chain here
            if nxt in visited:
                break
            chain.append(nxt)
            visited.add(nxt)
            curr = nxt
        if len(chain) >= 2:
            chains.append(chain)

    return chains


def _phase4b_isotonic_chains(
    G: nx.DiGraph,
    corrected: Dict[int, float],
    dn_node_facc: Dict[int, float],
    anchor_overrides: Optional[Dict[int, float]] = None,
) -> Tuple[Dict[int, float], Dict[int, Tuple[float, float]]]:
    """
    Chain-wise isotonic regression to enforce facc monotonicity.

    For each maximal 1:1 chain, runs PAVA (non-decreasing) in log-space
    on the corrected values. This adjusts values both up and down,
    minimizing total distortion while enforcing monotonicity.

    anchor_overrides: optional {reach_id: value} to pin certain chain
    heads (e.g. after junction floor). Implemented as high-weight
    observations.

    Returns (corrected, adjusted) where adjusted = {reach_id: (old, new)}.
    """
    EPS = 1.0  # floor for log-space (1 km²)
    ANCHOR_WEIGHT = 1000.0

    chains = _extract_1to1_chains(G)
    adjusted: Dict[int, Tuple[float, float]] = {}
    n_chains_with_violations = 0

    if anchor_overrides is None:
        anchor_overrides = {}

    for chain in chains:
        values = [corrected.get(rid, 0.0) for rid in chain]

        # Check if chain has any violations at all
        has_violation = False
        for i in range(len(values) - 1):
            if values[i] > values[i + 1] + 1.0:
                has_violation = True
                break
        if not has_violation:
            continue

        n_chains_with_violations += 1

        # Log-space transform
        log_vals = [math.log(max(v, EPS)) for v in values]
        weights = [1.0] * len(chain)

        # Anchor overrides get high weight
        for i, rid in enumerate(chain):
            if rid in anchor_overrides:
                log_vals[i] = math.log(max(anchor_overrides[rid], EPS))
                weights[i] = ANCHOR_WEIGHT

        # Run PAVA (non-decreasing)
        fitted = _pava_nondecreasing(log_vals, weights)

        # Back to linear space and apply
        for i, rid in enumerate(chain):
            new_val = math.exp(fitted[i])
            old_val = corrected[rid]
            if abs(new_val - old_val) > 0.5:  # only record meaningful changes
                corrected[rid] = new_val
                adjusted[rid] = (old_val, new_val)

    print(
        f"    {len(chains)} chains extracted, "
        f"{n_chains_with_violations} with violations"
    )
    return corrected, adjusted


def _phase4c_junction_floor(
    G: nx.DiGraph,
    corrected: Dict[int, float],
) -> Tuple[Dict[int, float], Dict[int, Tuple[float, float]]]:
    """
    Re-enforce junction conservation after imputation.

    For every junction node v (in_degree >= 2):
        corrected[v] = max(corrected[v], sum(corrected[upstream]))

    Runs in topo order so downstream junctions see already-fixed upstream values.

    Returns (corrected, floored) where floored = {reach_id: (old, new)}.
    """
    topo_order = list(nx.topological_sort(G))
    floored: Dict[int, Tuple[float, float]] = {}
    for node in topo_order:
        preds = list(G.predecessors(node))
        if len(preds) < 2:
            continue
        floor = sum(corrected.get(p, 0.0) for p in preds)
        if corrected[node] < floor - 1.0:
            old = corrected[node]
            corrected[node] = floor
            floored[node] = (old, floor)
    return corrected, floored


def _phase_final_consistency(
    G: nx.DiGraph,
    corrected: Dict[int, float],
    dn_node_facc: Dict[int, float],
) -> Tuple[Dict[int, float], Dict[int, Tuple[float, float, str]]]:
    """
    Final topological pass after all correction phases.

    Phases 4b (isotonic), 4c (junction floor), and 4d (node validation)
    adjust values after Phase 2 but don't propagate downstream. This
    pass re-enforces all three topology rules using final values:

    1. Junctions: floor to sum(corrected upstream) + lateral (only raises)
    2. Bifurcation children: width-proportional share of corrected parent
    3. 1:1 links: corrected[parent] + lateral, guaranteed >= parent (only raises)

    Single topological-order pass processes parents before children.
    """
    topo_order = list(nx.topological_sort(G))

    # Precompute bifurc shares
    bifurc_share: Dict[Tuple[int, int], float] = {}
    for node in G.nodes():
        succs = list(G.successors(node))
        if len(succs) < 2:
            continue
        widths = [max(G.nodes[c].get("width", 0.0), 0.0) for c in succs]
        total_w = sum(widths)
        if total_w <= 0:
            for c in succs:
                bifurc_share[(node, c)] = 1.0 / len(succs)
        else:
            for c, w in zip(succs, widths):
                bifurc_share[(node, c)] = w / total_w

    changes: Dict[int, Tuple[float, float, str]] = {}

    for node in topo_order:
        base = max(dn_node_facc.get(node, 0.0), 0.0)
        preds = list(G.predecessors(node))

        if not preds:
            continue

        old_val = corrected[node]

        if len(preds) >= 2:
            # Junction: floor to sum(corrected upstream) + lateral
            floor = sum(corrected.get(p, 0.0) for p in preds)
            sum_base_up = sum(max(dn_node_facc.get(p, 0.0), 0.0) for p in preds)
            lateral = max(base - sum_base_up, 0.0)
            new_val = floor + lateral
            if new_val > corrected[node] + 1.0:
                corrected[node] = new_val
        else:
            parent = preds[0]
            if G.out_degree(parent) >= 2:
                # Bifurcation child: recompute width-proportional share
                share = bifurc_share.get((parent, node), 1.0 / G.out_degree(parent))
                new_val = corrected[parent] * share
                corrected[node] = new_val
            else:
                # 1:1 link: propagate parent + lateral
                parent_base = max(dn_node_facc.get(parent, 0.0), 0.0)
                lateral = max(base - parent_base, 0.0)
                new_val = corrected[parent] + lateral
                if new_val > corrected[node] + 1.0:
                    corrected[node] = new_val

        if abs(corrected[node] - old_val) > 0.01:
            if len(preds) >= 2:
                tag = "final_junction"
            elif G.out_degree(preds[0]) >= 2:
                tag = "final_bifurc"
            else:
                tag = "final_1to1"
            changes[node] = (old_val, corrected[node], tag)

    return corrected, changes


def _phase4d_node_validation(
    G: nx.DiGraph,
    corrected: Dict[int, float],
    node_max_lookup: Dict[int, float],
    ratio_threshold: float = 0.1,
) -> Tuple[Dict[int, float], Dict[int, Tuple[float, float]]]:
    """
    Post-correction validation: compare corrected facc against node-level evidence.

    If corrected facc is wildly below node MAX (< ratio_threshold * node_max),
    the pipeline corrupted the value (e.g., Phase 4 re-sample followed wrong
    tributary, then Phase 4b isotonic smoothed the chain DOWN to match).

    Replace with node MAX — safe because node-level facc comes directly from
    MERIT D8 sampling at the reach's actual location.

    Parameters
    ----------
    G : reach graph
    corrected : {reach_id: corrected_facc}
    node_max_lookup : {reach_id: MAX(node_facc)} from v17b
    ratio_threshold : flag if corrected < threshold * node_max (default 0.1 = 10%)

    Returns
    -------
    corrected : updated dict
    overrides : {reach_id: (old_corrected, node_max)} for overridden reaches
    """
    overrides: Dict[int, Tuple[float, float]] = {}
    for rid in G.nodes():
        node_max = node_max_lookup.get(rid, 0.0)
        corr = corrected.get(rid, 0.0)
        if node_max > 0 and corr < node_max * ratio_threshold:
            overrides[rid] = (corr, node_max)
            corrected[rid] = node_max
    return corrected, overrides


# ---------------------------------------------------------------------------
# Collect remaining T003 flags
# ---------------------------------------------------------------------------


def _collect_t003_flags(
    G: nx.DiGraph,
    corrected: Dict[int, float],
) -> Dict[int, str]:
    """
    Identify remaining non-bifurcation T003 violations and classify them.

    Returns {downstream_reach_id: reason} where reason is one of:
      chain, junction_adjacent, non_isolated.
    """
    flags: Dict[int, str] = {}
    for u, v in G.edges():
        if corrected.get(u, 0.0) <= corrected.get(v, 0.0) + 1.0:
            continue
        if G.out_degree(u) >= 2:
            continue  # bifurcation — expected
        # Classify
        succs_v = list(G.successors(v))
        feeds_junction = any(G.in_degree(w) >= 2 for w in succs_v)
        would_break_dn = any(
            corrected.get(u, 0.0) > corrected.get(w, 0.0) + 1.0 for w in succs_v
        )
        if feeds_junction:
            reason = "junction_adjacent"
        elif would_break_dn:
            reason = "chain"
        else:
            reason = "non_isolated"
        flags[v] = reason
    return flags


# ---------------------------------------------------------------------------
# Export remaining T003 violations
# ---------------------------------------------------------------------------


def _export_remaining_t003(
    conn: duckdb.DuckDBPyConnection,
    G: nx.DiGraph,
    corrected: Dict[int, float],
    v17b_facc: Dict[int, float],
    region: str,
    out_path: Path,
) -> None:
    """
    Export remaining non-bifurcation T003 violations as GeoJSON for visual audit.
    """
    # Collect violating edges
    violations = []
    for u, v in G.edges():
        corr_u = corrected.get(u, 0.0)
        corr_v = corrected.get(v, 0.0)
        if corr_u <= corr_v + 1.0:
            continue
        if G.out_degree(u) >= 2:
            continue  # bifurcation — expected
        violations.append((u, v))

    if not violations:
        print("\n  No remaining T003 violations to export")
        return

    # Classify each violation
    downstream_ids = sorted(set(v for _, v in violations))

    # Load geometries for the downstream reaches (the ones we'd display)
    geom_data = _load_reach_geometries(conn, downstream_ids, region)

    # Build rows
    rows = []
    for u, v in violations:
        corr_u = corrected.get(u, 0.0)
        corr_v = corrected.get(v, 0.0)
        base_v = max(v17b_facc.get(v, 0.0), 0.0)

        # Classify reason
        succs_v = list(G.successors(v))
        would_break_dn = any(corr_u > corrected.get(w, 0.0) + 1.0 for w in succs_v)
        feeds_junction = any(G.in_degree(w) >= 2 for w in succs_v)
        is_chain = would_break_dn and not feeds_junction

        if feeds_junction:
            reason = "junction_adjacent"
        elif is_chain:
            reason = "chain"
        else:
            reason = "non_isolated"

        # Downstream ratio (self / best downstream)
        ratio_dn = None
        for w in succs_v:
            cw = corrected.get(w, 0.0)
            if cw > 0:
                ratio_dn = round(corr_v / cw, 3)
                break

        rows.append(
            {
                "reach_id": v,
                "upstream_id": u,
                "downstream_id": succs_v[0] if succs_v else None,
                "facc_base": round(base_v, 2),
                "facc_corrected": round(corr_v, 2),
                "facc_upstream": round(corr_u, 2),
                "delta": round(corr_v - base_v, 2),
                "ratio_up": round(corr_u / corr_v, 3) if corr_v > 0 else None,
                "ratio_dn": ratio_dn,
                "reason": reason,
                "is_junction_child": G.in_degree(v) >= 2,
                "is_junction_parent": any(G.in_degree(w) >= 2 for w in succs_v),
                "n_preds": G.in_degree(v),
                "n_succs": G.out_degree(v),
            }
        )

    df = pd.DataFrame(rows)
    reason_counts = df["reason"].value_counts()

    print(f"\n  Remaining T003 violations: {len(df)}")
    for r, c in reason_counts.items():
        print(f"    {r:25s} {c:,}")

    # Build GeoJSON
    features = []
    for _, row in df.iterrows():
        rid = row["reach_id"]
        if rid not in geom_data:
            continue
        wkt, _ = geom_data[rid]
        if wkt is None:
            continue
        try:
            from shapely import wkt as shapely_wkt
            from shapely.geometry import mapping

            geom = shapely_wkt.loads(wkt)
            props = {k: v for k, v in row.items() if k != "reach_id"}
            props["reach_id"] = int(rid)
            # Convert numpy/pandas types to native Python for JSON
            for k in props:
                val = props[k]
                if isinstance(val, (np.integer,)):
                    props[k] = int(val)
                elif isinstance(val, (np.floating,)):
                    props[k] = float(val)
                elif isinstance(val, (np.bool_,)):
                    props[k] = bool(val)
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geom),
                    "properties": props,
                }
            )
        except Exception:
            continue

    if features:
        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }
        out_path.mkdir(parents=True, exist_ok=True)
        geojson_path = out_path / f"remaining_t003_{region}.geojson"
        with open(geojson_path, "w") as f:
            json.dump(geojson, f)
        print(f"  Saved: {geojson_path} ({len(features)} features)")


# ---------------------------------------------------------------------------
# Stage B: Propagation + Refinement (reusable helper)
# ---------------------------------------------------------------------------


def _run_stage_b(
    G: nx.DiGraph,
    dn_node_facc: Dict[int, float],
    node_max_lookup: Dict[int, float],
) -> Tuple[
    Dict[int, float],
    Dict[int, Tuple[float, float, str]],
    Dict[int, Tuple[float, float]],
    Dict[int, Tuple[float, float]],
    Dict[int, Tuple[float, float]],
    Dict[int, Tuple[float, float, str]],
]:
    """
    Propagation + refinement on a cleaned baseline.

    Returns (corrected, changes, adjusted, floored, node_overrides, final_changes).
    """
    # B1: Full topology propagation
    corrected, changes = _phase2_topo_correction(G, dn_node_facc)

    # B2: Isotonic smoothing with bifurc anchors
    anchors: Dict[int, float] = {}
    for node in G.nodes():
        preds = list(G.predecessors(node))
        if len(preds) == 1 and G.out_degree(preds[0]) >= 2:
            anchors[node] = corrected[node]
    corrected, adjusted = _phase4b_isotonic_chains(
        G, corrected, dn_node_facc, anchor_overrides=anchors
    )

    # B3: Junction floor
    corrected, floored = _phase4c_junction_floor(G, corrected)

    # B4: Node validation
    corrected, node_overrides = _phase4d_node_validation(
        G, corrected, node_max_lookup, ratio_threshold=0.1
    )

    # B5: Final consistency pass
    corrected, final_changes = _phase_final_consistency(G, corrected, dn_node_facc)

    return corrected, changes, adjusted, floored, node_overrides, final_changes


# ---------------------------------------------------------------------------
# Phase 6: Validation
# ---------------------------------------------------------------------------


def _phase6_validate(
    G: nx.DiGraph,
    corrected: Dict[int, float],
    v17b_facc: Dict[int, float],
) -> Dict[str, int]:
    """
    Run inline validation checks with T003 split by edge type.

    Returns dict of check_name -> violation_count.
    """
    results = {}

    # F006: Junction conservation (facc >= sum upstream)
    n_f006 = 0
    for node in G.nodes():
        preds = list(G.predecessors(node))
        if len(preds) < 2:
            continue
        floor = sum(corrected.get(p, 0.0) for p in preds)
        if corrected.get(node, 0.0) < floor - 1.0:
            n_f006 += 1
    results["F006_junction_conservation"] = n_f006

    # T003: facc monotonicity — exclude bifurcation edges (expected drops)
    n_t003 = 0
    n_t003_bifurc = 0
    for u, v in G.edges():
        if corrected.get(u, 0.0) <= corrected.get(v, 0.0) + 1.0:
            continue
        if G.out_degree(u) >= 2:
            n_t003_bifurc += 1  # Not counted as violation
        else:
            n_t003 += 1

    results["T003_monotonicity"] = n_t003
    results["T003_bifurc_excluded"] = n_t003_bifurc

    # Junction raise count (corrected > v17b * 2)
    n_raise = 0
    for node in G.nodes():
        preds = list(G.predecessors(node))
        if len(preds) < 2:
            continue
        base = max(v17b_facc.get(node, 0.0), 0.0)
        if base > 0 and corrected.get(node, 0.0) > base * 2:
            n_raise += 1
    results["junction_raises_gt_2x"] = n_raise

    # 1:1 link drops
    n_link_drop = 0
    for u, v in G.edges():
        if G.out_degree(u) == 1 and G.in_degree(v) == 1:
            if corrected.get(u, 0.0) > corrected.get(v, 0.0) + 1.0:
                n_link_drop += 1
    results["link_drops_1to1"] = n_link_drop

    print("    Phase 6 — validation:")
    for check, count in sorted(results.items()):
        status = "PASS" if count == 0 else f"{count:,}"
        print(f"      {check:35s} {status}")

    return results


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

    conn.execute("DROP TABLE IF EXISTS _v3_facc")
    conn.execute(
        "CREATE TEMP TABLE _v3_facc (  reach_id BIGINT PRIMARY KEY, new_facc DOUBLE)"
    )
    data = list(
        zip(
            corrections_df["reach_id"].astype(int),
            corrections_df["corrected_facc"].astype(float),
        )
    )
    conn.executemany("INSERT INTO _v3_facc VALUES (?, ?)", data)
    conn.execute(
        "UPDATE reaches SET facc = t.new_facc "
        "FROM _v3_facc t WHERE reaches.reach_id = t.reach_id"
    )
    n = len(data)
    conn.execute("DROP TABLE IF EXISTS _v3_facc")

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

    conn.execute("DROP TABLE IF EXISTS _v3_tag_ids")
    conn.execute("CREATE TEMP TABLE _v3_tag_ids (reach_id BIGINT PRIMARY KEY)")
    conn.executemany(
        "INSERT INTO _v3_tag_ids VALUES (?)",
        [(int(rid),) for rid in corrections_df["reach_id"].astype(int)],
    )

    if "edit_flag" in cols:
        conn.execute(
            "UPDATE reaches SET edit_flag = 'facc_denoise_v3' "
            "FROM _v3_tag_ids t WHERE reaches.reach_id = t.reach_id"
        )
    if "facc_quality" in cols:
        conn.execute(
            "UPDATE reaches SET facc_quality = 'denoise_v3' "
            "FROM _v3_tag_ids t WHERE reaches.reach_id = t.reach_id"
        )
    conn.execute("DROP TABLE IF EXISTS _v3_tag_ids")

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


def _clear_old_tags(
    conn: duckdb.DuckDBPyConnection,
    region: str,
) -> None:
    """Clear old facc_quality and edit_flag tags for a region."""
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
        "'conservation_corrected_p3','denoise_v3',"
        "'traced','suspect'"
    )
    old_edit = (
        "'facc_conservation_p1','facc_conservation_p2',"
        "'facc_conservation_p3','facc_conservation_single','facc_denoise_v3'"
    )
    if "facc_quality" in cols:
        conn.execute(
            f"UPDATE reaches SET facc_quality = NULL "
            f"WHERE region = ? AND facc_quality IN ({old_quality})",
            [region],
        )
        print("    Cleared facc_quality tags")
    if "edit_flag" in cols:
        conn.execute(
            f"UPDATE reaches SET edit_flag = NULL "
            f"WHERE region = ? AND edit_flag IN ({old_edit})",
            [region],
        )
        print("    Cleared edit_flag tags")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def correct_facc_denoise(
    db_path: str,
    v17b_path: str,
    region: str,
    dry_run: bool = True,
    merit_path: Optional[str] = None,
    output_dir: str = "output/facc_detection",
    log_threshold: float = 1.0,
    variability_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Run v3 facc denoising pipeline for one region.

    Three-stage architecture:
      Stage A: Baseline cleanup (node init, validation, outlier detection, isotonic)
      Stage B: Propagation + refinement (topo correction, isotonic, junction floor,
               node validation, final consistency)
      Stage C: Optional MERIT refinement (resample T003 violations, re-run Stage B)

    Parameters
    ----------
    db_path : path to v17c DuckDB
    v17b_path : path to v17b DuckDB (read-only baseline)
    region : region code (NA, SA, EU, AF, AS, OC)
    dry_run : if True, don't modify DB
    merit_path : path to MERIT Hydro base dir (enables Stage C)
    output_dir : where to write CSV + JSON
    log_threshold : min log-deviation for outlier flagging (default 1.0 ~ 2.7x)
    variability_threshold : min MAX/MIN ratio to trigger dn-node denoising (default 2.0)
    """
    region = region.upper()
    out_path = Path(output_dir)
    mode_str = "DRY RUN" if dry_run else "APPLYING TO DB"
    merit_str = "+ MERIT re-sampling" if merit_path else "(no MERIT)"

    print(f"\n{'=' * 60}")
    print(f"Facc Denoising v3 — {region} [{mode_str}] {merit_str}")
    print(f"{'=' * 60}")

    # Load v17b baseline
    print("  Loading v17b baseline...")
    v17b_facc = _load_v17b_facc(v17b_path, region)
    print(f"    {len(v17b_facc)} v17b reaches")

    # Load topology + reaches from v17c
    conn = duckdb.connect(db_path, read_only=dry_run)
    try:
        print("  Loading topology...")
        topo_df = _load_topology(conn, region)
        print("  Loading reaches...")
        reaches_df = _load_reaches(conn, region)
        print(f"    {len(reaches_df)} reaches")

        if len(reaches_df) == 0:
            return pd.DataFrame()

        # Load node-level facc stats from v17b (original MERIT D8 samples)
        print("  Loading node-level facc stats from v17b...")
        node_stats = _load_node_facc_stats(v17b_path, region)
        n_noisy = int((node_stats["variability_ratio"] > variability_threshold).sum())
        print(
            f"    {len(node_stats)} reaches, {n_noisy} with variability >{variability_threshold}x"
        )

        # Build node MAX lookup for sanity guards (Phase 4 + Phase 4d)
        node_max_lookup: Dict[int, float] = dict(
            zip(
                node_stats["reach_id"].astype(int),
                node_stats["max_facc"].astype(float),
            )
        )

        # Build graph
        print("  Building graph...")
        G = _build_graph(topo_df, reaches_df)
        n_bifurc = sum(1 for n in G.nodes() if G.out_degree(n) >= 2)
        print(
            f"    {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
            f"{n_bifurc} bifurcations"
        )

        # ==============================================================
        # Stage A: Baseline Cleanup (internal data only, no propagation)
        # ==============================================================

        # A1: Node denoising (existing Phase 1)
        print(
            f"\n  A1: Node-based initialization (dn-node for var>{variability_threshold}x)"
        )
        dn_node_facc, denoised_ids = _phase1_node_init(
            G,
            v17b_facc,
            node_stats,
            variability_threshold=variability_threshold,
        )
        n_node_diff = len(denoised_ids)
        print(f"    {n_node_diff} reaches denoised (switched from MAX to dn-node)")

        # A2: Node validation on baseline — catch extreme denoising errors
        print("\n  A2: Node validation on baseline")
        dn_node_facc, baseline_overrides = _phase4d_node_validation(
            G, dn_node_facc, node_max_lookup, ratio_threshold=0.1
        )
        print(f"    {len(baseline_overrides)} baseline reaches overridden by node MAX")

        # A3: Outlier detection on baseline — diagnostics only
        print("\n  A3: Outlier detection on baseline (diagnostics only)")
        flagged = _phase3_outlier_detection(
            G,
            dn_node_facc,
            dn_node_facc,
            log_threshold=log_threshold,
        )

        # A4: Isotonic regression on baseline chains — smooth raw MERIT noise
        print("\n  A4: Isotonic regression on baseline chains")
        dn_node_facc, baseline_isotonic = _phase4b_isotonic_chains(
            G, dn_node_facc, dn_node_facc, anchor_overrides=None
        )
        n_bl_raised = sum(1 for old, new in baseline_isotonic.values() if new > old)
        n_bl_lowered = sum(1 for old, new in baseline_isotonic.values() if new < old)
        print(
            f"    {len(baseline_isotonic)} baseline reaches adjusted "
            f"({n_bl_raised} raised, {n_bl_lowered} lowered)"
        )

        # ==============================================================
        # Stage B: Propagation + Refinement
        # ==============================================================

        print("\n  Stage B: Propagation + Refinement")
        corrected, changes, adjusted, floored, node_overrides, final_changes = (
            _run_stage_b(G, dn_node_facc, node_max_lookup)
        )
        n_final_junc = sum(
            1 for _, _, t in final_changes.values() if t == "final_junction"
        )
        n_final_bifurc = sum(
            1 for _, _, t in final_changes.values() if t == "final_bifurc"
        )
        n_final_1to1 = sum(1 for _, _, t in final_changes.values() if t == "final_1to1")
        print(f"    Phase 2 changes: {len(changes)}")
        print(f"    Isotonic adjusted: {len(adjusted)}")
        print(f"    Junctions re-floored: {len(floored)}")
        print(f"    Node MAX overrides: {len(node_overrides)}")
        print(
            f"    Final consistency: {len(final_changes)} "
            f"({n_final_junc} junc, {n_final_bifurc} bifurc, {n_final_1to1} 1:1)"
        )

        # ==============================================================
        # Stage C: Optional MERIT refinement
        # ==============================================================

        print("\n  Stage C: Optional MERIT refinement")
        merit_searcher = create_merit_search(merit_path)
        resampled_ids: Set[int] = set()
        n_resampled = 0
        resample_stats: Dict[str, int] = {}
        resample_diag: Dict[int, Dict] = {}

        if merit_searcher:
            dn_node_facc, n_resampled, resample_stats, resample_diag = (
                _phase4_resample_t003(
                    conn,
                    G,
                    corrected,
                    dn_node_facc,
                    region,
                    merit_searcher,
                    node_max_lookup=node_max_lookup,
                )
            )
            # Track which reaches were actually resampled
            for rid in G.nodes():
                base_val = max(v17b_facc.get(rid, 0.0), 0.0)
                if abs(dn_node_facc.get(rid, 0.0) - base_val) > 0.01:
                    if rid not in denoised_ids:
                        resampled_ids.add(rid)

            if n_resampled > 0:
                print("\n    Re-running Stage B on MERIT-updated baseline...")
                (
                    corrected,
                    changes,
                    adjusted,
                    floored,
                    node_overrides,
                    final_changes,
                ) = _run_stage_b(G, dn_node_facc, node_max_lookup)
                n_final_junc = sum(
                    1 for _, _, t in final_changes.values() if t == "final_junction"
                )
                n_final_bifurc = sum(
                    1 for _, _, t in final_changes.values() if t == "final_bifurc"
                )
                n_final_1to1 = sum(
                    1 for _, _, t in final_changes.values() if t == "final_1to1"
                )
                print(f"    Phase 2 changes: {len(changes)}")
                print(f"    Isotonic adjusted: {len(adjusted)}")
                print(f"    Junctions re-floored: {len(floored)}")
                print(f"    Node MAX overrides: {len(node_overrides)}")
                print(
                    f"    Final consistency: {len(final_changes)} "
                    f"({n_final_junc} junc, {n_final_bifurc} bifurc, "
                    f"{n_final_1to1} 1:1)"
                )
        else:
            print("    SKIPPED (no MERIT path provided)")

        # ==============================================================
        # Post-pipeline: validation, export, summary
        # ==============================================================

        baseline_override_ids = set(baseline_overrides.keys())
        baseline_isotonic_ids = set(baseline_isotonic.keys())
        node_override_ids = set(node_overrides.keys())
        final_ids = set(final_changes.keys())

        # ---- Collect T003 flags ----
        t003_flags = _collect_t003_flags(G, corrected)
        print(f"\n  T003 flags: {len(t003_flags)} remaining violations (metadata only)")
        from collections import Counter

        for reason, cnt in sorted(Counter(t003_flags.values()).items()):
            print(f"    {reason:25s} {cnt:,}")

        # ---- Export remaining T003 violations ----
        _export_remaining_t003(conn, G, corrected, v17b_facc, region, out_path)

        # ---- Validation ----
        print("\n  Validation:")
        validation = _phase6_validate(G, corrected, v17b_facc)
        validation["baseline_node_overrides"] = len(baseline_overrides)
        validation["baseline_isotonic_adjusted"] = len(baseline_isotonic)
        validation["phase3_flagged"] = len(flagged)
        validation["isotonic_adjusted"] = len(adjusted)
        validation["t003_flagged"] = len(t003_flags)
        validation["floored_junctions"] = len(floored)
        validation["node_max_overrides"] = len(node_overrides)
        validation["final_consistency_changes"] = len(final_changes)
        print(f"      {'baseline_node_overrides':35s} {len(baseline_overrides):,}")
        print(f"      {'baseline_isotonic_adjusted':35s} {len(baseline_isotonic):,}")
        print(f"      {'phase3_flagged':35s} {len(flagged):,}")
        print(f"      {'isotonic_adjusted':35s} {len(adjusted):,}")
        print(f"      {'t003_flagged':35s} {len(t003_flags):,}")
        print(f"      {'floored_junctions':35s} {len(floored):,}")
        print(f"      {'node_max_overrides':35s} {len(node_overrides):,}")
        print(f"      {'final_consistency_changes':35s} {len(final_changes):,}")

        # Build corrections DataFrame — compare corrected to v17b
        rows = []
        for rid in G.nodes():
            base_v17b = max(v17b_facc.get(rid, 0.0), 0.0)
            corr = corrected.get(rid, 0.0)
            if abs(corr - base_v17b) > 0.01 or rid in t003_flags:
                delta = corr - base_v17b
                delta_pct = (
                    100.0 * delta / base_v17b
                    if base_v17b > 0
                    else (float("inf") if delta > 0 else 0.0)
                )
                # Determine correction type (last-write-wins order)
                if rid in final_ids:
                    ctype = final_changes[rid][2]
                elif rid in node_override_ids:
                    ctype = "node_max_override"
                elif rid in floored:
                    ctype = "junction_floor_post"
                elif rid in adjusted:
                    ctype = "isotonic_regression"
                elif rid in changes:
                    ctype = changes[rid][2]
                elif rid in resampled_ids:
                    ctype = "upa_resample"
                elif rid in baseline_isotonic_ids:
                    ctype = "baseline_isotonic"
                elif rid in baseline_override_ids:
                    ctype = "baseline_node_override"
                elif rid in denoised_ids:
                    ctype = "node_denoise"
                elif rid in t003_flags:
                    ctype = "t003_flagged_only"
                else:
                    ctype = "unknown"
                diag = resample_diag.get(rid, {})
                rows.append(
                    {
                        "reach_id": rid,
                        "region": region,
                        "original_facc": round(base_v17b, 4),
                        "dn_node_facc": round(dn_node_facc.get(rid, 0.0), 4),
                        "corrected_facc": round(corr, 4),
                        "delta": round(delta, 4),
                        "delta_pct": round(delta_pct, 2),
                        "correction_type": ctype,
                        "t003_flag": rid in t003_flags,
                        "t003_reason": t003_flags.get(rid),
                        "was_resampled": rid in resampled_ids,
                        "resample_reason": diag.get("reason"),
                        "resample_method": diag.get("method"),
                        "resample_target_min": diag.get("target_min"),
                        "resample_best_candidate": diag.get("best_candidate"),
                        "resample_delta_to_fix": diag.get("delta_to_fix"),
                        "resample_d8_steps": diag.get("d8_steps"),
                    }
                )

        corrections_df = pd.DataFrame(rows) if rows else pd.DataFrame()

        if len(corrections_df) == 0:
            print("\n  No changes needed")
            return corrections_df

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

        print("\n  Summary:")
        print(f"    Total corrections: {len(corrections_df)}")
        print(f"    Raised:  {n_raised}")
        print(f"    Lowered: {n_lowered}")
        print(f"    Net facc change: {total_delta:>+,.0f} km^2 ({pct:+.3f}%)")
        print(f"    Phase 3 flagged: {len(flagged)}")
        print(f"    Baseline node overrides: {len(baseline_overrides)}")
        print(f"    Baseline isotonic adjusted: {len(baseline_isotonic)}")
        print(f"    T003 targeted resampled: {n_resampled}")
        if resample_stats:
            print(
                f"    T003 fixed monotonicity: {resample_stats.get('downstream_fixed_mono', 0)}"
            )
            print(
                f"    T003 reduced gap: {resample_stats.get('downstream_reduced_gap', 0)}"
            )
        for ctype, row in by_type.iterrows():
            print(
                f"      {ctype:25s}  n={int(row['count']):>6,}  "
                f"med_delta={row['median_delta']:>+12,.1f} km^2"
            )

        # Apply to DB
        if not dry_run:
            print("\n  Restoring v17b baseline...")
            _restore_v17b(conn, v17b_facc, region)
            print("  Clearing old tags...")
            _clear_old_tags(conn, region)
            print("  Applying v3 corrections...")
            _apply_to_db(conn, corrections_df)
            print("  Done.")

        # Save outputs
        out_path.mkdir(parents=True, exist_ok=True)
        csv_path = out_path / f"facc_denoise_v3_{region}.csv"
        corrections_df.to_csv(csv_path, index=False)
        print(f"\n  Saved CSV: {csv_path} ({len(corrections_df)} rows)")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "region": region,
            "algorithm": "denoise_v3",
            "dry_run": dry_run,
            "merit_resample": merit_path is not None,
            "total_reaches": len(reaches_df),
            "bifurcations": n_bifurc,
            "node_stats_coverage": len(node_stats),
            "noisy_reaches": n_noisy,
            "denoised_reaches": n_node_diff,
            "corrections": len(corrections_df),
            "raised": n_raised,
            "lowered": n_lowered,
            "total_facc_before": total_before,
            "net_facc_change": total_delta,
            "pct_change": round(pct, 4),
            "baseline_node_overrides": len(baseline_overrides),
            "baseline_isotonic_adjusted": len(baseline_isotonic),
            "phase3_flagged": len(flagged),
            "t003_resample_stats": resample_stats,
            "t003_flagged": len(t003_flags),
            "floored_junctions": len(floored),
            "node_max_overrides": len(node_overrides),
            "final_consistency_changes": len(final_changes),
            "final_junctions": n_final_junc,
            "final_bifurc": n_final_bifurc,
            "final_1to1": n_final_1to1,
            "validation": validation,
            "by_type": {
                ctype: {
                    "count": int(row["count"]),
                    "median_delta": round(float(row["median_delta"]), 2),
                }
                for ctype, row in by_type.iterrows()
            },
            "variability_threshold": variability_threshold,
            "log_threshold": log_threshold,
            "db_path": str(db_path),
            "v17b_path": str(v17b_path),
        }
        summary_path = out_path / f"facc_denoise_v3_summary_{region}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved summary: {summary_path}")

        return corrections_df

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Facc denoising v3 — topology-aware with UPA re-sampling"
    )
    parser.add_argument("--db", required=True, help="v17c DuckDB path")
    parser.add_argument(
        "--v17b",
        default="data/duckdb/sword_v17b.duckdb",
        help="v17b DuckDB path (baseline)",
    )
    parser.add_argument("--region", help="Single region")
    parser.add_argument("--all", action="store_true", help="Process all regions")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write corrections to DB (default: dry run)",
    )
    parser.add_argument(
        "--merit", help="MERIT Hydro base path (enables Phase 4 re-sampling)"
    )
    parser.add_argument(
        "--output-dir", default="output/facc_detection", help="Output directory"
    )
    parser.add_argument(
        "--log-threshold",
        type=float,
        default=1.0,
        help="Log-deviation threshold for outlier detection (default: 1.0)",
    )
    parser.add_argument(
        "--var-threshold",
        type=float,
        default=2.0,
        help="Variability threshold for node denoising (default: 2.0)",
    )

    args = parser.parse_args()
    if not args.region and not args.all:
        parser.error("Specify --region or --all")

    regions = REGIONS if args.all else [args.region.upper()]
    all_corrections = []
    for region in regions:
        df = correct_facc_denoise(
            db_path=args.db,
            v17b_path=args.v17b,
            region=region,
            dry_run=not args.apply,
            merit_path=args.merit,
            output_dir=args.output_dir,
            log_threshold=args.log_threshold,
            variability_threshold=args.var_threshold,
        )
        if len(df) > 0:
            all_corrections.append(df)

    if all_corrections:
        combined = pd.concat(all_corrections, ignore_index=True)
        n_up = int((combined["delta"] > 0).sum())
        n_dn = int((combined["delta"] < 0).sum())
        n_resamp = (
            int(combined["was_resampled"].sum())
            if "was_resampled" in combined.columns
            else 0
        )
        print(f"\n{'=' * 60}")
        print(
            f"GRAND TOTAL: {len(combined)} modifications "
            f"({n_up} raised, {n_dn} lowered, {n_resamp} resampled)"
        )
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
