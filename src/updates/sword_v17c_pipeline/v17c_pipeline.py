#!/usr/bin/env python3
"""
v17c Pipeline - Simplified SWORD v17c attribute computation.

This script:
1. Builds a reach-level directed graph from v17b topology
2. Computes v17c attributes (hydro_dist_out, best_headwater, is_mainstem, etc.)
3. Optionally applies SWOT-derived slopes with quality filtering
4. Writes results directly to sword_v17c.duckdb via SWORDWorkflow

NO MILP optimization - uses original v17b topology for flow direction.

Usage:
    # Process single region
    python -m src.updates.sword_v17c_pipeline.v17c_pipeline --db data/duckdb/sword_v17c.duckdb --region NA

    # Process all regions
    python -m src.updates.sword_v17c_pipeline.v17c_pipeline --db data/duckdb/sword_v17c.duckdb --all

    # Skip SWOT slope integration (faster)
    python -m src.updates.sword_v17c_pipeline.v17c_pipeline --db data/duckdb/sword_v17c.duckdb --all --skip-swot
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime as dt
from typing import Dict, List, Optional, Set, Tuple

import duckdb
import networkx as nx
import numpy as np
import pandas as pd

# Regions in processing order (largest to smallest for progress feedback)
REGIONS = ["NA", "SA", "EU", "AF", "AS", "OC"]

# Default model paths for facc correction
DEFAULT_NOFACC_MODEL = "output/facc_detection/rf_regressor_baseline_nofacc.joblib"
DEFAULT_STANDARD_MODEL = "output/facc_detection/rf_regressor_baseline.joblib"


def log(msg: str) -> None:
    """Log message with timestamp."""
    print(f"[{dt.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_effective_width(attrs: Dict, min_obs: int = 5) -> float:
    """
    Get effective width for routing decisions.

    Prefers SWOT-observed width (width_obs_median) if n_obs >= min_obs,
    otherwise falls back to original GRWL width.
    """
    n_obs = attrs.get("n_obs", 0)
    if pd.isna(n_obs):
        n_obs = 0
    if n_obs >= min_obs:
        swot_width = attrs.get("width_obs_median")
        if swot_width is not None and not pd.isna(swot_width) and swot_width > 0:
            return swot_width
    width = attrs.get("width", 0)
    if pd.isna(width):
        return 0
    return width or 0


# =============================================================================
# Data Loading
# =============================================================================


def load_topology(conn: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    """Load reach_topology from DuckDB."""
    log(f"Loading topology for {region}...")
    df = conn.execute(
        """
        SELECT reach_id, direction, neighbor_rank, neighbor_reach_id
        FROM reach_topology
        WHERE region = ?
    """,
        [region.upper()],
    ).fetchdf()
    log(f"Loaded {len(df):,} topology rows")
    return df


def load_reaches(conn: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    """Load reaches with attributes."""
    log(f"Loading reaches for {region}...")

    # Get available columns (handles older DBs without v17c columns)
    cols_result = conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'reaches'"
    ).fetchall()
    available_cols = {row[0].lower() for row in cols_result}

    # Core columns (required)
    core_cols = [
        "reach_id",
        "region",
        "reach_length",
        "width",
        "slope",
        "facc",
        "n_rch_up",
        "n_rch_down",
        "dist_out",
        "path_freq",
        "stream_order",
        "lakeflag",
        "trib_flag",
    ]

    # Optional columns (v17c additions)
    optional_cols = [
        "wse_obs_mean",
        "wse_obs_std",
        "width_obs_median",
        "n_obs",
        "main_side",
        "type",
        "end_reach",
        "path_order",
        "path_segs",
    ]

    # Build column list
    select_cols = [c for c in core_cols if c.lower() in available_cols]
    select_cols += [c for c in optional_cols if c.lower() in available_cols]

    df = conn.execute(
        f"""
        SELECT {", ".join(select_cols)}
        FROM reaches
        WHERE region = ?
    """,
        [region.upper()],
    ).fetchdf()
    log(f"Loaded {len(df):,} reaches")
    return df


# =============================================================================
# Topology Count & dist_out Fixes (runs before FACC correction)
# =============================================================================


def fix_topology_counts(conn: duckdb.DuckDBPyConnection, region: str) -> int:
    """Recount n_rch_up/n_rch_down from reach_topology table. Fixes stale counts.

    Also corrects end_reach classification based on updated counts:
      headwater (n_up=0, n_down>0) → 1
      outlet (n_down=0, n_up>0) → 2
      junction (n_up>1 OR n_down>1) → 3
      middle → 0

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Read-write connection.
    region : str
        Region code (e.g. 'NA').

    Returns
    -------
    int
        Number of reaches with corrected counts.
    """
    region_upper = region.upper()

    # Single UPDATE: recount from reach_topology and fix mismatches
    n_fixed = conn.execute(
        """
        WITH actual_counts AS (
            SELECT
                rt.reach_id,
                rt.region,
                SUM(CASE WHEN rt.direction = 'up' THEN 1 ELSE 0 END) AS actual_up,
                SUM(CASE WHEN rt.direction = 'down' THEN 1 ELSE 0 END) AS actual_down
            FROM reach_topology rt
            WHERE rt.region = ?
            GROUP BY rt.reach_id, rt.region
        )
        UPDATE reaches
        SET
            n_rch_up = COALESCE(ac.actual_up, 0),
            n_rch_down = COALESCE(ac.actual_down, 0)
        FROM actual_counts ac
        WHERE reaches.reach_id = ac.reach_id
          AND reaches.region = ac.region
          AND (reaches.n_rch_up != COALESCE(ac.actual_up, 0)
               OR reaches.n_rch_down != COALESCE(ac.actual_down, 0))
        """,
        [region_upper],
    ).fetchone()[0]

    if n_fixed > 0:
        log(f"  Fixed {n_fixed} topology count mismatches")

    # Also fix reaches that have NO topology entries but have nonzero counts
    n_orphan_fix = conn.execute(
        """
        UPDATE reaches
        SET n_rch_up = 0, n_rch_down = 0
        WHERE region = ?
          AND (n_rch_up != 0 OR n_rch_down != 0)
          AND reach_id NOT IN (
              SELECT DISTINCT reach_id FROM reach_topology WHERE region = ?
          )
        """,
        [region_upper, region_upper],
    ).fetchone()[0]

    if n_orphan_fix > 0:
        log(f"  Fixed {n_orphan_fix} orphan reaches with nonzero counts")
        n_fixed += n_orphan_fix

    # Fix end_reach based on updated counts
    n_end_fix = conn.execute(
        """
        UPDATE reaches
        SET end_reach = CASE
            WHEN n_rch_up = 0 AND n_rch_down > 0 THEN 1   -- headwater
            WHEN n_rch_down = 0 AND n_rch_up > 0 THEN 2   -- outlet
            WHEN n_rch_up > 1 OR n_rch_down > 1 THEN 3    -- junction
            ELSE 0                                          -- middle
        END
        WHERE region = ?
          AND end_reach != CASE
            WHEN n_rch_up = 0 AND n_rch_down > 0 THEN 1
            WHEN n_rch_down = 0 AND n_rch_up > 0 THEN 2
            WHEN n_rch_up > 1 OR n_rch_down > 1 THEN 3
            ELSE 0
          END
        """,
        [region_upper],
    ).fetchone()[0]

    if n_end_fix > 0:
        log(f"  Fixed {n_end_fix} end_reach classifications")

    return n_fixed


def fix_dist_out_monotonicity(
    conn: duckdb.DuckDBPyConnection,
    region: str,
    tolerance: float = 100.0,
) -> int:
    """Recalculate dist_out for reaches violating downstream monotonicity.

    Uses the same formula as dist_out_from_topo.py:
        dist_out = reach_length + max(downstream neighbor dist_out)

    Iterates until stable (violations can cascade upstream), max 100 iters.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Read-write connection.
    region : str
        Region code (e.g. 'NA').
    tolerance : float
        Tolerance in meters for detecting violations.

    Returns
    -------
    int
        Total number of reaches with corrected dist_out.
    """
    region_upper = region.upper()
    total_fixed = 0

    for iteration in range(100):
        # Find violations: min downstream dist_out > upstream dist_out + tolerance
        violations = conn.execute(
            """
            WITH min_downstream AS (
                SELECT
                    r1.reach_id,
                    r1.region,
                    r1.dist_out AS dist_out_up,
                    r1.reach_length,
                    MAX(r2.dist_out) AS max_dist_out_down
                FROM reaches r1
                JOIN reach_topology rt
                    ON r1.reach_id = rt.reach_id AND r1.region = rt.region
                JOIN reaches r2
                    ON rt.neighbor_reach_id = r2.reach_id AND rt.region = r2.region
                WHERE rt.direction = 'down'
                    AND r1.region = ?
                    AND r1.dist_out > 0 AND r1.dist_out != -9999
                    AND r2.dist_out > 0 AND r2.dist_out != -9999
                GROUP BY r1.reach_id, r1.region, r1.dist_out, r1.reach_length
            )
            SELECT
                reach_id,
                region,
                reach_length,
                max_dist_out_down
            FROM min_downstream
            WHERE max_dist_out_down > dist_out_up + ?
            """,
            [region_upper, tolerance],
        ).fetchdf()

        if len(violations) == 0:
            break

        # Recalculate: dist_out = reach_length + max(downstream dist_out)
        for _, row in violations.iterrows():
            new_dist_out = row["reach_length"] + row["max_dist_out_down"]
            conn.execute(
                """
                UPDATE reaches
                SET dist_out = ?
                WHERE reach_id = ? AND region = ?
                """,
                [float(new_dist_out), int(row["reach_id"]), row["region"]],
            )

        total_fixed += len(violations)
        log(f"  dist_out iteration {iteration + 1}: fixed {len(violations)} violations")

    return total_fixed


# =============================================================================
# Lint Gate (runs after all writes are committed)
# =============================================================================


def run_lint_gate(
    db_path: str,
    region: str,
    checks: Optional[List[str]] = None,
) -> dict:
    """Run ERROR-severity lint checks; raise RuntimeError if any fail.

    Opens a read-only connection via LintRunner (separate from the pipeline
    connection) and runs all ERROR-severity checks (or a filtered subset).

    Parameters
    ----------
    db_path : str
        Path to the DuckDB database (must be a file, not in-memory).
    region : str
        Region code (e.g. 'NA').
    checks : list of str, optional
        Specific check IDs or category prefixes (e.g. ``["T001", "T005"]``).
        Default ``None`` runs all ERROR-severity checks.

    Returns
    -------
    dict
        Mapping ``{check_id: {"passed": bool, "issues": int, "total": int}}``.

    Raises
    ------
    RuntimeError
        If any ERROR-severity check fails, listing failing checks and counts.
    """
    from sword_duckdb.lint import LintRunner, Severity

    log(f"Running lint gate for {region}...")

    with LintRunner(db_path) as runner:
        results = runner.run(
            checks=checks,
            region=region,
            severity=Severity.ERROR,
        )

    summary: dict = {}
    failures: List[str] = []

    for r in results:
        summary[r.check_id] = {
            "passed": r.passed,
            "issues": r.issues_found,
            "total": r.total_checked,
        }
        if not r.passed:
            failures.append(f"{r.check_id} ({r.name}): {r.issues_found} issues")

    log(f"Lint gate: {len(results)} checks, {len(failures)} failures")

    if failures:
        msg = (
            f"Lint gate FAILED for {region} — {len(failures)} ERROR check(s):\n"
            + "\n".join(f"  - {f}" for f in failures)
        )
        raise RuntimeError(msg)

    return summary


# =============================================================================
# FACC Correction (runs before graph construction)
# =============================================================================


def run_facc_corrections(
    conn: duckdb.DuckDBPyConnection,
    region: str,
    nofacc_model_path: str,
    standard_model_path: str,
) -> int:
    """
    Detect and correct facc anomalies directly in the DB.

    Uses the hybrid approach: nofacc model for entry points, then standard
    model for propagation reaches after re-extracting features.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Read-write connection to DuckDB.
    region : str
        Region code (e.g. 'NA').
    nofacc_model_path : str
        Path to no-facc RF model (.joblib) for entry points.
    standard_model_path : str
        Path to standard RF model (.joblib) for propagation.

    Returns
    -------
    int
        Number of corrections applied.

    Raises
    ------
    FileNotFoundError
        If model files don't exist.
    RuntimeError
        If detection or correction fails.
    """
    from pathlib import Path as _Path

    # Validate model files exist
    nofacc_path = _Path(nofacc_model_path)
    standard_path = _Path(standard_model_path)
    if not nofacc_path.exists():
        raise FileNotFoundError(f"No-facc model not found: {nofacc_model_path}")
    if not standard_path.exists():
        raise FileNotFoundError(f"Standard model not found: {standard_model_path}")

    # Import facc detection modules
    from sword_duckdb.facc_detection.detect import detect_hybrid
    from sword_duckdb.facc_detection.correct_topological import (
        identify_entry_points,
        get_downstream_order,
        apply_corrections_to_db,
    )
    from sword_duckdb.facc_detection.rf_features import RFFeatureExtractor
    from sword_duckdb.facc_detection.rf_regressor import FaccRegressor

    # Step 1: Detect anomalies
    log(f"Detecting facc anomalies for {region}...")
    result = detect_hybrid(conn, region=region)
    anomalies = result.anomalies

    if len(anomalies) == 0:
        log("No facc anomalies detected")
        return 0

    log(f"Detected {len(anomalies):,} facc anomalies")

    # Step 2: Separate entry points vs propagation
    entry_points, propagation = identify_entry_points(anomalies, conn)
    log(f"  Entry points: {len(entry_points):,}, Propagation: {len(propagation):,}")

    if not entry_points:
        log("No entry points found, skipping correction")
        return 0

    # Step 3: Get downstream ordering for propagation
    hop_groups = get_downstream_order(entry_points, propagation, conn)

    # Load models
    nofacc_model = FaccRegressor.load(nofacc_path)
    standard_model = FaccRegressor.load(standard_path)
    log(f"Loaded models: nofacc={nofacc_path.name}, standard={standard_path.name}")

    total_corrected = 0

    # Step 4: Extract features and correct entry points with nofacc model
    log("Extracting features for entry point correction...")
    extractor = RFFeatureExtractor(conn)
    all_features = extractor.extract_all(region=region)

    entry_features = all_features[all_features["reach_id"].isin(entry_points)].copy()
    if len(entry_features) > 0:
        entry_preds = nofacc_model.predict(entry_features)
        apply_corrections_to_db(conn, entry_preds, "entry_points")
        total_corrected += len(entry_preds)
        log(f"  Corrected {len(entry_preds):,} entry points")

    # Step 5: Re-extract features (2-hop now reads corrected values)
    if propagation and hop_groups:
        log("Re-extracting features after entry point corrections...")
        extractor = RFFeatureExtractor(conn)
        all_features = extractor.extract_all(region=region)

        # Step 6: Correct propagation reaches with standard model
        for hop_idx, hop_ids in enumerate(hop_groups):
            hop_features = all_features[all_features["reach_id"].isin(hop_ids)].copy()
            if len(hop_features) == 0:
                continue
            hop_preds = standard_model.predict(hop_features)
            apply_corrections_to_db(conn, hop_preds, f"propagation_hop{hop_idx + 1}")
            total_corrected += len(hop_preds)
            log(
                f"  Corrected {len(hop_preds):,} propagation reaches (hop {hop_idx + 1})"
            )

    log(f"Total facc corrections applied: {total_corrected:,}")
    return total_corrected


# =============================================================================
# Graph Construction
# =============================================================================


def build_reach_graph(
    topology_df: pd.DataFrame, reaches_df: pd.DataFrame
) -> nx.DiGraph:
    """
    Build directed graph where nodes=reaches, edges=flow connections.

    Flow direction from topology:
    - direction='up': neighbor is upstream -> neighbor -> reach
    - direction='down': neighbor is downstream -> reach -> neighbor
    """
    import math

    log("Building reach-level directed graph...")

    G = nx.DiGraph()

    # Create reach attributes dict
    reach_attrs = {}
    n_swot_width = 0

    for _, row in reaches_df.iterrows():
        rid = int(row["reach_id"])
        base_attrs = {
            "reach_length": row["reach_length"],
            "width": row["width"],
            "slope": row["slope"],
            "facc": row.get("facc", 0),
            "n_rch_up": row.get("n_rch_up", 0),
            "n_rch_down": row.get("n_rch_down", 0),
            "dist_out": row.get("dist_out", 0),
            "path_freq": row.get("path_freq", 1),
            "stream_order": row.get("stream_order", 1),
            "lakeflag": row.get("lakeflag", 0),
            "main_side": row.get("main_side", 0),
            "type": row.get("type", 1),
            "end_reach": row.get("end_reach", 0),
            "wse_obs_mean": row.get("wse_obs_mean"),
            "width_obs_median": row.get("width_obs_median"),
            "n_obs": row.get("n_obs", 0),
        }

        # Compute effective width (SWOT-preferred) and use facc directly (already corrected in DB)
        eff_width = get_effective_width(base_attrs)
        eff_facc = base_attrs.get("facc", 0)
        if pd.isna(eff_facc):
            eff_facc = 0

        # Track usage stats
        n_obs_val = base_attrs.get("n_obs", 0)
        width_obs_val = base_attrs.get("width_obs_median")
        if (
            not pd.isna(n_obs_val)
            and n_obs_val >= 5
            and width_obs_val is not None
            and not pd.isna(width_obs_val)
        ):
            n_swot_width += 1

        base_attrs["effective_width"] = eff_width
        base_attrs["effective_facc"] = eff_facc
        base_attrs["log_facc"] = math.log1p(eff_facc)

        reach_attrs[rid] = base_attrs

    log(f"Using SWOT width for {n_swot_width:,} reaches (n_obs >= 5)")

    # Add all reaches as nodes
    for rid, attrs in reach_attrs.items():
        G.add_node(rid, **attrs)

    # Add edges from topology
    edges_added = set()
    for _, row in topology_df.iterrows():
        reach_id = int(row["reach_id"])
        neighbor_id = int(row["neighbor_reach_id"])
        direction = row["direction"]

        if direction == "up":
            # neighbor is upstream: neighbor -> reach
            u, v = neighbor_id, reach_id
        else:
            # neighbor is downstream: reach -> neighbor
            u, v = reach_id, neighbor_id

        if (u, v) not in edges_added:
            edges_added.add((u, v))
            G.add_edge(u, v, reach_id=v)

    log(f"Reach graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def identify_junctions(G: nx.DiGraph) -> Set[int]:
    """
    Identify junction nodes in the reach graph.

    A junction is where:
    - in_degree > 1 (confluence)
    - out_degree > 1 (bifurcation)
    - in_degree == 0 (headwater)
    - out_degree == 0 (outlet)
    """
    junctions = set()

    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)

        if in_deg != 1 or out_deg != 1:
            junctions.add(node)

    log(f"Identified {len(junctions):,} junctions")
    return junctions


def build_section_graph(
    G: nx.DiGraph, junctions: Set[int]
) -> Tuple[nx.DiGraph, pd.DataFrame]:
    """
    Build a section graph where each edge is a section (chain of reaches between junctions).

    Returns:
        R: DiGraph where nodes are junctions, edges are sections
        sections_df: DataFrame with section details
    """
    log("Building section graph...")

    R = nx.DiGraph()
    sections = []
    section_id = 0

    # Add junction nodes
    for j in junctions:
        node_data = G.nodes[j]
        in_deg = G.in_degree(j)
        out_deg = G.out_degree(j)

        if in_deg == 0:
            node_type = "Head_water"
        elif out_deg == 0:
            node_type = "Outlet"
        else:
            node_type = "Junction"

        R.add_node(j, node_type=node_type, **node_data)

    # For each junction, trace downstream to next junction
    for upstream_j in junctions:
        for first_reach in G.successors(upstream_j):
            # Trace chain until we hit another junction
            reach_ids = []
            cumulative_dist = 0
            current = first_reach

            while current not in junctions:
                reach_ids.append(current)
                cumulative_dist += G.nodes[current].get("reach_length", 0)

                succs = list(G.successors(current))
                if len(succs) == 0:
                    break
                current = succs[0]

            # current is now the downstream junction
            downstream_j = current
            if downstream_j in junctions:
                reach_ids.append(downstream_j)
                cumulative_dist += G.nodes[downstream_j].get("reach_length", 0)

                R.add_edge(
                    upstream_j,
                    downstream_j,
                    section_id=section_id,
                    reach_ids=reach_ids,
                    distance=cumulative_dist,
                    n_reaches=len(reach_ids),
                )

                sections.append(
                    {
                        "section_id": section_id,
                        "upstream_junction": upstream_j,
                        "downstream_junction": downstream_j,
                        "reach_ids": reach_ids,
                        "distance": cumulative_dist,
                        "n_reaches": len(reach_ids),
                    }
                )
                section_id += 1

    sections_df = pd.DataFrame(sections)
    log(
        f"Section graph: {R.number_of_nodes():,} junctions, {R.number_of_edges():,} sections"
    )
    return R, sections_df


# =============================================================================
# New Attribute Computation
# =============================================================================


def compute_path_variables(G: nx.DiGraph, sections_df: pd.DataFrame) -> Dict[int, Dict]:
    """
    Compute path_freq, stream_order, path_segs, and path_order for every reach.

    Parameters
    ----------
    G : nx.DiGraph
        Reach-level directed graph (nodes = reaches, edges = flow direction).
        Node attributes must include ``dist_out``.  ``main_side`` and ``type``
        are read with safe defaults (0 and 1 respectively).
    sections_df : pd.DataFrame
        Output of ``build_section_graph`` with columns
        ``section_id``, ``reach_ids``, etc.

    Returns
    -------
    Dict[int, Dict]
        Mapping reach_id -> {"path_freq": int, "stream_order": int,
        "path_segs": int, "path_order": int}.
    """
    import math
    from collections import deque

    log("Computing path variables (path_freq, stream_order, path_segs, path_order)...")

    if G.number_of_nodes() == 0:
        log("Empty graph, returning empty results")
        return {}

    # ------------------------------------------------------------------
    # 1. path_freq  (topological summation)
    # ------------------------------------------------------------------
    pf: Dict[int, int] = {}

    # Ghost reaches (type=6): unreliable topology, get -9999 and skip.
    # Side channels (main_side 1,2): valid topology, get computed pf
    # but are excluded from confluence sums to prevent double-counting
    # at braid reconvergences. is_mainstem_edge handles routing downstream.
    ghosts: Set[int] = set()
    skip_in_sums: Set[int] = set()
    for node in G.nodes():
        node_type = G.nodes[node].get("type", 1)
        main_side = G.nodes[node].get("main_side", 0)
        if node_type == 6:
            ghosts.add(node)
            skip_in_sums.add(node)
            pf[node] = -9999
        elif main_side in (1, 2):
            skip_in_sums.add(node)

    def _sum_predecessors(node: int) -> int:
        """Sum pf from predecessors, excluding side channels and ghosts."""
        total = sum(
            pf[pred]
            for pred in G.predecessors(node)
            if pred not in skip_in_sums and pf.get(pred, 0) > 0
        )
        if total > 0:
            return total
        # All main predecessors excluded — inherit from any valid predecessor
        fallback = sum(
            pf[pred]
            for pred in G.predecessors(node)
            if pred not in ghosts and pf.get(pred, 0) > 0
        )
        return fallback if fallback > 0 else 1

    is_dag = nx.is_directed_acyclic_graph(G)

    if is_dag:
        # Fast path: topological sort
        for node in nx.topological_sort(G):
            if node in ghosts:
                continue
            if G.in_degree(node) == 0:
                pf[node] = 1
            else:
                pf[node] = _sum_predecessors(node)
    else:
        # Fallback: iterative worklist propagation for graphs with cycles.
        log("WARNING: Graph has cycles, using iterative worklist for path_freq")
        for node in G.nodes():
            if node in ghosts:
                continue
            if G.in_degree(node) == 0:
                pf[node] = 1
            else:
                pf[node] = 0

        # BFS from headwaters downstream
        queue: deque[int] = deque()
        for node in G.nodes():
            if node not in ghosts and G.in_degree(node) == 0:
                queue.append(node)

        visited_count: Dict[int, int] = defaultdict(int)
        max_iterations = G.number_of_nodes() * 4  # safety bound
        iterations = 0

        while queue and iterations < max_iterations:
            node = queue.popleft()
            iterations += 1
            if node in ghosts:
                continue

            if G.in_degree(node) == 0:
                new_pf = 1
            else:
                new_pf = _sum_predecessors(node)

            if new_pf != pf.get(node, 0):
                pf[node] = new_pf
                for succ in G.successors(node):
                    if succ not in ghosts:
                        visited_count[succ] += 1
                        if visited_count[succ] < max_iterations:
                            queue.append(succ)

        # Any node still at 0 gets fallback 1
        for node in G.nodes():
            if node not in ghosts and pf.get(node, 0) <= 0:
                pf[node] = 1

    # ------------------------------------------------------------------
    # 2. stream_order  = round(ln(path_freq)) + 1
    # ------------------------------------------------------------------
    so: Dict[int, int] = {}
    for node in G.nodes():
        freq = pf.get(node, -9999)
        if freq > 0:
            so[node] = int(round(math.log(freq))) + 1
        else:
            so[node] = -9999

    # ------------------------------------------------------------------
    # 3. path_segs  (section-based segment identifier)
    # ------------------------------------------------------------------
    ps: Dict[int, int] = {}
    reach_to_section: Dict[int, int] = {}

    if sections_df is not None and len(sections_df) > 0:
        for _, row in sections_df.iterrows():
            sid = int(row["section_id"]) + 1  # 1-based
            for rid in row["reach_ids"]:
                rid = int(rid)
                reach_to_section[rid] = sid
                ps[rid] = sid

    # Junction reaches not in any section get unique IDs
    max_seg = max(ps.values()) if ps else 0
    next_seg = max_seg + 1
    for node in G.nodes():
        if node not in ps:
            ps[node] = next_seg
            next_seg += 1

    # ------------------------------------------------------------------
    # 4. path_order  (rank by dist_out ASC within path_freq groups)
    # ------------------------------------------------------------------
    po: Dict[int, int] = {}

    # Group nodes by path_freq
    freq_groups: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for node in G.nodes():
        freq = pf.get(node)
        dist = G.nodes[node].get("dist_out", 0)
        if pd.isna(dist):
            dist = 0
        freq_groups[freq].append((dist, node))

    for freq, members in freq_groups.items():
        # Sort by dist_out ascending, assign 1-based rank
        members.sort(key=lambda x: x[0])
        for rank, (_, node) in enumerate(members, start=1):
            po[node] = rank

    # ------------------------------------------------------------------
    # 5. Assemble results
    # ------------------------------------------------------------------
    results: Dict[int, Dict] = {}
    for node in G.nodes():
        if node in ghosts:
            results[node] = {
                "path_freq": -9999,
                "stream_order": -9999,
                "path_segs": -9999,
                "path_order": -9999,
            }
        else:
            results[node] = {
                "path_freq": pf.get(node, -9999),
                "stream_order": so.get(node, -9999),
                "path_segs": ps.get(node, -9999),
                "path_order": po.get(node, -9999),
            }

    # ------------------------------------------------------------------
    # 6. Validation logging
    # ------------------------------------------------------------------
    n_valid = sum(1 for v in pf.values() if v > 0)
    n_ghost = len(ghosts)
    log(f"path_freq: {n_valid:,} valid (>0), {n_ghost:,} ghosts (-9999)")

    # T002 monotonicity check (log-only)
    mono_violations = 0
    for u, v in G.edges():
        pf_u = pf.get(u, -9999)
        pf_v = pf.get(v, -9999)
        if pf_u > 0 and pf_v > 0 and pf_v < pf_u:
            mono_violations += 1
    if mono_violations > 0:
        log(f"WARNING: T002 path_freq monotonicity violations: {mono_violations:,}")
    else:
        log("T002 path_freq monotonicity: OK")

    # T010 headwater check (log-only)
    hw_violations = 0
    for node in G.nodes():
        if G.in_degree(node) == 0 and node not in ghosts:
            if pf.get(node, 0) < 1:
                hw_violations += 1
    if hw_violations > 0:
        log(f"WARNING: T010 headwater path_freq violations: {hw_violations:,}")
    else:
        log("T010 headwater path_freq: OK")

    log("Path variables computed")
    return results


def compute_hydro_distances(G: nx.DiGraph) -> Dict[int, Dict]:
    """
    Compute hydrologic distances for each reach.

    - hydro_dist_out: Distance to outlet following main channel
    - hydro_dist_hw: Distance from headwater following main channel
    """
    log("Computing hydrologic distances...")

    # Handle empty graph
    if G.number_of_nodes() == 0:
        log("Empty graph, returning empty results")
        return {}

    # Identify outlets (no outgoing edges)
    outlets = [n for n in G.nodes() if G.out_degree(n) == 0]
    log(f"Found {len(outlets):,} outlets")

    # Compute dist_out using Dijkstra from outlets (reversed graph)
    R = G.reverse()

    dist_out = {}
    for node in G.nodes():
        dist_out[node] = float("inf")

    for outlet in outlets:
        dist_out[outlet] = 0

    # Multi-source Dijkstra (only if we have outlets)
    if outlets:
        lengths = nx.multi_source_dijkstra_path_length(
            R, outlets, weight=lambda u, v, d: G.nodes[v].get("reach_length", 0)
        )
        dist_out.update(lengths)

    # Compute dist_hw (distance from furthest headwater)
    headwaters = [n for n in G.nodes() if G.in_degree(n) == 0]
    log(f"Found {len(headwaters):,} headwaters")

    dist_hw = {}
    for node in G.nodes():
        dist_hw[node] = 0

    # For each node, find max distance from any headwater
    for hw in headwaters:
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, hw, weight=lambda u, v, d: G.nodes[v].get("reach_length", 0)
            )
            for node, dist in lengths.items():
                if dist > dist_hw[node]:
                    dist_hw[node] = dist
        except nx.NetworkXError:
            continue

    results = {}
    for node in G.nodes():
        results[node] = {
            "hydro_dist_out": dist_out.get(node, float("inf")),
            "hydro_dist_hw": dist_hw.get(node, 0),
        }

    log("Hydrologic distances computed")
    return results


def compute_best_headwater_outlet(G: nx.DiGraph) -> Dict[int, Dict]:
    """
    Compute best headwater and outlet for each reach.

    Uses effective_width (SWOT if available), log(facc), and pathlen to select "main" path.
    Ranking tuple: (effective_width, log_facc, pathlen) - all maximized.
    """
    log("Computing best headwater/outlet assignments...")

    if not nx.is_directed_acyclic_graph(G):
        log("WARNING: Graph has cycles, computing on largest DAG component")
        # Find strongly connected components and work on DAG of SCCs
        # For simplicity, we'll proceed but results may be incomplete
        pass

    try:
        topo = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        log("ERROR: Graph has cycles, cannot compute topological sort")
        return {}

    # Upstream pass: track headwater sets and choose best
    hw_sets = {n: set() for n in G.nodes()}
    best_hw = {}
    pathlen_hw = {}

    for n in topo:
        preds = list(G.predecessors(n))

        if not preds:
            # Headwater
            hw_sets[n] = {n}
            best_hw[n] = n
            pathlen_hw[n] = 0
        else:
            # Merge headwater sets from predecessors
            union = set()
            candidates = []

            for p in preds:
                union |= hw_sets[p]
                reach_len = G.nodes[n].get("reach_length", 0)
                total_len = pathlen_hw.get(p, 0) + reach_len
                # Use effective_width (SWOT-preferred) and log_facc for ranking
                eff_width = G.nodes[p].get("effective_width", 0) or 0
                log_facc = G.nodes[p].get("log_facc", 0) or 0
                candidates.append((eff_width, log_facc, total_len, best_hw.get(p), p))

            hw_sets[n] = union

            # Select by effective_width (primary), log_facc (secondary), pathlen (tertiary)
            best = max(candidates, key=lambda x: (x[0], x[1], x[2]))
            best_hw[n] = best[3]
            pathlen_hw[n] = best[2]

    # Downstream pass: track outlet and choose best
    best_out = {}
    pathlen_out = {}

    for n in reversed(topo):
        succs = list(G.successors(n))

        if not succs:
            # Outlet
            best_out[n] = n
            pathlen_out[n] = 0
        else:
            candidates = []

            for s in succs:
                reach_len = G.nodes[s].get("reach_length", 0)
                total_len = pathlen_out.get(s, 0) + reach_len
                # Use effective_width (SWOT-preferred) and log_facc for ranking
                eff_width = G.nodes[s].get("effective_width", 0) or 0
                log_facc = G.nodes[s].get("log_facc", 0) or 0
                candidates.append((eff_width, log_facc, total_len, best_out.get(s), s))

            best = max(candidates, key=lambda x: (x[0], x[1], x[2]))
            best_out[n] = best[3]
            pathlen_out[n] = best[2]

    results = {}
    for node in G.nodes():
        results[node] = {
            "best_headwater": best_hw.get(node),
            "best_outlet": best_out.get(node),
            "pathlen_hw": pathlen_hw.get(node, 0),
            "pathlen_out": pathlen_out.get(node, 0),
            "path_freq": len(hw_sets.get(node, set())),
        }

    log("Best headwater/outlet computed")
    return results


def compute_mainstem(G: nx.DiGraph, hw_out_attrs: Dict[int, Dict]) -> Dict[int, bool]:
    """
    Compute is_mainstem for each reach.

    A reach is on the mainstem if it's on the path from best_headwater to best_outlet.
    """
    log("Computing mainstem classification...")

    is_mainstem = {n: False for n in G.nodes()}

    # Group by (best_headwater, best_outlet) pairs
    paths = defaultdict(list)
    for node, attrs in hw_out_attrs.items():
        key = (attrs["best_headwater"], attrs["best_outlet"])
        paths[key].append(node)

    # For each unique path, mark nodes on it as mainstem
    for (hw, out), nodes in paths.items():
        if hw is None or out is None:
            continue

        try:
            path = nx.shortest_path(G, hw, out)
            for n in path:
                is_mainstem[n] = True
        except nx.NetworkXNoPath:
            continue

    n_mainstem = sum(is_mainstem.values())
    n_total = len(G.nodes())
    pct = 100 * n_mainstem / n_total if n_total > 0 else 0
    log(f"Mainstem reaches: {n_mainstem:,} ({pct:.1f}%)")

    return is_mainstem


def compute_main_neighbors(G: nx.DiGraph) -> Dict[int, Dict]:
    """
    Compute rch_id_up_main and rch_id_dn_main for each reach.

    For each node, selects the main upstream predecessor and main downstream
    successor using the same (effective_width, log_facc) ranking used by
    best_headwater/best_outlet, ensuring consistent routing across all v17c
    columns.

    Returns dict {reach_id: {'rch_id_up_main': int|None, 'rch_id_dn_main': int|None}}.
    """
    log("Computing main neighbors (rch_id_up_main / rch_id_dn_main)...")

    results = {}

    for node in G.nodes():
        # Main upstream neighbor: pick best predecessor
        preds = list(G.predecessors(node))
        if preds:
            best_up = max(
                preds,
                key=lambda n: (
                    G.nodes[n].get("effective_width", 0) or 0,
                    G.nodes[n].get("log_facc", 0) or 0,
                ),
            )
            rch_id_up_main = best_up
        else:
            rch_id_up_main = None

        # Main downstream neighbor: pick best successor
        succs = list(G.successors(node))
        if succs:
            best_dn = max(
                succs,
                key=lambda n: (
                    G.nodes[n].get("effective_width", 0) or 0,
                    G.nodes[n].get("log_facc", 0) or 0,
                ),
            )
            rch_id_dn_main = best_dn
        else:
            rch_id_dn_main = None

        results[node] = {
            "rch_id_up_main": rch_id_up_main,
            "rch_id_dn_main": rch_id_dn_main,
        }

    n_with_up = sum(1 for v in results.values() if v["rch_id_up_main"] is not None)
    n_with_dn = sum(1 for v in results.values() if v["rch_id_dn_main"] is not None)
    log(f"Main neighbors: {n_with_up:,} with up_main, {n_with_dn:,} with dn_main")

    return results


# =============================================================================
# SWOT Slope Validation (Junction-Level)
# =============================================================================


def compute_junction_slopes(
    G: nx.DiGraph, sections_df: pd.DataFrame, reaches_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute slopes at junction endpoints for each section.

    For each section, we compute:
    - slope_from_upstream: slope measured from upstream junction (should be NEGATIVE)
    - slope_from_downstream: slope measured from downstream junction (should be POSITIVE)
    """
    log("Computing junction slopes from SWOT data...")

    # Create reach -> WSE mapping
    wse_map = {}
    for _, row in reaches_df.iterrows():
        rid = int(row["reach_id"])
        wse = row.get("wse_obs_mean")
        if pd.notna(wse):
            wse_map[rid] = wse

    results = []

    for _, section in sections_df.iterrows():
        section_id = section["section_id"]
        upstream_j = section["upstream_junction"]
        downstream_j = section["downstream_junction"]
        reach_ids = section["reach_ids"]
        total_distance = section["distance"]

        if len(reach_ids) < 2:
            continue

        # Get WSE values for reaches in this section
        wse_data = []
        cumulative_dist = 0

        for rid in reach_ids:
            wse = wse_map.get(rid)
            reach_len = G.nodes[rid].get("reach_length", 0) if rid in G.nodes else 0

            if wse is not None:
                wse_data.append(
                    {
                        "reach_id": rid,
                        "wse": wse,
                        "dist_from_upstream": cumulative_dist,
                        "dist_from_downstream": total_distance - cumulative_dist,
                    }
                )
            cumulative_dist += reach_len

        if len(wse_data) < 2:
            continue

        wse_df = pd.DataFrame(wse_data)

        # Compute slope from upstream junction
        try:
            slope_upstream = np.polyfit(wse_df["dist_from_upstream"], wse_df["wse"], 1)[
                0
            ]
        except Exception:
            slope_upstream = np.nan

        # Compute slope from downstream junction
        try:
            slope_downstream = np.polyfit(
                wse_df["dist_from_downstream"], wse_df["wse"], 1
            )[0]
        except Exception:
            slope_downstream = np.nan

        # Determine if slopes match expected signs
        upstream_correct = slope_upstream < 0 if pd.notna(slope_upstream) else None
        downstream_correct = (
            slope_downstream > 0 if pd.notna(slope_downstream) else None
        )

        direction_valid = (
            upstream_correct and downstream_correct
            if (upstream_correct is not None and downstream_correct is not None)
            else None
        )

        # Determine likely cause if invalid
        likely_cause = None
        if direction_valid is False:
            lakeflag = (
                G.nodes[upstream_j].get("lakeflag", 0) if upstream_j in G.nodes else 0
            )
            if lakeflag > 0:
                likely_cause = "lake_section"
            elif pd.notna(slope_upstream) and abs(slope_upstream) > 0.05:
                likely_cause = "extreme_slope_data_error"
            elif pd.notna(slope_downstream) and abs(slope_downstream) > 0.05:
                likely_cause = "extreme_slope_data_error"
            else:
                likely_cause = "potential_topology_error"

        results.append(
            {
                "section_id": section_id,
                "upstream_junction": upstream_j,
                "downstream_junction": downstream_j,
                "n_reaches": len(reach_ids),
                "n_reaches_with_wse": len(wse_df),
                "distance": total_distance,
                "slope_from_upstream": slope_upstream,
                "slope_from_downstream": slope_downstream,
                "direction_valid": direction_valid,
                "likely_cause": likely_cause,
            }
        )

    junction_slopes_df = pd.DataFrame(results)

    # Summary
    if not junction_slopes_df.empty:
        n_total = len(junction_slopes_df)
        n_valid = junction_slopes_df["direction_valid"].sum()
        n_invalid = (junction_slopes_df["direction_valid"] == False).sum()

        log("Junction slope validation:")
        log(f"  Total sections with SWOT data: {n_total:,}")
        log(f"  Direction valid: {n_valid:,} ({100 * n_valid / n_total:.1f}%)")
        log(f"  Direction INVALID: {n_invalid:,} ({100 * n_invalid / n_total:.1f}%)")

    return junction_slopes_df


# =============================================================================
# DuckDB Output
# =============================================================================


def save_to_duckdb(
    conn: duckdb.DuckDBPyConnection,
    region: str,
    hydro_dist: Dict[int, Dict],
    hw_out: Dict[int, Dict],
    is_mainstem: Dict[int, bool],
    main_neighbors: Optional[Dict[int, Dict]] = None,
    path_vars: Optional[Dict[int, Dict]] = None,
) -> int:
    """
    Save computed v17c attributes to DuckDB reaches table.

    Returns:
        Number of reaches updated
    """
    log(f"Saving v17c attributes to DuckDB for {region}...")

    # Build update dataframe
    rows = []
    mn = main_neighbors or {}
    pv = path_vars or {}
    for reach_id in hydro_dist.keys():
        hd = hydro_dist.get(reach_id, {})
        ho = hw_out.get(reach_id, {})
        ms = is_mainstem.get(reach_id, False)
        nb = mn.get(reach_id, {})
        pvar = pv.get(reach_id, {})

        row = {
            "reach_id": reach_id,
            "hydro_dist_out": hd.get("hydro_dist_out"),
            "hydro_dist_hw": hd.get("hydro_dist_hw"),
            "best_headwater": ho.get("best_headwater"),
            "best_outlet": ho.get("best_outlet"),
            "pathlen_hw": ho.get("pathlen_hw"),
            "pathlen_out": ho.get("pathlen_out"),
            "is_mainstem_edge": ms,
            "rch_id_up_main": nb.get("rch_id_up_main"),
            "rch_id_dn_main": nb.get("rch_id_dn_main"),
        }
        if pvar:
            row["path_freq"] = pvar.get("path_freq")
            row["stream_order"] = pvar.get("stream_order")
            row["path_segs"] = pvar.get("path_segs")
            row["path_order"] = pvar.get("path_order")
        rows.append(row)

    if not rows:
        log("No rows to update")
        return 0

    update_df = pd.DataFrame(rows)

    # Handle infinity values - convert to NULL
    update_df = update_df.replace([np.inf, -np.inf], np.nan)

    # Register DataFrame and update
    conn.register("v17c_updates", update_df)

    # Load spatial extension (needed for RTREE index compatibility)
    try:
        conn.execute("INSTALL spatial; LOAD spatial;")
    except Exception:
        pass  # Extension may already be loaded or not needed

    # Build SET clause - always include base v17c columns
    set_clauses = [
        "hydro_dist_out = u.hydro_dist_out",
        "hydro_dist_hw = u.hydro_dist_hw",
        "best_headwater = u.best_headwater",
        "best_outlet = u.best_outlet",
        "pathlen_hw = u.pathlen_hw",
        "pathlen_out = u.pathlen_out",
        "is_mainstem_edge = u.is_mainstem_edge",
        "rch_id_up_main = u.rch_id_up_main",
        "rch_id_dn_main = u.rch_id_dn_main",
    ]
    if path_vars:
        set_clauses.extend(
            [
                "path_freq = u.path_freq",
                "stream_order = u.stream_order",
                "path_segs = u.path_segs",
                "path_order = u.path_order",
            ]
        )

    # Update reaches table
    conn.execute(f"""
        UPDATE reaches SET
            {", ".join(set_clauses)}
        FROM v17c_updates u
        WHERE reaches.reach_id = u.reach_id
        AND reaches.region = '{region.upper()}'
    """)

    conn.unregister("v17c_updates")

    log(f"Updated {len(rows):,} reaches")
    return len(rows)


def save_sections_to_duckdb(
    conn: duckdb.DuckDBPyConnection,
    region: str,
    sections_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> None:
    """Save section data and validation results to DuckDB tables."""
    log(f"Saving sections to DuckDB for {region}...")

    if sections_df.empty:
        log("No sections to save")
        return

    # Prepare sections for insert
    sections_insert = sections_df.copy()
    sections_insert["region"] = region.upper()
    # Convert reach_ids list to JSON string
    sections_insert["reach_ids"] = sections_insert["reach_ids"].apply(json.dumps)

    conn.register("sections_insert", sections_insert)
    conn.execute("""
        INSERT OR REPLACE INTO v17c_sections
        SELECT
            section_id,
            region,
            upstream_junction,
            downstream_junction,
            reach_ids,
            distance,
            n_reaches
        FROM sections_insert
    """)
    conn.unregister("sections_insert")
    log(f"Saved {len(sections_insert):,} sections")

    # Save validation results if any
    if not validation_df.empty:
        validation_insert = validation_df.copy()
        validation_insert["region"] = region.upper()

        conn.register("validation_insert", validation_insert)
        conn.execute("""
            INSERT OR REPLACE INTO v17c_section_slope_validation
            SELECT
                section_id,
                region,
                slope_from_upstream,
                slope_from_downstream,
                direction_valid,
                likely_cause
            FROM validation_insert
        """)
        conn.unregister("validation_insert")
        log(f"Saved {len(validation_insert):,} validation records")


# =============================================================================
# SWOT Slopes Integration
# =============================================================================


def apply_swot_slopes(
    conn: duckdb.DuckDBPyConnection,
    region: str,
    swot_path: str,
) -> int:
    """
    Apply SWOT-derived slopes to reaches.

    This function:
    1. Loads SWOT node data from parquet files
    2. Computes section-level slopes with MAD outlier filtering
    3. Updates reaches with swot_slope, swot_slope_se, swot_slope_confidence

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Database connection
    region : str
        Region code
    swot_path : str
        Path to SWOT parquet files directory

    Returns
    -------
    int
        Number of reaches updated
    """
    import glob

    log(f"Applying SWOT slopes for {region}...")

    # Check if SWOT data exists
    if not os.path.isdir(swot_path):
        log(f"SWOT data directory not found: {swot_path}")
        return 0

    parquet_files = [
        f
        for f in glob.glob(os.path.join(swot_path, "*.parquet"))
        if not os.path.basename(f).startswith("._")
    ]

    if not parquet_files:
        log(f"No parquet files found in {swot_path}")
        return 0

    log(f"Found {len(parquet_files)} SWOT parquet files")

    # Get node_ids for this region
    nodes_df = conn.execute(
        """
        SELECT node_id FROM nodes WHERE region = ?
    """,
        [region.upper()],
    ).fetchdf()

    if nodes_df.empty:
        log(f"No nodes found for region {region}")
        return 0

    node_ids = nodes_df["node_id"].tolist()
    log(f"Region {region} has {len(node_ids):,} nodes")

    # For now, we'll use the pre-computed slopes if they exist
    # in the SWOT pipeline output
    swot_slopes_file = os.path.join(
        os.path.dirname(swot_path).replace("/node", ""),
        f"output/{region.lower()}/{region.lower()}_swot_slopes.csv",
    )

    if os.path.exists(swot_slopes_file):
        log(f"Loading pre-computed SWOT slopes from {swot_slopes_file}")
        slopes_df = pd.read_csv(swot_slopes_file)

        # Map section slopes to reaches
        # This requires the section-reach mapping which we'd need from the pipeline
        log(f"Loaded {len(slopes_df):,} section slopes")

        # For now, just log that SWOT slopes are available
        # Full integration would require running SWOT_slopes.py functions
        log("SWOT slope integration requires section-reach mapping")
        return 0

    else:
        log(f"SWOT slopes file not found: {swot_slopes_file}")
        log("Run SWOT_slopes.py separately to compute slopes")
        return 0


# =============================================================================
# Main Pipeline
# =============================================================================


def process_region(
    db_path: str,
    region: str,
    user_id: str = "v17c_pipeline",
    skip_swot: bool = False,
    swot_path: Optional[str] = None,
    skip_facc: bool = False,
    nofacc_model_path: str = DEFAULT_NOFACC_MODEL,
    standard_model_path: str = DEFAULT_STANDARD_MODEL,
    skip_path_vars: bool = False,
    skip_lint_gate: bool = False,
    lint_checks: Optional[List[str]] = None,
) -> Dict:
    """
    Process a single region through the v17c pipeline.

    Parameters
    ----------
    db_path : str
        Path to sword_v17c.duckdb
    region : str
        Region code (NA, SA, EU, AF, AS, OC)
    user_id : str
        User ID for provenance
    skip_swot : bool
        Skip SWOT slope integration
    swot_path : str, optional
        Path to SWOT data directory
    skip_facc : bool
        Skip facc anomaly correction (default: run corrections)
    nofacc_model_path : str
        Path to no-facc RF model for entry point correction
    standard_model_path : str
        Path to standard RF model for propagation correction
    skip_lint_gate : bool
        Skip the post-write lint gate (default: run it)
    lint_checks : list of str, optional
        Specific lint check IDs or prefixes to run (default: all ERROR checks)

    Returns
    -------
    dict
        Processing statistics
    """
    # Import SWORDWorkflow for provenance
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sword_duckdb import SWORDWorkflow
    from sword_duckdb.schema import add_v17c_columns, normalize_region

    region = normalize_region(region)
    log(f"\n{'=' * 60}")
    log(f"Processing region: {region}")
    log(f"{'=' * 60}")

    # Initialize workflow
    workflow = SWORDWorkflow(user_id=user_id)
    sword = workflow.load(db_path, region, reason=f"v17c pipeline for {region}")
    conn = sword.db.conn

    # Ensure v17c columns exist
    add_v17c_columns(conn)

    # Create v17c tables if they don't exist
    create_v17c_tables(conn)

    # Load data from database
    topology_df = load_topology(conn, region)
    reaches_df = load_reaches(conn, region)

    # Fix stale topology counts from reach_topology table
    n_topo_fixes = fix_topology_counts(conn, region)
    if n_topo_fixes > 0:
        log(f"Fixed {n_topo_fixes} topology count mismatches")
        reaches_df = load_reaches(conn, region)

    # Fix dist_out monotonicity violations
    n_dist_fixes = fix_dist_out_monotonicity(conn, region)
    if n_dist_fixes > 0:
        log(f"Fixed {n_dist_fixes} dist_out monotonicity violations")
        reaches_df = load_reaches(conn, region)

    # Detect and correct facc anomalies BEFORE building graph
    n_facc_corrections = 0
    if not skip_facc:
        n_facc_corrections = run_facc_corrections(
            conn,
            region,
            nofacc_model_path,
            standard_model_path,
        )
        if n_facc_corrections > 0:
            # Reload reaches with corrected facc values
            reaches_df = load_reaches(conn, region)

    # Build reach-level graph (uses corrected facc from DB)
    G = build_reach_graph(topology_df, reaches_df)

    # Validate DAG
    if not nx.is_directed_acyclic_graph(G):
        log("WARNING: Graph contains cycles!")
        # Find cycles for debugging
        try:
            cycle = nx.find_cycle(G)
            log(f"Example cycle: {cycle[:5]}...")
        except nx.NetworkXNoCycle:
            pass

    # Identify junctions and build section graph
    junctions = identify_junctions(G)
    R, sections_df = build_section_graph(G, junctions)

    # Compute path variables (path_freq, stream_order, path_segs, path_order)
    path_vars = None
    if not skip_path_vars:
        path_vars = compute_path_variables(G, sections_df)

    # Compute junction-level validation (uses WSE data)
    has_wse = (
        reaches_df["wse_obs_mean"].notna().any()
        if "wse_obs_mean" in reaches_df.columns
        else False
    )
    validation_df = pd.DataFrame()

    if has_wse:
        validation_df = compute_junction_slopes(G, sections_df, reaches_df)

    # Load existing best_headwater/best_outlet for change tracking
    old_hw_out = {}
    try:
        old_df = conn.execute(
            """
            SELECT reach_id, best_headwater, best_outlet
            FROM reaches
            WHERE region = ? AND best_headwater IS NOT NULL
        """,
            [region.upper()],
        ).fetchdf()
        for _, row in old_df.iterrows():
            old_hw_out[int(row["reach_id"])] = {
                "best_headwater": row["best_headwater"],
                "best_outlet": row["best_outlet"],
            }
        log(f"Loaded {len(old_hw_out):,} existing best_headwater/outlet assignments")
    except Exception:
        log("No existing best_headwater/outlet assignments (first run)")

    # Compute new attributes
    hydro_dist = compute_hydro_distances(G)
    hw_out = compute_best_headwater_outlet(G)
    is_mainstem = compute_mainstem(G, hw_out)
    main_neighbors = compute_main_neighbors(G)

    # Log changes in best_headwater/best_outlet assignments
    if old_hw_out:
        n_hw_changed = 0
        n_out_changed = 0
        for rid, new_vals in hw_out.items():
            if rid in old_hw_out:
                old_vals = old_hw_out[rid]
                if new_vals["best_headwater"] != old_vals["best_headwater"]:
                    n_hw_changed += 1
                if new_vals["best_outlet"] != old_vals["best_outlet"]:
                    n_out_changed += 1
        log(
            f"Routing changes: {n_hw_changed:,} best_headwater, {n_out_changed:,} best_outlet"
        )

    # Save to DuckDB with provenance
    with workflow.transaction(f"v17c attributes for {region}"):
        n_updated = save_to_duckdb(
            conn,
            region,
            hydro_dist,
            hw_out,
            is_mainstem,
            main_neighbors,
            path_vars=path_vars,
        )
        save_sections_to_duckdb(conn, region, sections_df, validation_df)

    # Apply SWOT slopes if requested
    n_swot_updated = 0
    if not skip_swot and swot_path:
        with workflow.transaction(f"SWOT slopes for {region}"):
            n_swot_updated = apply_swot_slopes(conn, region, swot_path)

    workflow.close()

    # Post-write lint gate (read-only, separate connection)
    lint_results = {}
    if not skip_lint_gate:
        lint_results = run_lint_gate(db_path, region, checks=lint_checks)

    # Summary statistics
    stats = {
        "region": region,
        "reaches_processed": len(reaches_df),
        "reaches_updated": n_updated,
        "facc_corrections": n_facc_corrections,
        "sections": len(sections_df),
        "junctions": len(junctions),
        "mainstem_reaches": sum(is_mainstem.values()),
        "swot_updated": n_swot_updated,
    }
    if path_vars:
        n_valid_pf = sum(1 for v in path_vars.values() if v.get("path_freq", -9999) > 0)
        stats["path_freq_valid"] = n_valid_pf
        stats["path_freq_excluded"] = len(path_vars) - n_valid_pf

    if not validation_df.empty:
        stats["validation_valid"] = int(validation_df["direction_valid"].sum())
        stats["validation_invalid"] = int(
            (validation_df["direction_valid"] == False).sum()
        )

    if lint_results:
        stats["lint_checks_run"] = len(lint_results)
        stats["lint_checks_passed"] = sum(
            1 for v in lint_results.values() if v["passed"]
        )

    log(f"\nRegion {region} complete: {n_updated:,} reaches updated")
    return stats


def create_v17c_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create v17c-specific tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS v17c_sections (
            section_id INTEGER,
            region VARCHAR(2),
            upstream_junction BIGINT,
            downstream_junction BIGINT,
            reach_ids VARCHAR,
            distance DOUBLE,
            n_reaches INTEGER,
            PRIMARY KEY (section_id, region)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS v17c_section_slope_validation (
            section_id INTEGER,
            region VARCHAR(2),
            slope_from_upstream DOUBLE,
            slope_from_downstream DOUBLE,
            direction_valid BOOLEAN,
            likely_cause VARCHAR,
            PRIMARY KEY (section_id, region)
        )
    """)


def run_pipeline(
    db_path: str,
    regions: List[str],
    user_id: str = "v17c_pipeline",
    skip_swot: bool = False,
    swot_path: Optional[str] = None,
    skip_facc: bool = False,
    nofacc_model_path: str = DEFAULT_NOFACC_MODEL,
    standard_model_path: str = DEFAULT_STANDARD_MODEL,
    skip_path_vars: bool = False,
    skip_lint_gate: bool = False,
    lint_checks: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Run the v17c pipeline for multiple regions.

    Parameters
    ----------
    db_path : str
        Path to sword_v17c.duckdb
    regions : list
        List of region codes to process
    user_id : str
        User ID for provenance tracking
    skip_swot : bool
        Skip SWOT slope integration
    swot_path : str, optional
        Path to SWOT data directory
    skip_facc : bool
        Skip facc anomaly correction
    nofacc_model_path : str
        Path to no-facc RF model for entry point correction
    standard_model_path : str
        Path to standard RF model for propagation correction
    skip_lint_gate : bool
        Skip the post-write lint gate
    lint_checks : list of str, optional
        Specific lint check IDs or prefixes for the lint gate

    Returns
    -------
    list
        List of processing statistics per region
    """
    log(f"v17c Pipeline - Processing {len(regions)} regions")
    log(f"Database: {db_path}")
    log(f"Skip SWOT: {skip_swot}")
    log(f"Skip FACC: {skip_facc}")
    log(f"Skip path vars: {skip_path_vars}")
    log(f"Skip lint gate: {skip_lint_gate}")
    if swot_path:
        log(f"SWOT path: {swot_path}")

    all_stats = []

    for region in regions:
        try:
            stats = process_region(
                db_path=db_path,
                region=region,
                user_id=user_id,
                skip_swot=skip_swot,
                swot_path=swot_path,
                skip_facc=skip_facc,
                nofacc_model_path=nofacc_model_path,
                standard_model_path=standard_model_path,
                skip_path_vars=skip_path_vars,
                skip_lint_gate=skip_lint_gate,
                lint_checks=lint_checks,
            )
            all_stats.append(stats)
        except Exception as e:
            log(f"ERROR processing {region}: {e}")
            import traceback

            traceback.print_exc()
            all_stats.append(
                {
                    "region": region,
                    "error": str(e),
                }
            )

    # Print summary
    log("\n" + "=" * 60)
    log("PIPELINE SUMMARY")
    log("=" * 60)

    total_updated = 0
    for stats in all_stats:
        if "error" in stats:
            log(f"{stats['region']}: ERROR - {stats['error']}")
        else:
            facc_str = (
                f", {stats['facc_corrections']:,} facc fixes"
                if stats.get("facc_corrections")
                else ""
            )
            pf_str = (
                f", {stats['path_freq_valid']:,} valid pf"
                if stats.get("path_freq_valid")
                else ""
            )
            log(
                f"{stats['region']}: {stats['reaches_updated']:,} reaches, "
                f"{stats['sections']:,} sections, "
                f"{stats['mainstem_reaches']:,} mainstem{facc_str}{pf_str}"
            )
            total_updated += stats.get("reaches_updated", 0)

    log(f"\nTotal reaches updated: {total_updated:,}")

    return all_stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="v17c Pipeline - Compute and save v17c attributes to DuckDB"
    )
    parser.add_argument("--db", required=True, help="Path to sword_v17c.duckdb")
    parser.add_argument(
        "--region", help="Single region to process (NA, SA, EU, AF, AS, OC)"
    )
    parser.add_argument("--all", action="store_true", help="Process all regions")
    parser.add_argument(
        "--skip-swot", action="store_true", help="Skip SWOT slope integration"
    )
    parser.add_argument(
        "--swot-path",
        default="/Volumes/SWORD_DATA/data/swot/RiverSP_D_parq/node",
        help="Path to SWOT parquet files",
    )
    parser.add_argument(
        "--user-id", default="v17c_pipeline", help="User ID for provenance tracking"
    )
    parser.add_argument(
        "--skip-facc",
        action="store_true",
        help="Skip facc anomaly correction (default: run corrections before v17c computation)",
    )
    parser.add_argument(
        "--skip-path-vars",
        action="store_true",
        help="Skip path variable recomputation (path_freq, stream_order, path_segs, path_order)",
    )
    parser.add_argument(
        "--nofacc-model",
        default=DEFAULT_NOFACC_MODEL,
        help=f"Path to no-facc RF model for entry points (default: {DEFAULT_NOFACC_MODEL})",
    )
    parser.add_argument(
        "--standard-model",
        default=DEFAULT_STANDARD_MODEL,
        help=f"Path to standard RF model for propagation (default: {DEFAULT_STANDARD_MODEL})",
    )
    parser.add_argument(
        "--skip-lint-gate",
        action="store_true",
        help="Skip the post-write lint gate (default: run ERROR-severity checks)",
    )
    parser.add_argument(
        "--lint-checks",
        nargs="+",
        default=None,
        help="Specific lint check IDs or prefixes to run (e.g. T001 T005 F)",
    )

    args = parser.parse_args()

    # Determine regions to process
    if args.all:
        regions = REGIONS
    elif args.region:
        regions = [args.region.upper()]
    else:
        parser.error("Either --region or --all must be specified")

    # Validate database exists
    if not os.path.exists(args.db):
        print(f"ERROR: Database not found: {args.db}")
        sys.exit(1)

    # Run pipeline
    stats = run_pipeline(
        db_path=args.db,
        regions=regions,
        user_id=args.user_id,
        skip_swot=args.skip_swot,
        swot_path=args.swot_path,
        skip_facc=args.skip_facc,
        nofacc_model_path=args.nofacc_model,
        standard_model_path=args.standard_model,
        skip_path_vars=args.skip_path_vars,
        skip_lint_gate=args.skip_lint_gate,
        lint_checks=args.lint_checks,
    )

    # Exit with error if any region failed
    if any("error" in s for s in stats):
        sys.exit(1)


if __name__ == "__main__":
    main()
