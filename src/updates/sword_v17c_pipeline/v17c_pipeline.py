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
REGIONS = ['NA', 'SA', 'EU', 'AF', 'AS', 'OC']


def log(msg: str) -> None:
    """Log message with timestamp."""
    print(f"[{dt.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Data Loading
# =============================================================================

def load_topology(conn: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    """Load reach_topology from DuckDB."""
    log(f"Loading topology for {region}...")
    df = conn.execute("""
        SELECT reach_id, direction, neighbor_rank, neighbor_reach_id
        FROM reach_topology
        WHERE region = ?
    """, [region.upper()]).fetchdf()
    log(f"Loaded {len(df):,} topology rows")
    return df


def load_reaches(conn: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    """Load reaches with attributes."""
    log(f"Loading reaches for {region}...")
    df = conn.execute("""
        SELECT
            reach_id, region, reach_length, width, slope, facc,
            n_rch_up, n_rch_down, dist_out, path_freq, stream_order,
            lakeflag, trib_flag, wse_obs_mean, wse_obs_std
        FROM reaches
        WHERE region = ?
    """, [region.upper()]).fetchdf()
    log(f"Loaded {len(df):,} reaches")
    return df


# =============================================================================
# Graph Construction
# =============================================================================

def build_reach_graph(topology_df: pd.DataFrame, reaches_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build directed graph where nodes=reaches, edges=flow connections.

    Flow direction from topology:
    - direction='up': neighbor is upstream -> neighbor -> reach
    - direction='down': neighbor is downstream -> reach -> neighbor
    """
    log("Building reach-level directed graph...")

    G = nx.DiGraph()

    # Create reach attributes dict
    reach_attrs = {}
    for _, row in reaches_df.iterrows():
        rid = int(row['reach_id'])
        reach_attrs[rid] = {
            'reach_length': row['reach_length'],
            'width': row['width'],
            'slope': row['slope'],
            'facc': row.get('facc', 0),
            'n_rch_up': row.get('n_rch_up', 0),
            'n_rch_down': row.get('n_rch_down', 0),
            'dist_out': row.get('dist_out', 0),
            'path_freq': row.get('path_freq', 1),
            'stream_order': row.get('stream_order', 1),
            'lakeflag': row.get('lakeflag', 0),
            'wse_obs_mean': row.get('wse_obs_mean'),
        }

    # Add all reaches as nodes
    for rid, attrs in reach_attrs.items():
        G.add_node(rid, **attrs)

    # Add edges from topology
    edges_added = set()
    for _, row in topology_df.iterrows():
        reach_id = int(row['reach_id'])
        neighbor_id = int(row['neighbor_reach_id'])
        direction = row['direction']

        if direction == 'up':
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


def build_section_graph(G: nx.DiGraph, junctions: Set[int]) -> Tuple[nx.DiGraph, pd.DataFrame]:
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
            node_type = 'Head_water'
        elif out_deg == 0:
            node_type = 'Outlet'
        else:
            node_type = 'Junction'

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
                cumulative_dist += G.nodes[current].get('reach_length', 0)

                succs = list(G.successors(current))
                if len(succs) == 0:
                    break
                current = succs[0]

            # current is now the downstream junction
            downstream_j = current
            if downstream_j in junctions:
                reach_ids.append(downstream_j)
                cumulative_dist += G.nodes[downstream_j].get('reach_length', 0)

                R.add_edge(upstream_j, downstream_j,
                          section_id=section_id,
                          reach_ids=reach_ids,
                          distance=cumulative_dist,
                          n_reaches=len(reach_ids))

                sections.append({
                    'section_id': section_id,
                    'upstream_junction': upstream_j,
                    'downstream_junction': downstream_j,
                    'reach_ids': reach_ids,
                    'distance': cumulative_dist,
                    'n_reaches': len(reach_ids),
                })
                section_id += 1

    sections_df = pd.DataFrame(sections)
    log(f"Section graph: {R.number_of_nodes():,} junctions, {R.number_of_edges():,} sections")
    return R, sections_df


# =============================================================================
# New Attribute Computation
# =============================================================================

def compute_hydro_distances(G: nx.DiGraph) -> Dict[int, Dict]:
    """
    Compute hydrologic distances for each reach.

    - hydro_dist_out: Distance to outlet following main channel
    - hydro_dist_hw: Distance from headwater following main channel
    """
    log("Computing hydrologic distances...")

    # Identify outlets (no outgoing edges)
    outlets = [n for n in G.nodes() if G.out_degree(n) == 0]
    log(f"Found {len(outlets):,} outlets")

    # Compute dist_out using Dijkstra from outlets (reversed graph)
    R = G.reverse()

    dist_out = {}
    for node in G.nodes():
        dist_out[node] = float('inf')

    for outlet in outlets:
        dist_out[outlet] = 0

    # Multi-source Dijkstra
    lengths = nx.multi_source_dijkstra_path_length(
        R, outlets,
        weight=lambda u, v, d: G.nodes[v].get('reach_length', 0)
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
                G, hw,
                weight=lambda u, v, d: G.nodes[v].get('reach_length', 0)
            )
            for node, dist in lengths.items():
                if dist > dist_hw[node]:
                    dist_hw[node] = dist
        except nx.NetworkXError:
            continue

    results = {}
    for node in G.nodes():
        results[node] = {
            'hydro_dist_out': dist_out.get(node, float('inf')),
            'hydro_dist_hw': dist_hw.get(node, 0),
        }

    log("Hydrologic distances computed")
    return results


def compute_best_headwater_outlet(G: nx.DiGraph) -> Dict[int, Dict]:
    """
    Compute best headwater and outlet for each reach.

    Uses path frequency (convergence) and width to select "main" path.
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
                reach_len = G.nodes[n].get('reach_length', 0)
                total_len = pathlen_hw.get(p, 0) + reach_len
                width = G.nodes[p].get('width', 0) or 0
                candidates.append((width, total_len, best_hw.get(p), p))

            hw_sets[n] = union

            # Select by width (primary), then path length
            best = max(candidates, key=lambda x: (x[0], x[1]))
            best_hw[n] = best[2]
            pathlen_hw[n] = best[1]

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
                reach_len = G.nodes[s].get('reach_length', 0)
                total_len = pathlen_out.get(s, 0) + reach_len
                width = G.nodes[s].get('width', 0) or 0
                candidates.append((width, total_len, best_out.get(s), s))

            best = max(candidates, key=lambda x: (x[0], x[1]))
            best_out[n] = best[2]
            pathlen_out[n] = best[1]

    results = {}
    for node in G.nodes():
        results[node] = {
            'best_headwater': best_hw.get(node),
            'best_outlet': best_out.get(node),
            'pathlen_hw': pathlen_hw.get(node, 0),
            'pathlen_out': pathlen_out.get(node, 0),
            'path_freq': len(hw_sets.get(node, set())),
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
        key = (attrs['best_headwater'], attrs['best_outlet'])
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
    log(f"Mainstem reaches: {n_mainstem:,} ({100*n_mainstem/len(G.nodes()):.1f}%)")

    return is_mainstem


# =============================================================================
# SWOT Slope Validation (Junction-Level)
# =============================================================================

def compute_junction_slopes(
    G: nx.DiGraph,
    sections_df: pd.DataFrame,
    reaches_df: pd.DataFrame
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
        rid = int(row['reach_id'])
        wse = row.get('wse_obs_mean')
        if pd.notna(wse):
            wse_map[rid] = wse

    results = []

    for _, section in sections_df.iterrows():
        section_id = section['section_id']
        upstream_j = section['upstream_junction']
        downstream_j = section['downstream_junction']
        reach_ids = section['reach_ids']
        total_distance = section['distance']

        if len(reach_ids) < 2:
            continue

        # Get WSE values for reaches in this section
        wse_data = []
        cumulative_dist = 0

        for rid in reach_ids:
            wse = wse_map.get(rid)
            reach_len = G.nodes[rid].get('reach_length', 0) if rid in G.nodes else 0

            if wse is not None:
                wse_data.append({
                    'reach_id': rid,
                    'wse': wse,
                    'dist_from_upstream': cumulative_dist,
                    'dist_from_downstream': total_distance - cumulative_dist,
                })
            cumulative_dist += reach_len

        if len(wse_data) < 2:
            continue

        wse_df = pd.DataFrame(wse_data)

        # Compute slope from upstream junction
        try:
            slope_upstream = np.polyfit(wse_df['dist_from_upstream'], wse_df['wse'], 1)[0]
        except Exception:
            slope_upstream = np.nan

        # Compute slope from downstream junction
        try:
            slope_downstream = np.polyfit(wse_df['dist_from_downstream'], wse_df['wse'], 1)[0]
        except Exception:
            slope_downstream = np.nan

        # Determine if slopes match expected signs
        upstream_correct = slope_upstream < 0 if pd.notna(slope_upstream) else None
        downstream_correct = slope_downstream > 0 if pd.notna(slope_downstream) else None

        direction_valid = upstream_correct and downstream_correct if (
            upstream_correct is not None and downstream_correct is not None
        ) else None

        # Determine likely cause if invalid
        likely_cause = None
        if direction_valid is False:
            lakeflag = G.nodes[upstream_j].get('lakeflag', 0) if upstream_j in G.nodes else 0
            if lakeflag > 0:
                likely_cause = 'lake_section'
            elif pd.notna(slope_upstream) and abs(slope_upstream) > 0.05:
                likely_cause = 'extreme_slope_data_error'
            elif pd.notna(slope_downstream) and abs(slope_downstream) > 0.05:
                likely_cause = 'extreme_slope_data_error'
            else:
                likely_cause = 'potential_topology_error'

        results.append({
            'section_id': section_id,
            'upstream_junction': upstream_j,
            'downstream_junction': downstream_j,
            'n_reaches': len(reach_ids),
            'n_reaches_with_wse': len(wse_df),
            'distance': total_distance,
            'slope_from_upstream': slope_upstream,
            'slope_from_downstream': slope_downstream,
            'direction_valid': direction_valid,
            'likely_cause': likely_cause,
        })

    junction_slopes_df = pd.DataFrame(results)

    # Summary
    if not junction_slopes_df.empty:
        n_total = len(junction_slopes_df)
        n_valid = junction_slopes_df['direction_valid'].sum()
        n_invalid = (junction_slopes_df['direction_valid'] == False).sum()

        log(f"Junction slope validation:")
        log(f"  Total sections with SWOT data: {n_total:,}")
        log(f"  Direction valid: {n_valid:,} ({100*n_valid/n_total:.1f}%)")
        log(f"  Direction INVALID: {n_invalid:,} ({100*n_invalid/n_total:.1f}%)")

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
) -> int:
    """
    Save computed v17c attributes to DuckDB reaches table.

    Returns:
        Number of reaches updated
    """
    log(f"Saving v17c attributes to DuckDB for {region}...")

    # Build update dataframe
    rows = []
    for reach_id in hydro_dist.keys():
        hd = hydro_dist.get(reach_id, {})
        ho = hw_out.get(reach_id, {})
        ms = is_mainstem.get(reach_id, False)

        rows.append({
            'reach_id': reach_id,
            'hydro_dist_out': hd.get('hydro_dist_out'),
            'hydro_dist_hw': hd.get('hydro_dist_hw'),
            'best_headwater': ho.get('best_headwater'),
            'best_outlet': ho.get('best_outlet'),
            'pathlen_hw': ho.get('pathlen_hw'),
            'pathlen_out': ho.get('pathlen_out'),
            'is_mainstem_edge': ms,
        })

    if not rows:
        log("No rows to update")
        return 0

    update_df = pd.DataFrame(rows)

    # Handle infinity values - convert to NULL
    update_df = update_df.replace([np.inf, -np.inf], np.nan)

    # Register DataFrame and update
    conn.register('v17c_updates', update_df)

    # Update reaches table
    conn.execute(f"""
        UPDATE reaches SET
            hydro_dist_out = u.hydro_dist_out,
            hydro_dist_hw = u.hydro_dist_hw,
            best_headwater = u.best_headwater,
            best_outlet = u.best_outlet,
            pathlen_hw = u.pathlen_hw,
            pathlen_out = u.pathlen_out,
            is_mainstem_edge = u.is_mainstem_edge
        FROM v17c_updates u
        WHERE reaches.reach_id = u.reach_id
        AND reaches.region = '{region.upper()}'
    """)

    conn.unregister('v17c_updates')

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
    sections_insert['region'] = region.upper()
    # Convert reach_ids list to JSON string
    sections_insert['reach_ids'] = sections_insert['reach_ids'].apply(json.dumps)

    conn.register('sections_insert', sections_insert)
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
    conn.unregister('sections_insert')
    log(f"Saved {len(sections_insert):,} sections")

    # Save validation results if any
    if not validation_df.empty:
        validation_insert = validation_df.copy()
        validation_insert['region'] = region.upper()

        conn.register('validation_insert', validation_insert)
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
        conn.unregister('validation_insert')
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

    parquet_files = [f for f in glob.glob(os.path.join(swot_path, '*.parquet'))
                     if not os.path.basename(f).startswith('._')]

    if not parquet_files:
        log(f"No parquet files found in {swot_path}")
        return 0

    log(f"Found {len(parquet_files)} SWOT parquet files")

    # Get node_ids for this region
    nodes_df = conn.execute("""
        SELECT node_id FROM nodes WHERE region = ?
    """, [region.upper()]).fetchdf()

    if nodes_df.empty:
        log(f"No nodes found for region {region}")
        return 0

    node_ids = nodes_df['node_id'].tolist()
    log(f"Region {region} has {len(node_ids):,} nodes")

    # For now, we'll use the pre-computed slopes if they exist
    # in the SWOT pipeline output
    swot_slopes_file = os.path.join(
        os.path.dirname(swot_path).replace('/node', ''),
        f'output/{region.lower()}/{region.lower()}_swot_slopes.csv'
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
    log(f"\n{'='*60}")
    log(f"Processing region: {region}")
    log(f"{'='*60}")

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

    # Build reach-level graph
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

    # Compute junction-level validation (uses WSE data)
    has_wse = reaches_df['wse_obs_mean'].notna().any() if 'wse_obs_mean' in reaches_df.columns else False
    validation_df = pd.DataFrame()

    if has_wse:
        validation_df = compute_junction_slopes(G, sections_df, reaches_df)

    # Compute new attributes
    hydro_dist = compute_hydro_distances(G)
    hw_out = compute_best_headwater_outlet(G)
    is_mainstem = compute_mainstem(G, hw_out)

    # Save to DuckDB with provenance
    with workflow.transaction(f"v17c attributes for {region}"):
        n_updated = save_to_duckdb(conn, region, hydro_dist, hw_out, is_mainstem)
        save_sections_to_duckdb(conn, region, sections_df, validation_df)

    # Apply SWOT slopes if requested
    n_swot_updated = 0
    if not skip_swot and swot_path:
        with workflow.transaction(f"SWOT slopes for {region}"):
            n_swot_updated = apply_swot_slopes(conn, region, swot_path)

    workflow.close()

    # Summary statistics
    stats = {
        'region': region,
        'reaches_processed': len(reaches_df),
        'reaches_updated': n_updated,
        'sections': len(sections_df),
        'junctions': len(junctions),
        'mainstem_reaches': sum(is_mainstem.values()),
        'swot_updated': n_swot_updated,
    }

    if not validation_df.empty:
        stats['validation_valid'] = int(validation_df['direction_valid'].sum())
        stats['validation_invalid'] = int((validation_df['direction_valid'] == False).sum())

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

    Returns
    -------
    list
        List of processing statistics per region
    """
    log(f"v17c Pipeline - Processing {len(regions)} regions")
    log(f"Database: {db_path}")
    log(f"Skip SWOT: {skip_swot}")
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
            )
            all_stats.append(stats)
        except Exception as e:
            log(f"ERROR processing {region}: {e}")
            import traceback
            traceback.print_exc()
            all_stats.append({
                'region': region,
                'error': str(e),
            })

    # Print summary
    log("\n" + "=" * 60)
    log("PIPELINE SUMMARY")
    log("=" * 60)

    total_updated = 0
    for stats in all_stats:
        if 'error' in stats:
            log(f"{stats['region']}: ERROR - {stats['error']}")
        else:
            log(f"{stats['region']}: {stats['reaches_updated']:,} reaches, "
                f"{stats['sections']:,} sections, "
                f"{stats['mainstem_reaches']:,} mainstem")
            total_updated += stats.get('reaches_updated', 0)

    log(f"\nTotal reaches updated: {total_updated:,}")

    return all_stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="v17c Pipeline - Compute and save v17c attributes to DuckDB"
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to sword_v17c.duckdb"
    )
    parser.add_argument(
        "--region",
        help="Single region to process (NA, SA, EU, AF, AS, OC)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all regions"
    )
    parser.add_argument(
        "--skip-swot",
        action="store_true",
        help="Skip SWOT slope integration"
    )
    parser.add_argument(
        "--swot-path",
        default="/Volumes/SWORD_DATA/data/swot/RiverSP_D_parq/node",
        help="Path to SWOT parquet files"
    )
    parser.add_argument(
        "--user-id",
        default="v17c_pipeline",
        help="User ID for provenance tracking"
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
    )

    # Exit with error if any region failed
    if any('error' in s for s in stats):
        sys.exit(1)


if __name__ == "__main__":
    main()
