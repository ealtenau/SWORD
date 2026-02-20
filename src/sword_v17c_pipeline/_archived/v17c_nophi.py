#!/usr/bin/env python3
"""
v17c attribute computation WITHOUT phi optimization.

This script:
1. Creates a reduced (junction-based) graph from v17b topology
2. Validates SWOT slopes at junction level against flow direction
3. Computes new v17c attributes (hydro_dist_out, best_headwater, is_mainstem, etc.)

The reduced graph collapses chains of reaches into single edges between junctions.
Junctions are nodes where:
- Multiple reaches converge (confluence)
- Multiple reaches diverge (bifurcation)
- Headwaters (no upstream)
- Outlets (no downstream)

SWOT Slope Validation (Junction-Level):
- For each section (junction to junction), SWOT slopes are computed relative to distance
  from each junction endpoint
- Slope at UPSTREAM junction should be NEGATIVE (WSE decreases as you move away)
- Slope at DOWNSTREAM junction should be POSITIVE (WSE increases as you move away)
- If signs don't match expected direction, it indicates a potential topology error

Usage:
    python v17c_nophi.py --continent NA --db-path data/duckdb/sword_v17b.duckdb
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime as dt
from typing import Dict, List, Optional, Set, Tuple

import duckdb
import networkx as nx
import numpy as np
import pandas as pd


def log(msg):
    print(f"[{dt.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Data Loading
# =============================================================================

def load_topology(db_path: str, continent: str) -> pd.DataFrame:
    """Load reach_topology from v17b."""
    log(f"Loading topology for {continent}...")
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute(f"""
        SELECT reach_id, direction, neighbor_rank, neighbor_reach_id
        FROM reach_topology
        WHERE region = '{continent.upper()}'
    """).fetchdf()
    conn.close()
    log(f"Loaded {len(df):,} topology rows")
    return df


def load_reaches(db_path: str, continent: str, v17c_db_path: str = None) -> pd.DataFrame:
    """Load reaches with attributes, including WSE from v17c if available."""
    log(f"Loading reaches for {continent}...")
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute(f"""
        SELECT
            reach_id, region, reach_length, width, slope, facc,
            n_rch_up, n_rch_down, dist_out, path_freq, stream_order,
            lakeflag, trib_flag
        FROM reaches
        WHERE region = '{continent.upper()}'
    """).fetchdf()
    conn.close()
    log(f"Loaded {len(df):,} reaches")

    # Try to get WSE data from v17c
    if v17c_db_path and os.path.exists(v17c_db_path):
        try:
            conn = duckdb.connect(v17c_db_path, read_only=True)
            wse_df = conn.execute(f"""
                SELECT reach_id, wse_obs_mean, wse_obs_std
                FROM reaches
                WHERE region = '{continent.upper()}'
                  AND wse_obs_mean IS NOT NULL
            """).fetchdf()
            conn.close()

            if not wse_df.empty:
                df = df.merge(wse_df, on='reach_id', how='left')
                log(f"Added WSE data for {wse_df['wse_obs_mean'].notna().sum():,} reaches")
        except Exception as e:
            log(f"Could not load WSE from v17c: {e}")

    return df


def load_swot_slopes(db_path: str, continent: str, v17c_db_path: str = None) -> pd.DataFrame:
    """
    Load SWOT slope observations.

    First tries the primary db_path. If no slopes found and v17c_db_path is provided,
    falls back to v17c database (which has computed SWOT stats).
    """
    log(f"Loading SWOT slopes for {continent}...")

    def try_load_slopes(path: str) -> pd.DataFrame:
        conn = duckdb.connect(path, read_only=True)

        # Check if slope columns exist
        cols = conn.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'reaches'
        """).fetchdf()['column_name'].tolist()

        if 'slope_obs_mean' not in cols:
            conn.close()
            return pd.DataFrame()

        df = conn.execute(f"""
            SELECT
                reach_id,
                slope_obs_mean,
                slope_obs_median,
                slope_obs_std,
                slope_obs_range,
                n_obs
            FROM reaches
            WHERE region = '{continent.upper()}'
              AND slope_obs_mean IS NOT NULL
              AND slope_obs_mean != 0
        """).fetchdf()
        conn.close()
        return df

    # Try primary db
    df = try_load_slopes(db_path)

    if df.empty and v17c_db_path and os.path.exists(v17c_db_path):
        log(f"No slopes in v17b, trying v17c: {v17c_db_path}")
        df = try_load_slopes(v17c_db_path)

    if df.empty:
        log("WARNING: No SWOT slope data found")
    else:
        log(f"Loaded SWOT slopes for {len(df):,} reaches")

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
            'wse_obs_mean': row.get('wse_obs_mean'),  # From v17c SWOT
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
            # Edge attributes from downstream reach
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


def build_reduced_graph(G: nx.DiGraph, junctions: Set[int]) -> nx.DiGraph:
    """
    Build reduced graph where edges connect junctions directly.

    Each edge in reduced graph represents a chain of reaches.
    Edge attributes include the list of reach_ids in the chain.
    """
    log("Building reduced (junction-to-junction) graph...")

    R = nx.DiGraph()

    # Add junction nodes
    for j in junctions:
        R.add_node(j, **G.nodes[j])

    # For each junction, trace downstream to next junction
    for j in junctions:
        for successor in G.successors(j):
            # Trace chain until we hit another junction
            chain = [j]
            current = successor
            total_length = 0
            total_width = 0
            width_count = 0
            reach_ids = []

            while current not in junctions:
                chain.append(current)
                reach_ids.append(current)

                node_data = G.nodes[current]
                total_length += node_data.get('reach_length', 0)
                w = node_data.get('width', 0)
                if w and w > 0:
                    total_width += w
                    width_count += 1

                # Move to next
                succs = list(G.successors(current))
                if len(succs) == 0:
                    # Dead end (shouldn't happen if junctions identified correctly)
                    break
                current = succs[0]

            # current is now the downstream junction
            chain.append(current)
            reach_ids.append(current)

            if current in junctions:
                avg_width = total_width / width_count if width_count > 0 else 0
                R.add_edge(j, current,
                    reach_ids=reach_ids,
                    chain_length=total_length,
                    avg_width=avg_width,
                    n_reaches=len(reach_ids),
                )

    log(f"Reduced graph: {R.number_of_nodes():,} nodes, {R.number_of_edges():,} edges")
    return R


# =============================================================================
# SWOT Slope Validation (Junction-Level)
# =============================================================================

def build_section_graph(G: nx.DiGraph, junctions: Set[int]) -> Tuple[nx.DiGraph, pd.DataFrame]:
    """
    Build a section graph where each edge is a section (chain of reaches between junctions).

    Returns:
        R: DiGraph where nodes are junctions, edges are sections
        sections_df: DataFrame with section details (section_id, upstream_junction,
                     downstream_junction, reach_ids, distance)
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
                # Add the downstream junction reach if it's part of the chain
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


def compute_junction_slopes(
    G: nx.DiGraph,
    sections_df: pd.DataFrame,
    reaches_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute slopes at junction endpoints for each section.

    For each section, we compute:
    - slope_upstream: slope measured from upstream junction (should be NEGATIVE)
    - slope_downstream: slope measured from downstream junction (should be POSITIVE)

    The slope is computed as: change in WSE / distance from junction

    In the topology-defined direction:
    - From upstream junction, moving downstream, WSE should DECREASE -> negative slope
    - From downstream junction, moving upstream, WSE should DECREASE -> positive slope
    """
    log("Computing junction slopes from SWOT data...")

    # Create reach -> WSE mapping
    wse_map = {}
    dist_map = {}  # reach_id -> distance from start of its section

    for _, row in reaches_df.iterrows():
        rid = int(row['reach_id'])
        wse = row.get('wse_obs_mean')
        if pd.notna(wse):
            wse_map[rid] = wse
        dist_map[rid] = row.get('dist_out', 0)

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

        # Compute slope from upstream junction (regress WSE on distance from upstream)
        # Negative slope = WSE decreases as distance increases = correct
        try:
            slope_upstream = np.polyfit(wse_df['dist_from_upstream'], wse_df['wse'], 1)[0]
        except:
            slope_upstream = np.nan

        # Compute slope from downstream junction (regress WSE on distance from downstream)
        # Positive slope = WSE increases as distance increases = correct
        try:
            slope_downstream = np.polyfit(wse_df['dist_from_downstream'], wse_df['wse'], 1)[0]
        except:
            slope_downstream = np.nan

        # Determine if slopes match expected signs
        # Upstream slope should be NEGATIVE (WSE drops as you go downstream from upstream junction)
        # Downstream slope should be POSITIVE (WSE rises as you go upstream from downstream junction)
        upstream_correct = slope_upstream < 0 if pd.notna(slope_upstream) else None
        downstream_correct = slope_downstream > 0 if pd.notna(slope_downstream) else None

        # Overall direction check: both should agree
        direction_valid = upstream_correct and downstream_correct if (
            upstream_correct is not None and downstream_correct is not None
        ) else None

        results.append({
            'section_id': section_id,
            'upstream_junction': upstream_j,
            'downstream_junction': downstream_j,
            'n_reaches': len(reach_ids),
            'n_reaches_with_wse': len(wse_df),
            'distance': total_distance,
            'slope_from_upstream': slope_upstream,
            'slope_from_downstream': slope_downstream,
            'upstream_sign_correct': upstream_correct,
            'downstream_sign_correct': downstream_correct,
            'direction_valid': direction_valid,
            'wse_range': wse_df['wse'].max() - wse_df['wse'].min(),
            'wse_upstream': wse_df.loc[wse_df['dist_from_upstream'].idxmin(), 'wse'],
            'wse_downstream': wse_df.loc[wse_df['dist_from_downstream'].idxmin(), 'wse'],
        })

    junction_slopes_df = pd.DataFrame(results)

    # Summary
    if not junction_slopes_df.empty:
        n_total = len(junction_slopes_df)
        n_valid = junction_slopes_df['direction_valid'].sum()
        n_invalid = (junction_slopes_df['direction_valid'] == False).sum()
        n_unknown = junction_slopes_df['direction_valid'].isna().sum()

        log(f"Junction slope validation:")
        log(f"  Total sections with SWOT data: {n_total:,}")
        log(f"  Direction valid (slopes match expected): {n_valid:,} ({100*n_valid/n_total:.1f}%)")
        log(f"  Direction INVALID (potential topology error): {n_invalid:,} ({100*n_invalid/n_total:.1f}%)")
        log(f"  Unknown (insufficient data): {n_unknown:,} ({100*n_unknown/n_total:.1f}%)")

    return junction_slopes_df


def validate_slopes_junction_level(
    G: nx.DiGraph,
    R: nx.DiGraph,
    sections_df: pd.DataFrame,
    reaches_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate topology using junction-level SWOT slopes.

    This is the CORRECT validation approach:
    - For each section (junction to junction), compute slopes from both endpoints
    - Upstream junction slope should be NEGATIVE
    - Downstream junction slope should be POSITIVE
    - If both match, topology is correct for this section

    Returns:
        section_validation: DataFrame with validation results per section
        reach_validation: DataFrame mapping reaches to their section validation status
    """
    log("Validating topology using junction-level SWOT slopes...")

    # Compute junction slopes
    junction_slopes = compute_junction_slopes(G, sections_df, reaches_df)

    if junction_slopes.empty:
        log("No junction slopes computed - insufficient SWOT data")
        return pd.DataFrame(), pd.DataFrame()

    # Create reach-level validation by mapping from sections
    reach_results = []
    for _, section in sections_df.iterrows():
        section_id = section['section_id']
        reach_ids = section['reach_ids']

        # Get validation status for this section
        section_val = junction_slopes[junction_slopes['section_id'] == section_id]
        if section_val.empty:
            continue

        section_row = section_val.iloc[0]
        direction_valid = section_row['direction_valid']
        slope_upstream = section_row['slope_from_upstream']
        slope_downstream = section_row['slope_from_downstream']

        for rid in reach_ids:
            reach_results.append({
                'reach_id': rid,
                'section_id': section_id,
                'upstream_junction': section['upstream_junction'],
                'downstream_junction': section['downstream_junction'],
                'direction_valid': direction_valid,
                'slope_from_upstream': slope_upstream,
                'slope_from_downstream': slope_downstream,
            })

    reach_validation = pd.DataFrame(reach_results)

    return junction_slopes, reach_validation


def analyze_suspect_sections(
    junction_slopes: pd.DataFrame,
    G: nx.DiGraph
) -> pd.DataFrame:
    """
    Analyze sections with potential topology issues.

    Flags sections where:
    - Direction is invalid (slopes don't match expected signs)
    - Extreme slope magnitudes (data quality issues)
    """
    if junction_slopes.empty:
        return pd.DataFrame()

    log("Analyzing suspect sections...")

    # Filter to problematic sections
    suspect = junction_slopes[
        (junction_slopes['direction_valid'] == False) |
        (junction_slopes['slope_from_upstream'].abs() > 0.05) |
        (junction_slopes['slope_from_downstream'].abs() > 0.05)
    ].copy()

    if suspect.empty:
        log("No suspect sections found - topology looks good!")
        return pd.DataFrame()

    # Add context
    causes = []
    for _, row in suspect.iterrows():
        slope_up = row['slope_from_upstream']
        slope_dn = row['slope_from_downstream']

        if pd.notna(slope_up) and abs(slope_up) > 0.05:
            causes.append('extreme_slope_data_error')
        elif pd.notna(slope_dn) and abs(slope_dn) > 0.05:
            causes.append('extreme_slope_data_error')
        elif row['direction_valid'] == False:
            # Check if it's a lake section
            upstream_j = row['upstream_junction']
            lakeflag = G.nodes[upstream_j].get('lakeflag', 0) if upstream_j in G.nodes else 0
            if lakeflag > 0:
                causes.append('lake_section')
            else:
                causes.append('potential_topology_error')
        else:
            causes.append('unknown')

    suspect['likely_cause'] = causes

    # Summary
    cause_counts = suspect['likely_cause'].value_counts()
    log("Suspect section causes:")
    for cause, count in cause_counts.items():
        log(f"  {cause}: {count:,}")

    n_topology_errors = (suspect['likely_cause'] == 'potential_topology_error').sum()
    if n_topology_errors > 0:
        log(f"HIGH PRIORITY: {n_topology_errors:,} sections may have topology errors")

    return suspect


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
        log("ERROR: Graph has cycles, cannot compute")
        return {}

    topo = list(nx.topological_sort(G))

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
                total_len = pathlen_hw[p] + reach_len
                width = G.nodes[p].get('width', 0)
                candidates.append((width, total_len, best_hw[p], p))

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
                total_len = pathlen_out[s] + reach_len
                width = G.nodes[s].get('width', 0)
                candidates.append((width, total_len, best_out[s], s))

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
            # Find shortest path from hw to out
            path = nx.shortest_path(G, hw, out)
            for n in path:
                is_mainstem[n] = True
        except nx.NetworkXNoPath:
            continue

    n_mainstem = sum(is_mainstem.values())
    log(f"Mainstem reaches: {n_mainstem:,} ({100*n_mainstem/len(G.nodes()):.1f}%)")

    return is_mainstem


# =============================================================================
# Output
# =============================================================================

def save_results(
    reaches_df: pd.DataFrame,
    hydro_dist: Dict[int, Dict],
    hw_out: Dict[int, Dict],
    is_mainstem: Dict[int, bool],
    section_validation: pd.DataFrame,
    reach_validation: pd.DataFrame,
    suspect_sections: pd.DataFrame,
    sections_df: pd.DataFrame,
    output_dir: str,
    continent: str
):
    """Save computed attributes to files."""

    os.makedirs(output_dir, exist_ok=True)

    # Merge all attributes into reaches
    log("Merging computed attributes...")

    results = []
    for _, row in reaches_df.iterrows():
        rid = int(row['reach_id'])

        result = {
            'reach_id': rid,
            'region': row['region'],
            # Original attributes
            'reach_length': row['reach_length'],
            'width': row['width'],
            'dist_out_original': row['dist_out'],
            'path_freq_original': row.get('path_freq', 1),
        }

        # Add computed attributes
        if rid in hydro_dist:
            result.update(hydro_dist[rid])

        if rid in hw_out:
            result.update(hw_out[rid])

        result['is_mainstem'] = is_mainstem.get(rid, False)

        results.append(result)

    results_df = pd.DataFrame(results)

    # Save main results
    out_path = os.path.join(output_dir, f"{continent.lower()}_v17c_nophi_attrs.parquet")
    results_df.to_parquet(out_path, index=False)
    log(f"Saved attributes to {out_path}")

    # Save sections
    if not sections_df.empty:
        sections_path = os.path.join(output_dir, f"{continent.lower()}_sections.parquet")
        # Convert reach_ids list to string for parquet
        sections_save = sections_df.copy()
        sections_save['reach_ids'] = sections_save['reach_ids'].apply(lambda x: ','.join(map(str, x)))
        sections_save.to_parquet(sections_path, index=False)
        log(f"Saved {len(sections_df):,} sections to {sections_path}")

    # Save section-level slope validation
    if not section_validation.empty:
        slope_path = os.path.join(output_dir, f"{continent.lower()}_section_slope_validation.parquet")
        section_validation.to_parquet(slope_path, index=False)
        log(f"Saved section slope validation to {slope_path}")

    # Save reach-level validation mapping
    if not reach_validation.empty:
        reach_val_path = os.path.join(output_dir, f"{continent.lower()}_reach_slope_validation.parquet")
        reach_validation.to_parquet(reach_val_path, index=False)
        log(f"Saved reach validation to {reach_val_path}")

    # Save suspect sections for manual review
    if not suspect_sections.empty:
        suspect_path = os.path.join(output_dir, f"{continent.lower()}_suspect_sections.parquet")
        suspect_sections.to_parquet(suspect_path, index=False)
        log(f"Saved {len(suspect_sections):,} suspect sections to {suspect_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("COMPUTATION SUMMARY")
    print("=" * 60)
    print(f"Reaches processed: {len(results_df):,}")
    print(f"Mainstem reaches: {results_df['is_mainstem'].sum():,}")
    if not sections_df.empty:
        print(f"Sections (junction-to-junction): {len(sections_df):,}")
    if not section_validation.empty:
        n_valid = section_validation['direction_valid'].sum()
        n_invalid = (section_validation['direction_valid'] == False).sum()
        n_unknown = section_validation['direction_valid'].isna().sum()
        n_total = len(section_validation)
        print(f"Junction slope validation:")
        print(f"  Direction VALID: {n_valid:,} ({100*n_valid/n_total:.1f}%)")
        print(f"  Direction INVALID: {n_invalid:,} ({100*n_invalid/n_total:.1f}%)")
        print(f"  Unknown (insufficient data): {n_unknown:,}")
    if not suspect_sections.empty:
        n_topology_errors = (suspect_sections['likely_cause'] == 'potential_topology_error').sum()
        print(f"Suspect sections requiring review: {len(suspect_sections):,}")
        print(f"  Potential topology errors: {n_topology_errors:,}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Compute v17c attributes without phi")
    ap.add_argument("--continent", required=True, help="Continent code")
    ap.add_argument("--db-path", required=True, help="Path to v17b database")
    ap.add_argument("--v17c-db-path", default=None, help="Path to v17c database (for SWOT WSE)")
    ap.add_argument("--output-dir", default="output", help="Output directory")
    ap.add_argument("--skip-slopes", action="store_true", help="Skip slope validation")

    args = ap.parse_args()

    continent = args.continent.upper()

    # Default v17c path
    v17c_db_path = args.v17c_db_path
    if v17c_db_path is None:
        # Try default location
        default_v17c = os.path.join(os.path.dirname(args.db_path), "sword_v17c.duckdb")
        if os.path.exists(default_v17c):
            v17c_db_path = default_v17c

    log(f"=== v17c Attribute Computation (no phi) ===")
    log(f"Continent: {continent}")
    log(f"Database: {args.db_path}")
    if v17c_db_path:
        log(f"v17c DB (for SWOT WSE): {v17c_db_path}")

    # Load data
    topology_df = load_topology(args.db_path, continent)
    reaches_df = load_reaches(args.db_path, continent, v17c_db_path)

    # Build reach-level graph
    G = build_reach_graph(topology_df, reaches_df)

    # Validate DAG
    if not nx.is_directed_acyclic_graph(G):
        log("ERROR: Graph contains cycles!")
        sys.exit(1)

    # Identify junctions and build section graph
    junctions = identify_junctions(G)
    R, sections_df = build_section_graph(G, junctions)

    # Junction-level topology validation using SWOT WSE
    section_validation = pd.DataFrame()
    reach_validation = pd.DataFrame()
    suspect_sections = pd.DataFrame()

    if not args.skip_slopes:
        # Check if we have WSE data
        has_wse = reaches_df['wse_obs_mean'].notna().any() if 'wse_obs_mean' in reaches_df.columns else False

        if has_wse:
            section_validation, reach_validation = validate_slopes_junction_level(
                G, R, sections_df, reaches_df
            )
            if not section_validation.empty:
                suspect_sections = analyze_suspect_sections(section_validation, G)
        else:
            log("WARNING: No SWOT WSE data available - skipping slope validation")
            log("  To enable validation, ensure v17c database has wse_obs_mean column")

    # Compute new attributes
    hydro_dist = compute_hydro_distances(G)
    hw_out = compute_best_headwater_outlet(G)
    is_mainstem = compute_mainstem(G, hw_out)

    # Save results
    output_dir = os.path.join(args.output_dir, continent.lower())
    save_results(
        reaches_df, hydro_dist, hw_out, is_mainstem,
        section_validation, reach_validation, suspect_sections,
        sections_df, output_dir, continent
    )

    # Save graphs for inspection
    graph_path = os.path.join(output_dir, f"{continent.lower()}_reach_graph.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)
    log(f"Saved reach graph to {graph_path}")

    section_graph_path = os.path.join(output_dir, f"{continent.lower()}_section_graph.pkl")
    with open(section_graph_path, "wb") as f:
        pickle.dump(R, f)
    log(f"Saved section graph to {section_graph_path}")

    log("Done!")


if __name__ == "__main__":
    main()
