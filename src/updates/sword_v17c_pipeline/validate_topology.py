#!/usr/bin/env python3
"""
Validate topology from directed graph.

Performs these checks:
1. DAG validation (no cycles)
2. dist_out monotonicity (decreases downstream)
3. path_freq monotonicity (increases downstream)
4. facc monotonicity (increases downstream)
5. Width trend (generally increases downstream)
6. WSE/elevation monotonicity (decreases downstream)
7. Slope reasonableness (no negative, flag extremes)
8. Connected components analysis
9. Orphan reach detection
10. Lake sandwich detection (river between lakes)
11. n_rch_up/n_rch_down consistency with graph
12. Reach length outliers
13. Attribute completeness

Can compare original v17b topology vs phi-optimized topology.

Usage:
    # Validate single graph
    python validate_topology.py --graph output/na/na_v17b_directed.pkl

    # Compare v17b vs phi
    python validate_topology.py \
        --graph output/na/na_v17b_directed.pkl \
        --compare output/na/na_MultiDirected_refined.pkl

    # Full validation with all heuristics
    python validate_topology.py --graph graph.pkl --full

    # JSON output for CI
    python validate_topology.py --graph graph.pkl --json
"""

import argparse
import pickle
import sys
from collections import defaultdict
from datetime import datetime as dt

import networkx as nx
import numpy as np


def log(msg):
    print(f"[{dt.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_graph(path: str) -> nx.MultiDiGraph:
    """Load pickled graph."""
    log(f"Loading graph from {path}...")
    with open(path, "rb") as f:
        G = pickle.load(f)
    log(f"Loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def validate_dag(G: nx.MultiDiGraph) -> dict:
    """Check if graph is a DAG."""
    log("Checking DAG property...")

    is_dag = nx.is_directed_acyclic_graph(G)

    result = {
        'is_dag': is_dag,
        'n_cycles': 0,
        'sample_cycles': [],
    }

    if not is_dag:
        # Find some cycles
        try:
            cycles = list(nx.simple_cycles(G))
            result['n_cycles'] = len(cycles)
            result['sample_cycles'] = cycles[:5]
            log(f"FAIL: Found {len(cycles)} cycles")
        except Exception as e:
            log(f"FAIL: Has cycles but couldn't enumerate: {e}")
    else:
        log("PASS: Graph is a valid DAG")

    return result


def get_node_or_edge_attr(G: nx.MultiDiGraph, node, edge_data, attr: str):
    """Get attribute from node first, then edge data as fallback."""
    val = G.nodes[node].get(attr)
    if val is not None:
        return val
    return edge_data.get(attr)


def get_edge_endpoint_attrs(G: nx.MultiDiGraph, u, v, edge_data, attr: str):
    """
    Get attribute values for edge endpoints.

    For edge u -> v, tries to get attr for both endpoints.
    Handles cases where attr is stored on:
    - Nodes directly
    - Edges (uses same edge's attr as proxy for source node)
    - Need to look up connected edges for target node

    Returns (val_u, val_v) or (None, None) if not found.
    """
    # Try node attributes first
    val_u = G.nodes[u].get(attr)
    val_v = G.nodes[v].get(attr)

    if val_u is not None and val_v is not None:
        return val_u, val_v

    # If on edges, we need the edge's own value for u, and a downstream edge's value for v
    if val_u is None:
        val_u = edge_data.get(attr)

    if val_v is None:
        # Look for any outgoing edge from v to get its dist_out
        for _, _, _, succ_data in G.out_edges(v, keys=True, data=True):
            val_v = succ_data.get(attr)
            if val_v is not None:
                break

    return val_u, val_v


def validate_dist_out_monotonicity(G: nx.MultiDiGraph) -> dict:
    """
    Check that dist_out decreases as we move downstream.

    For each edge u -> v: dist_out[u] should be > dist_out[v]
    """
    log("Checking dist_out monotonicity...")

    violations = []
    checked_edges = 0
    skipped_edges = 0

    for u, v, k, d in G.edges(keys=True, data=True):
        # Get dist_out for both endpoints
        dist_u, dist_v = get_edge_endpoint_attrs(G, u, v, d, 'dist_out')

        if dist_u is None or dist_v is None:
            skipped_edges += 1
            continue

        checked_edges += 1

        # Should decrease downstream (u -> v means u is upstream)
        if dist_u <= dist_v:
            violations.append({
                'edge': (u, v, k),
                'dist_u': dist_u,
                'dist_v': dist_v,
                'diff': dist_v - dist_u,
            })

    n_violations = len(violations)
    pct = 100 * n_violations / checked_edges if checked_edges > 0 else 0

    result = {
        'checked_edges': checked_edges,
        'skipped_edges': skipped_edges,
        'violations': n_violations,
        'violation_pct': pct,
        'sample_violations': violations[:10],
    }

    if skipped_edges > 0:
        log(f"INFO: Skipped {skipped_edges:,} edges (missing dist_out on endpoint)")

    if n_violations == 0:
        log(f"PASS: All {checked_edges:,} checked edges have decreasing dist_out")
    else:
        log(f"WARN: {n_violations:,} edges ({pct:.2f}%) violate dist_out monotonicity")

    return result


def validate_path_freq_monotonicity(G: nx.MultiDiGraph) -> dict:
    """
    Check that path_freq increases as we move downstream.

    For each edge u -> v: path_freq[v] should be >= path_freq[u]
    (More paths converge as we move toward outlets)
    """
    log("Checking path_freq monotonicity...")

    violations = []
    checked_edges = 0
    skipped_edges = 0

    for u, v, k, d in G.edges(keys=True, data=True):
        freq_u, freq_v = get_edge_endpoint_attrs(G, u, v, d, 'path_freq')

        if freq_u is None or freq_v is None:
            skipped_edges += 1
            continue

        checked_edges += 1

        # Should increase or stay same downstream
        if freq_v < freq_u:
            violations.append({
                'edge': (u, v, k),
                'freq_u': freq_u,
                'freq_v': freq_v,
            })

    if checked_edges == 0:
        log("SKIP: No path_freq attribute found")
        return {'skipped': True}

    n_violations = len(violations)
    pct = 100 * n_violations / checked_edges if checked_edges > 0 else 0

    result = {
        'checked_edges': checked_edges,
        'skipped_edges': skipped_edges,
        'violations': n_violations,
        'violation_pct': pct,
        'sample_violations': violations[:10],
    }

    if n_violations == 0:
        log(f"PASS: All {checked_edges:,} edges have non-decreasing path_freq")
    else:
        log(f"INFO: {n_violations:,} edges ({pct:.2f}%) have decreasing path_freq (expected at bifurcations)")

    return result


def validate_connected_components(G: nx.MultiDiGraph) -> dict:
    """Analyze connected components."""
    log("Analyzing connected components...")

    weak_components = list(nx.weakly_connected_components(G))
    component_sizes = sorted([len(c) for c in weak_components], reverse=True)

    result = {
        'n_components': len(weak_components),
        'largest_size': component_sizes[0] if component_sizes else 0,
        'size_distribution': component_sizes[:20],  # Top 20
        'singleton_count': sum(1 for s in component_sizes if s == 1),
    }

    log(f"INFO: {len(weak_components):,} weakly connected components")
    log(f"INFO: Largest component: {result['largest_size']:,} nodes")
    log(f"INFO: Singletons: {result['singleton_count']:,}")

    return result


def validate_headwaters_outlets(G: nx.MultiDiGraph) -> dict:
    """Analyze headwaters (in_degree=0) and outlets (out_degree=0)."""
    log("Analyzing headwaters and outlets...")

    headwaters = [n for n in G.nodes() if G.in_degree(n) == 0]
    outlets = [n for n in G.nodes() if G.out_degree(n) == 0]
    orphans = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]

    result = {
        'n_headwaters': len(headwaters),
        'n_outlets': len(outlets),
        'n_orphans': len(orphans),
    }

    log(f"INFO: {len(headwaters):,} headwaters (in_degree=0)")
    log(f"INFO: {len(outlets):,} outlets (out_degree=0)")
    if orphans:
        log(f"WARN: {len(orphans):,} orphan nodes (isolated, no connections)")

    return result


# =============================================================================
# NEW HEURISTIC VALIDATIONS
# =============================================================================


def validate_facc_monotonicity(G: nx.MultiDiGraph) -> dict:
    """
    Check that facc (flow accumulation) increases downstream.

    For each edge u -> v: facc[v] should be >= facc[u]
    (More drainage area accumulates as we move toward outlets)
    """
    log("Checking facc monotonicity...")

    violations = []
    checked_edges = 0
    skipped_edges = 0

    for u, v, k, d in G.edges(keys=True, data=True):
        facc_u, facc_v = get_edge_endpoint_attrs(G, u, v, d, 'facc')

        if facc_u is None or facc_v is None:
            skipped_edges += 1
            continue

        checked_edges += 1

        # Should increase or stay same downstream
        if facc_v < facc_u:
            violations.append({
                'edge': (u, v, k),
                'facc_u': facc_u,
                'facc_v': facc_v,
                'diff': facc_u - facc_v,
            })

    if checked_edges == 0:
        log("SKIP: No facc attribute found")
        return {'skipped': True}

    n_violations = len(violations)
    pct = 100 * n_violations / checked_edges if checked_edges > 0 else 0

    result = {
        'checked_edges': checked_edges,
        'skipped_edges': skipped_edges,
        'violations': n_violations,
        'violation_pct': pct,
        'sample_violations': violations[:10],
    }

    if n_violations == 0:
        log(f"PASS: All {checked_edges:,} edges have non-decreasing facc")
    else:
        log(f"WARN: {n_violations:,} edges ({pct:.2f}%) violate facc monotonicity")

    return result


def validate_width_trend(G: nx.MultiDiGraph, tolerance_pct: float = 50.0) -> dict:
    """
    Check that width generally increases downstream.

    Uses tolerance because natural width variation is expected.
    Only flags significant decreases (> tolerance_pct).

    Args:
        tolerance_pct: Allow width to decrease by this % without flagging
    """
    log("Checking width trend...")

    violations = []
    checked_edges = 0
    skipped_edges = 0

    for u, v, k, d in G.edges(keys=True, data=True):
        width_u, width_v = get_edge_endpoint_attrs(G, u, v, d, 'width')

        if width_u is None or width_v is None or width_u <= 0:
            skipped_edges += 1
            continue

        checked_edges += 1

        # Check for significant decrease
        decrease_pct = 100 * (width_u - width_v) / width_u
        if decrease_pct > tolerance_pct:
            violations.append({
                'edge': (u, v, k),
                'width_u': width_u,
                'width_v': width_v,
                'decrease_pct': decrease_pct,
            })

    if checked_edges == 0:
        log("SKIP: No width attribute found")
        return {'skipped': True}

    n_violations = len(violations)
    pct = 100 * n_violations / checked_edges if checked_edges > 0 else 0

    result = {
        'checked_edges': checked_edges,
        'skipped_edges': skipped_edges,
        'violations': n_violations,
        'violation_pct': pct,
        'tolerance_pct': tolerance_pct,
        'sample_violations': violations[:10],
    }

    if n_violations == 0:
        log(f"PASS: No significant width decreases (>{tolerance_pct}%)")
    else:
        log(f"INFO: {n_violations:,} edges ({pct:.2f}%) have significant width decrease")

    return result


def validate_wse_monotonicity(G: nx.MultiDiGraph) -> dict:
    """
    Check that WSE (water surface elevation) decreases downstream.

    For each edge u -> v: wse[u] should be > wse[v]
    (Water flows downhill)
    """
    log("Checking WSE monotonicity...")

    violations = []
    checked_edges = 0
    skipped_edges = 0

    for u, v, k, d in G.edges(keys=True, data=True):
        # Try multiple possible attribute names
        wse_u, wse_v = get_edge_endpoint_attrs(G, u, v, d, 'wse')
        if wse_u is None or wse_v is None:
            wse_u, wse_v = get_edge_endpoint_attrs(G, u, v, d, 'wse_mean')

        if wse_u is None or wse_v is None:
            skipped_edges += 1
            continue

        checked_edges += 1

        # Should decrease downstream
        if wse_u < wse_v:
            violations.append({
                'edge': (u, v, k),
                'wse_u': wse_u,
                'wse_v': wse_v,
                'diff': wse_v - wse_u,
            })

    if checked_edges == 0:
        log("SKIP: No WSE attribute found")
        return {'skipped': True}

    n_violations = len(violations)
    pct = 100 * n_violations / checked_edges if checked_edges > 0 else 0

    result = {
        'checked_edges': checked_edges,
        'skipped_edges': skipped_edges,
        'violations': n_violations,
        'violation_pct': pct,
        'sample_violations': violations[:10],
    }

    if n_violations == 0:
        log(f"PASS: All {checked_edges:,} edges have decreasing WSE")
    else:
        log(f"WARN: {n_violations:,} edges ({pct:.2f}%) violate WSE monotonicity")

    return result


def validate_slope_reasonableness(G: nx.MultiDiGraph,
                                   min_slope: float = -0.001,
                                   max_slope_mpm: float = 0.01) -> dict:
    """
    Check that slopes are within reasonable bounds.

    Slope units in SWORD can vary:
    - m/m: typical range 0.00001 to 0.01
    - cm/km: typical range 1 to 1000

    This function auto-detects units based on median value.

    Args:
        min_slope: Minimum expected slope (allow small negative for measurement error)
        max_slope_mpm: Maximum expected slope in m/m (0.01 = 1% grade)
    """
    log("Checking slope reasonableness...")

    slope_values = []
    total_edges = 0

    for u, v, k, d in G.edges(keys=True, data=True):
        total_edges += 1
        slope = d.get('slope', G.nodes[u].get('slope'))
        if slope is not None:
            slope_values.append(slope)

    if len(slope_values) == 0:
        log("SKIP: No slope attribute found")
        return {'skipped': True}

    slope_arr = np.array(slope_values)
    median_slope = float(np.median(slope_arr))

    # Auto-detect units: if median > 1, likely cm/km; if < 0.1, likely m/m
    if median_slope > 1.0:
        unit = "cm/km"
        # Convert thresholds to cm/km
        effective_min = min_slope * 100000  # m/m to cm/km
        effective_max = max_slope_mpm * 100000
    else:
        unit = "m/m"
        effective_min = min_slope
        effective_max = max_slope_mpm

    negative_slopes = []
    extreme_slopes = []

    for u, v, k, d in G.edges(keys=True, data=True):
        slope = d.get('slope', G.nodes[u].get('slope'))
        if slope is not None:
            if slope < effective_min:
                negative_slopes.append({
                    'edge': (u, v, k),
                    'slope': slope,
                })
            elif slope > effective_max:
                extreme_slopes.append({
                    'edge': (u, v, k),
                    'slope': slope,
                })

    result = {
        'total_edges': len(slope_values),
        'detected_unit': unit,
        'negative_count': len(negative_slopes),
        'extreme_count': len(extreme_slopes),
        'min_slope': float(np.min(slope_arr)),
        'max_slope': float(np.max(slope_arr)),
        'mean_slope': float(np.mean(slope_arr)),
        'median_slope': median_slope,
        'p95_slope': float(np.percentile(slope_arr, 95)),
        'sample_negative': negative_slopes[:10],
        'sample_extreme': extreme_slopes[:10],
    }

    if len(negative_slopes) == 0:
        log(f"PASS: No negative slopes")
    else:
        log(f"WARN: {len(negative_slopes):,} edges have negative slope")

    log(f"INFO: Detected slope unit: {unit} (median={median_slope:.4f})")

    if len(extreme_slopes) == 0:
        log(f"PASS: No extreme slopes")
    else:
        log(f"INFO: {len(extreme_slopes):,} edges have extreme slope (>{effective_max:.4f} {unit})")

    return result


def validate_lake_sandwich(G: nx.MultiDiGraph) -> dict:
    """
    Detect lake sandwich pattern: river reach between two lake reaches.

    Pattern: lake -> river -> lake
    This can indicate misclassification or topology issues.

    lakeflag: 0=river, 1=lake, 2=canal, 3=tidal

    Note: lakeflag may be stored on edges or nodes depending on graph format.
    """
    log("Checking for lake sandwiches...")

    sandwiches = []
    total_river_reaches = 0
    has_lakeflag = False

    # Build a lookup for lakeflag by node (from edges or nodes)
    node_lakeflag = {}

    # First try nodes
    for n in G.nodes():
        lf = G.nodes[n].get('lakeflag', G.nodes[n].get('lake_flag'))
        if lf is not None:
            has_lakeflag = True
            node_lakeflag[n] = lf

    # If not on nodes, try to get from edges (use edge's lakeflag for source node)
    if not has_lakeflag:
        for u, v, k, d in G.edges(keys=True, data=True):
            lf = d.get('lakeflag', d.get('lake_flag'))
            if lf is not None:
                has_lakeflag = True
                node_lakeflag[u] = lf

    if not has_lakeflag:
        log("SKIP: No lakeflag attribute found")
        return {'skipped': True}

    # Now check for sandwiches
    for n, lakeflag in node_lakeflag.items():
        # Only check river reaches (lakeflag == 0)
        if lakeflag == 0:
            total_river_reaches += 1

            # Get predecessors and successors
            preds = list(G.predecessors(n))
            succs = list(G.successors(n))

            # Check if any predecessor is a lake
            pred_lake = any(
                node_lakeflag.get(p) == 1
                for p in preds
            )

            # Check if any successor is a lake
            succ_lake = any(
                node_lakeflag.get(s) == 1
                for s in succs
            )

            if pred_lake and succ_lake:
                sandwiches.append({
                    'reach': n,
                    'predecessors': preds,
                    'successors': succs,
                })

    n_sandwiches = len(sandwiches)
    pct = 100 * n_sandwiches / total_river_reaches if total_river_reaches > 0 else 0

    result = {
        'total_river_reaches': total_river_reaches,
        'sandwich_count': n_sandwiches,
        'sandwich_pct': pct,
        'sample_sandwiches': sandwiches[:20],
    }

    if n_sandwiches == 0:
        log(f"PASS: No lake sandwiches found")
    else:
        log(f"WARN: {n_sandwiches:,} lake sandwiches ({pct:.2f}% of river reaches)")

    return result


def validate_neighbor_counts(G: nx.MultiDiGraph) -> dict:
    """
    Check that n_rch_up and n_rch_down match actual graph topology.

    n_rch_up should equal in_degree
    n_rch_down should equal out_degree

    Note: Attributes may be on edges; we use edge attrs for the source node.
    """
    log("Checking neighbor count consistency...")

    up_mismatches = []
    down_mismatches = []
    has_counts = False

    # Build lookup from edges if not on nodes
    node_n_up = {}
    node_n_down = {}

    # Try nodes first
    for n in G.nodes():
        n_up = G.nodes[n].get('n_rch_up')
        n_down = G.nodes[n].get('n_rch_down')
        if n_up is not None:
            has_counts = True
            node_n_up[n] = n_up
        if n_down is not None:
            has_counts = True
            node_n_down[n] = n_down

    # If not on nodes, try edges
    if not has_counts:
        for u, v, k, d in G.edges(keys=True, data=True):
            n_up = d.get('n_rch_up')
            n_down = d.get('n_rch_down')
            if n_up is not None:
                has_counts = True
                node_n_up[u] = n_up
            if n_down is not None:
                has_counts = True
                node_n_down[u] = n_down

    if not has_counts:
        log("SKIP: No n_rch_up/n_rch_down attributes found")
        return {'skipped': True}

    # Check consistency
    for n in set(node_n_up.keys()) | set(node_n_down.keys()):
        actual_up = G.in_degree(n)
        actual_down = G.out_degree(n)

        if n in node_n_up and node_n_up[n] != actual_up:
            up_mismatches.append({
                'node': n,
                'n_rch_up': node_n_up[n],
                'actual_in_degree': actual_up,
            })

        if n in node_n_down and node_n_down[n] != actual_down:
            down_mismatches.append({
                'node': n,
                'n_rch_down': node_n_down[n],
                'actual_out_degree': actual_down,
            })

    total_checked = len(set(node_n_up.keys()) | set(node_n_down.keys()))

    result = {
        'total_checked': total_checked,
        'up_mismatches': len(up_mismatches),
        'down_mismatches': len(down_mismatches),
        'sample_up_mismatches': up_mismatches[:10],
        'sample_down_mismatches': down_mismatches[:10],
    }

    if len(up_mismatches) == 0 and len(down_mismatches) == 0:
        log(f"PASS: All {total_checked:,} neighbor counts match graph topology")
    else:
        if up_mismatches:
            log(f"WARN: {len(up_mismatches):,} nodes have n_rch_up mismatch")
        if down_mismatches:
            log(f"WARN: {len(down_mismatches):,} nodes have n_rch_down mismatch")

    return result


def validate_reach_length(G: nx.MultiDiGraph,
                          min_length: float = 10.0,
                          max_length: float = 100000.0) -> dict:
    """
    Check reach lengths are within reasonable bounds.

    Args:
        min_length: Minimum expected reach length (m)
        max_length: Maximum expected reach length (m)
    """
    log("Checking reach length bounds...")

    too_short = []
    too_long = []
    lengths = []

    # Try multiple attribute names
    attr_names = ['reach_length', 'length', 'reach_len', 'len']

    for u, v, k, d in G.edges(keys=True, data=True):
        length = None
        for attr in attr_names:
            length = d.get(attr)
            if length is not None:
                break
            length = G.nodes[u].get(attr)
            if length is not None:
                break

        if length is not None:
            lengths.append(length)

            if length < min_length:
                too_short.append({
                    'edge': (u, v, k),
                    'length': length,
                })
            elif length > max_length:
                too_long.append({
                    'edge': (u, v, k),
                    'length': length,
                })

    if len(lengths) == 0:
        log("SKIP: No reach_length attribute found")
        return {'skipped': True}

    length_arr = np.array(lengths)

    result = {
        'total_edges': len(lengths),
        'too_short': len(too_short),
        'too_long': len(too_long),
        'min_length': float(np.min(length_arr)),
        'max_length': float(np.max(length_arr)),
        'mean_length': float(np.mean(length_arr)),
        'median_length': float(np.median(length_arr)),
        'sample_too_short': too_short[:10],
        'sample_too_long': too_long[:10],
    }

    if len(too_short) == 0:
        log(f"PASS: No reaches shorter than {min_length}m")
    else:
        log(f"INFO: {len(too_short):,} reaches shorter than {min_length}m")

    if len(too_long) == 0:
        log(f"PASS: No reaches longer than {max_length}m")
    else:
        log(f"INFO: {len(too_long):,} reaches longer than {max_length}m")

    return result


def validate_attribute_completeness(G: nx.MultiDiGraph,
                                     required_attrs: list = None) -> dict:
    """
    Check that required attributes are present on nodes/edges.

    Args:
        required_attrs: List of attribute names to check.
                       If None, uses default SWORD attributes.
    """
    log("Checking attribute completeness...")

    if required_attrs is None:
        required_attrs = [
            'dist_out', 'facc', 'width', 'reach_length',
            'lakeflag', 'n_rch_up', 'n_rch_down',
        ]

    node_coverage = {}
    edge_coverage = {}

    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()

    for attr in required_attrs:
        # Check nodes
        node_count = sum(1 for n in G.nodes() if G.nodes[n].get(attr) is not None)
        node_coverage[attr] = {
            'count': node_count,
            'pct': 100 * node_count / total_nodes if total_nodes > 0 else 0,
        }

        # Check edges
        edge_count = sum(1 for u, v, k, d in G.edges(keys=True, data=True) if d.get(attr) is not None)
        edge_coverage[attr] = {
            'count': edge_count,
            'pct': 100 * edge_count / total_edges if total_edges > 0 else 0,
        }

    # Report
    for attr in required_attrs:
        nc = node_coverage[attr]
        ec = edge_coverage[attr]
        if nc['count'] > 0:
            log(f"INFO: {attr} on nodes: {nc['count']:,} ({nc['pct']:.1f}%)")
        elif ec['count'] > 0:
            log(f"INFO: {attr} on edges: {ec['count']:,} ({ec['pct']:.1f}%)")
        else:
            log(f"WARN: {attr} not found")

    result = {
        'required_attrs': required_attrs,
        'node_coverage': node_coverage,
        'edge_coverage': edge_coverage,
        'total_nodes': total_nodes,
        'total_edges': total_edges,
    }

    return result


def validate_trib_flag_consistency(G: nx.MultiDiGraph) -> dict:
    """
    Check that trib_flag matches actual tributaries.

    trib_flag should be 1 if in_degree > 1 (has tributary joining)
    """
    log("Checking trib_flag consistency...")

    mismatches = []
    has_trib_flag = False

    # Build lookup from edges if not on nodes
    node_trib_flag = {}

    for n in G.nodes():
        tf = G.nodes[n].get('trib_flag')
        if tf is not None:
            has_trib_flag = True
            node_trib_flag[n] = tf

    if not has_trib_flag:
        for u, v, k, d in G.edges(keys=True, data=True):
            tf = d.get('trib_flag')
            if tf is not None:
                has_trib_flag = True
                node_trib_flag[u] = tf

    if not has_trib_flag:
        log("SKIP: No trib_flag attribute found")
        return {'skipped': True}

    for n, trib_flag in node_trib_flag.items():
        actual_in_degree = G.in_degree(n)
        expected_trib = 1 if actual_in_degree > 1 else 0

        if trib_flag != expected_trib:
            mismatches.append({
                'node': n,
                'trib_flag': trib_flag,
                'in_degree': actual_in_degree,
                'expected': expected_trib,
            })

    result = {
        'total_checked': len(node_trib_flag),
        'mismatches': len(mismatches),
        'sample_mismatches': mismatches[:10],
    }

    if len(mismatches) == 0:
        log(f"PASS: All {len(node_trib_flag):,} trib_flag values consistent with topology")
    else:
        log(f"WARN: {len(mismatches):,} nodes have trib_flag mismatch")

    return result


def compare_topologies(G1: nx.MultiDiGraph, G2: nx.MultiDiGraph, name1: str, name2: str) -> dict:
    """
    Compare two topologies to find differences.

    Looks at:
    - Edge direction differences
    - Missing/extra edges
    - Attribute differences
    """
    log(f"Comparing {name1} vs {name2}...")

    # Get edge sets (as frozensets to ignore key differences)
    edges1 = set()
    edges2 = set()

    for u, v, k in G1.edges(keys=True):
        edges1.add((u, v))

    for u, v, k in G2.edges(keys=True):
        edges2.add((u, v))

    # Find differences
    only_in_1 = edges1 - edges2
    only_in_2 = edges2 - edges1

    # Check for reversed edges
    reversed_edges = []
    for u, v in only_in_1:
        if (v, u) in only_in_2:
            reversed_edges.append((u, v))

    result = {
        'edges_g1': len(edges1),
        'edges_g2': len(edges2),
        'only_in_g1': len(only_in_1),
        'only_in_g2': len(only_in_2),
        'reversed': len(reversed_edges),
        'sample_reversed': reversed_edges[:10],
    }

    common = edges1 & edges2
    log(f"INFO: {name1} has {len(edges1):,} edges")
    log(f"INFO: {name2} has {len(edges2):,} edges")
    log(f"INFO: Common edges: {len(common):,}")
    log(f"INFO: Only in {name1}: {len(only_in_1):,}")
    log(f"INFO: Only in {name2}: {len(only_in_2):,}")
    log(f"INFO: Reversed direction: {len(reversed_edges):,}")

    return result


def run_validation(graph_path: str, compare_path: str = None, full: bool = False) -> dict:
    """Run all validations on a graph.

    Args:
        graph_path: Path to pickle file
        compare_path: Optional second graph for comparison
        full: If True, run all heuristic validations
    """

    G = load_graph(graph_path)

    results = {
        'graph_path': graph_path,
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
    }

    # Core validations (always run)
    results['dag'] = validate_dag(G)
    results['dist_out'] = validate_dist_out_monotonicity(G)
    results['path_freq'] = validate_path_freq_monotonicity(G)
    results['components'] = validate_connected_components(G)
    results['hw_outlets'] = validate_headwaters_outlets(G)

    # Extended heuristics (run with --full or always if data available)
    if full:
        log("\n--- Extended Heuristics ---")
        results['facc'] = validate_facc_monotonicity(G)
        results['width'] = validate_width_trend(G)
        results['wse'] = validate_wse_monotonicity(G)
        results['slope'] = validate_slope_reasonableness(G)
        results['lake_sandwich'] = validate_lake_sandwich(G)
        results['neighbor_counts'] = validate_neighbor_counts(G)
        results['reach_length'] = validate_reach_length(G)
        results['trib_flag'] = validate_trib_flag_consistency(G)
        results['completeness'] = validate_attribute_completeness(G)

    # Compare if second graph provided
    if compare_path:
        G2 = load_graph(compare_path)
        results['comparison'] = compare_topologies(
            G, G2,
            graph_path.split('/')[-1],
            compare_path.split('/')[-1]
        )

    return results


def print_summary(results: dict):
    """Print validation summary."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\nGraph: {results['graph_path']}")
    print(f"Nodes: {results['nodes']:,}")
    print(f"Edges: {results['edges']:,}")

    # DAG check
    dag = results['dag']
    status = "✓ PASS" if dag['is_dag'] else "✗ FAIL"
    print(f"\n[{status}] DAG validation")
    if not dag['is_dag']:
        print(f"    {dag['n_cycles']} cycles found")

    # dist_out
    dist = results['dist_out']
    if dist.get('skipped_edges', 0) > 0:
        print(f"    (checked {dist.get('checked_edges', 0):,}, skipped {dist['skipped_edges']:,})")
    if dist['violations'] == 0:
        print(f"\n[✓ PASS] dist_out monotonicity")
    else:
        pct = dist['violation_pct']
        print(f"\n[⚠ WARN] dist_out monotonicity: {dist['violations']:,} violations ({pct:.2f}%)")

    # path_freq
    pf = results.get('path_freq', {})
    if pf.get('skipped'):
        print(f"\n[- SKIP] path_freq monotonicity (no data)")
    elif pf.get('violations', 0) == 0:
        print(f"\n[✓ PASS] path_freq monotonicity")
    else:
        print(f"\n[ℹ INFO] path_freq: {pf['violations']:,} decreases (expected at bifurcations)")

    # Components
    comp = results['components']
    print(f"\n[ℹ INFO] Connected components: {comp['n_components']:,}")
    print(f"    Largest: {comp['largest_size']:,} nodes")
    print(f"    Singletons: {comp['singleton_count']:,}")

    # Headwaters/outlets
    hw = results['hw_outlets']
    print(f"\n[ℹ INFO] Headwaters: {hw['n_headwaters']:,}")
    print(f"[ℹ INFO] Outlets: {hw['n_outlets']:,}")
    if hw['n_orphans'] > 0:
        print(f"[⚠ WARN] Orphan nodes: {hw['n_orphans']:,}")

    # === Extended Heuristics ===
    if 'facc' in results:
        print("\n" + "-" * 40)
        print("EXTENDED HEURISTICS")
        print("-" * 40)

        # facc
        facc = results.get('facc', {})
        if facc.get('skipped'):
            print(f"\n[- SKIP] facc monotonicity (no data)")
        elif facc.get('violations', 0) == 0:
            print(f"\n[✓ PASS] facc monotonicity")
        else:
            print(f"\n[⚠ WARN] facc monotonicity: {facc['violations']:,} violations ({facc['violation_pct']:.2f}%)")

        # width
        width = results.get('width', {})
        if width.get('skipped'):
            print(f"\n[- SKIP] width trend (no data)")
        elif width.get('violations', 0) == 0:
            print(f"\n[✓ PASS] width trend (no significant decreases)")
        else:
            print(f"\n[ℹ INFO] width: {width['violations']:,} significant decreases ({width['violation_pct']:.2f}%)")

        # wse
        wse = results.get('wse', {})
        if wse.get('skipped'):
            print(f"\n[- SKIP] WSE monotonicity (no data)")
        elif wse.get('violations', 0) == 0:
            print(f"\n[✓ PASS] WSE monotonicity")
        else:
            print(f"\n[⚠ WARN] WSE monotonicity: {wse['violations']:,} violations ({wse['violation_pct']:.2f}%)")

        # slope
        slope = results.get('slope', {})
        if slope.get('skipped'):
            print(f"\n[- SKIP] slope reasonableness (no data)")
        else:
            unit = slope.get('detected_unit', 'm/m')
            if slope.get('negative_count', 0) == 0:
                print(f"\n[✓ PASS] slope: no negative values")
            else:
                print(f"\n[⚠ WARN] slope: {slope['negative_count']:,} negative values")
            if slope.get('extreme_count', 0) > 0:
                print(f"[ℹ INFO] slope: {slope['extreme_count']:,} extreme values")
            print(f"    Unit: {unit}")
            print(f"    Range: {slope['min_slope']:.4f} to {slope['max_slope']:.4f}")
            print(f"    Median: {slope['median_slope']:.4f}, P95: {slope.get('p95_slope', 0):.4f}")

        # lake sandwich
        ls = results.get('lake_sandwich', {})
        if ls.get('skipped'):
            print(f"\n[- SKIP] lake sandwich (no lakeflag data)")
        elif ls.get('sandwich_count', 0) == 0:
            print(f"\n[✓ PASS] lake sandwich: none found")
        else:
            print(f"\n[⚠ WARN] lake sandwich: {ls['sandwich_count']:,} ({ls['sandwich_pct']:.2f}%)")

        # neighbor counts
        nc = results.get('neighbor_counts', {})
        if nc.get('skipped'):
            print(f"\n[- SKIP] neighbor counts (no n_rch_up/down data)")
        elif nc.get('up_mismatches', 0) == 0 and nc.get('down_mismatches', 0) == 0:
            print(f"\n[✓ PASS] neighbor counts match topology")
        else:
            print(f"\n[⚠ WARN] neighbor counts: {nc.get('up_mismatches', 0)} up, {nc.get('down_mismatches', 0)} down mismatches")

        # reach length
        rl = results.get('reach_length', {})
        if rl.get('skipped'):
            print(f"\n[- SKIP] reach length (no data)")
        else:
            issues = rl.get('too_short', 0) + rl.get('too_long', 0)
            if issues == 0:
                print(f"\n[✓ PASS] reach length: all within bounds")
            else:
                print(f"\n[ℹ INFO] reach length: {rl.get('too_short', 0)} too short, {rl.get('too_long', 0)} too long")
            print(f"    Range: {rl['min_length']:.1f}m to {rl['max_length']:.1f}m")
            print(f"    Median: {rl['median_length']:.1f}m")

        # trib_flag
        tf = results.get('trib_flag', {})
        if tf.get('skipped'):
            print(f"\n[- SKIP] trib_flag (no data)")
        elif tf.get('mismatches', 0) == 0:
            print(f"\n[✓ PASS] trib_flag consistent with topology")
        else:
            print(f"\n[⚠ WARN] trib_flag: {tf['mismatches']:,} mismatches")

        # completeness
        comp = results.get('completeness', {})
        if comp:
            print(f"\n[ℹ INFO] Attribute completeness:")
            for attr in comp.get('required_attrs', []):
                nc = comp['node_coverage'].get(attr, {})
                ec = comp['edge_coverage'].get(attr, {})
                if nc.get('count', 0) > 0:
                    print(f"    {attr}: {nc['pct']:.0f}% nodes")
                elif ec.get('count', 0) > 0:
                    print(f"    {attr}: {ec['pct']:.0f}% edges")
                else:
                    print(f"    {attr}: NOT FOUND")

    # Comparison
    if 'comparison' in results:
        cmp = results['comparison']
        print(f"\n--- Topology Comparison ---")
        print(f"Reversed edges: {cmp['reversed']:,}")
        print(f"Only in first: {cmp['only_in_g1']:,}")
        print(f"Only in second: {cmp['only_in_g2']:,}")

    print("\n" + "=" * 60)


def main():
    ap = argparse.ArgumentParser(
        description="Validate topology graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python validate_topology.py --graph output/na/na.pkl

  # Full validation with all heuristics
  python validate_topology.py --graph output/na/na.pkl --full

  # Compare two graphs
  python validate_topology.py --graph v17b.pkl --compare v17c.pkl

  # JSON output for CI
  python validate_topology.py --graph graph.pkl --full --json
        """
    )
    ap.add_argument("--graph", required=True, help="Path to graph pickle")
    ap.add_argument("--compare", default=None, help="Optional second graph to compare")
    ap.add_argument("--full", action="store_true",
                    help="Run all extended heuristics (facc, width, WSE, slope, lake sandwich, etc.)")
    ap.add_argument("--json", action="store_true", help="Output as JSON")

    args = ap.parse_args()

    results = run_validation(args.graph, args.compare, full=args.full)

    if args.json:
        import json
        # Convert non-serializable items
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [clean(x) for x in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj

        print(json.dumps(clean(results), indent=2))
    else:
        print_summary(results)


if __name__ == "__main__":
    main()
