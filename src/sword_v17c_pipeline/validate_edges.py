#!/usr/bin/env python3
"""
Validation script for SWORD graph edge attributes.

Computes statistics on edge attributes across all continents:
- Total R edges and percentage
- Changed edges that are R vs U
- Percentage of changed edges that are U
- Total changed edges
"""

import pickle
from pathlib import Path
import argparse
import networkx as nx


def validate_continent(continent_code, output_dir):
    """
    Validate edge attributes for a single continent.
    
    Parameters:
    -----------
    continent_code : str
        Two-letter continent code (na, sa, eu, af, oc, as)
    output_dir : Path
        Base output directory containing continent subdirectories
        
    Returns:
    --------
    dict : Statistics dictionary
    """
    pkl_path = output_dir / continent_code / f"{continent_code}.pkl"
    
    if not pkl_path.exists():
        raise FileNotFoundError(f"Graph file not found: {pkl_path}")
    
    print(f"Loading {continent_code} graph from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        G = pickle.load(f)
    
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Identify edges that are part of cycles (cyclomatic complexity)
    # An edge is part of a cycle if removing it doesn't disconnect the graph
    # This means finding edges that are NOT bridges (cut edges)
    print(f"  Identifying cycle edges (non-bridge edges)...")
    cycle_edges = set()
    edge_cycle_complexity = {}  # Map (u, v, k) -> cycle complexity metrics
    
    # Convert to undirected simple graph for cycle detection
    G_undirected = nx.Graph()
    edge_to_keys = {}  # Map (u, v) -> list of keys in original graph
    
    for u, v, k, data in G.edges(keys=True, data=True):
        # Use canonical ordering for undirected graph
        edge_pair = (u, v) if u < v else (v, u)
        G_undirected.add_edge(u, v)
        if edge_pair not in edge_to_keys:
            edge_to_keys[edge_pair] = []
        edge_to_keys[edge_pair].append((u, v, k))
    
    # Find bridges (edges whose removal disconnects the graph)
    bridges = list(nx.bridges(G_undirected))
    bridge_set = set()
    for u, v in bridges:
        bridge_set.add((u, v) if u < v else (v, u))
    
    # Find cycle basis to measure complexity
    cycle_basis = nx.cycle_basis(G_undirected)
    
    # Count cycle participation and measure cycle sizes for each edge
    edge_cycle_count = {}  # How many cycles each edge participates in
    edge_cycle_sizes = {}  # Sizes of cycles each edge participates in
    
    for cycle in cycle_basis:
        cycle_size = len(cycle)
        # Create edges from cycle (wrap around)
        cycle_edge_pairs = set()
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            edge_pair = (u, v) if u < v else (v, u)
            cycle_edge_pairs.add(edge_pair)
        
        # Count participation for each edge in this cycle
        for edge_pair in cycle_edge_pairs:
            if edge_pair in edge_to_keys:
                for orig_u, orig_v, k in edge_to_keys[edge_pair]:
                    edge_key = (orig_u, orig_v, k)
                    if edge_key not in edge_cycle_count:
                        edge_cycle_count[edge_key] = 0
                        edge_cycle_sizes[edge_key] = []
                    edge_cycle_count[edge_key] += 1
                    edge_cycle_sizes[edge_key].append(cycle_size)
    
    # All non-bridge edges are part of cycles
    for u, v in G_undirected.edges():
        edge_pair = (u, v) if u < v else (v, u)
        if edge_pair not in bridge_set:
            # This edge is part of a cycle - add all keys for this edge pair
            if edge_pair in edge_to_keys:
                for orig_u, orig_v, k in edge_to_keys[edge_pair]:
                    edge_key = (orig_u, orig_v, k)
                    cycle_edges.add(edge_key)
                    # Store complexity metrics
                    cycle_count = edge_cycle_count.get(edge_key, 0)
                    cycle_sizes = edge_cycle_sizes.get(edge_key, [])
                    avg_cycle_size = sum(cycle_sizes) / len(cycle_sizes) if cycle_sizes else 0
                    max_cycle_size = max(cycle_sizes) if cycle_sizes else 0
                    edge_cycle_complexity[edge_key] = {
                        'cycle_count': cycle_count,
                        'avg_cycle_size': avg_cycle_size,
                        'max_cycle_size': max_cycle_size,
                    }
    
    print(f"  Found {len(cycle_edges):,} cycle edges ({100*len(cycle_edges)/G.number_of_edges():.2f}% of total)")
    print(f"  Found {len(bridge_set):,} bridge edges ({100*len(bridge_set)/G.number_of_edges():.2f}% of total)")
    
    # Initialize counters
    total_edges = 0
    r_edges = 0
    u_edges = 0
    changed_r = 0
    changed_u = 0
    changed_total = 0
    changed_no_conf = 0  # Changed edges without confidence data
    unchanged_r = 0
    unchanged_u = 0
    edges_without_confidence = 0
    
    # Phi direction change analysis
    phi_flipped_total = 0
    phi_flip_confirmed = 0  # phi flipped AND final changed (kept)
    phi_flip_reversed = 0   # phi flipped BUT final NOT changed (reversed)
    phi_flip_confirmed_r = 0
    phi_flip_confirmed_u = 0
    phi_flip_reversed_r = 0
    phi_flip_reversed_u = 0
    phi_flip_no_reliable = 0  # phi flipped but no reliability data
    
    # SWORD-PHI agreement analysis
    phi_agreed_with_sword = 0  # phi did NOT flip (agrees with SWORD)
    phi_disagreed_with_sword = 0  # phi flipped (disagrees with SWORD)
    phi_disagreed_r = 0
    phi_disagreed_u = 0
    phi_disagreed_no_reliable = 0
    
    # Cycle vs non-cycle analysis
    cycle_total = 0
    cycle_phi_agreed = 0
    cycle_phi_disagreed = 0
    cycle_r_total = 0
    cycle_r_phi_agreed = 0
    cycle_r_phi_disagreed = 0
    cycle_u_total = 0
    cycle_u_phi_agreed = 0
    cycle_u_phi_disagreed = 0
    non_cycle_total = 0
    non_cycle_phi_agreed = 0
    non_cycle_phi_disagreed = 0
    non_cycle_r_total = 0
    non_cycle_r_phi_agreed = 0
    non_cycle_r_phi_disagreed = 0
    non_cycle_u_total = 0
    non_cycle_u_phi_agreed = 0
    non_cycle_u_phi_disagreed = 0
    
    # Cycle complexity analysis - group by cycle participation count
    complexity_groups = {}  # cycle_count -> {total, agreed, disagreed, r_total, r_agreed, r_disagreed}
    
    # Baseline validation: exclude edges where type = 6
    baseline_total = 0  # Total edges excluding type=6
    baseline_r_total = 0
    baseline_r_phi_agreed = 0
    baseline_r_phi_disagreed = 0
    
    # Count R edges with negative slope
    r_negative_slope = 0
    
    # Iterate through all edges (handles MultiDiGraph)
    for u, v, k, data in G.edges(keys=True, data=True):
        # Skip edges with type = 6 for baseline validation
        edge_type = data.get('type', None)
        is_baseline = True
        if edge_type == 6:
            is_baseline = False
        else:
            baseline_total += 1
        
        total_edges += 1
        
        # Get attributes (default to None if missing)
        direction_change = data.get('swot_direction_change', None)
        phi_direction_change = data.get('phi_direction_change', None)
        path_seg_reliable = data.get('path_seg_reliable', None)
        path_seg_slope = data.get('path_seg_slope', None)
        
        # Handle boolean values (could be True/False or string "T"/"F")
        is_changed = False
        if direction_change is True or (isinstance(direction_change, str) and direction_change.upper() == 'T'):
            is_changed = True
            changed_total += 1
        elif direction_change is False or (isinstance(direction_change, str) and direction_change.upper() == 'F'):
            is_changed = False
        # If None, treat as unchanged (conservative)
        
        # Handle phi_direction_change FIRST (needed for baseline calculation)
        phi_flipped = False
        if phi_direction_change is True or (isinstance(phi_direction_change, str) and phi_direction_change.upper() == 'T'):
            phi_flipped = True
            phi_flipped_total += 1
            phi_disagreed_with_sword += 1
        else:
            # Phi did not flip, so it agrees with SWORD
            phi_agreed_with_sword += 1
        
        # Handle path_seg_reliable (could be 'R', 'U', or None)
        has_reliable = path_seg_reliable is not None
        is_reliable = False
        if has_reliable:
            reliable_upper = str(path_seg_reliable).upper()
            if reliable_upper == 'R':
                is_reliable = True
                r_edges += 1
                # Check for negative slope
                if path_seg_slope is not None:
                    try:
                        slope_val = float(path_seg_slope)
                        if slope_val < 0:
                            r_negative_slope += 1
                    except (ValueError, TypeError):
                        pass
                # Track baseline R edges (excluding type = 6)
                if is_baseline:
                    baseline_r_total += 1
                    if not phi_flipped:
                        baseline_r_phi_agreed += 1
                    else:
                        baseline_r_phi_disagreed += 1
                # Count phi disagreements by reliability
                if phi_flipped:
                    phi_disagreed_r += 1
            elif reliable_upper == 'U':
                is_reliable = False
                u_edges += 1
                # Count phi disagreements by reliability
                if phi_flipped:
                    phi_disagreed_u += 1
        else:
            edges_without_confidence += 1
            if phi_flipped:
                phi_disagreed_no_reliable += 1
        
        # Track cycle vs non-cycle phi agreement
        is_cycle_edge = (u, v, k) in cycle_edges
        if is_cycle_edge:
            cycle_total += 1
            if phi_flipped:
                cycle_phi_disagreed += 1
            else:
                cycle_phi_agreed += 1
            
            # Get cycle complexity metrics
            complexity = edge_cycle_complexity.get((u, v, k), {})
            cycle_count = complexity.get('cycle_count', 0)
            
            # Track by complexity group
            if cycle_count not in complexity_groups:
                complexity_groups[cycle_count] = {
                    'total': 0,
                    'agreed': 0,
                    'disagreed': 0,
                    'r_total': 0,
                    'r_agreed': 0,
                    'r_disagreed': 0,
                }
            
            complexity_groups[cycle_count]['total'] += 1
            if phi_flipped:
                complexity_groups[cycle_count]['disagreed'] += 1
            else:
                complexity_groups[cycle_count]['agreed'] += 1
            
            # Track by reliability
            if has_reliable:
                if is_reliable:
                    cycle_r_total += 1
                    complexity_groups[cycle_count]['r_total'] += 1
                    if phi_flipped:
                        cycle_r_phi_disagreed += 1
                        complexity_groups[cycle_count]['r_disagreed'] += 1
                    else:
                        cycle_r_phi_agreed += 1
                        complexity_groups[cycle_count]['r_agreed'] += 1
                else:
                    cycle_u_total += 1
                    if phi_flipped:
                        cycle_u_phi_disagreed += 1
                    else:
                        cycle_u_phi_agreed += 1
        else:
            non_cycle_total += 1
            if phi_flipped:
                non_cycle_phi_disagreed += 1
            else:
                non_cycle_phi_agreed += 1
            # Track by reliability
            if has_reliable:
                if is_reliable:
                    non_cycle_r_total += 1
                    if phi_flipped:
                        non_cycle_r_phi_disagreed += 1
                    else:
                        non_cycle_r_phi_agreed += 1
                else:
                    non_cycle_u_total += 1
                    if phi_flipped:
                        non_cycle_u_phi_disagreed += 1
                    else:
                        non_cycle_u_phi_agreed += 1
        
        # Count combinations - only count R/U if we have path_seg_reliable data
        if is_changed:
            if has_reliable:
                if is_reliable:
                    changed_r += 1
                else:
                    changed_u += 1
            else:
                changed_no_conf += 1
        else:
            if has_reliable:
                if is_reliable:
                    unchanged_r += 1
                else:
                    unchanged_u += 1
        
        # Analyze phi flips
        if phi_flipped:
            if is_changed:
                # Phi flip was confirmed (kept in final)
                phi_flip_confirmed += 1
                if has_reliable:
                    if is_reliable:
                        phi_flip_confirmed_r += 1
                    else:
                        phi_flip_confirmed_u += 1
                else:
                    phi_flip_no_reliable += 1
            else:
                # Phi flip was reversed (not in final)
                phi_flip_reversed += 1
                if has_reliable:
                    if is_reliable:
                        phi_flip_reversed_r += 1
                    else:
                        phi_flip_reversed_u += 1
                else:
                    phi_flip_no_reliable += 1
    
    # Calculate percentages
    pct_r = (r_edges / total_edges * 100) if total_edges > 0 else 0.0
    # Percentage of changed edges that are U (only counting edges with path_seg_reliable)
    changed_with_reliable = changed_r + changed_u
    pct_changed_u = (changed_u / changed_with_reliable * 100) if changed_with_reliable > 0 else 0.0
    pct_total_changed = (changed_total / total_edges * 100) if total_edges > 0 else 0.0
    
    if edges_without_confidence > 0:
        print(f"  Warning: {edges_without_confidence:,} edges ({100*edges_without_confidence/total_edges:.2f}%) without path_seg_reliable data")
    if changed_no_conf > 0:
        print(f"  Note: {changed_no_conf:,} changed edges ({100*changed_no_conf/changed_total:.2f}% of changed) without path_seg_reliable data")
    
    # Calculate phi-related percentages
    pct_phi_flip_confirmed = (phi_flip_confirmed / phi_flipped_total * 100) if phi_flipped_total > 0 else 0.0
    pct_phi_flip_reversed = (phi_flip_reversed / phi_flipped_total * 100) if phi_flipped_total > 0 else 0.0
    phi_confirmed_with_r = phi_flip_confirmed_r + phi_flip_confirmed_u
    phi_reversed_with_r = phi_flip_reversed_r + phi_flip_reversed_u
    pct_phi_confirmed_r = (phi_flip_confirmed_r / phi_confirmed_with_r * 100) if phi_confirmed_with_r > 0 else 0.0
    pct_phi_reversed_r = (phi_flip_reversed_r / phi_reversed_with_r * 100) if phi_reversed_with_r > 0 else 0.0
    
    # Calculate SWORD-PHI agreement percentages
    pct_phi_agreed = (phi_agreed_with_sword / total_edges * 100) if total_edges > 0 else 0.0
    pct_phi_disagreed = (phi_disagreed_with_sword / total_edges * 100) if total_edges > 0 else 0.0
    phi_disagreed_with_r = phi_disagreed_r + phi_disagreed_u
    pct_phi_disagreed_r = (phi_disagreed_r / phi_disagreed_with_r * 100) if phi_disagreed_with_r > 0 else 0.0
    
    # Calculate cycle vs non-cycle phi agreement
    pct_cycle_phi_agreed = (cycle_phi_agreed / cycle_total * 100) if cycle_total > 0 else 0.0
    pct_non_cycle_phi_agreed = (non_cycle_phi_agreed / non_cycle_total * 100) if non_cycle_total > 0 else 0.0
    pct_cycle_r_phi_agreed = (cycle_r_phi_agreed / cycle_r_total * 100) if cycle_r_total > 0 else 0.0
    pct_cycle_u_phi_agreed = (cycle_u_phi_agreed / cycle_u_total * 100) if cycle_u_total > 0 else 0.0
    pct_non_cycle_r_phi_agreed = (non_cycle_r_phi_agreed / non_cycle_r_total * 100) if non_cycle_r_total > 0 else 0.0
    pct_non_cycle_u_phi_agreed = (non_cycle_u_phi_agreed / non_cycle_u_total * 100) if non_cycle_u_total > 0 else 0.0
    
    # Calculate baseline R agreement (excluding type = 6)
    pct_baseline_r_phi_agreed = (baseline_r_phi_agreed / baseline_r_total * 100) if baseline_r_total > 0 else 0.0
    
    stats = {
        'continent': continent_code,
        'total_edges': total_edges,
        'r_edges': r_edges,
        'u_edges': u_edges,
        'pct_r': pct_r,
        'changed_r': changed_r,
        'changed_u': changed_u,
        'pct_changed_u': pct_changed_u,
        'changed_total': changed_total,
        'changed_with_reliable': changed_with_reliable,
        'changed_no_conf': changed_no_conf,
        'pct_total_changed': pct_total_changed,
        'unchanged_r': unchanged_r,
        'unchanged_u': unchanged_u,
        'edges_without_confidence': edges_without_confidence,
        # Phi analysis
        'phi_flipped_total': phi_flipped_total,
        'phi_flip_confirmed': phi_flip_confirmed,
        'phi_flip_reversed': phi_flip_reversed,
        'phi_flip_confirmed_r': phi_flip_confirmed_r,
        'phi_flip_confirmed_u': phi_flip_confirmed_u,
        'phi_flip_reversed_r': phi_flip_reversed_r,
        'phi_flip_reversed_u': phi_flip_reversed_u,
        'phi_flip_no_reliable': phi_flip_no_reliable,
        'pct_phi_flip_confirmed': pct_phi_flip_confirmed,
        'pct_phi_flip_reversed': pct_phi_flip_reversed,
        'pct_phi_confirmed_r': pct_phi_confirmed_r,
        'pct_phi_reversed_r': pct_phi_reversed_r,
        # SWORD-PHI agreement
        'phi_agreed_with_sword': phi_agreed_with_sword,
        'phi_disagreed_with_sword': phi_disagreed_with_sword,
        'phi_disagreed_r': phi_disagreed_r,
        'phi_disagreed_u': phi_disagreed_u,
        'phi_disagreed_no_reliable': phi_disagreed_no_reliable,
        'pct_phi_agreed': pct_phi_agreed,
        'pct_phi_disagreed': pct_phi_disagreed,
        'pct_phi_disagreed_r': pct_phi_disagreed_r,
        # Cycle analysis
        'cycle_total': cycle_total,
        'cycle_phi_agreed': cycle_phi_agreed,
        'cycle_phi_disagreed': cycle_phi_disagreed,
        'pct_cycle_phi_agreed': pct_cycle_phi_agreed,
        'cycle_r_total': cycle_r_total,
        'cycle_r_phi_agreed': cycle_r_phi_agreed,
        'cycle_r_phi_disagreed': cycle_r_phi_disagreed,
        'pct_cycle_r_phi_agreed': pct_cycle_r_phi_agreed,
        'cycle_u_total': cycle_u_total,
        'cycle_u_phi_agreed': cycle_u_phi_agreed,
        'cycle_u_phi_disagreed': cycle_u_phi_disagreed,
        'pct_cycle_u_phi_agreed': pct_cycle_u_phi_agreed,
        'non_cycle_total': non_cycle_total,
        'non_cycle_phi_agreed': non_cycle_phi_agreed,
        'non_cycle_phi_disagreed': non_cycle_phi_disagreed,
        'pct_non_cycle_phi_agreed': pct_non_cycle_phi_agreed,
        'non_cycle_r_total': non_cycle_r_total,
        'non_cycle_r_phi_agreed': non_cycle_r_phi_agreed,
        'non_cycle_r_phi_disagreed': non_cycle_r_phi_disagreed,
        'pct_non_cycle_r_phi_agreed': pct_non_cycle_r_phi_agreed,
        'non_cycle_u_total': non_cycle_u_total,
        'non_cycle_u_phi_agreed': non_cycle_u_phi_agreed,
        'non_cycle_u_phi_disagreed': non_cycle_u_phi_disagreed,
        'pct_non_cycle_u_phi_agreed': pct_non_cycle_u_phi_agreed,
        # Cycle complexity analysis
        'complexity_groups': complexity_groups,
        # Baseline validation (excluding type = 6)
        'baseline_total': baseline_total,
        'baseline_r_total': baseline_r_total,
        'baseline_r_phi_agreed': baseline_r_phi_agreed,
        'baseline_r_phi_disagreed': baseline_r_phi_disagreed,
        'pct_baseline_r_phi_agreed': pct_baseline_r_phi_agreed,
        'r_negative_slope': r_negative_slope,
    }
    
    return stats


def print_stats(stats_dict):
    """Print formatted statistics for a continent."""
    s = stats_dict
    print(f"\n{'='*60}")
    print(f"CONTINENT: {s['continent'].upper()}")
    print(f"{'='*60}")
    print(f"Total edges:                    {s['total_edges']:>12,}")
    print(f"R edges (reliable):              {s['r_edges']:>12,} ({s['pct_r']:>6.2f}%)")
    print(f"U edges (unreliable):            {s['u_edges']:>12,} ({100-s['pct_r']:>6.2f}%)")
    print(f"")
    print(f"Changed edges (total):           {s['changed_total']:>12,} ({s['pct_total_changed']:>6.2f}%)")
    print(f"  Changed + R:                   {s['changed_r']:>12,}")
    print(f"  Changed + U:                   {s['changed_u']:>12,} ({s['pct_changed_u']:>6.2f}% of changed with path_seg_reliable)")
    if s.get('changed_no_conf', 0) > 0:
        print(f"  Changed + no path_seg_reliable: {s['changed_no_conf']:>12,}")
    print(f"")
    print(f"Unchanged edges:                 {s['total_edges'] - s['changed_total']:>12,}")
    print(f"  Unchanged + R:                 {s['unchanged_r']:>12,}")
    print(f"  Unchanged + U:                 {s['unchanged_u']:>12,}")
    print(f"")
    print(f"{'BASELINE VALIDATION (excluding type=6)':<60}")
    print(f"{'-'*60}")
    print(f"Total baseline edges (type != 6): {s['baseline_total']:>12,}")
    print(f"R edges (baseline):               {s['baseline_r_total']:>12,}")
    print(f"  R edges that agreed with SWORD: {s['baseline_r_phi_agreed']:>12,} ({s['pct_baseline_r_phi_agreed']:>6.2f}%)")
    print(f"  R edges that disagreed:        {s['baseline_r_phi_disagreed']:>12,}")
    print(f"R edges with negative slope:     {s['r_negative_slope']:>12,}")
    print(f"")
    print(f"{'SWORD-PHI AGREEMENT ANALYSIS':<60}")
    print(f"{'-'*60}")
    print(f"Phi AGREED with SWORD:           {s['phi_agreed_with_sword']:>12,} ({s['pct_phi_agreed']:>6.2f}%)")
    print(f"Phi DISAGREED with SWORD:        {s['phi_disagreed_with_sword']:>12,} ({s['pct_phi_disagreed']:>6.2f}%)")
    if s['phi_disagreed_with_sword'] > 0:
        print(f"  Disagreed + R:                 {s['phi_disagreed_r']:>12,} ({s['pct_phi_disagreed_r']:>6.2f}% of disagreed)")
        print(f"  Disagreed + U:                 {s['phi_disagreed_u']:>12,}")
        if s.get('phi_disagreed_no_reliable', 0) > 0:
            print(f"  Disagreed + no reliability:    {s['phi_disagreed_no_reliable']:>12,}")
    print(f"")
    print(f"{'CYCLE VS NON-CYCLE PHI ACCURACY':<60}")
    print(f"{'-'*60}")
    print(f"Cycle edges:                     {s['cycle_total']:>12,} ({100*s['cycle_total']/s['total_edges']:>6.2f}% of total)")
    print(f"  Phi agreement on cycles:       {s['cycle_phi_agreed']:>12,} ({s['pct_cycle_phi_agreed']:>6.2f}%)")
    print(f"  Phi disagreement on cycles:    {s['cycle_phi_disagreed']:>12,}")
    if s['cycle_r_total'] > 0:
        print(f"  Cycle R edges:                  {s['cycle_r_total']:>12,}")
        print(f"    Phi agreement on cycle R:     {s['cycle_r_phi_agreed']:>12,} ({s['pct_cycle_r_phi_agreed']:>6.2f}%)")
    if s['cycle_u_total'] > 0:
        print(f"  Cycle U edges:                  {s['cycle_u_total']:>12,}")
        print(f"    Phi agreement on cycle U:     {s['cycle_u_phi_agreed']:>12,} ({s['pct_cycle_u_phi_agreed']:>6.2f}%)")
    print(f"Non-cycle edges:                 {s['non_cycle_total']:>12,} ({100*s['non_cycle_total']/s['total_edges']:>6.2f}% of total)")
    print(f"  Phi agreement on non-cycles:   {s['non_cycle_phi_agreed']:>12,} ({s['pct_non_cycle_phi_agreed']:>6.2f}%)")
    print(f"  Phi disagreement on non-cycles:{s['non_cycle_phi_disagreed']:>12,}")
    if s['non_cycle_r_total'] > 0:
        print(f"  Non-cycle R edges:              {s['non_cycle_r_total']:>12,}")
        print(f"    Phi agreement on non-cycle R: {s['non_cycle_r_phi_agreed']:>12,} ({s['pct_non_cycle_r_phi_agreed']:>6.2f}%)")
    if s.get('non_cycle_u_total', 0) > 0:
        print(f"  Non-cycle U edges:              {s['non_cycle_u_total']:>12,}")
        print(f"    Phi agreement on non-cycle U: {s['non_cycle_u_phi_agreed']:>12,} ({s['pct_non_cycle_u_phi_agreed']:>6.2f}%)")
    print(f"")
    
    # Print cycle complexity breakdown
    if s.get('complexity_groups'):
        print(f"{'CYCLE COMPLEXITY ANALYSIS':<60}")
        print(f"{'-'*60}")
        print(f"{'Cycles':>8} {'Total':>12} {'Agreed':>12} {'%Agree':>10} {'R Total':>12} {'R Agreed':>12} {'%R Agree':>10}")
        print(f"{'-'*60}")
        for cycle_count in sorted(s['complexity_groups'].keys()):
            cg = s['complexity_groups'][cycle_count]
            pct_agree = (cg['agreed'] / cg['total'] * 100) if cg['total'] > 0 else 0.0
            pct_r_agree = (cg['r_agreed'] / cg['r_total'] * 100) if cg['r_total'] > 0 else 0.0
            print(f"{cycle_count:>8} {cg['total']:>12,} {cg['agreed']:>12,} {pct_agree:>9.2f}% "
                  f"{cg['r_total']:>12,} {cg['r_agreed']:>12,} {pct_r_agree:>9.2f}%")
        print(f"")
    print(f"{'PHI DIRECTION CHANGE ANALYSIS':<60}")
    print(f"{'-'*60}")
    print(f"Phi flipped edges (total):        {s['phi_flipped_total']:>12,}")
    print(f"  Phi flip CONFIRMED (kept):      {s['phi_flip_confirmed']:>12,} ({s['pct_phi_flip_confirmed']:>6.2f}%)")
    print(f"    Confirmed + R:               {s['phi_flip_confirmed_r']:>12,} ({s['pct_phi_confirmed_r']:>6.2f}% of confirmed)")
    print(f"    Confirmed + U:               {s['phi_flip_confirmed_u']:>12,}")
    print(f"  Phi flip REVERSED (flipped back): {s['phi_flip_reversed']:>12,} ({s['pct_phi_flip_reversed']:>6.2f}%)")
    print(f"    Reversed + R:                {s['phi_flip_reversed_r']:>12,} ({s['pct_phi_reversed_r']:>6.2f}% of reversed)")
    print(f"    Reversed + U:                {s['phi_flip_reversed_u']:>12,}")
    if s.get('phi_flip_no_reliable', 0) > 0:
        print(f"  Phi flip + no reliability:     {s['phi_flip_no_reliable']:>12,}")


def print_summary(all_stats):
    """Print summary table across all continents."""
    print(f"\n{'='*80}")
    print(f"SUMMARY ACROSS ALL CONTINENTS")
    print(f"{'='*80}")
    print(f"{'Continent':<12} {'Total':>12} {'R':>12} {'%R':>8} {'Changed':>12} {'Ch+R':>8} {'Ch+U':>8} {'%ChU':>8}")
    print(f"{'-'*80}")
    
    for stats in all_stats:
        s = stats
        print(f"{s['continent']:<12} {s['total_edges']:>12,} {s['r_edges']:>12,} "
              f"{s['pct_r']:>7.2f}% {s['changed_total']:>12,} {s['changed_r']:>8,} "
              f"{s['changed_u']:>8,} {s['pct_changed_u']:>7.2f}%")
    
    # Global totals
    total_all = sum(s['total_edges'] for s in all_stats)
    r_all = sum(s['r_edges'] for s in all_stats)
    changed_all = sum(s['changed_total'] for s in all_stats)
    changed_r_all = sum(s['changed_r'] for s in all_stats)
    changed_u_all = sum(s['changed_u'] for s in all_stats)
    changed_with_reliable_all = sum(s.get('changed_with_reliable', s['changed_r'] + s['changed_u']) for s in all_stats)
    
    pct_r_all = (r_all / total_all * 100) if total_all > 0 else 0.0
    pct_changed_u_all = (changed_u_all / changed_with_reliable_all * 100) if changed_with_reliable_all > 0 else 0.0
    
    print(f"{'-'*80}")
    print(f"{'GLOBAL TOTAL':<12} {total_all:>12,} {r_all:>12,} "
          f"{pct_r_all:>7.2f}% {changed_all:>12,} {changed_r_all:>8,} "
          f"{changed_u_all:>8,} {pct_changed_u_all:>7.2f}%")
    
    # Phi analysis summary
    print(f"\n{'='*80}")
    print(f"PHI DIRECTION CHANGE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Continent':<12} {'PhiFlipped':>12} {'Confirmed':>12} {'%Conf':>8} {'Rev+R':>8} {'Rev+U':>8} {'Conf+R':>8} {'Conf+U':>8}")
    print(f"{'-'*80}")
    
    for stats in all_stats:
        s = stats
        print(f"{s['continent']:<12} {s['phi_flipped_total']:>12,} {s['phi_flip_confirmed']:>12,} "
              f"{s['pct_phi_flip_confirmed']:>7.2f}% {s['phi_flip_reversed_r']:>8,} "
              f"{s['phi_flip_reversed_u']:>8,} {s['phi_flip_confirmed_r']:>8,} "
              f"{s['phi_flip_confirmed_u']:>8,}")
    
    # Global phi totals
    phi_flipped_all = sum(s['phi_flipped_total'] for s in all_stats)
    phi_confirmed_all = sum(s['phi_flip_confirmed'] for s in all_stats)
    phi_reversed_all = sum(s['phi_flip_reversed'] for s in all_stats)
    phi_confirmed_r_all = sum(s['phi_flip_confirmed_r'] for s in all_stats)
    phi_confirmed_u_all = sum(s['phi_flip_confirmed_u'] for s in all_stats)
    phi_reversed_r_all = sum(s['phi_flip_reversed_r'] for s in all_stats)
    phi_reversed_u_all = sum(s['phi_flip_reversed_u'] for s in all_stats)
    
    pct_phi_confirmed_all = (phi_confirmed_all / phi_flipped_all * 100) if phi_flipped_all > 0 else 0.0
    phi_confirmed_with_r = phi_confirmed_r_all + phi_confirmed_u_all
    phi_reversed_with_r = phi_reversed_r_all + phi_reversed_u_all
    pct_phi_confirmed_r_all = (phi_confirmed_r_all / phi_confirmed_with_r * 100) if phi_confirmed_with_r > 0 else 0.0
    pct_phi_reversed_r_all = (phi_reversed_r_all / phi_reversed_with_r * 100) if phi_reversed_with_r > 0 else 0.0
    
    print(f"{'-'*80}")
    print(f"{'GLOBAL TOTAL':<12} {phi_flipped_all:>12,} {phi_confirmed_all:>12,} "
          f"{pct_phi_confirmed_all:>7.2f}% {phi_reversed_r_all:>8,} "
          f"{phi_reversed_u_all:>8,} {phi_confirmed_r_all:>8,} "
          f"{phi_confirmed_u_all:>8,}")
    print(f"\n  Of {phi_confirmed_all:,} confirmed phi flips: {phi_confirmed_r_all:,} ({pct_phi_confirmed_r_all:.2f}%) are R")
    print(f"  Of {phi_reversed_all:,} reversed phi flips: {phi_reversed_r_all:,} ({pct_phi_reversed_r_all:.2f}%) are R")
    
    # SWORD-PHI agreement summary
    print(f"\n{'='*80}")
    print(f"SWORD-PHI AGREEMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Continent':<12} {'Total':>12} {'Agreed':>12} {'%Agreed':>10} {'Disagreed':>12} {'Dis+R':>8} {'Dis+U':>8} {'%DisR':>8}")
    print(f"{'-'*80}")
    
    for stats in all_stats:
        s = stats
        print(f"{s['continent']:<12} {s['total_edges']:>12,} {s['phi_agreed_with_sword']:>12,} "
              f"{s['pct_phi_agreed']:>9.2f}% {s['phi_disagreed_with_sword']:>12,} "
              f"{s['phi_disagreed_r']:>8,} {s['phi_disagreed_u']:>8,} "
              f"{s['pct_phi_disagreed_r']:>7.2f}%")
    
    # Global SWORD-PHI agreement totals
    phi_agreed_all = sum(s['phi_agreed_with_sword'] for s in all_stats)
    phi_disagreed_all = sum(s['phi_disagreed_with_sword'] for s in all_stats)
    phi_disagreed_r_all = sum(s['phi_disagreed_r'] for s in all_stats)
    phi_disagreed_u_all = sum(s['phi_disagreed_u'] for s in all_stats)
    
    pct_phi_agreed_all = (phi_agreed_all / total_all * 100) if total_all > 0 else 0.0
    pct_phi_disagreed_all = (phi_disagreed_all / total_all * 100) if total_all > 0 else 0.0
    phi_disagreed_with_r_all = phi_disagreed_r_all + phi_disagreed_u_all
    pct_phi_disagreed_r_all = (phi_disagreed_r_all / phi_disagreed_with_r_all * 100) if phi_disagreed_with_r_all > 0 else 0.0
    
    print(f"{'-'*80}")
    print(f"{'GLOBAL TOTAL':<12} {total_all:>12,} {phi_agreed_all:>12,} "
          f"{pct_phi_agreed_all:>9.2f}% {phi_disagreed_all:>12,} "
          f"{phi_disagreed_r_all:>8,} {phi_disagreed_u_all:>8,} "
          f"{pct_phi_disagreed_r_all:>7.2f}%")
    print(f"\n  Phi agreement with SWORD: {pct_phi_agreed_all:.2f}%")
    print(f"  Phi disagreement with SWORD: {pct_phi_disagreed_all:.2f}%")
    if phi_disagreed_with_r_all > 0:
        print(f"  Of {phi_disagreed_all:,} phi disagreements: {phi_disagreed_r_all:,} ({pct_phi_disagreed_r_all:.2f}%) are on R edges")
    
    # Cycle vs non-cycle summary
    print(f"\n{'='*80}")
    print(f"CYCLE VS NON-CYCLE PHI ACCURACY SUMMARY")
    print(f"{'='*80}")
    print(f"{'Continent':<12} {'Total':>12} {'Cycles':>12} {'%Cyc':>8} {'CycAgree':>12} {'%CycAg':>8} {'NonCycAgree':>12} {'%NonCycAg':>10}")
    print(f"{'-'*80}")
    
    for stats in all_stats:
        s = stats
        print(f"{s['continent']:<12} {s['total_edges']:>12,} {s['cycle_total']:>12,} "
              f"{100*s['cycle_total']/s['total_edges']:>7.2f}% {s['cycle_phi_agreed']:>12,} "
              f"{s['pct_cycle_phi_agreed']:>7.2f}% {s['non_cycle_phi_agreed']:>12,} "
              f"{s['pct_non_cycle_phi_agreed']:>9.2f}%")
    
    # Global cycle vs non-cycle totals
    cycle_total_all = sum(s['cycle_total'] for s in all_stats)
    cycle_phi_agreed_all = sum(s['cycle_phi_agreed'] for s in all_stats)
    cycle_phi_disagreed_all = sum(s['cycle_phi_disagreed'] for s in all_stats)
    non_cycle_total_all = sum(s['non_cycle_total'] for s in all_stats)
    non_cycle_phi_agreed_all = sum(s['non_cycle_phi_agreed'] for s in all_stats)
    non_cycle_phi_disagreed_all = sum(s['non_cycle_phi_disagreed'] for s in all_stats)
    
    pct_cycle_phi_agreed_all = (cycle_phi_agreed_all / cycle_total_all * 100) if cycle_total_all > 0 else 0.0
    pct_non_cycle_phi_agreed_all = (non_cycle_phi_agreed_all / non_cycle_total_all * 100) if non_cycle_total_all > 0 else 0.0
    
    print(f"{'-'*80}")
    print(f"{'GLOBAL TOTAL':<12} {total_all:>12,} {cycle_total_all:>12,} "
          f"{100*cycle_total_all/total_all:>7.2f}% {cycle_phi_agreed_all:>12,} "
          f"{pct_cycle_phi_agreed_all:>7.2f}% {non_cycle_phi_agreed_all:>12,} "
          f"{pct_non_cycle_phi_agreed_all:>9.2f}%")
    # Calculate R-specific totals
    cycle_r_total_all = sum(s['cycle_r_total'] for s in all_stats)
    cycle_r_phi_agreed_all = sum(s['cycle_r_phi_agreed'] for s in all_stats)
    non_cycle_r_total_all = sum(s['non_cycle_r_total'] for s in all_stats)
    non_cycle_r_phi_agreed_all = sum(s['non_cycle_r_phi_agreed'] for s in all_stats)
    
    pct_cycle_r_phi_agreed_all = (cycle_r_phi_agreed_all / cycle_r_total_all * 100) if cycle_r_total_all > 0 else 0.0
    pct_non_cycle_r_phi_agreed_all = (non_cycle_r_phi_agreed_all / non_cycle_r_total_all * 100) if non_cycle_r_total_all > 0 else 0.0
    
    print(f"\n  Overall phi agreement: {pct_phi_agreed_all:.2f}%")
    print(f"  Cycle edges phi agreement: {pct_cycle_phi_agreed_all:.2f}%")
    print(f"  Non-cycle edges phi agreement: {pct_non_cycle_phi_agreed_all:.2f}%")
    print(f"  Difference (non-cycle - cycle): {pct_non_cycle_phi_agreed_all - pct_cycle_phi_agreed_all:.2f} percentage points")
    print(f"\n  RELIABLE (R) EDGES:")
    print(f"    Cycle R edges phi agreement: {pct_cycle_r_phi_agreed_all:.2f}% ({cycle_r_phi_agreed_all:,}/{cycle_r_total_all:,})")
    print(f"    Non-cycle R edges phi agreement: {pct_non_cycle_r_phi_agreed_all:.2f}% ({non_cycle_r_phi_agreed_all:,}/{non_cycle_r_total_all:,})")
    print(f"    Difference (non-cycle R - cycle R): {pct_non_cycle_r_phi_agreed_all - pct_cycle_r_phi_agreed_all:.2f} percentage points")
    
    # Cycle complexity summary
    print(f"\n{'='*80}")
    print(f"CYCLE COMPLEXITY ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Cycles':>8} {'Total':>12} {'Agreed':>12} {'%Agree':>10} {'R Total':>12} {'R Agreed':>12} {'%R Agree':>10}")
    print(f"{'-'*80}")
    
    # Aggregate complexity groups across all continents
    global_complexity = {}
    for stats in all_stats:
        for cycle_count, cg in stats.get('complexity_groups', {}).items():
            if cycle_count not in global_complexity:
                global_complexity[cycle_count] = {
                    'total': 0,
                    'agreed': 0,
                    'disagreed': 0,
                    'r_total': 0,
                    'r_agreed': 0,
                    'r_disagreed': 0,
                }
            global_complexity[cycle_count]['total'] += cg['total']
            global_complexity[cycle_count]['agreed'] += cg['agreed']
            global_complexity[cycle_count]['disagreed'] += cg['disagreed']
            global_complexity[cycle_count]['r_total'] += cg['r_total']
            global_complexity[cycle_count]['r_agreed'] += cg['r_agreed']
            global_complexity[cycle_count]['r_disagreed'] += cg['r_disagreed']
    
    for cycle_count in sorted(global_complexity.keys()):
        cg = global_complexity[cycle_count]
        pct_agree = (cg['agreed'] / cg['total'] * 100) if cg['total'] > 0 else 0.0
        pct_r_agree = (cg['r_agreed'] / cg['r_total'] * 100) if cg['r_total'] > 0 else 0.0
        print(f"{cycle_count:>8} {cg['total']:>12,} {cg['agreed']:>12,} {pct_agree:>9.2f}% "
              f"{cg['r_total']:>12,} {cg['r_agreed']:>12,} {pct_r_agree:>9.2f}%")
    
    print(f"{'-'*80}")
    print(f"\n  Interpretation:")
    if len(global_complexity) > 1:
        cycle_counts = sorted(global_complexity.keys())
        # Find meaningful low and high complexity (with sufficient samples)
        low_complexity = None
        high_complexity = None
        for cc in cycle_counts:
            if global_complexity[cc]['total'] >= 100:  # At least 100 edges
                if low_complexity is None:
                    low_complexity = cc
                high_complexity = cc
        
        if low_complexity is not None and high_complexity is not None and low_complexity != high_complexity:
            low_cg = global_complexity[low_complexity]
            high_cg = global_complexity[high_complexity]
            low_pct = (low_cg['agreed'] / low_cg['total'] * 100) if low_cg['total'] > 0 else 0.0
            high_pct = (high_cg['agreed'] / high_cg['total'] * 100) if high_cg['total'] > 0 else 0.0
            low_r_pct = (low_cg['r_agreed'] / low_cg['r_total'] * 100) if low_cg['r_total'] > 0 else 0.0
            high_r_pct = (high_cg['r_agreed'] / high_cg['r_total'] * 100) if high_cg['r_total'] > 0 else 0.0
            print(f"    Low complexity ({low_complexity} cycles, n={low_cg['total']:,}): {low_pct:.2f}% agreement, {low_r_pct:.2f}% R agreement")
            print(f"    High complexity ({high_complexity} cycles, n={high_cg['total']:,}): {high_pct:.2f}% agreement, {high_r_pct:.2f}% R agreement")
            print(f"    Accuracy change: {low_pct - high_pct:.2f} percentage points")
            print(f"    R accuracy change: {low_r_pct - high_r_pct:.2f} percentage points")
        
        # Also show trend: compare 1 cycle vs 2+ cycles
        single_cycle = global_complexity.get(1, {'total': 0, 'agreed': 0, 'r_total': 0, 'r_agreed': 0})
        multi_cycle = {'total': 0, 'agreed': 0, 'r_total': 0, 'r_agreed': 0}
        for cc in cycle_counts:
            if cc > 1:
                cg = global_complexity[cc]
                multi_cycle['total'] += cg['total']
                multi_cycle['agreed'] += cg['agreed']
                multi_cycle['r_total'] += cg['r_total']
                multi_cycle['r_agreed'] += cg['r_agreed']
        
        if single_cycle['total'] > 0 and multi_cycle['total'] > 0:
            single_pct = (single_cycle['agreed'] / single_cycle['total'] * 100)
            multi_pct = (multi_cycle['agreed'] / multi_cycle['total'] * 100)
            single_r_pct = (single_cycle['r_agreed'] / single_cycle['r_total'] * 100) if single_cycle['r_total'] > 0 else 0.0
            multi_r_pct = (multi_cycle['r_agreed'] / multi_cycle['r_total'] * 100) if multi_cycle['r_total'] > 0 else 0.0
            print(f"\n  Summary comparison:")
            print(f"    Single cycle (1): {single_pct:.2f}% agreement, {single_r_pct:.2f}% R agreement (n={single_cycle['total']:,})")
            print(f"    Multi-cycle (2+): {multi_pct:.2f}% agreement, {multi_r_pct:.2f}% R agreement (n={multi_cycle['total']:,})")
            print(f"    Difference: {single_pct - multi_pct:.2f} percentage points (R: {single_r_pct - multi_r_pct:.2f} pp)")
    
    # Baseline validation summary
    baseline_total_all = sum(s['baseline_total'] for s in all_stats)
    baseline_r_total_all = sum(s['baseline_r_total'] for s in all_stats)
    baseline_r_phi_agreed_all = sum(s['baseline_r_phi_agreed'] for s in all_stats)
    baseline_r_phi_disagreed_all = sum(s['baseline_r_phi_disagreed'] for s in all_stats)
    pct_baseline_r_phi_agreed_all = (baseline_r_phi_agreed_all / baseline_r_total_all * 100) if baseline_r_total_all > 0 else 0.0
    
    print(f"\n{'='*80}")
    print(f"BASELINE VALIDATION SUMMARY (excluding type=6)")
    print(f"{'='*80}")
    print(f"{'Continent':<12} {'Total':>12} {'R Total':>12} {'R Agreed':>12} {'R Disagreed':>12} {'%Agreed':>10}")
    print(f"{'-'*80}")
    
    for stats in all_stats:
        s = stats
        print(f"{s['continent']:<12} {s['baseline_total']:>12,} {s['baseline_r_total']:>12,} {s['baseline_r_phi_agreed']:>12,} "
              f"{s['baseline_r_phi_disagreed']:>12,} {s['pct_baseline_r_phi_agreed']:>9.2f}%")
    
    print(f"{'-'*80}")
    print(f"{'GLOBAL TOTAL':<12} {baseline_total_all:>12,} {baseline_r_total_all:>12,} {baseline_r_phi_agreed_all:>12,} "
          f"{baseline_r_phi_disagreed_all:>12,} {pct_baseline_r_phi_agreed_all:>9.2f}%")
    # Count R edges with negative slope
    r_negative_slope_all = sum(s['r_negative_slope'] for s in all_stats)
    
    print(f"\n  BASELINE VALIDATION RESULTS:")
    print(f"    Total baseline edges (excluding type=6): {baseline_total_all:,}")
    print(f"    Total R edges (baseline): {baseline_r_total_all:,}")
    print(f"    R edges that agreed with SWORD: {baseline_r_phi_agreed_all:,}")
    print(f"    Percentage agreement: {pct_baseline_r_phi_agreed_all:.2f}%")
    print(f"\n  R EDGES WITH NEGATIVE SLOPE:")
    print(f"    Total R edges with negative path_seg_slope: {r_negative_slope_all:,}")
    print(f"    Percentage of R edges: {100*r_negative_slope_all/r_all:.2f}%" if r_all > 0 else "    Percentage of R edges: N/A")
    
    # Interpretation
    if pct_cycle_phi_agreed_all < 70 and pct_non_cycle_phi_agreed_all > 95:
        print(f"\n  ⚠️  SCENARIO A: Topology constrains tree parts, but φ is weak for complex braids.")
    elif pct_cycle_phi_agreed_all > 90:
        print(f"\n  ✅ SCENARIO B: φ works well even in complex multi-channel systems!")


def main():
    parser = argparse.ArgumentParser(
        description="Validate SWORD graph edge attributes across continents"
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='output',
        help='Output directory containing continent subdirectories (default: output)'
    )
    parser.add_argument(
        '--continent',
        type=str,
        choices=['na', 'sa', 'eu', 'af', 'oc', 'as', 'all'],
        default='all',
        help='Continent to validate (default: all)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.outdir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    continents = ['na', 'sa', 'eu', 'af', 'oc', 'as'] if args.continent == 'all' else [args.continent]
    
    all_stats = []
    
    for continent in continents:
        try:
            stats = validate_continent(continent, output_dir)
            print_stats(stats)
            all_stats.append(stats)
        except Exception as e:
            print(f"\nERROR processing {continent}: {e}")
            continue
    
    if len(all_stats) > 1:
        print_summary(all_stats)
    
    print(f"\n{'='*80}")
    print("Validation complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

