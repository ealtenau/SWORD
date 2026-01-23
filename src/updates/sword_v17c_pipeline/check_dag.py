#!/usr/bin/env python3
"""
Check if a NetworkX graph stored in a pickle file is a DAG (Directed Acyclic Graph).
"""

import pickle
import argparse
import networkx as nx
from pathlib import Path


def check_dag(pickle_path):
    """Load graph from pickle and check if it's a DAG."""
    pickle_path = Path(pickle_path)
    
    if not pickle_path.exists():
        raise FileNotFoundError(f"File not found: {pickle_path}")
    
    print(f"Loading graph from: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        G = pickle.load(f)
    
    print(f"\nGraph type: {type(G).__name__}")
    print(f"Number of nodes: {G.number_of_nodes():,}")
    print(f"Number of edges: {G.number_of_edges():,}")
    print(f"Is directed: {G.is_directed()}")
    print(f"Is multigraph: {G.is_multigraph()}")
    
    # Check for NaN path_seg values
    import math
    nan_path_seg_count = 0
    nan_path_seg_edges = []
    total_edges_checked = 0
    
    for u, v, k, data in G.edges(keys=True, data=True):
        total_edges_checked += 1
        path_seg = data.get('path_seg', None)
        if path_seg is not None and isinstance(path_seg, float) and math.isnan(path_seg):
            nan_path_seg_count += 1
            if len(nan_path_seg_edges) < 10:  # Store first 10 examples
                nan_path_seg_edges.append((u, v, k, data.get('reach_id', 'N/A')))
    
    print(f"\nPath_seg NaN check:")
    print(f"  Total edges checked: {total_edges_checked:,}")
    print(f"  Edges with NaN path_seg: {nan_path_seg_count:,} ({100*nan_path_seg_count/total_edges_checked:.2f}%)")
    if nan_path_seg_edges:
        print(f"  Sample edges with NaN path_seg:")
        for u, v, k, rid in nan_path_seg_edges[:10]:
            print(f"    {u} -> {v} (key={k}): reach_id={rid}")
    
    # Convert to simple DiGraph if it's a MultiDiGraph for cycle detection
    if G.is_multigraph():
        print("\nConverting MultiDiGraph to simple DiGraph for cycle detection...")
        G_simple = nx.DiGraph()
        for u, v, k, data in G.edges(keys=True, data=True):
            if G_simple.has_edge(u, v):
                # Keep first edge if parallel edges exist
                continue
            # Copy all attributes, handling NaN values
            import math
            edge_attrs = {}
            for key, value in data.items():
                # Skip NaN values
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    edge_attrs[key] = value
            G_simple.add_edge(u, v, **edge_attrs)
        print(f"Simple graph edges: {G_simple.number_of_edges():,}")
        
        # Show sample edge attributes to debug
        sample_edge = next(iter(G_simple.edges(data=True)))
        print(f"Sample edge attributes: {list(sample_edge[2].keys())}")
    else:
        G_simple = G
    
    # Check if it's a DAG
    is_dag = nx.is_directed_acyclic_graph(G_simple)
    
    print(f"\n{'='*60}")
    print(f"Is DAG: {is_dag}")
    print(f"{'='*60}")
    
    if not is_dag:
        print("\n⚠️  Graph contains cycles!")
        
        # Find cycles
        try:
            cycles = list(nx.simple_cycles(G_simple))
            print(f"\nFound {len(cycles):,} simple cycle(s):")
            
            # Show cycles with reach_ids
            for i, cycle in enumerate(cycles[:10], 1):
                print(f"\n  Cycle {i} ({len(cycle)} nodes):")
                print(f"    Nodes: {cycle}")
                
                # Get reach_ids for edges in this cycle
                reach_ids = []
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]  # Wrap around to close the cycle
                    
                    # Try to get reach_id from the edge
                    if G_simple.has_edge(u, v):
                        edge_data = G_simple[u][v]
                        reach_id = edge_data.get('reach_id', 'N/A')
                        reach_ids.append(reach_id)
                    else:
                        reach_ids.append('N/A')
                
                print(f"    Reach IDs: {reach_ids}")
                
                # Also show edge details if available
                if any(rid != 'N/A' for rid in reach_ids):
                    print(f"    Edge details:")
                    for j in range(len(cycle)):
                        u = cycle[j]
                        v = cycle[(j + 1) % len(cycle)]
                        if G_simple.has_edge(u, v):
                            edge_data = G_simple[u][v]
                            reach_id = edge_data.get('reach_id', 'N/A')
                            
                            # Handle NaN values properly - check both section_id and path_seg
                            import math
                            section_id = edge_data.get('section_id', None)
                            path_seg = edge_data.get('path_seg', None)
                            
                            # Format section_id
                            if section_id is None:
                                section_id_str = 'N/A'
                            elif isinstance(section_id, float) and math.isnan(section_id):
                                section_id_str = 'NaN'
                            else:
                                section_id_str = str(section_id)
                            
                            # Format path_seg (often used instead of section_id)
                            if path_seg is None:
                                path_seg_str = 'N/A'
                            elif isinstance(path_seg, float) and math.isnan(path_seg):
                                path_seg_str = 'NaN'
                            else:
                                path_seg_str = str(path_seg)
                            
                            # Check original MultiDiGraph for all parallel edges if it exists
                            if G.is_multigraph():
                                parallel_edges = list(G[u][v].items())  # Get all edge keys and data
                                if len(parallel_edges) > 1:
                                    print(f"      {u} -> {v}: reach_id={reach_id}, section_id={section_id_str}, path_seg={path_seg_str} (⚠️ {len(parallel_edges)} parallel edges)")
                                    # Show all parallel edges
                                    for kk, dd in parallel_edges:
                                        par_reach = dd.get('reach_id', 'N/A')
                                        par_section = dd.get('section_id', None)
                                        par_path_seg = dd.get('path_seg', None)
                                        
                                        # Format section_id
                                        if par_section is None:
                                            par_section_str = 'N/A'
                                        elif isinstance(par_section, float) and math.isnan(par_section):
                                            par_section_str = 'NaN'
                                        else:
                                            par_section_str = str(par_section)
                                        
                                        # Format path_seg
                                        if par_path_seg is None:
                                            par_path_seg_str = 'N/A'
                                        elif isinstance(par_path_seg, float) and math.isnan(par_path_seg):
                                            par_path_seg_str = 'NaN'
                                        else:
                                            par_path_seg_str = str(par_path_seg)
                                        
                                        print(f"        Key {kk}: reach_id={par_reach}, section_id={par_section_str}, path_seg={par_path_seg_str}")
                                else:
                                    print(f"      {u} -> {v}: reach_id={reach_id}, section_id={section_id_str}, path_seg={path_seg_str}")
                            else:
                                print(f"      {u} -> {v}: reach_id={reach_id}, section_id={section_id_str}, path_seg={path_seg_str}")
            
            if len(cycles) > 10:
                print(f"\n  ... and {len(cycles) - 10} more cycles")
            
            # Find strongly connected components (SCCs) - cycles are in SCCs with >1 node
            sccs = list(nx.strongly_connected_components(G_simple))
            sccs_with_cycles = [scc for scc in sccs if len(scc) > 1]
            print(f"\nStrongly connected components with cycles: {len(sccs_with_cycles):,}")
            if sccs_with_cycles:
                print(f"Largest SCC size: {max(len(scc) for scc in sccs_with_cycles)}")
                print(f"Sample SCCs:")
                for i, scc in enumerate(sccs_with_cycles[:5], 1):
                    print(f"  SCC {i}: {len(scc)} nodes - {list(scc)[:10]}...")
        
        except Exception as e:
            print(f"Error finding cycles: {e}")
    
    return is_dag


def main():
    parser = argparse.ArgumentParser(description="Check if a NetworkX graph is a DAG")
    parser.add_argument("pickle_path", type=str, help="Path to pickle file containing the graph")
    args = parser.parse_args()
    
    try:
        is_dag = check_dag(args.pickle_path)
        exit(0 if is_dag else 1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()

