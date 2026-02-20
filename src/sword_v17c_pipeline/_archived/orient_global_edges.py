#!/usr/bin/env python3
"""
Orient LineString geometries in global_edges.gpkg to flow from upstream to downstream.

For each edge, checks if the LineString starts at the upstream node (up_network_node)
and ends at the downstream node (dn_network_node). If not, reverses the LineString.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime as dt
from shapely.geometry import LineString

def orient_edges_upstream_to_downstream(edges_file: str, nodes_file: str, output_file: str = None, tolerance: float = 1e-6):
    """
    Orient all edge LineStrings to flow from upstream to downstream.
    
    Args:
        edges_file: Path to edges GeoPackage file
        nodes_file: Path to nodes GeoPackage file
        output_file: Path to output file (if None, overwrites edges_file)
        tolerance: Coordinate matching tolerance in degrees
    """
    if output_file is None:
        output_file = edges_file
    
    print(f"[{dt.now().strftime('%H:%M:%S')}] Loading edges from {edges_file}...", flush=True)
    edges_gdf = gpd.read_file(edges_file)
    print(f"[{dt.now().strftime('%H:%M:%S')}] Loaded {len(edges_gdf):,} edges", flush=True)
    
    print(f"[{dt.now().strftime('%H:%M:%S')}] Loading nodes from {nodes_file}...", flush=True)
    nodes_gdf = gpd.read_file(nodes_file)
    print(f"[{dt.now().strftime('%H:%M:%S')}] Loaded {len(nodes_gdf):,} nodes", flush=True)
    
    # Create mapping from node_id to (x, y) coordinates
    print(f"[{dt.now().strftime('%H:%M:%S')}] Creating node coordinate mapping...", flush=True)
    node_coords = {}
    for idx, row in nodes_gdf.iterrows():
        node_id = row['node_id']
        if 'x' in row and 'y' in row and pd.notna(row['x']) and pd.notna(row['y']):
            node_coords[node_id] = (row['x'], row['y'])
        elif row.geometry is not None:
            # Fallback to geometry if x/y columns not available
            node_coords[node_id] = (row.geometry.x, row.geometry.y)
    
    print(f"[{dt.now().strftime('%H:%M:%S')}] Mapped {len(node_coords):,} node coordinates", flush=True)
    
    # Track statistics
    reversed_count = 0
    missing_upstream = 0
    missing_downstream = 0
    no_match_count = 0
    
    print(f"[{dt.now().strftime('%H:%M:%S')}] Orienting edge geometries...", flush=True)
    
    # Process each edge
    for idx, row in edges_gdf.iterrows():
        up_node_id = row['up_network_node']
        dn_node_id = row['dn_network_node']
        geom = row.geometry
        
        if geom is None or not isinstance(geom, LineString):
            continue
        
        # Get node coordinates
        up_coords = node_coords.get(up_node_id)
        dn_coords = node_coords.get(dn_node_id)
        
        if up_coords is None:
            missing_upstream += 1
            continue
        if dn_coords is None:
            missing_downstream += 1
            continue
        
        # Get LineString endpoints
        start_coords = geom.coords[0]
        end_coords = geom.coords[-1]
        
        # Check if LineString starts at upstream and ends at downstream
        start_at_up = (abs(start_coords[0] - up_coords[0]) < tolerance and 
                      abs(start_coords[1] - up_coords[1]) < tolerance)
        end_at_dn = (abs(end_coords[0] - dn_coords[0]) < tolerance and 
                    abs(end_coords[1] - dn_coords[1]) < tolerance)
        
        # If correctly oriented, no change needed
        if start_at_up and end_at_dn:
            continue
        
        # Check if reversed (starts at downstream, ends at upstream)
        start_at_dn = (abs(start_coords[0] - dn_coords[0]) < tolerance and 
                      abs(start_coords[1] - dn_coords[1]) < tolerance)
        end_at_up = (abs(end_coords[0] - up_coords[0]) < tolerance and 
                    abs(end_coords[1] - up_coords[1]) < tolerance)
        
        if start_at_dn and end_at_up:
            # Reverse the LineString
            edges_gdf.at[idx, 'geometry'] = LineString(list(geom.coords)[::-1])
            reversed_count += 1
        else:
            # Neither endpoint matches exactly - might be a multi-segment edge
            # Check which endpoint is closer to upstream
            dist_start_to_up = np.sqrt((start_coords[0] - up_coords[0])**2 + 
                                      (start_coords[1] - up_coords[1])**2)
            dist_end_to_up = np.sqrt((end_coords[0] - up_coords[0])**2 + 
                                     (end_coords[1] - up_coords[1])**2)
            
            # If end is closer to upstream, reverse
            if dist_end_to_up < dist_start_to_up:
                edges_gdf.at[idx, 'geometry'] = LineString(list(geom.coords)[::-1])
                reversed_count += 1
            else:
                no_match_count += 1
        
        # Progress indicator
        if (idx + 1) % 10000 == 0:
            print(f"[{dt.now().strftime('%H:%M:%S')}]  Processed {idx + 1:,} edges...", flush=True)
    
    print(f"[{dt.now().strftime('%H:%M:%S')}] Orientation complete!", flush=True)
    print(f"[{dt.now().strftime('%H:%M:%S')}] Statistics:", flush=True)
    print(f"  Reversed: {reversed_count:,}", flush=True)
    print(f"  Missing upstream node: {missing_upstream:,}", flush=True)
    print(f"  Missing downstream node: {missing_downstream:,}", flush=True)
    print(f"  No clear match: {no_match_count:,}", flush=True)
    
    # Write output
    print(f"[{dt.now().strftime('%H:%M:%S')}] Writing oriented edges to {output_file}...", flush=True)
    edges_gdf.to_file(output_file, driver='GPKG')
    print(f"[{dt.now().strftime('%H:%M:%S')}] Successfully wrote {len(edges_gdf):,} edges", flush=True)

if __name__ == "__main__":
    edges_file = "output/global_edges.gpkg"
    nodes_file = "output/global_nodes.gpkg"
    
    orient_edges_upstream_to_downstream(edges_file, nodes_file)

