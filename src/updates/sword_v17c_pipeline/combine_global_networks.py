#!/usr/bin/env python3
"""
Combine all continent-specific network edges and nodes into global files.

Reads all {continent}/{continent}_network_edges.gpkg and 
{continent}/{continent}_network_nodes.gpkg files from the output directory
and combines them into global_edges.gpkg and global_nodes.gpkg.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from datetime import datetime as dt

def combine_global_networks(output_dir: str = "output"):
    """
    Combine all continent network files into global files.
    
    Args:
        output_dir: Directory containing continent subdirectories
    """
    output_path = Path(output_dir)
    
    # Define continent codes
    continents = ['af', 'as', 'eu', 'na', 'oc', 'sa']
    
    # Lists to store GeoDataFrames
    edges_list = []
    nodes_list = []
    
    print(f"[{dt.now().strftime('%H:%M:%S')}] Starting global network combination...", flush=True)
    
    # Read and combine edges
    print(f"[{dt.now().strftime('%H:%M:%S')}] Reading edges files...", flush=True)
    for continent in continents:
        edges_file = output_path / continent / f"{continent}_network_edges.gpkg"
        if edges_file.exists():
            print(f"[{dt.now().strftime('%H:%M:%S')}]  Reading {edges_file}...", flush=True)
            gdf_edges = gpd.read_file(edges_file)
            # Add continent identifier
            gdf_edges['continent'] = continent
            edges_list.append(gdf_edges)
            print(f"[{dt.now().strftime('%H:%M:%S')}]    Loaded {len(gdf_edges):,} edges", flush=True)
        else:
            print(f"[{dt.now().strftime('%H:%M:%S')}]  WARNING: {edges_file} not found, skipping", flush=True)
    
    # Read and combine nodes
    print(f"[{dt.now().strftime('%H:%M:%S')}] Reading nodes files...", flush=True)
    for continent in continents:
        nodes_file = output_path / continent / f"{continent}_network_nodes.gpkg"
        if nodes_file.exists():
            print(f"[{dt.now().strftime('%H:%M:%S')}]  Reading {nodes_file}...", flush=True)
            gdf_nodes = gpd.read_file(nodes_file)
            # Add continent identifier
            gdf_nodes['continent'] = continent
            nodes_list.append(gdf_nodes)
            print(f"[{dt.now().strftime('%H:%M:%S')}]    Loaded {len(gdf_nodes):,} nodes", flush=True)
        else:
            print(f"[{dt.now().strftime('%H:%M:%S')}]  WARNING: {nodes_file} not found, skipping", flush=True)
    
    # Combine edges
    if edges_list:
        print(f"[{dt.now().strftime('%H:%M:%S')}] Combining {len(edges_list)} edges GeoDataFrames...", flush=True)
        global_edges = pd.concat(edges_list, ignore_index=True)
        print(f"[{dt.now().strftime('%H:%M:%S')}] Combined edges shape: {global_edges.shape}", flush=True)
        
        # Ensure CRS is set
        if global_edges.crs is None:
            global_edges.set_crs(epsg=4326, inplace=True)
        
        # Write global edges
        global_edges_file = output_path / "global_edges.gpkg"
        print(f"[{dt.now().strftime('%H:%M:%S')}] Writing {global_edges_file}...", flush=True)
        global_edges.to_file(global_edges_file, driver='GPKG')
        print(f"[{dt.now().strftime('%H:%M:%S')}] Successfully wrote {len(global_edges):,} edges to {global_edges_file}", flush=True)
    else:
        print(f"[{dt.now().strftime('%H:%M:%S')}] ERROR: No edges files found!", flush=True)
    
    # Combine nodes
    if nodes_list:
        print(f"[{dt.now().strftime('%H:%M:%S')}] Combining {len(nodes_list)} nodes GeoDataFrames...", flush=True)
        global_nodes = pd.concat(nodes_list, ignore_index=True)
        print(f"[{dt.now().strftime('%H:%M:%S')}] Combined nodes shape: {global_nodes.shape}", flush=True)
        
        # Ensure CRS is set
        if global_nodes.crs is None:
            global_nodes.set_crs(epsg=4326, inplace=True)
        
        # Write global nodes
        global_nodes_file = output_path / "global_nodes.gpkg"
        print(f"[{dt.now().strftime('%H:%M:%S')}] Writing {global_nodes_file}...", flush=True)
        global_nodes.to_file(global_nodes_file, driver='GPKG')
        print(f"[{dt.now().strftime('%H:%M:%S')}] Successfully wrote {len(global_nodes):,} nodes to {global_nodes_file}", flush=True)
    else:
        print(f"[{dt.now().strftime('%H:%M:%S')}] ERROR: No nodes files found!", flush=True)
    
    print(f"[{dt.now().strftime('%H:%M:%S')}] Global network combination complete!", flush=True)

if __name__ == "__main__":
    combine_global_networks()


