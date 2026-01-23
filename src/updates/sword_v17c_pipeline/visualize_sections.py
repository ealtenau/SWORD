"""
Visualize SWOT sections with water surface elevation data and computed slopes.

This script loads SWOT node data from parquet files and visualizes:
- All WSE observations colored by cyclePass
- Regression line computed using compute_clean_slopes
- Section metadata and slope statistics

Usage:
    # List available sections
    python visualize_sections.py --dir /path/to/project --continent af --list
    
    # Visualize a specific section
    python visualize_sections.py --dir /path/to/project --continent af --section 6043
    
    # Visualize a specific section with a specific junction
    python visualize_sections.py --dir /path/to/project --continent af --section 6043 --junction 12345
"""

import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
import argparse
import os
import glob
from pathlib import Path

# Import compute_clean_slopes from SWOT_slopes
from SWOT_slopes import compute_clean_slopes


def load_section_data(directory, continent, section_id, junction_id=None):
    """
    Load SWOT node data for a specific section from parquet file.
    
    Parameters:
    -----------
    directory : str
        Base directory containing output folders
    continent : str
        Continent code (e.g., 'af', 'na', 'as')
    section_id : int
        Section ID to visualize
    junction_id : int, optional
        If provided, filter to this junction_id only
        
    Returns:
    --------
    pd.DataFrame
        Filtered SWOT node data for the section
    """
    parquet_path = os.path.join(directory, 'output', continent, f'{continent}_swot_nodes.parquet')
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    # Load data using DuckDB for efficiency
    con = duckdb.connect(":memory:")
    
    query = f"""
        SELECT *
        FROM read_parquet('{parquet_path}')
        WHERE section_id = {section_id}
    """
    
    if junction_id is not None:
        query += f" AND junction_id = {junction_id}"
    
    df = con.execute(query).df()
    con.close()
    
    if df.empty:
        raise ValueError(f"No data found for section_id={section_id}" + 
                        (f", junction_id={junction_id}" if junction_id else ""))
    
    return df


def plotly_scatter(D, xCol, yCol, juns, dfSlope, upstream_junction_id=None, downstream_junction_id=None):
    """
    Create interactive Plotly scatter plot with regression line.
    
    Parameters:
    -----------
    D : pd.DataFrame
        Data to plot (all observations)
    xCol : str
        Column name for x-axis (typically 'distance')
    yCol : str
        Column name for y-axis (typically 'wse')
    juns : list or array
        Junction IDs for title
    dfSlope : pd.DataFrame
        Slope results from compute_clean_slopes (should have one row)
    """
    # Assign discrete colors using Plotly's built-in qualitative palettes
    palette = px.colors.qualitative.Set2
    
    groups = sorted(D['cyclePass'].unique())
    color_map = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
    
    # Create figure
    fig = go.Figure()
    
    # Add each group as a separate trace
    for g in groups:
        sub = D[D['cyclePass'] == g]
        
        fig.add_trace(go.Scatter(
            x=sub[xCol],
            y=sub[yCol],
            mode='markers',
            name=f"Pass {g}",
            marker=dict(
                color=color_map[g],
                size=6,
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            legendgroup=int(g),
            customdata=sub[['pass_size', 'wse_u', 'node_id']].values,
            hovertemplate=(
                "<b>Pass:</b> %{text}<br>"
                "<b>Distance:</b> %{x:.2f}<br>"
                "<b>WSE:</b> %{y:.3f}<br>"
                "<b>Pass size:</b> %{customdata[0]}<br>"
                "<b>WSE unc:</b> %{customdata[1]:.3f}<br>"
                "<b>Node ID:</b> %{customdata[2]}<extra></extra>"
            ),
            text=[g] * len(sub)
        ))
    
    # Add regression line if slope data is available
    if not dfSlope.empty and dfSlope['slope'].notna().any():
        row = dfSlope.iloc[0]
        slope = row['slope']
        intercept = row['intercept']
        
        if pd.notna(slope) and pd.notna(intercept):
            X_unique = np.sort(D[xCol].unique())
            pred = (X_unique * slope) + intercept
            
            customdata = np.array([[slope, row['SE'], row['slopeF'], row.get('slopeP', 0)]] * len(X_unique))
            
            fig.add_trace(go.Scatter(
                x=X_unique,
                y=pred,
                mode='lines',
                name="Regression Line",
                line=dict(color='rgb(255,0,0)', width=4, dash='dash'),
                legendgroup=9999,
                customdata=customdata,
                hovertemplate=(
                    "<b>Regression Line</b><br>"
                    "<b>Slope:</b> %{customdata[0]:.2e}<br>"
                    "<b>Slope Fraction:</b> %{customdata[2]:.3f}<br>"
                    "<b>Slope SE:</b> %{customdata[1]:.2e}<br>"
                    "<b>Slope P:</b> %{customdata[3]:.2e}<extra></extra>"
                ),
                text='Regression'
            ))
    
    # Build title with section and junction info
    title_parts = [f"Section: {D['section_id'].iloc[0]}"]
    
    # Determine slope sign to position junction IDs correctly
    slope_positive = False
    if not dfSlope.empty and dfSlope['slope'].notna().any():
        row = dfSlope.iloc[0]
        slope_positive = row['slope'] > 0
        title_parts.append(f"Slope: {row['slope']:.2e} ± {row['SE']:.2e}")
    
    # Add junction IDs to title positioned based on slope direction
    # Positive slope: upstream junction ID on right side of plot
    # Negative slope: upstream junction ID on left side of plot
    if upstream_junction_id is not None and downstream_junction_id is not None:
        if slope_positive:
            # Positive slope: upstream on right, downstream on left
            title_parts.append(f"L↓: {downstream_junction_id} → R↑: {upstream_junction_id}")
        else:
            # Negative slope: upstream on left, downstream on right
            title_parts.append(f"L↑: {upstream_junction_id} → R↓: {downstream_junction_id}")
    elif upstream_junction_id is not None:
        title_parts.append(f"Upstream: {upstream_junction_id}")
    elif downstream_junction_id is not None:
        title_parts.append(f"Downstream: {downstream_junction_id}")
    
    # Enable single-group selection ("solo mode") in the legend
    fig.update_layout(
        title=dict(
            text=" | ".join(title_parts),
            x=0.5,  # Center the title
            xanchor='center',
            font=dict(size=12)
        ),
        template="plotly_white",
        legend=dict(
            title="Passes",
            itemsizing='constant',
            itemclick='toggleothers',  # Only one group visible at a time
            itemdoubleclick='toggle',  # Double-click toggles all
        ),
        xaxis_title=xCol,
        yaxis_title=yCol,
        hovermode='closest',
        width=1200,
        height=700,
        margin=dict(t=80),  # Increase top margin for longer titles
    )
    
    fig.show()


def plot(directory, continent, section_id, junction_id=None):
    """
    Main plotting function for a section.
    
    Parameters:
    -----------
    directory : str
        Base directory containing output folders
    continent : str
        Continent code
    section_id : int
        Section ID to visualize
    junction_id : int, optional
        If provided, filter to this junction_id only
    """
    # Load section data
    print(f"Loading data for section_id={section_id}...")
    dfSection = load_section_data(directory, continent, section_id, junction_id)
    
    print(f"Loaded {len(dfSection):,} observations")
    print(f"Unique cyclePass values: {sorted(dfSection['cyclePass'].unique())}")
    print(f"Unique junction_id values: {sorted(dfSection['junction_id'].unique())}")
    
    # If junction_id not specified but multiple exist, use first one
    if junction_id is None:
        juns = sorted(dfSection['junction_id'].unique())
        if len(juns) > 1:
            print(f"Multiple junctions found: {juns}. Using first: {juns[0]}")
            dfSection = dfSection[dfSection['junction_id'] == juns[0]]
            juns = [juns[0]]
        else:
            juns = juns.tolist()
    else:
        juns = [junction_id]
    
    # Remove NaN WSE values
    dfSection = dfSection[dfSection['wse'].notna()].copy()
    
    if dfSection.empty:
        raise ValueError("No valid WSE data after filtering NaN values")
    
    print(f"Valid observations after filtering: {len(dfSection):,}")
    
    # Load network edges to get path_start_node (upstream) and path_end_node (downstream)
    network_edges_path = os.path.join(directory, 'output', continent, f'{continent}_network_edges.gpkg')
    upstream_junction_id = None
    downstream_junction_id = None
    
    if os.path.exists(network_edges_path):
        try:
            dfNetworkEdge = gpd.read_file(network_edges_path)
            # Get path_start_node and path_end_node for this section_id (path_seg)
            section_edges = dfNetworkEdge[dfNetworkEdge['path_seg'] == section_id]
            if not section_edges.empty:
                upstream_junction_id = section_edges.iloc[0]['path_start_node']
                downstream_junction_id = section_edges.iloc[0]['path_end_node']
                print(f"Upstream junction_id (path_start_node): {upstream_junction_id}")
                print(f"Downstream junction_id (path_end_node): {downstream_junction_id}")
        except Exception as e:
            print(f"Warning: Could not load junction IDs: {e}")
    else:
        print(f"Warning: Network edges file not found: {network_edges_path}")
    
    # Compute clean slopes using compute_clean_slopes
    print("Computing slopes using compute_clean_slopes...")
    result = compute_clean_slopes(dfSection)
    dfSlope = result["section_stats_clean"]
    
    # Filter to the specific section and junction
    dfSlope = dfSlope[
        (dfSlope['section_id'] == section_id) & 
        (dfSlope['junction_id'] == juns[0])
    ]
    
    if dfSlope.empty:
        print("WARNING: No slope results found for this section/junction combination")
        dfSlope = pd.DataFrame(columns=['junction_id', 'section_id', 'distance', 'slope', 'SE', 'intercept', 'slopeF', 'outlier_removed_flag'])
    else:
        print(f"Slope: {dfSlope['slope'].iloc[0]:.2e} ± {dfSlope['SE'].iloc[0]:.2e}")
        print(f"Slope fraction: {dfSlope['slopeF'].iloc[0]:.3f}")
    
    # Create visualization
    plotly_scatter(dfSection, 'distance', 'wse', juns, dfSlope, upstream_junction_id, downstream_junction_id)


def list_available_sections(directory, continent):
    """
    List all available sections in a continent's parquet file.
    
    Parameters:
    -----------
    directory : str
        Base directory containing output folders
    continent : str
        Continent code
    """
    parquet_path = os.path.join(directory, 'output', continent, f'{continent}_swot_nodes.parquet')
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    con = duckdb.connect(":memory:")
    
    query = f"""
        SELECT 
            section_id,
            junction_id,
            COUNT(*) as n_obs,
            COUNT(DISTINCT cyclePass) as n_passes,
            MIN(distance) as min_dist,
            MAX(distance) as max_dist
        FROM read_parquet('{parquet_path}')
        GROUP BY section_id, junction_id
        ORDER BY section_id, junction_id
    """
    
    df = con.execute(query).df()
    con.close()
    
    print(f"\nAvailable sections in {continent}:")
    print(df.to_string(index=False))
    print(f"\nTotal sections: {df['section_id'].nunique()}")
    print(f"Total section-junction combinations: {len(df)}")
    
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Visualize SWOT sections with WSE data and computed slopes"
    )
    ap.add_argument("--dir", required=True, help="Base directory containing output folders")
    ap.add_argument("--continent", required=True, help="Continent code (e.g., 'af', 'na', 'as')")
    ap.add_argument("--section", type=int, help="Section ID to visualize")
    ap.add_argument("--junction", type=int, help="Junction ID (optional, filters if multiple exist)")
    ap.add_argument("--list", action="store_true", help="List all available sections and exit")
    
    args = ap.parse_args()
    
    if args.list:
        list_available_sections(args.dir, args.continent)
    elif args.section:
        plot(args.dir, args.continent, args.section, args.junction)
    else:
        print("Error: Either --section or --list must be provided")
        print("Use --list to see available sections")
        ap.print_help()

