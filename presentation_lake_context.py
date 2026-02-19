#!/usr/bin/env python3
"""
Generate lake context visualization showing misclassified river reaches within lakes.
"""

import duckdb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import contextily as cx

DB_PATH = Path('/Users/jakegearon/projects/SWORD/data/duckdb/sword_v17b.duckdb')
OUTPUT_DIR = Path('/Users/jakegearon/projects/SWORD/outputs/presentation')


def get_connection():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    conn.execute("LOAD spatial")
    return conn


def get_reach_coords(reach_id, region):
    """Get reach centerline coordinates."""
    conn = get_connection()
    result = conn.execute('''
        SELECT c.x, c.y
        FROM reaches r
        JOIN centerlines c ON c.region = r.region
            AND c.cl_id BETWEEN r.cl_id_min AND r.cl_id_max
        WHERE r.reach_id = ? AND r.region = ?
        ORDER BY c.cl_id
    ''', [reach_id, region]).fetchdf()
    conn.close()
    if result.empty:
        return [], []
    return list(result['x']), list(result['y'])


def plot_lake_context(reach_id, region, output_path, buffer_deg=0.1):
    """
    Plot a sandwiched reach with ALL surrounding reaches color-coded by lakeflag.
    Shows the broader lake context.
    """
    conn = get_connection()

    # Get main reach info
    main = conn.execute('''
        SELECT reach_id, x, y, lakeflag, reach_length, width, river_name
        FROM reaches WHERE reach_id = ? AND region = ?
    ''', [reach_id, region]).fetchdf()

    if main.empty:
        print(f"Reach {reach_id} not found")
        return

    x_center, y_center = main.iloc[0]['x'], main.iloc[0]['y']

    # Get ALL reaches in the bounding box
    all_reaches = conn.execute('''
        SELECT reach_id, x, y, lakeflag, reach_length, width
        FROM reaches
        WHERE region = ?
          AND x BETWEEN ? AND ?
          AND y BETWEEN ? AND ?
    ''', [region, x_center - buffer_deg, x_center + buffer_deg,
          y_center - buffer_deg, y_center + buffer_deg]).fetchdf()

    conn.close()

    print(f"Found {len(all_reaches)} reaches in area")
    print(f"  Lakes (lakeflag=1): {len(all_reaches[all_reaches['lakeflag'] == 1])}")
    print(f"  Rivers (lakeflag=0): {len(all_reaches[all_reaches['lakeflag'] == 0])}")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Color scheme
    colors = {0: '#3498db', 1: '#27ae60', 2: '#f39c12', 3: '#9b59b6'}
    labels = {0: 'River', 1: 'Lake', 2: 'Canal', 3: 'Tidal'}

    # Plot all reaches by lakeflag, with lakes first (background)
    for lakeflag in [1, 0, 2, 3]:
        subset = all_reaches[all_reaches['lakeflag'] == lakeflag]
        for _, row in subset.iterrows():
            if row['reach_id'] == reach_id:
                continue  # Plot main reach last
            lons, lats = get_reach_coords(row['reach_id'], region)
            if lons:
                lw = max(1, min(6, row['width'] / 100))
                ax.plot(lons, lats, color=colors.get(lakeflag, 'gray'),
                       linewidth=lw, alpha=0.7)

    # Plot main reach (the sandwiched one) prominently
    main_row = main.iloc[0]
    lons, lats = get_reach_coords(reach_id, region)
    if lons:
        ax.plot(lons, lats, color='red', linewidth=5, alpha=0.95)
        ax.scatter([main_row['x']], [main_row['y']], c='red', s=300, zorder=10,
                  edgecolors='white', linewidths=2, marker='*')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='red', edgecolor='white', linewidth=2,
                      label=f'ISSUE: River in Lake ({reach_id})'),
        mpatches.Patch(facecolor=colors[1], label='Lake (lakeflag=1)'),
        mpatches.Patch(facecolor=colors[0], label='River (lakeflag=0)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    # Title
    ax.set_title(f"Misclassified River Reach Within Lake System\n"
                f"Reach {reach_id} ({region}) - {main_row['reach_length']:.0f}m long, "
                f"{main_row['width']:.0f}m wide\n"
                f"This river reach (red) is sandwiched between lake reaches (green)",
                fontsize=13)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Add basemap
    try:
        cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.Esri.WorldImagery, zoom='auto')
    except Exception as e:
        print(f"Warning: Could not add basemap: {e}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    # Lake Erie example - Presque Isle area
    print("Generating Lake Erie context map...")
    plot_lake_context(74269900341, 'NA',
                     OUTPUT_DIR / 'lake_context_erie.png', buffer_deg=0.15)

    # Russia example
    print("\nGenerating Russia context map...")
    plot_lake_context(26120600131, 'EU',
                     OUTPUT_DIR / 'lake_context_russia.png', buffer_deg=0.1)

    # Amazon example
    print("\nGenerating Amazon context map...")
    plot_lake_context(62259000041, 'SA',
                     OUTPUT_DIR / 'lake_context_amazon.png', buffer_deg=0.1)
