#!/usr/bin/env python3
"""
High-resolution lake context visualization with proper projection.
"""

import duckdb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import contextily as cx
from pyproj import Transformer

DB_PATH = Path('/Users/jakegearon/projects/SWORD/data/duckdb/sword_v17b.duckdb')
OUTPUT_DIR = Path('/Users/jakegearon/projects/SWORD/outputs/presentation')

# Transformer for WGS84 -> Web Mercator
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def get_connection():
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    conn.execute("LOAD spatial")
    return conn


def to_mercator(lons, lats):
    """Convert lon/lat arrays to Web Mercator."""
    if not lons:
        return [], []
    xs, ys = transformer.transform(lons, lats)
    return list(xs), list(ys)


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


def plot_lake_hires(reach_id, region, output_path, buffer_deg=0.1, zoom=14):
    """
    Plot a sandwiched reach with high-res imagery and Web Mercator projection.
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
    print(f"  Lakes: {len(all_reaches[all_reaches['lakeflag'] == 1])}")
    print(f"  Rivers: {len(all_reaches[all_reaches['lakeflag'] == 0])}")

    # Create figure with larger size for hi-res
    fig, ax = plt.subplots(figsize=(16, 14))

    # Color scheme - distinct and visible on satellite
    colors = {
        0: '#FF6B35',  # River - orange/coral
        1: '#00D4FF',  # Lake - cyan
        2: '#FFE66D',  # Canal - yellow
        3: '#A855F7',  # Tidal - purple
    }

    # Plot all reaches by lakeflag, lakes first (background)
    for lakeflag in [1, 0, 2, 3]:
        subset = all_reaches[all_reaches['lakeflag'] == lakeflag]
        for _, row in subset.iterrows():
            lons, lats = get_reach_coords(row['reach_id'], region)
            if lons:
                xs, ys = to_mercator(lons, lats)
                lw = max(2, min(8, row['width'] / 80))
                # White outline for visibility on imagery
                ax.plot(xs, ys, color='white', linewidth=lw + 2, alpha=0.9,
                       solid_capstyle='round')
                ax.plot(xs, ys, color=colors.get(lakeflag, 'gray'),
                       linewidth=lw, alpha=0.95, solid_capstyle='round')

    main_row = main.iloc[0]

    # Legend with better styling
    legend_elements = [
        mpatches.Patch(facecolor=colors[0], edgecolor='white', linewidth=2,
                      label='River (lakeflag=0)'),
        mpatches.Patch(facecolor=colors[1], edgecolor='white', linewidth=2,
                      label='Lake (lakeflag=1)'),
    ]
    leg = ax.legend(handles=legend_elements, loc='upper right', fontsize=13,
                   framealpha=0.95, edgecolor='gray', fancybox=True)
    leg.get_frame().set_linewidth(1.5)

    # Title
    ax.set_title(f"SWORD Reach Classification\n"
                f"Reach {reach_id} ({region}) — {main_row['reach_length']:.0f}m long, "
                f"{main_row['width']:.0f}m wide",
                fontsize=15, fontweight='bold', pad=15)

    # Add high-res basemap
    try:
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zoom=zoom)
    except Exception as e:
        print(f"Warning: Could not add basemap at zoom {zoom}, trying auto: {e}")
        try:
            cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zoom='auto')
        except Exception as e2:
            print(f"Basemap failed: {e2}")

    # Clean up axes and fix aspect ratio
    ax.set_axis_off()
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    # Poyang Lake, China - very clear example
    print("Generating Poyang Lake (AS) hi-res...")
    plot_lake_hires(43442200071, 'AS', OUTPUT_DIR / 'hires_poyang_lake.png',
                   buffer_deg=0.15, zoom=13)

    # Lake Erie - Presque Isle
    print("\nGenerating Lake Erie hi-res...")
    plot_lake_hires(74269900341, 'NA', OUTPUT_DIR / 'hires_lake_erie.png',
                   buffer_deg=0.12, zoom=14)

    # Romania/Danube area
    print("\nGenerating Romania hi-res...")
    plot_lake_hires(22733000184, 'EU', OUTPUT_DIR / 'hires_romania.png',
                   buffer_deg=0.12, zoom=14)

    # South America lake complex
    print("\nGenerating SA lake complex hi-res...")
    plot_lake_hires(61650801471, 'SA', OUTPUT_DIR / 'hires_sa_lakes.png',
                   buffer_deg=0.2, zoom=12)
