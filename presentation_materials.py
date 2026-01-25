#!/usr/bin/env python3
"""
SWORD v17c Presentation Materials Generator
============================================

Generates visualizations and exports for SWORD data quality issues
and topology improvements documentation.
"""

import duckdb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import json
import contextily as cx

# Database path
DB_PATH = Path('/Users/jakegearon/projects/SWORD/data/duckdb/sword_v17b.duckdb')
OUTPUT_DIR = Path('/Users/jakegearon/projects/SWORD/outputs/presentation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_connection():
    """Get database connection."""
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    conn.execute("LOAD spatial")
    return conn


def find_sandwiched_reaches(limit=50):
    """Find river reaches sandwiched between lakes."""
    conn = get_connection()
    query = '''
    WITH river_reaches AS (
        SELECT reach_id, region, lakeflag, x, y, reach_length, width, geom
        FROM reaches WHERE lakeflag = 0
    ),
    upstream_lakes AS (
        SELECT DISTINCT rt.reach_id, rt.region
        FROM reach_topology rt
        JOIN reaches r ON rt.neighbor_reach_id = r.reach_id AND rt.region = r.region
        WHERE rt.direction = 'up' AND r.lakeflag = 1
    ),
    downstream_lakes AS (
        SELECT DISTINCT rt.reach_id, rt.region
        FROM reach_topology rt
        JOIN reaches r ON rt.neighbor_reach_id = r.reach_id AND rt.region = r.region
        WHERE rt.direction = 'down' AND r.lakeflag = 1
    )
    SELECT rr.reach_id, rr.region, rr.x as lon, rr.y as lat,
           rr.reach_length, rr.width, ST_AsText(rr.geom) as wkt
    FROM river_reaches rr
    JOIN upstream_lakes ul ON rr.reach_id = ul.reach_id AND rr.region = ul.region
    JOIN downstream_lakes dl ON rr.reach_id = dl.reach_id AND rr.region = dl.region
    ORDER BY rr.reach_length DESC
    LIMIT ?
    '''
    df = conn.execute(query, [limit]).fetchdf()
    conn.close()
    return df


def find_river_to_lake_transitions(limit=30):
    """Find river reaches directly feeding into lakes."""
    conn = get_connection()
    query = '''
    SELECT
        r.reach_id, r.region, r.x as lon, r.y as lat,
        r.reach_length, r.width,
        r.lakeflag as river_lakeflag,
        lake.reach_id as lake_reach_id,
        lake.lakeflag as lake_lakeflag,
        lake.reach_length as lake_length
    FROM reaches r
    JOIN reach_topology rt ON r.reach_id = rt.reach_id AND r.region = rt.region
    JOIN reaches lake ON rt.neighbor_reach_id = lake.reach_id AND rt.region = lake.region
    WHERE r.lakeflag = 0
      AND lake.lakeflag = 1
      AND rt.direction = 'down'
    ORDER BY r.reach_length DESC
    LIMIT ?
    '''
    df = conn.execute(query, [limit]).fetchdf()
    conn.close()
    return df


def get_reach_with_neighbors(reach_id, region):
    """Get a reach and its upstream/downstream neighbors for visualization."""
    conn = get_connection()

    # Get the main reach with centerline coordinates
    main = conn.execute('''
        SELECT r.reach_id, r.region, r.x, r.y, r.reach_length, r.width, r.lakeflag,
               r.cl_id_min, r.cl_id_max
        FROM reaches r WHERE r.reach_id = ? AND r.region = ?
    ''', [reach_id, region]).fetchdf()

    # Get upstream neighbors
    upstream = conn.execute('''
        SELECT r.reach_id, r.region, r.x, r.y, r.reach_length, r.width, r.lakeflag,
               r.cl_id_min, r.cl_id_max
        FROM reach_topology rt
        JOIN reaches r ON rt.neighbor_reach_id = r.reach_id AND rt.region = r.region
        WHERE rt.reach_id = ? AND rt.region = ? AND rt.direction = 'up'
    ''', [reach_id, region]).fetchdf()

    # Get downstream neighbors
    downstream = conn.execute('''
        SELECT r.reach_id, r.region, r.x, r.y, r.reach_length, r.width, r.lakeflag,
               r.cl_id_min, r.cl_id_max
        FROM reach_topology rt
        JOIN reaches r ON rt.neighbor_reach_id = r.reach_id AND rt.region = r.region
        WHERE rt.reach_id = ? AND rt.region = ? AND rt.direction = 'down'
    ''', [reach_id, region]).fetchdf()

    conn.close()
    return main, upstream, downstream


def get_centerline_coords(cl_id_min, cl_id_max, region):
    """Get centerline coordinates for a reach."""
    conn = get_connection()
    result = conn.execute('''
        SELECT x, y FROM centerlines
        WHERE region = ? AND cl_id BETWEEN ? AND ?
        ORDER BY cl_id
    ''', [region, cl_id_min, cl_id_max]).fetchdf()
    conn.close()
    return list(result['x']), list(result['y'])


def get_reach_linestring(reach_id, region):
    """Construct reach linestring from centerline points."""
    conn = get_connection()

    # First try using geometry (only works for NA region)
    result = conn.execute('''
        SELECT ST_AsText(ST_MakeLine(list(c.geom ORDER BY c.cl_id))) as wkt
        FROM reaches r
        JOIN centerlines c ON c.region = r.region
            AND c.cl_id BETWEEN r.cl_id_min AND r.cl_id_max
        WHERE r.reach_id = ? AND r.region = ?
        GROUP BY r.reach_id
    ''', [reach_id, region]).fetchdf()

    if not result.empty and result['wkt'].iloc[0] and 'EMPTY' not in result['wkt'].iloc[0]:
        wkt = result['wkt'].iloc[0]
        # Handle different geometry types
        if wkt.startswith('LINESTRING'):
            coords_str = wkt.replace('LINESTRING (', '').replace(')', '')
        elif wkt.startswith('MULTILINESTRING'):
            import re
            match = re.search(r'\(\(([^)]+)\)', wkt)
            if match:
                coords_str = match.group(1)
            else:
                coords_str = None
        else:
            coords_str = None

        if coords_str:
            coords = []
            for pt in coords_str.split(', '):
                parts = pt.strip().split(' ')
                if len(parts) >= 2:
                    try:
                        coords.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
            if coords:
                conn.close()
                return [c[0] for c in coords], [c[1] for c in coords]

    # Fallback: use x,y coordinates directly from centerlines table
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


def get_lakeflag_distribution():
    """Get lakeflag distribution across all reaches."""
    conn = get_connection()
    df = conn.execute('''
        SELECT lakeflag,
               CASE lakeflag
                   WHEN 0 THEN 'river'
                   WHEN 1 THEN 'lake'
                   WHEN 2 THEN 'canal'
                   WHEN 3 THEN 'tidal'
                   ELSE 'unknown'
               END as type,
               COUNT(*) as count
        FROM reaches
        GROUP BY lakeflag
        ORDER BY lakeflag
    ''').fetchdf()
    conn.close()
    return df


def plot_sandwiched_reach_example(reach_id, region, output_path):
    """Plot a detailed view of a sandwiched reach with its neighbors."""
    main, upstream, downstream = get_reach_with_neighbors(reach_id, region)

    if main.empty:
        print(f"Reach {reach_id} not found")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    # Color mapping for lakeflag
    colors = {0: '#1f77b4', 1: '#2ca02c', 2: '#ff7f0e', 3: '#9467bd'}
    labels = {0: 'River', 1: 'Lake', 2: 'Canal', 3: 'Tidal'}

    # Plot upstream neighbors
    for _, row in upstream.iterrows():
        lons, lats = get_reach_linestring(row['reach_id'], region)
        if lons:
            ax.plot(lons, lats, color=colors.get(row['lakeflag'], 'gray'),
                   linewidth=max(2, min(8, row['width']/50)), alpha=0.7)
            ax.scatter([row['x']], [row['y']], c=colors.get(row['lakeflag'], 'gray'),
                      s=100, zorder=5, edgecolors='black')
            ax.annotate(f"{int(row['reach_id'])}\n({labels.get(row['lakeflag'], '?')})",
                       (row['x'], row['y']), fontsize=8, ha='center')

    # Plot main reach (the sandwiched one)
    for _, row in main.iterrows():
        lons, lats = get_reach_linestring(row['reach_id'], region)
        if lons:
            ax.plot(lons, lats, color='red', linewidth=4, alpha=0.9)
            ax.scatter([row['x']], [row['y']], c='red', s=200, zorder=10,
                      edgecolors='black', marker='*')
            ax.annotate(f"{int(row['reach_id'])}\n(SANDWICHED RIVER)",
                       (row['x'], row['y']), fontsize=10, ha='center',
                       fontweight='bold', color='red')

    # Plot downstream neighbors
    for _, row in downstream.iterrows():
        lons, lats = get_reach_linestring(row['reach_id'], region)
        if lons:
            ax.plot(lons, lats, color=colors.get(row['lakeflag'], 'gray'),
                   linewidth=max(2, min(8, row['width']/50)), alpha=0.7)
            ax.scatter([row['x']], [row['y']], c=colors.get(row['lakeflag'], 'gray'),
                      s=100, zorder=5, edgecolors='black')
            ax.annotate(f"{int(row['reach_id'])}\n({labels.get(row['lakeflag'], '?')})",
                       (row['x'], row['y']), fontsize=8, ha='center')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='red', label='Sandwiched River (Issue)'),
        mpatches.Patch(facecolor=colors[0], label='River (lakeflag=0)'),
        mpatches.Patch(facecolor=colors[1], label='Lake (lakeflag=1)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Get main reach info for title
    main_row = main.iloc[0]
    ax.set_title(f"Lake-Sandwiched River Issue\n"
                f"Reach {reach_id} ({region})\n"
                f"Length: {main_row['reach_length']:.0f}m, Width: {main_row['width']:.0f}m",
                fontsize=14)
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


def plot_lakeflag_distribution(output_path):
    """Plot lakeflag distribution pie chart."""
    df = get_lakeflag_distribution()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']

    # Pie chart
    ax1.pie(df['count'], labels=df['type'], autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax1.set_title('SWORD Reach Type Distribution (Global)', fontsize=14)

    # Bar chart with counts
    bars = ax2.bar(df['type'], df['count'], color=colors)
    ax2.set_ylabel('Count')
    ax2.set_title('Reach Counts by Type', fontsize=14)
    ax2.set_xlabel('Reach Type (lakeflag)')

    # Add count labels on bars
    for bar, count in zip(bars, df['count']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{count:,}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_sandwiched_examples_overview(output_path, n_examples=20):
    """Create overview map of sandwiched reach examples."""
    df = find_sandwiched_reaches(n_examples)

    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot world background (simple)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Plot each example
    scatter = ax.scatter(df['lon'], df['lat'], c=df['reach_length'],
                        cmap='viridis', s=100, edgecolors='red', linewidths=2,
                        alpha=0.8)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Reach Length (m)')

    # Annotate top examples
    for i, row in df.head(10).iterrows():
        ax.annotate(f"{row['region']}", (row['lon'], row['lat']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_title(f'River Reaches Sandwiched Between Lakes (Top {n_examples} by Length)\n'
                f'Total Found: {len(df)}', fontsize=14)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def export_examples_geojson(output_path):
    """Export sandwiched reaches as GeoJSON for external visualization."""
    df = find_sandwiched_reaches(100)

    features = []
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row['lon'], row['lat']]
            },
            "properties": {
                "reach_id": int(row['reach_id']),
                "region": row['region'],
                "reach_length": row['reach_length'],
                "width": row['width'],
                "issue_type": "sandwiched_river"
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"Exported {len(features)} features to {output_path}")


def create_topology_summary():
    """Create summary of topology improvements."""
    summary = """
# SWORD v17b → v17c Topology Improvements

## Key Algorithm Changes

### 1. Centerline Neighbor Recalculation (KDTree)
**Commit**: 777f4aa
**File**: reactive.py

Uses scipy.spatial.cKDTree for spatial neighbor queries instead of
manual distance calculations. Improves accuracy of `node_id_neighbors` field.

### 2. Main/Side Classification
**Commit**: 853c867
**File**: reactive.py:615

Reactive recalculation of `main_side` field:
- 0 = main channel (higher path_freq)
- 1 = side channel (lower path_freq)
- 2 = secondary outlet

### 3. Tributary Flag Correction
**Commit**: 6303ef2
**File**: reconstruction.py

Fixed k-value mismatch in trib_flag calculation. Made delta threshold
configurable for different river systems.

### 4. Distance to Outlet (dist_out)
**File**: reconstruction.py:1382-1497

BFS traversal from outlets upstream, accumulating reach lengths.
Now properly handles disconnected networks.

## Topology Validation Checks

The `check_topo_consistency()` function validates:
- Type 1: n_rch_up/down count matches actual neighbors
- Type 2: Bidirectional neighbor references (A→B implies B→A)
- Type 5: No self-references
- Type 3 (warn): Ghost reaches with both up+down neighbors
- Type 4 (warn): Non-ghost reaches missing upstream

## Reactive Update System

```
Dependency Graph:
centerline.geometry → reach.len, reach.bounds
                   → node.len, node.xy
                   → centerline.node_id_neighbors

reach.topology → reach.dist_out
              → reach.end_rch
              → reach.main_side
              → node.dist_out, node.end_rch, node.main_side
```
"""
    return summary


def generate_all_materials():
    """Generate all presentation materials."""
    print("=" * 60)
    print("SWORD v17c Presentation Materials Generator")
    print("=" * 60)

    # 1. Lakeflag distribution
    print("\n1. Generating lakeflag distribution chart...")
    plot_lakeflag_distribution(OUTPUT_DIR / 'lakeflag_distribution.png')

    # 2. Overview map of sandwiched examples
    print("\n2. Generating sandwiched reaches overview map...")
    plot_sandwiched_examples_overview(OUTPUT_DIR / 'sandwiched_overview.png')

    # 3. Detailed example: Lake Erie area
    print("\n3. Generating detailed example: Lake Erie (74269900341)...")
    plot_sandwiched_reach_example(74269900341, 'NA',
                                  OUTPUT_DIR / 'example_lake_erie.png')

    # 4. Detailed example: Russia
    print("\n4. Generating detailed example: Russia (26120600131)...")
    plot_sandwiched_reach_example(26120600131, 'EU',
                                  OUTPUT_DIR / 'example_russia.png')

    # 5. Detailed example: Amazon
    print("\n5. Generating detailed example: Amazon (62259000041)...")
    plot_sandwiched_reach_example(62259000041, 'SA',
                                  OUTPUT_DIR / 'example_amazon.png')

    # 6. Export GeoJSON
    print("\n6. Exporting sandwiched reaches to GeoJSON...")
    export_examples_geojson(OUTPUT_DIR / 'sandwiched_reaches.geojson')

    # 7. Generate topology summary
    print("\n7. Writing topology improvements summary...")
    summary = create_topology_summary()
    with open(OUTPUT_DIR / 'topology_improvements.md', 'w') as f:
        f.write(summary)
    print(f"Saved: {OUTPUT_DIR / 'topology_improvements.md'}")

    # 8. Export CSV of all examples
    print("\n8. Exporting examples to CSV...")
    df = find_sandwiched_reaches(100)
    df.to_csv(OUTPUT_DIR / 'sandwiched_reaches.csv', index=False)
    print(f"Saved: {OUTPUT_DIR / 'sandwiched_reaches.csv'}")

    print("\n" + "=" * 60)
    print(f"All materials saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    generate_all_materials()
