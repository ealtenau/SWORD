#!/usr/bin/env python3
"""
Compare SWORD v17b vs v17c topology differences.
"""

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

V17B_PATH = '/Users/jakegearon/projects/SWORD/data/duckdb/sword_v17b.duckdb'
V17C_PATH = '/Users/jakegearon/projects/SWORD/data/duckdb/sword_v17c.duckdb'
OUTPUT_DIR = Path('/Users/jakegearon/projects/SWORD/outputs/presentation')


def compare_reach_attribute(attr: str, region: str = 'NA'):
    """Compare a reach attribute between v17b and v17c."""
    v17b = duckdb.connect(V17B_PATH, read_only=True)
    v17c = duckdb.connect(V17C_PATH, read_only=True)

    df_b = v17b.execute(f'''
        SELECT reach_id, {attr} as val_v17b
        FROM reaches WHERE region = ?
    ''', [region]).fetchdf()

    df_c = v17c.execute(f'''
        SELECT reach_id, {attr} as val_v17c
        FROM reaches WHERE region = ?
    ''', [region]).fetchdf()

    v17b.close()
    v17c.close()

    merged = df_b.merge(df_c, on='reach_id')
    merged['diff'] = merged['val_v17c'] - merged['val_v17b']
    merged['abs_diff'] = merged['diff'].abs()

    return merged


def summarize_changes(attr: str, region: str = 'NA'):
    """Summarize changes for an attribute."""
    merged = compare_reach_attribute(attr, region)

    total = len(merged)
    changed = (merged['abs_diff'] > 0.001).sum()
    pct_changed = 100 * changed / total

    print(f"\n{attr} ({region}):")
    print(f"  Total reaches: {total:,}")
    print(f"  Changed: {changed:,} ({pct_changed:.1f}%)")

    if changed > 0:
        changes = merged[merged['abs_diff'] > 0.001]['diff']
        print(f"  Mean change: {changes.mean():.2f}")
        print(f"  Min change: {changes.min():.2f}")
        print(f"  Max change: {changes.max():.2f}")

    return merged


def compare_all_topology(region: str = 'NA'):
    """Compare all topology attributes."""
    attrs = [
        'dist_out',
        'n_rch_up',
        'n_rch_down',
        'path_freq',
        'path_order',
        'path_segs',
        'stream_order',
        'main_side',
        'end_reach',
        'network',
    ]

    results = {}
    for attr in attrs:
        try:
            merged = summarize_changes(attr, region)
            results[attr] = merged
        except Exception as e:
            print(f"\n{attr}: ERROR - {e}")

    return results


def plot_dist_out_changes(region: str = 'NA'):
    """Create visualization of dist_out changes."""
    merged = compare_reach_attribute('dist_out', region)

    # Get coordinates for mapping
    v17b = duckdb.connect(V17B_PATH, read_only=True)
    coords = v17b.execute('''
        SELECT reach_id, x, y FROM reaches WHERE region = ?
    ''', [region]).fetchdf()
    v17b.close()

    merged = merged.merge(coords, on='reach_id')
    changed = merged[merged['abs_diff'] > 0.001]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram of changes
    ax1 = axes[0]
    ax1.hist(changed['diff'] / 1000, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Change in dist_out (km)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Distribution of dist_out Changes ({region})\n'
                 f'{len(changed):,} reaches changed out of {len(merged):,}')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)

    # Map of changes
    ax2 = axes[1]
    scatter = ax2.scatter(
        changed['x'], changed['y'],
        c=changed['diff'] / 1000,
        cmap='RdBu_r',
        s=1,
        alpha=0.5,
        vmin=-100, vmax=100
    )
    plt.colorbar(scatter, ax=ax2, label='Change in dist_out (km)')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title(f'Spatial Distribution of dist_out Changes ({region})')
    ax2.set_aspect('equal')

    plt.tight_layout()
    output_path = OUTPUT_DIR / f'topology_dist_out_changes_{region}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")


def generate_topology_report(region: str = 'NA'):
    """Generate full topology comparison report."""
    print("=" * 60)
    print(f"SWORD v17b → v17c Topology Comparison ({region})")
    print("=" * 60)

    results = compare_all_topology(region)

    # Create visualization
    plot_dist_out_changes(region)

    return results


if __name__ == '__main__':
    import sys
    region = sys.argv[1] if len(sys.argv) > 1 else 'NA'
    generate_topology_report(region)
