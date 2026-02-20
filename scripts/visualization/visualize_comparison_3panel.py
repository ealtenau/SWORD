#!/usr/bin/env python
"""Three-panel comparison: RGB+lines, drift magnitude, water mask+lines."""

import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_rgb_basemap(bbox, stac, cog):
    """Fetch RGB satellite basemap for visualization."""
    items = stac.search_by_bbox(
        bbox,
        start_date="2024-01-01",
        end_date="2024-12-31",
        max_cloud_cover=20,
        limit=5,
    )
    if not items:
        return None

    # Use clearest scene
    items_sorted = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
    item = items_sorted[0]

    bands = {}
    for band in ['red', 'green', 'blue']:
        data, transform, crs = cog.read_band(item, band, bbox)
        bands[band] = data

    # Stack and normalize
    rgb = np.stack([bands['red'], bands['green'], bands['blue']], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0)

    # Stretch for visualization
    p2, p98 = np.percentile(rgb[rgb > 0], [2, 98])
    rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    rgb = np.clip(rgb * 1.5, 0, 1)  # Slight brightness boost

    return rgb


def main():
    from src.sword_duckdb.imagery.river_tracer import RiverTracer
    from src.sword_duckdb.imagery import SentinelSTACClient, COGReader
    import duckdb

    # Rhine near Cologne - same test region
    bbox = (6.8, 50.85, 7.1, 51.05)  # ~30km stretch

    # Fetch RGB basemap first
    logger.info("Fetching RGB basemap...")
    stac = SentinelSTACClient()
    cog = COGReader()
    rgb_basemap = fetch_rgb_basemap(bbox, stac, cog)
    logger.info(f"Basemap shape: {rgb_basemap.shape if rgb_basemap is not None else None}")

    logger.info("Running RiverTracer...")
    tracer = RiverTracer(
        patch_size_deg=0.1,
        overlap_deg=0.01,
        min_votes=4,
        n_workers=4,
    )
    result = tracer.trace(bbox, exit_sides="n,s")

    logger.info(f"Extracted centerline: {len(result.centerline_coords) if result.centerline_coords is not None else 0} points")
    logger.info(f"Total length: {result.total_length_km:.1f} km")

    # Load SWORD centerlines from DuckDB
    logger.info("Loading SWORD centerlines...")
    db_path = "/Users/jakegearon/projects/SWORD/data/duckdb/sword_v17c.duckdb"

    conn = duckdb.connect(db_path, read_only=True)
    conn.execute("LOAD spatial;")

    # Query centerline points directly
    query = f"""
    SELECT x, y, reach_id
    FROM centerlines
    WHERE x BETWEEN {bbox[0]} AND {bbox[2]}
      AND y BETWEEN {bbox[1]} AND {bbox[3]}
    ORDER BY reach_id, cl_id
    """

    sword_points = conn.execute(query).fetchall()
    conn.close()

    logger.info(f"Found {len(sword_points)} SWORD centerline points in bbox")

    # Convert to array
    sword_coords_all = []
    for x, y, reach_id in sword_points:
        sword_coords_all.append([x, y])

    sword_coords = np.array(sword_coords_all) if sword_coords_all else None
    logger.info(f"SWORD total points: {len(sword_coords) if sword_coords is not None else 0}")

    # Compute drift
    extracted = result.centerline_coords
    if extracted is not None and sword_coords is not None:
        sword_tree = cKDTree(sword_coords)
        dists_deg, _ = sword_tree.query(extracted)

        lat_center = (bbox[1] + bbox[3]) / 2
        m_per_deg = 111000 * np.cos(np.radians(lat_center))
        drifts_m = dists_deg * m_per_deg

        logger.info(f"Drift stats: mean={np.mean(drifts_m):.1f}m, median={np.median(drifts_m):.1f}m, max={np.max(drifts_m):.1f}m")
    else:
        drifts_m = None

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Get mosaic info
    mosaic = result.mosaic
    h, w = mosaic.mask.shape
    extent = [bbox[0], bbox[2], bbox[1], bbox[3]]

    # --- Panel 1: RGB + both centerlines ---
    ax1 = axes[0]

    # Use satellite basemap
    if rgb_basemap is not None:
        ax1.imshow(rgb_basemap, extent=extent, origin='upper')
    else:
        # Fallback
        rgb_bg = np.zeros((h, w, 3))
        rgb_bg[mosaic.mask > 0] = [0.2, 0.4, 0.8]
        rgb_bg[mosaic.mask == 0] = [0.9, 0.9, 0.85]
        ax1.imshow(rgb_bg, extent=extent, origin='upper')

    # Plot centerlines
    if sword_coords is not None:
        ax1.scatter(sword_coords[:, 0], sword_coords[:, 1], c='red', s=1, alpha=0.7, label='SWORD')
    if extracted is not None:
        ax1.plot(extracted[:, 0], extracted[:, 1], 'g-', linewidth=1.5, alpha=0.8, label='Extracted')

    ax1.set_xlim(bbox[0], bbox[2])
    ax1.set_ylim(bbox[1], bbox[3])
    ax1.legend(loc='upper right')
    ax1.set_title('Centerline Comparison\nSWORD (red) vs Extracted (green)')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    # Scale bar
    scalebar1 = ScaleBar(m_per_deg, units='m', location='lower left', length_fraction=0.2)
    ax1.add_artist(scalebar1)

    # --- Panel 2: Drift magnitude ---
    ax2 = axes[1]

    if rgb_basemap is not None:
        ax2.imshow(rgb_basemap, extent=extent, origin='upper', alpha=0.7)
    else:
        ax2.imshow(rgb_bg, extent=extent, origin='upper', alpha=0.5)

    if extracted is not None and drifts_m is not None:
        sc = ax2.scatter(
            extracted[:, 0], extracted[:, 1],
            c=drifts_m, cmap='RdYlGn_r', s=8,
            vmin=0, vmax=100
        )
        cbar = plt.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Drift (m)')

    ax2.set_xlim(bbox[0], bbox[2])
    ax2.set_ylim(bbox[1], bbox[3])
    ax2.set_title(f'Drift Magnitude\nMean: {np.mean(drifts_m):.1f}m, Max: {np.max(drifts_m):.1f}m')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')

    scalebar2 = ScaleBar(m_per_deg, units='m', location='lower left', length_fraction=0.2)
    ax2.add_artist(scalebar2)

    # --- Panel 3: Water mask + both centerlines ---
    ax3 = axes[2]

    # Show water mask directly (no satellite)
    ax3.imshow(mosaic.mask, extent=extent, origin='upper', cmap='Blues')

    # Plot centerlines
    if sword_coords is not None:
        ax3.scatter(sword_coords[:, 0], sword_coords[:, 1], c='red', s=2, alpha=0.8, label='SWORD')
    if extracted is not None:
        ax3.plot(extracted[:, 0], extracted[:, 1], 'lime', linewidth=2, alpha=0.9, label='Extracted')

    ax3.set_xlim(bbox[0], bbox[2])
    ax3.set_ylim(bbox[1], bbox[3])
    ax3.legend(loc='upper right')
    ax3.set_title(f'Water Mask + Centerlines\n{mosaic.coverage_pct:.0f}% coverage, {result.total_length_km:.1f}km extracted')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')

    scalebar3 = ScaleBar(m_per_deg, units='m', location='lower left', length_fraction=0.2)
    ax3.add_artist(scalebar3)

    plt.tight_layout()
    plt.savefig('sword_comparison_3panel.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: sword_comparison_3panel.png")


if __name__ == "__main__":
    main()
