#!/usr/bin/env python3
"""
Generate the 9-panel RivGraph pipeline view for each sampled reach.
"""

import sys

sys.path.insert(
    0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent / "src")
)

import numpy as np
import duckdb
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_closing
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from skimage.morphology import disk, remove_small_objects, skeletonize
from skimage.draw import line
import tempfile
import os
import re

from pyproj import Transformer
import rasterio

from sword_duckdb.imagery import FusionCenterlineExtractor

# Parse results from log
log_path = "/tmp/stratified_sample.log"
log = open(log_path).read()

results = []
pattern = r"\[(\d+)/52\] Reach (\d+)\n  width=(\d+)m, slope=([0-9.e+-]+), facc=([0-9.e+-]+), n_chan=(\d+)"

for m in re.finditer(pattern, log):
    idx, reach_id, width, slope, facc, n_chan = m.groups()
    pos = m.end()
    result_match = re.search(
        r"→ mean_drift=([0-9.]+)m, pct_ok=([0-9.]+)%, pct_major=([0-9.]+)%",
        log[pos : pos + 500],
    )

    if result_match:
        results.append(
            {
                "idx": int(idx),
                "reach_id": int(reach_id),
                "width": float(width),
                "slope": float(slope),
                "n_chan": int(n_chan),
                "mean_drift": float(result_match.group(1)),
                "pct_ok": float(result_match.group(2)),
            }
        )

print(f"Found {len(results)} successful reaches")

# Connect to SWORD
sword_db = duckdb.connect(
    "/Users/jakegearon/projects/SWORD/data/duckdb/sword_v17b.duckdb", read_only=True
)
extractor = FusionCenterlineExtractor()

# Use ALL successful reaches
selected = sorted(results, key=lambda x: x["mean_drift"])

print(f"Processing ALL {len(selected)} reaches:")
for r in selected:
    status = (
        "GOOD"
        if r["mean_drift"] < 60
        else "MOD"
        if r["mean_drift"] < 500
        else "OUTLIER"
    )
    print(f"  {r['reach_id']}: {r['mean_drift']:.0f}m [{status}]")


def generate_pipeline_figure(reach_info, sword_db, extractor, output_path):
    """Generate 9-panel pipeline figure for one reach."""
    reach_id = reach_info["reach_id"]
    reach_width = reach_info["width"]

    # Get SWORD centerline
    nodes = sword_db.execute(
        """
        SELECT cl_id, x, y FROM centerlines WHERE reach_id = ? ORDER BY cl_id
    """,
        [reach_id],
    ).fetchall()

    sword_nodes = np.array([(cl_id, x, y) for cl_id, x, y in nodes])
    sword_wgs84 = sword_nodes[:, 1:3]

    # Create bbox
    buffer = 0.02
    bbox = (
        sword_wgs84[:, 0].min() - buffer,
        sword_wgs84[:, 1].min() - buffer,
        sword_wgs84[:, 0].max() + buffer,
        sword_wgs84[:, 1].max() + buffer,
    )

    # Get water mask
    water_result = extractor.get_fused_water_mask(bbox, "2023-01-01", "2024-12-31")
    water_mask_raw = water_result.mask
    h, w = water_mask_raw.shape

    # Transform SWORD to pixel
    xform = Transformer.from_crs("EPSG:4326", water_result.crs, always_xy=True)
    sword_utm = np.array([xform.transform(x, y) for x, y in sword_wgs84])
    inv_t = ~water_result.transform
    sword_px = np.array([inv_t * (x, y) for x, y in sword_utm])[:, ::-1]
    sword_px[:, 0] = np.clip(sword_px[:, 0], 0, h - 1)
    sword_px[:, 1] = np.clip(sword_px[:, 1], 0, w - 1)

    # Corridor constraint
    corridor_factor = 5.0
    min_corridor_m = 400.0
    corridor_buffer_m = max(reach_width * corridor_factor, min_corridor_m)
    corridor_buffer_px = int(corridor_buffer_m / 10)

    sword_line = np.zeros((h, w), dtype=bool)
    for r, c in sword_px:
        ri, ci = int(r), int(c)
        if 0 <= ri < h and 0 <= ci < w:
            sword_line[ri, ci] = True

    corridor = binary_dilation(sword_line, disk(corridor_buffer_px))
    water_in_corridor = water_mask_raw & corridor

    # SWORD-path bridging
    water_filled = water_in_corridor.copy()
    bridge_width = 2

    for i in range(len(sword_px) - 1):
        r1, c1 = int(sword_px[i, 0]), int(sword_px[i, 1])
        r2, c2 = int(sword_px[i + 1, 0]), int(sword_px[i + 1, 1])
        rr, cc = line(r1, c1, r2, c2)
        rr = np.clip(rr, 0, h - 1)
        cc = np.clip(cc, 0, w - 1)
        gap_pixels = ~water_in_corridor[rr, cc]
        if np.any(gap_pixels):
            for dr in range(-bridge_width, bridge_width + 1):
                for dc in range(-bridge_width, bridge_width + 1):
                    if dr * dr + dc * dc <= bridge_width * bridge_width:
                        rr_off = np.clip(rr + dr, 0, h - 1)
                        cc_off = np.clip(cc + dc, 0, w - 1)
                        water_filled[rr_off, cc_off] = True

    # Cleanup
    water_closed = binary_closing(water_filled, disk(2))
    water_mask = remove_small_objects(water_closed, min_size=100)

    # Distance transform
    dt = distance_transform_edt(water_mask)

    # Skeleton
    skeleton = skeletonize(water_mask)

    # RivGraph extraction
    rivgraph_cl_px = None
    try:
        from rivgraph.classes import river

        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = os.path.join(tmpdir, "mask.tif")
            with rasterio.open(
                mask_path,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype="uint8",
                crs=water_result.crs,
                transform=water_result.transform,
            ) as dst:
                dst.write(water_mask.astype("uint8"), 1)

            # Exit sides
            start_r, start_c = sword_px[0]
            end_r, end_c = sword_px[-1]
            exit_sides = set()

            for r, c in [(start_r, start_c), (end_r, end_c)]:
                if r < h * 0.15:
                    exit_sides.add("n")
                elif r > h * 0.85:
                    exit_sides.add("s")
                if c < w * 0.15:
                    exit_sides.add("w")
                elif c > w * 0.85:
                    exit_sides.add("e")

            if not exit_sides:
                exit_sides = {"n", "s"}

            riv = river(
                name="sword_update",
                path_to_mask=mask_path,
                results_folder=tmpdir,
                exit_sides=",".join(exit_sides),
            )
            riv.compute_network()

            try:
                riv.compute_centerline()
            except:
                pass

            if hasattr(riv, "centerline") and riv.centerline is not None:
                cl = riv.centerline
                if isinstance(cl, tuple) and len(cl) >= 2:
                    x_coords, y_coords = cl[0], cl[1]
                    cl_geo = np.column_stack([x_coords, y_coords])
                    rivgraph_cl_px = np.array([inv_t * (x, y) for x, y in cl_geo])[
                        :, ::-1
                    ]

    except Exception as e:
        print(f"    RivGraph failed: {e}")

    # DT snapping
    snapped_cl_px = None
    if rivgraph_cl_px is not None:
        snapped_cl_px = rivgraph_cl_px.copy()
        dt_snap_radius = 8

        for i in range(len(rivgraph_cl_px)):
            r, c = rivgraph_cl_px[i]
            ri, ci = int(r), int(c)
            best_r, best_c = ri, ci
            best_dt = 0

            for dr in range(-dt_snap_radius, dt_snap_radius + 1):
                for dc in range(-dt_snap_radius, dt_snap_radius + 1):
                    nr, nc = ri + dr, ci + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if dt[nr, nc] > best_dt:
                            best_dt = dt[nr, nc]
                            best_r, best_c = nr, nc

            snapped_cl_px[i] = [best_r, best_c]

        # Smooth
        if len(snapped_cl_px) > 20:
            snapped_cl_px[:, 0] = savgol_filter(
                snapped_cl_px[:, 0], min(15, len(snapped_cl_px) // 2 * 2 + 1), 3
            )
            snapped_cl_px[:, 1] = savgol_filter(
                snapped_cl_px[:, 1], min(15, len(snapped_cl_px) // 2 * 2 + 1), 3
            )

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))

    # 1. Raw water mask
    ax = axes[0, 0]
    ax.imshow(water_mask_raw, cmap="Blues")
    ax.plot(sword_px[:, 1], sword_px[:, 0], "r-", lw=1.5, label="SWORD")
    ax.set_title(f"1. Raw Water Mask\n({np.sum(water_mask_raw):,} px)")
    ax.legend(fontsize=8)
    ax.axis("off")

    # 2. Corridor constraint
    ax = axes[0, 1]
    ax.imshow(corridor, cmap="Oranges", alpha=0.5)
    ax.imshow(water_mask_raw, cmap="Blues", alpha=0.5)
    ax.plot(sword_px[:, 1], sword_px[:, 0], "r-", lw=1.5)
    ax.set_title(f"2. Corridor ({corridor_buffer_m:.0f}m)")
    ax.axis("off")

    # 3. Water in corridor
    ax = axes[0, 2]
    ax.imshow(water_in_corridor, cmap="Blues")
    ax.plot(sword_px[:, 1], sword_px[:, 0], "r-", lw=1.5)
    ax.set_title(f"3. Water in Corridor\n({np.sum(water_in_corridor):,} px)")
    ax.axis("off")

    # 4. After bridging
    ax = axes[1, 0]
    bridged_diff = water_filled & ~water_in_corridor
    ax.imshow(water_in_corridor, cmap="Blues")
    ax.imshow(np.ma.masked_where(~bridged_diff, bridged_diff), cmap="Reds", alpha=0.8)
    ax.plot(sword_px[:, 1], sword_px[:, 0], "r-", lw=0.5)
    ax.set_title("4. SWORD-Path Bridging\n(red = filled)")
    ax.axis("off")

    # 5. After cleanup
    ax = axes[1, 1]
    ax.imshow(water_mask, cmap="Blues")
    ax.set_title(f"5. After Cleanup\n({np.sum(water_mask):,} px)")
    ax.axis("off")

    # 6. Distance transform
    ax = axes[1, 2]
    dt_vis = np.ma.masked_where(~water_mask, dt)
    im = ax.imshow(dt_vis, cmap="hot")
    ax.set_title(f"6. Distance Transform\n(max={dt.max():.1f} px)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.axis("off")

    # 7. Skeleton + RivGraph
    ax = axes[2, 0]
    ax.imshow(water_mask, cmap="Blues", alpha=0.3)
    ax.imshow(np.ma.masked_where(~skeleton, skeleton), cmap="Greens", alpha=0.8)
    if rivgraph_cl_px is not None:
        ax.plot(
            rivgraph_cl_px[:, 1], rivgraph_cl_px[:, 0], "r-", lw=2, label="RivGraph"
        )
    ax.set_title("7. Skeleton + RivGraph")
    ax.legend(fontsize=8)
    ax.axis("off")

    # 8. DT snap
    ax = axes[2, 1]
    dt_vis = np.ma.masked_where(~water_mask, dt)
    ax.imshow(dt_vis, cmap="hot")
    if rivgraph_cl_px is not None:
        ax.plot(
            rivgraph_cl_px[:, 1],
            rivgraph_cl_px[:, 0],
            "b-",
            lw=1.5,
            alpha=0.7,
            label="Before",
        )
    if snapped_cl_px is not None:
        ax.plot(
            snapped_cl_px[:, 1], snapped_cl_px[:, 0], "g-", lw=1.5, label="After snap"
        )
    ax.set_title("8. DT Snap")
    ax.legend(fontsize=8)
    ax.axis("off")

    # 9. Final comparison
    ax = axes[2, 2]
    ax.imshow(water_mask, cmap="Blues", alpha=0.3)
    ax.plot(sword_px[:, 1], sword_px[:, 0], "r-", lw=2, label="SWORD")
    if snapped_cl_px is not None:
        ax.plot(snapped_cl_px[:, 1], snapped_cl_px[:, 0], "g-", lw=2, label="Observed")

        # Drift vectors
        tree = cKDTree(snapped_cl_px)
        for i in range(0, len(sword_px), max(1, len(sword_px) // 15)):
            r, c = sword_px[i]
            dist, idx = tree.query([r, c])
            nearest = snapped_cl_px[idx]
            ax.arrow(
                c,
                r,
                nearest[1] - c,
                nearest[0] - r,
                head_width=3,
                head_length=2,
                fc="orange",
                ec="orange",
                alpha=0.7,
            )

    ax.set_title("9. SWORD vs Observed\n(orange = drift)")
    ax.legend(fontsize=8)
    ax.axis("off")

    # Status
    status = (
        "GOOD"
        if reach_info["mean_drift"] < 60
        else "MODERATE"
        if reach_info["mean_drift"] < 500
        else "OUTLIER"
    )

    plt.suptitle(
        f"Reach {reach_id} | drift={reach_info['mean_drift']:.0f}m | %ok={reach_info['pct_ok']:.0f}% | "
        f"width={reach_info['width']:.0f}m | slope={reach_info['slope']:.1e} | n_chan={reach_info['n_chan']} | [{status}]",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


# Generate figures for each selected reach
output_dir = "/tmp/reach_pipelines"
os.makedirs(output_dir, exist_ok=True)

for i, reach_info in enumerate(selected):
    print(f"\n[{i + 1}/{len(selected)}] Processing reach {reach_info['reach_id']}...")
    output_path = f"{output_dir}/reach_{reach_info['reach_id']}.png"

    try:
        generate_pipeline_figure(reach_info, sword_db, extractor, output_path)
        print(f"  Saved to {output_path}")
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n\nAll figures saved to {output_dir}/")
print("Files:")
for f in sorted(os.listdir(output_dir)):
    print(f"  {output_dir}/{f}")
