#!/usr/bin/env python3
"""
Generate maps for each sampled reach showing SWORD vs observed centerlines.
"""

import sys

sys.path.insert(
    0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent / "src")
)

import numpy as np
import duckdb
import matplotlib.pyplot as plt
import re

from sword_duckdb.imagery import FusionCenterlineExtractor, SWORDUpdater
from scipy.ndimage import distance_transform_edt

# Parse results from log to get reach IDs
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

print(f"Found {len(results)} successful reaches to visualize")

# Connect to SWORD
sword_db = duckdb.connect(
    "/Users/jakegearon/projects/SWORD/data/duckdb/sword_v17b.duckdb", read_only=True
)

# Select subset to visualize (mix of good, moderate, outlier)
# Sort by drift and pick representative samples
sorted_results = sorted(results, key=lambda x: x["mean_drift"])

# Pick: 3 best, 3 moderate, 3 worst
n_each = 3
selected = []
selected.extend(sorted_results[:n_each])  # Best
mid_idx = len(sorted_results) // 2
selected.extend(sorted_results[mid_idx - 1 : mid_idx + 2])  # Middle
selected.extend(sorted_results[-n_each:])  # Worst

print(f"\nSelected {len(selected)} reaches for visualization:")
for r in selected:
    status = (
        "GOOD"
        if r["mean_drift"] < 60
        else "MODERATE"
        if r["mean_drift"] < 500
        else "OUTLIER"
    )
    print(f"  {r['reach_id']}: drift={r['mean_drift']:.0f}m ({status})")

# Create figure with subplots
n_cols = 3
n_rows = 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 18))
axes = axes.flatten()

extractor = FusionCenterlineExtractor()

for i, reach_info in enumerate(selected):
    if i >= len(axes):
        break

    ax = axes[i]
    reach_id = reach_info["reach_id"]

    print(f"\nProcessing {reach_id} ({i + 1}/{len(selected)})...")

    try:
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
        buffer = 0.015
        bbox = (
            sword_wgs84[:, 0].min() - buffer,
            sword_wgs84[:, 1].min() - buffer,
            sword_wgs84[:, 0].max() + buffer,
            sword_wgs84[:, 1].max() + buffer,
        )

        # Get water mask
        water_result = extractor.get_fused_water_mask(bbox, "2023-01-01", "2024-12-31")
        water_mask = water_result.mask

        # Compute DT for visualization
        dt = distance_transform_edt(water_mask)

        # Transform SWORD to pixel
        from pyproj import Transformer

        xform = Transformer.from_crs("EPSG:4326", water_result.crs, always_xy=True)
        sword_utm = np.array([xform.transform(x, y) for x, y in sword_wgs84])
        inv_t = ~water_result.transform
        sword_px = np.array([inv_t * (x, y) for x, y in sword_utm])[:, ::-1]

        # Try to run the updater to get observed centerline
        try:
            updater = SWORDUpdater(min_slope=0)  # Disable slope check for visualization
            result = updater.analyze_reach(
                sword_wgs84=sword_wgs84,
                sword_ids=sword_nodes[:, 0].astype(int),
                reach_width_m=reach_info["width"],
                reach_id=reach_id,
                start_date="2023-01-01",
                end_date="2024-12-31",
            )
            has_observed = True

            # Transform observed to pixel
            observed_utm = np.array(
                [xform.transform(x, y) for x, y in result.observed_wgs84]
            )
            observed_px = np.array([inv_t * (x, y) for x, y in observed_utm])[:, ::-1]

            # Transform proposed to pixel
            proposed_utm = np.array(
                [xform.transform(x, y) for x, y in result.proposed_wgs84]
            )
            proposed_px = np.array([inv_t * (x, y) for x, y in proposed_utm])[:, ::-1]

        except Exception as e:
            print(f"  Could not get observed centerline: {e}")
            has_observed = False

        # Plot
        h, w = water_mask.shape

        # Show DT (distance transform) as background
        dt_masked = np.ma.masked_where(~water_mask, dt)
        ax.imshow(dt_masked, cmap="Blues", alpha=0.8, extent=[0, w, h, 0])

        # Show water mask boundary
        ax.contour(water_mask, levels=[0.5], colors="blue", linewidths=0.5, alpha=0.5)

        # Plot SWORD centerline
        ax.plot(sword_px[:, 1], sword_px[:, 0], "r-", lw=2, label="SWORD", zorder=10)
        ax.scatter(
            sword_px[0, 1], sword_px[0, 0], c="red", s=100, marker="^", zorder=11
        )  # Start
        ax.scatter(
            sword_px[-1, 1], sword_px[-1, 0], c="red", s=100, marker="v", zorder=11
        )  # End

        if has_observed:
            # Plot observed centerline
            ax.plot(
                observed_px[:, 1],
                observed_px[:, 0],
                "g-",
                lw=2,
                label="Observed",
                zorder=9,
            )

            # Plot drift vectors (every 10th node)
            for j in range(0, len(sword_px), max(1, len(sword_px) // 15)):
                ax.annotate(
                    "",
                    xy=(proposed_px[j, 1], proposed_px[j, 0]),
                    xytext=(sword_px[j, 1], sword_px[j, 0]),
                    arrowprops=dict(arrowstyle="->", color="orange", lw=1.5),
                    zorder=8,
                )

        # Title with stats
        status = (
            "GOOD"
            if reach_info["mean_drift"] < 60
            else "MOD"
            if reach_info["mean_drift"] < 500
            else "OUTLIER"
        )
        ax.set_title(
            f"Reach {reach_id}\n"
            f"drift={reach_info['mean_drift']:.0f}m, %ok={reach_info['pct_ok']:.0f}%, "
            f"w={reach_info['width']:.0f}m, n_ch={reach_info['n_chan']}\n"
            f"slope={reach_info['slope']:.1e} [{status}]",
            fontsize=10,
        )

        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

    except Exception as e:
        print(f"  Error: {e}")
        ax.text(
            0.5,
            0.5,
            f"Error:\n{str(e)[:50]}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(f"Reach {reach_id} - ERROR")

plt.suptitle(
    "SWORD vs Observed Centerlines\n(Red=SWORD, Green=Observed, Orange arrows=drift, Blue=water DT)",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("/tmp/reach_maps.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved to /tmp/reach_maps.png")
