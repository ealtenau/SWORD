#!/usr/bin/env python3
"""
Visualize the stratified sample results.
"""

import sys

sys.path.insert(0, "/Users/jakegearon/projects/SWORD/src/updates")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Parse results from log
log_path = "/tmp/stratified_sample.log"
log = open(log_path).read()

# Extract reach info and results
results = []
pattern = r"\[(\d+)/52\] Reach (\d+)\n  width=(\d+)m, slope=([0-9.e+-]+), facc=([0-9.e+-]+), n_chan=(\d+)"

for m in re.finditer(pattern, log):
    idx, reach_id, width, slope, facc, n_chan = m.groups()

    # Find the result line after this
    pos = m.end()
    result_match = re.search(
        r"→ mean_drift=([0-9.]+)m, pct_ok=([0-9.]+)%, pct_major=([0-9.]+)%",
        log[pos : pos + 500],
    )
    fail_match = re.search(r"→ FAILED: (.+)", log[pos : pos + 500])

    if result_match:
        results.append(
            {
                "idx": int(idx),
                "reach_id": int(reach_id),
                "width": float(width),
                "slope": float(slope),
                "facc": float(facc),
                "n_chan": int(n_chan),
                "mean_drift": float(result_match.group(1)),
                "pct_ok": float(result_match.group(2)),
                "pct_major": float(result_match.group(3)),
                "status": "success",
            }
        )
    elif fail_match:
        results.append(
            {
                "idx": int(idx),
                "reach_id": int(reach_id),
                "width": float(width),
                "slope": float(slope),
                "facc": float(facc),
                "n_chan": int(n_chan),
                "error": fail_match.group(1),
                "status": "failed",
            }
        )

df = pd.DataFrame(results)
print(f"Parsed {len(df)} reaches from log")
print(
    f"Success: {(df['status'] == 'success').sum()}, Failed: {(df['status'] == 'failed').sum()}"
)

# Create bins
success = df[df["status"] == "success"].copy()
success["width_bin"] = pd.cut(
    success["width"],
    bins=[0, 100, 300, 1000, 10000],
    labels=["narrow", "medium", "wide", "very_wide"],
)
success["slope_bin"] = pd.cut(
    success["slope"],
    bins=[0, 1e-5, 1e-4, 1e-3, 1],
    labels=["flat", "low", "medium", "steep"],
)
success["nchan_bin"] = pd.cut(
    success["n_chan"],
    bins=[-1, 1, 2, 5, 100],
    labels=["single", "double", "multi", "braided"],
)

# Filter good vs outliers
success["is_outlier"] = success["mean_drift"] > 500

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 16))

# 1. Overall drift distribution
ax1 = fig.add_subplot(3, 4, 1)
ax1.hist(success["mean_drift"], bins=30, edgecolor="black", alpha=0.7)
ax1.axvline(60, color="orange", ls="--", label="Good threshold")
ax1.axvline(500, color="red", ls="--", label="Outlier threshold")
ax1.set_xlabel("Mean Drift (m)")
ax1.set_ylabel("Count")
ax1.set_title("Drift Distribution (all)")
ax1.legend(fontsize=8)

# 2. Drift by width (boxplot)
ax2 = fig.add_subplot(3, 4, 2)
width_order = ["narrow", "medium", "wide", "very_wide"]
success.boxplot(
    column="mean_drift",
    by="width_bin",
    ax=ax2,
    positions=[
        width_order.index(w) for w in success.groupby("width_bin").groups.keys()
    ],
)
ax2.set_title("Drift by Width")
ax2.set_xlabel("Width Category")
ax2.set_ylabel("Mean Drift (m)")
plt.suptitle("")

# 3. Drift by slope (boxplot)
ax3 = fig.add_subplot(3, 4, 3)
success.boxplot(column="mean_drift", by="slope_bin", ax=ax3)
ax3.set_title("Drift by Slope")
ax3.set_xlabel("Slope Category")
ax3.set_ylabel("Mean Drift (m)")
plt.suptitle("")

# 4. Drift by n_chan (boxplot)
ax4 = fig.add_subplot(3, 4, 4)
success.boxplot(column="mean_drift", by="nchan_bin", ax=ax4)
ax4.set_title("Drift by Channel Complexity")
ax4.set_xlabel("N Channels")
ax4.set_ylabel("Mean Drift (m)")
plt.suptitle("")

# 5. Width vs Drift scatter
ax5 = fig.add_subplot(3, 4, 5)
colors = ["green" if not o else "red" for o in success["is_outlier"]]
ax5.scatter(success["width"], success["mean_drift"], c=colors, alpha=0.7, s=50)
ax5.axhline(500, color="red", ls="--", alpha=0.5)
ax5.set_xlabel("Width (m)")
ax5.set_ylabel("Mean Drift (m)")
ax5.set_title("Width vs Drift\n(green=good, red=outlier)")
ax5.set_xscale("log")

# 6. Slope vs Drift scatter
ax6 = fig.add_subplot(3, 4, 6)
ax6.scatter(success["slope"], success["mean_drift"], c=colors, alpha=0.7, s=50)
ax6.axhline(500, color="red", ls="--", alpha=0.5)
ax6.axvline(1e-5, color="blue", ls="--", alpha=0.5, label="New min_slope")
ax6.set_xlabel("Slope")
ax6.set_ylabel("Mean Drift (m)")
ax6.set_title("Slope vs Drift\n(blue line = new threshold)")
ax6.set_xscale("log")
ax6.legend(fontsize=8)

# 7. pct_ok vs Drift
ax7 = fig.add_subplot(3, 4, 7)
ax7.scatter(
    success["pct_ok"],
    success["mean_drift"],
    c=success["n_chan"],
    cmap="viridis",
    alpha=0.7,
    s=50,
)
ax7.set_xlabel("% Nodes OK")
ax7.set_ylabel("Mean Drift (m)")
ax7.set_title("Accuracy vs Drift\n(color = n_chan)")
cb = plt.colorbar(ax7.collections[0], ax=ax7)
cb.set_label("N Channels")

# 8. Success rate by category
ax8 = fig.add_subplot(3, 4, 8)
categories = ["narrow", "medium", "wide", "very_wide"]
good_counts = [
    len(success[(success["width_bin"] == c) & (~success["is_outlier"])])
    for c in categories
]
outlier_counts = [
    len(success[(success["width_bin"] == c) & (success["is_outlier"])])
    for c in categories
]
x = np.arange(len(categories))
ax8.bar(x - 0.2, good_counts, 0.4, label="Good (<500m)", color="green", alpha=0.7)
ax8.bar(x + 0.2, outlier_counts, 0.4, label="Outlier (>500m)", color="red", alpha=0.7)
ax8.set_xticks(x)
ax8.set_xticklabels(categories)
ax8.set_xlabel("Width Category")
ax8.set_ylabel("Count")
ax8.set_title("Good vs Outlier by Width")
ax8.legend()

# 9. Individual reach results (sorted by drift)
ax9 = fig.add_subplot(3, 4, 9)
sorted_success = success.sort_values("mean_drift")
colors_sorted = [
    "green" if d < 60 else "orange" if d < 500 else "red"
    for d in sorted_success["mean_drift"]
]
ax9.barh(
    range(len(sorted_success)),
    sorted_success["mean_drift"],
    color=colors_sorted,
    alpha=0.7,
)
ax9.axvline(60, color="orange", ls="--", alpha=0.7)
ax9.axvline(500, color="red", ls="--", alpha=0.7)
ax9.set_xlabel("Mean Drift (m)")
ax9.set_ylabel("Reach (sorted)")
ax9.set_title("All Reaches by Drift")

# 10. Drift excluding outliers
ax10 = fig.add_subplot(3, 4, 10)
good = success[~success["is_outlier"]]
if len(good) > 0:
    ax10.hist(good["mean_drift"], bins=20, edgecolor="black", alpha=0.7, color="green")
    ax10.axvline(
        good["mean_drift"].mean(),
        color="red",
        ls="-",
        lw=2,
        label=f"Mean: {good['mean_drift'].mean():.1f}m",
    )
    ax10.axvline(
        good["mean_drift"].median(),
        color="blue",
        ls="--",
        lw=2,
        label=f"Median: {good['mean_drift'].median():.1f}m",
    )
ax10.set_xlabel("Mean Drift (m)")
ax10.set_ylabel("Count")
ax10.set_title(f"Good Reaches Only (n={len(good)})")
ax10.legend()

# 11. Failure analysis
ax11 = fig.add_subplot(3, 4, 11)
failed = df[df["status"] == "failed"]
if len(failed) > 0:
    error_counts = failed["error"].value_counts()
    # Shorten error messages
    short_errors = [e[:30] + "..." if len(e) > 30 else e for e in error_counts.index]
    ax11.barh(range(len(error_counts)), error_counts.values, color="red", alpha=0.7)
    ax11.set_yticks(range(len(error_counts)))
    ax11.set_yticklabels(short_errors, fontsize=8)
    ax11.set_xlabel("Count")
    ax11.set_title(f"Failure Types (n={len(failed)})")
else:
    ax11.text(
        0.5, 0.5, "No failures", ha="center", va="center", transform=ax11.transAxes
    )

# 12. Summary stats table
ax12 = fig.add_subplot(3, 4, 12)
ax12.axis("off")
stats_text = f"""
SUMMARY STATISTICS
------------------

Total processed: {len(df)}
Success: {len(success)} ({100 * len(success) / len(df):.0f}%)
Failed: {len(failed)} ({100 * len(failed) / len(df):.0f}%)

Good (<500m drift): {len(good)} ({100 * len(good) / len(success):.0f}% of success)
Outliers (>500m): {len(success[success["is_outlier"]])} ({100 * len(success[success["is_outlier"]]) / len(success):.0f}% of success)

Good reaches stats:
  Mean drift: {good["mean_drift"].mean():.1f}m
  Median drift: {good["mean_drift"].median():.1f}m
  Mean pct_ok: {good["pct_ok"].mean():.1f}%

By width (good only):
  narrow: {len(good[good["width_bin"] == "narrow"])} reaches
  medium: {len(good[good["width_bin"] == "medium"])} reaches
  wide: {len(good[good["width_bin"] == "wide"])} reaches
  very_wide: {len(good[good["width_bin"] == "very_wide"])} reaches

NEW FILTERS WOULD SKIP:
  slope < 1e-5: {len(success[success["slope"] < 1e-5])} reaches
"""
ax12.text(
    0.05,
    0.95,
    stats_text,
    transform=ax12.transAxes,
    fontsize=10,
    verticalalignment="top",
    fontfamily="monospace",
)

plt.suptitle(
    "SWORD Updater Stratified Sample Results (24 reaches)",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("/tmp/stratified_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved to /tmp/stratified_analysis.png")

# Also create a detailed table
print("\n" + "=" * 80)
print("DETAILED RESULTS")
print("=" * 80)
print(
    f"\n{'Reach ID':<15} {'Width':>8} {'Slope':>12} {'N_Chan':>7} {'Drift':>8} {'%OK':>6} {'Status':<10}"
)
print("-" * 80)
for _, r in df.iterrows():
    if r["status"] == "success":
        status = (
            "GOOD"
            if r["mean_drift"] < 60
            else "MODERATE"
            if r["mean_drift"] < 500
            else "OUTLIER"
        )
        print(
            f"{r['reach_id']:<15} {r['width']:>7.0f}m {r['slope']:>12.2e} {r['n_chan']:>7} {r['mean_drift']:>7.1f}m {r['pct_ok']:>5.1f}% {status:<10}"
        )
    else:
        print(
            f"{r['reach_id']:<15} {r['width']:>7.0f}m {r['slope']:>12.2e} {r['n_chan']:>7} {'--':>8} {'--':>6} FAILED"
        )
