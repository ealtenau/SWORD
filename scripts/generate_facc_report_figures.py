#!/usr/bin/env python3
"""
Generate figures for the facc correction technical report.

Reads v3 correction CSVs from output/facc_detection/ and produces 4 figures:
  1. v17b vs v17c scatter (log-log, colored by correction type)
  2. Correction type breakdown (stacked bar by region)
  3. Per-reach relative change distribution (histogram by region)
  4. Scalability comparison (conceptual diagram)

Usage:
    python scripts/generate_facc_report_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

REGIONS = ["NA", "SA", "EU", "AF", "AS", "OC"]
INPUT_DIR = Path("output/facc_detection")
OUTPUT_DIR = Path("output/facc_detection/figures")

# Consistent color palette for correction types
TYPE_COLORS = {
    "isotonic_regression": "#4C72B0",
    "junction_floor": "#55A868",
    "junction_floor_post": "#8172B2",
    "bifurc_share": "#C44E52",
    "lateral_lower": "#CCB974",
    "node_denoise": "#64B5CD",
    "upa_resample": "#DD8452",
    "t003_flagged_only": "#AAAAAA",
    "cascade_zero": "#999999",
    "unknown": "#DDDDDD",
}

# Friendly labels
TYPE_LABELS = {
    "isotonic_regression": "Isotonic regression",
    "junction_floor": "Junction floor (phase 2)",
    "junction_floor_post": "Junction floor (post-isotonic)",
    "bifurc_share": "Bifurcation split",
    "lateral_lower": "Lateral lower (cascade)",
    "node_denoise": "Node denoise",
    "upa_resample": "UPA re-sample",
    "t003_flagged_only": "T003 flagged only",
}


def load_all_csvs() -> pd.DataFrame:
    """Load and concatenate v3 correction CSVs for all regions."""
    frames = []
    for region in REGIONS:
        csv_path = INPUT_DIR / f"facc_denoise_v3_{region}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
            # Ensure region column is filled (pandas reads "NA" as NaN)
            if df["region"].isna().any() or (df["region"] == "").any():
                df["region"] = region
            frames.append(df)
        else:
            print(f"  WARNING: {csv_path} not found, skipping")
    if not frames:
        raise FileNotFoundError("No v3 correction CSVs found")
    return pd.concat(frames, ignore_index=True)


def load_summaries() -> dict:
    """Load summary JSONs for all regions."""
    summaries = {}
    for region in REGIONS:
        path = INPUT_DIR / f"facc_denoise_v3_summary_{region}.json"
        if path.exists():
            with open(path) as f:
                summaries[region] = json.load(f)
    return summaries


def fig1_scatter(df: pd.DataFrame) -> None:
    """Fig 1: v17b vs v17c scatter, log-log, colored by correction type."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot unchanged reaches (not in corrections) as background reference
    # We only have corrected reaches in the CSV, so plot them by type

    # Filter to reaches with positive values on both axes
    mask = (df["original_facc"] > 0) & (df["corrected_facc"] > 0)
    plot_df = df[mask].copy()

    # Major correction types to show (combine minor ones)
    major_types = [
        "isotonic_regression",
        "bifurc_share",
        "junction_floor",
        "junction_floor_post",
        "lateral_lower",
        "upa_resample",
    ]

    # Plot minor types first (background)
    minor_mask = ~plot_df["correction_type"].isin(major_types)
    if minor_mask.any():
        ax.scatter(
            plot_df.loc[minor_mask, "original_facc"],
            plot_df.loc[minor_mask, "corrected_facc"],
            s=3, alpha=0.3, c="#CCCCCC", label="Other", zorder=1,
            rasterized=True,
        )

    # Plot major types
    for ctype in major_types:
        sub = plot_df[plot_df["correction_type"] == ctype]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["original_facc"],
            sub["corrected_facc"],
            s=4, alpha=0.4,
            c=TYPE_COLORS.get(ctype, "#999999"),
            label=TYPE_LABELS.get(ctype, ctype),
            zorder=2,
            rasterized=True,
        )

    # 1:1 reference line
    lims = [1, max(plot_df["original_facc"].max(), plot_df["corrected_facc"].max()) * 1.5]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, zorder=3, label="1:1 line")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, lims[1])
    ax.set_ylim(1, lims[1])
    ax.set_xlabel("v17b facc (km$^2$)", fontsize=12)
    ax.set_ylabel("v17c facc (km$^2$)", fontsize=12)
    ax.set_title("Fig 1: v17b vs v17c Flow Accumulation (Corrected Reaches)", fontsize=13)
    ax.legend(fontsize=8, loc="upper left", markerscale=3, framealpha=0.9)
    ax.set_aspect("equal")

    fig.tight_layout()
    out = OUTPUT_DIR / "report_fig1.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig2_stacked_bar(summaries: dict) -> None:
    """Fig 2: Correction type breakdown, stacked bar by region."""
    # Extract by_type counts from summaries
    type_order = [
        "bifurc_share",
        "lateral_lower",
        "junction_floor",
        "junction_floor_post",
        "isotonic_regression",
        "upa_resample",
        "node_denoise",
        "t003_flagged_only",
    ]

    data = {ctype: [] for ctype in type_order}
    regions_present = []

    for region in REGIONS:
        if region not in summaries:
            continue
        regions_present.append(region)
        by_type = summaries[region].get("by_type", {})
        for ctype in type_order:
            count = by_type.get(ctype, {}).get("count", 0)
            data[ctype].append(count)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(regions_present))
    width = 0.6
    bottom = np.zeros(len(regions_present))

    for ctype in type_order:
        vals = np.array(data[ctype], dtype=float)
        if vals.sum() == 0:
            continue
        ax.bar(
            x, vals, width, bottom=bottom,
            label=TYPE_LABELS.get(ctype, ctype),
            color=TYPE_COLORS.get(ctype, "#999999"),
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(regions_present, fontsize=12)
    ax.set_ylabel("Number of corrections", fontsize=12)
    ax.set_title("Fig 2: Correction Type Breakdown by Region", fontsize=13)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.tight_layout()
    out = OUTPUT_DIR / "report_fig2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig3_change_distribution(df: pd.DataFrame) -> None:
    """Fig 3: Per-reach relative change distribution, faceted by region."""
    # Filter to reaches with meaningful changes
    mask = (df["original_facc"] > 0) & (df["delta_pct"].notna()) & (df["delta_pct"].abs() < 1e6)
    plot_df = df[mask].copy()

    # Clip extreme values for visualization
    plot_df["delta_pct_clipped"] = plot_df["delta_pct"].clip(-200, 200)

    regions_present = [r for r in REGIONS if r in plot_df["region"].values]
    n = len(regions_present)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), squeeze=False)

    for i, region in enumerate(regions_present):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        sub = plot_df[plot_df["region"] == region]["delta_pct_clipped"]

        ax.hist(sub, bins=80, color="#4C72B0", alpha=0.8, edgecolor="none")
        ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)

        median_val = sub.median()
        ax.axvline(median_val, color="#C44E52", ls="-", lw=1.2, alpha=0.8)

        ax.set_title(f"{region} (n={len(sub):,}, med={median_val:+.1f}%)", fontsize=11)
        ax.set_xlabel("% change from v17b", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Hide unused axes
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Fig 3: Per-Reach Relative Change Distribution (clipped to +/-200%)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / "report_fig3.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig4_scalability(summaries: dict) -> None:
    """Fig 4: Scalability comparison — runtime and reach count."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: conceptual complexity comparison
    basin_sizes = [10, 50, 100, 500, 1000, 5000]
    # Integrator: O(m^3) per basin, need to process ~N/m basins
    # Pipeline: O(N) total
    N = 248674

    integrator_ops = [m ** 3 * (N / m) for m in basin_sizes]  # total ops ~ N * m^2
    pipeline_ops = [N] * len(basin_sizes)  # constant

    ax1.plot(basin_sizes, integrator_ops, "o-", color="#C44E52", lw=2,
             label="Integrator (O(N*m$^2$) total)", markersize=6)
    ax1.plot(basin_sizes, pipeline_ops, "s-", color="#4C72B0", lw=2,
             label="v3 Pipeline (O(N) total)", markersize=6)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Basin size (reaches per basin)", fontsize=11)
    ax1.set_ylabel("Total operations", fontsize=11)
    ax1.set_title("Computational Complexity", fontsize=12)
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Right panel: actual corrections per region with reach context
    regions_present = [r for r in REGIONS if r in summaries]
    total_reaches = [summaries[r]["total_reaches"] for r in regions_present]
    corrections = [summaries[r]["corrections"] for r in regions_present]
    pct_corrected = [100 * c / t for c, t in zip(corrections, total_reaches)]

    x = np.arange(len(regions_present))
    bars1 = ax2.bar(x - 0.2, total_reaches, 0.35, label="Total reaches",
                    color="#4C72B0", alpha=0.7)
    bars2 = ax2.bar(x + 0.2, corrections, 0.35, label="Corrections",
                    color="#C44E52", alpha=0.7)

    # Annotate with percentage
    for i, (xi, pct) in enumerate(zip(x, pct_corrected)):
        ax2.text(xi + 0.2, corrections[i] + 500, f"{pct:.0f}%",
                 ha="center", va="bottom", fontsize=8, color="#C44E52")

    ax2.set_xticks(x)
    ax2.set_xticklabels(regions_present, fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Reaches and Corrections by Region", fontsize=12)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.suptitle("Fig 4: Scalability — v3 Pipeline Processes 248K Reaches Globally",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / "report_fig4.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading v3 correction CSVs...")
    df = load_all_csvs()
    print(f"  {len(df):,} total correction rows across {df['region'].nunique()} regions")

    print("Loading summary JSONs...")
    summaries = load_summaries()
    print(f"  {len(summaries)} region summaries loaded")

    print("\nGenerating figures...")
    fig1_scatter(df)
    fig2_stacked_bar(summaries)
    fig3_change_distribution(df)
    fig4_scalability(summaries)

    print("\nDone. Figures saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
