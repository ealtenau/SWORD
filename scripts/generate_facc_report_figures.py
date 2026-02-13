#!/usr/bin/env python3
"""
Generate figures for the facc correction technical report.

Reads v3 correction CSVs, summary JSONs, and DuckDB databases to produce
4 figures demonstrating why v17b facc is broken and how v17c fixes it.

  1. Before/after: junction conservation + bifurcation cloning (the core evidence)
  2. Correction type breakdown (stacked bar by region)
  3. Per-reach relative change distribution (histogram by region)
  4. Scalability comparison (conceptual diagram)

Usage:
    python scripts/generate_facc_report_figures.py

Requires: DuckDB files at data/duckdb/sword_v17b.duckdb and sword_v17c.duckdb
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

REGIONS = ["NA", "SA", "EU", "AF", "AS", "OC"]
INPUT_DIR = Path("output/facc_detection")
OUTPUT_DIR = Path("output/facc_detection/figures")
V17B_DB = Path("data/duckdb/sword_v17b.duckdb")
V17C_DB = Path("data/duckdb/sword_v17c.duckdb")

# Consistent color palette for correction types
TYPE_COLORS = {
    "lateral_propagate": "#CCB974",
    "junction_floor": "#55A868",
    "baseline_isotonic": "#8172B2",
    "bifurc_share": "#C44E52",
    "final_1to1": "#4C72B0",
    "final_bifurc": "#DD8452",
    "final_junction": "#64B5CD",
    "node_denoise": "#999999",
    "baseline_node_override": "#AAAAAA",
    "node_max_override": "#DDDDDD",
}

# Friendly labels
TYPE_LABELS = {
    "lateral_propagate": "Lateral propagate (Stage B)",
    "junction_floor": "Junction floor (Stage B)",
    "baseline_isotonic": "Baseline isotonic (Stage A)",
    "bifurc_share": "Bifurcation split (Stage B)",
    "final_1to1": "Final 1:1 consistency",
    "final_bifurc": "Final bifurcation consistency",
    "final_junction": "Final junction consistency",
    "node_denoise": "Node denoise (Stage A)",
    "baseline_node_override": "Baseline node override (Stage A)",
    "node_max_override": "Node max override",
}

V17B_COLOR = "#C44E52"  # red for broken
V17C_COLOR = "#4C72B0"  # blue for fixed


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


def _query_junction_conservation(db_path: Path) -> pd.DataFrame:
    """Query junction conservation ratios from a DuckDB database."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            WITH junction_upstream AS (
                SELECT
                    t.reach_id,
                    r.facc as junction_facc,
                    r.region,
                    SUM(r2.facc) as sum_upstream_facc,
                    COUNT(*) as n_upstream
                FROM reach_topology t
                JOIN reaches r ON t.reach_id = r.reach_id AND t.region = r.region
                JOIN reaches r2 ON t.neighbor_reach_id = r2.reach_id AND t.region = r2.region
                WHERE t.direction = 'up'
                GROUP BY t.reach_id, r.facc, r.region
                HAVING COUNT(*) >= 2
            )
            SELECT
                reach_id, region, junction_facc, sum_upstream_facc, n_upstream,
                CASE WHEN sum_upstream_facc > 0
                     THEN junction_facc / sum_upstream_facc
                     ELSE NULL END as conservation_ratio
            FROM junction_upstream
            WHERE sum_upstream_facc > 0
        """).fetchdf()
    finally:
        con.close()
    return df


def _query_bifurc_children(db_path: Path) -> pd.DataFrame:
    """Query bifurcation child/parent ratios from a DuckDB database."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            WITH bifurc_children AS (
                SELECT
                    t.reach_id as parent_id,
                    t.neighbor_reach_id as child_id,
                    r_parent.facc as parent_facc,
                    r_child.facc as child_facc,
                    r_child.width as child_width,
                    r_parent.region
                FROM reach_topology t
                JOIN reaches r_parent ON t.reach_id = r_parent.reach_id
                    AND t.region = r_parent.region
                JOIN reaches r_child ON t.neighbor_reach_id = r_child.reach_id
                    AND t.region = r_child.region
                WHERE t.direction = 'down'
                AND t.reach_id IN (
                    SELECT reach_id
                    FROM reach_topology
                    WHERE direction = 'down'
                    GROUP BY reach_id
                    HAVING COUNT(*) >= 2
                )
            )
            SELECT *,
                CASE WHEN parent_facc > 0
                     THEN child_facc / parent_facc
                     ELSE NULL END as child_parent_ratio
            FROM bifurc_children
            WHERE parent_facc > 0
        """).fetchdf()
    finally:
        con.close()
    return df


def fig1_before_after(
    junc_v17b: pd.DataFrame,
    junc_v17c: pd.DataFrame,
    bifurc_v17b: pd.DataFrame,
    bifurc_v17c: pd.DataFrame,
) -> None:
    """Fig 1: Before/after evidence — junction conservation + bifurcation cloning."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ---- Top left: Junction conservation ratio histogram (v17b) ----
    ax = axes[0][0]
    ratios_b = junc_v17b["conservation_ratio"].dropna().clip(0, 3)
    n_viol_b = (junc_v17b["conservation_ratio"].dropna() < 1.0).sum()
    n_total_b = len(junc_v17b["conservation_ratio"].dropna())

    ax.hist(ratios_b, bins=120, color=V17B_COLOR, alpha=0.8, edgecolor="none")
    ax.axvline(1.0, color="k", ls="--", lw=1.5, alpha=0.7)
    ax.fill_betweenx([0, ax.get_ylim()[1] * 1.5], 0, 1.0, alpha=0.08, color=V17B_COLOR)
    ax.set_xlim(0, 3)
    ax.set_xlabel("junction facc / sum(upstream facc)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(
        f"v17b: Junction Conservation\n"
        f"{n_viol_b:,} / {n_total_b:,} junctions violate conservation "
        f"({100 * n_viol_b / n_total_b:.0f}%)",
        fontsize=11,
        color=V17B_COLOR,
        fontweight="bold",
    )
    # Arrow pointing into the violation zone (left of x=1.0)
    ylim = ax.get_ylim()
    ax.annotate(
        "VIOLATION ZONE\n(ratio < 1.0)",
        xy=(0.5, ylim[1] * 0.3),
        xytext=(1.8, ylim[1] * 0.7),
        fontsize=10,
        color=V17B_COLOR,
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=V17B_COLOR, lw=2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=V17B_COLOR, alpha=0.9),
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # ---- Top right: Junction conservation ratio histogram (v17c) ----
    ax = axes[0][1]
    ratios_c = junc_v17c["conservation_ratio"].dropna().clip(0, 3)
    n_viol_c = (junc_v17c["conservation_ratio"].dropna() < (1.0 - 1e-3)).sum()
    n_total_c = len(junc_v17c["conservation_ratio"].dropna())

    ax.hist(ratios_c, bins=120, color=V17C_COLOR, alpha=0.8, edgecolor="none")
    ax.axvline(1.0, color="k", ls="--", lw=1.5, alpha=0.7)
    ax.fill_betweenx([0, ax.get_ylim()[1] * 1.5], 0, 1.0, alpha=0.08, color=V17C_COLOR)
    ax.set_xlim(0, 3)
    ax.set_xlabel("junction facc / sum(upstream facc)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(
        f"v17c: Junction Conservation\n"
        f"{n_viol_c:,} / {n_total_c:,} junctions violate conservation "
        f"({100 * n_viol_c / n_total_c:.1f}%)",
        fontsize=11,
        color=V17C_COLOR,
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # ---- Bottom left: Bifurcation child/parent ratio (v17b) ----
    ax = axes[1][0]
    ratios_b = bifurc_v17b["child_parent_ratio"].dropna().clip(0, 2)
    n_cloned_b = (
        (bifurc_v17b["child_parent_ratio"].dropna() > 0.9)
        & (bifurc_v17b["child_parent_ratio"].dropna() < 1.1)
    ).sum()
    n_bif_b = len(bifurc_v17b["child_parent_ratio"].dropna())

    ax.hist(ratios_b, bins=100, color=V17B_COLOR, alpha=0.8, edgecolor="none")
    ax.axvline(1.0, color="k", ls="--", lw=1.5, alpha=0.7, label="Ratio = 1.0 (cloned)")
    # Shade the cloning zone
    ax.axvspan(0.9, 1.1, alpha=0.15, color=V17B_COLOR)
    ax.set_xlim(0, 2)
    ax.set_xlabel("child facc / parent facc", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(
        f"v17b: Bifurcation Child/Parent Ratio\n"
        f"{n_cloned_b:,} / {n_bif_b:,} children cloned at ratio ~1.0 "
        f"({100 * n_cloned_b / n_bif_b:.0f}%)",
        fontsize=11,
        color=V17B_COLOR,
        fontweight="bold",
    )
    # Arrow pointing at the cloning spike near ratio=1.0
    ylim_b = ax.get_ylim()
    ax.annotate(
        "UPA CLONING\nPEAK",
        xy=(1.0, ylim_b[1] * 0.6),
        xytext=(1.5, ylim_b[1] * 0.8),
        fontsize=10,
        color=V17B_COLOR,
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=V17B_COLOR, lw=2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=V17B_COLOR, alpha=0.9),
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # ---- Bottom right: Bifurcation child/parent ratio (v17c) ----
    ax = axes[1][1]
    ratios_c = bifurc_v17c["child_parent_ratio"].dropna().clip(0, 2)
    n_cloned_c = (
        (bifurc_v17c["child_parent_ratio"].dropna() > 0.9)
        & (bifurc_v17c["child_parent_ratio"].dropna() < 1.1)
    ).sum()
    n_bif_c = len(bifurc_v17c["child_parent_ratio"].dropna())
    median_c = bifurc_v17c["child_parent_ratio"].dropna().median()

    ax.hist(ratios_c, bins=100, color=V17C_COLOR, alpha=0.8, edgecolor="none")
    ax.axvline(1.0, color="k", ls="--", lw=1.5, alpha=0.7)
    ax.axvline(
        median_c,
        color=V17C_COLOR,
        ls="-",
        lw=2,
        alpha=0.8,
        label=f"Median = {median_c:.2f}",
    )
    ax.axvspan(0.9, 1.1, alpha=0.08, color="#AAAAAA")
    ax.set_xlim(0, 2)
    ax.set_xlabel("child facc / parent facc", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(
        f"v17c: Bifurcation Child/Parent Ratio\n"
        f"Median = {median_c:.2f} (width-proportional split), "
        f"only {n_cloned_c:,} near 1.0 ({100 * n_cloned_c / n_bif_c:.1f}%)",
        fontsize=11,
        color=V17C_COLOR,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.suptitle(
        "Fig 1: v17b Has Systematic Errors — v17c Fixes Them",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out = OUTPUT_DIR / "report_fig1.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig2_stacked_bar(summaries: dict) -> None:
    """Fig 2: Correction type breakdown, stacked bar by region."""
    type_order = [
        "lateral_propagate",
        "junction_floor",
        "baseline_isotonic",
        "bifurc_share",
        "final_1to1",
        "final_bifurc",
        "final_junction",
        "node_denoise",
        "baseline_node_override",
        "node_max_override",
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
            x,
            vals,
            width,
            bottom=bottom,
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
    mask = (
        (df["original_facc"] > 0)
        & (df["delta_pct"].notna())
        & (df["delta_pct"].abs() < 1e6)
    )
    plot_df = df[mask].copy()
    plot_df["delta_pct_clipped"] = plot_df["delta_pct"].clip(-500, 500)

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

    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Fig 3: Per-Reach Relative Change Distribution (clipped to +/-500%)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    out = OUTPUT_DIR / "report_fig3.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig4_scalability(summaries: dict) -> None:
    """Fig 4: Scalability comparison — runtime and reach count."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    basin_sizes = [10, 50, 100, 500, 1000, 5000]
    N = 248674

    integrator_ops = [m**3 * (N / m) for m in basin_sizes]
    pipeline_ops = [N] * len(basin_sizes)

    ax1.plot(
        basin_sizes,
        integrator_ops,
        "o-",
        color="#C44E52",
        lw=2,
        label="Integrator (O(N*m$^2$) total)",
        markersize=6,
    )
    ax1.plot(
        basin_sizes,
        pipeline_ops,
        "s-",
        color="#4C72B0",
        lw=2,
        label="v3 Pipeline (O(N) total)",
        markersize=6,
    )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Basin size (reaches per basin)", fontsize=11)
    ax1.set_ylabel("Total operations", fontsize=11)
    ax1.set_title("Computational Complexity", fontsize=12)
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    regions_present = [r for r in REGIONS if r in summaries]
    total_reaches = [summaries[r]["total_reaches"] for r in regions_present]
    corrections = [summaries[r]["corrections"] for r in regions_present]
    pct_corrected = [100 * c / t for c, t in zip(corrections, total_reaches)]

    x = np.arange(len(regions_present))
    ax2.bar(
        x - 0.2, total_reaches, 0.35, label="Total reaches", color="#4C72B0", alpha=0.7
    )
    ax2.bar(x + 0.2, corrections, 0.35, label="Corrections", color="#C44E52", alpha=0.7)

    for i, (xi, pct) in enumerate(zip(x, pct_corrected)):
        ax2.text(
            xi + 0.2,
            corrections[i] + 500,
            f"{pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#C44E52",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(regions_present, fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Reaches and Corrections by Region", fontsize=12)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.suptitle(
        "Fig 4: Scalability — v3 Pipeline Processes 248K Reaches Globally",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    out = OUTPUT_DIR / "report_fig4.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig5_pava_example() -> None:
    """Fig 5: PAVA in action on a real 1:1 chain (SA example)."""
    # SA chain: 15 reaches with a clear drop from R8-R14 that PAVA pools through.
    # Data from chain starting at reach 62266502361 in South America.
    orig = [
        5589,
        5630,
        5670,
        5754,
        5836,
        5919,
        6000,
        6673,
        6242,
        5805,
        5367,
        4924,
        4493,
        4054,
        6776,
    ]
    corr = [
        6265,
        5666,
        5670,
        5754,
        5836,
        5919,
        6000,
        6448,
        6448,
        6448,
        6448,
        6523,
        6626,
        6681,
        6776,
    ]
    n = len(orig)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(11, 5))

    # Shade violation zones
    for i in range(n - 1):
        if orig[i] > orig[i + 1] * 1.001:
            ax.axvspan(i, i + 1, alpha=0.10, color=V17B_COLOR, zorder=0)

    ax.plot(
        x,
        orig,
        "o-",
        color=V17B_COLOR,
        lw=2,
        markersize=7,
        alpha=0.85,
        label="Before PAVA (Stage A baseline)",
        zorder=3,
    )
    ax.plot(
        x,
        corr,
        "s-",
        color=V17C_COLOR,
        lw=2.5,
        markersize=7,
        alpha=0.85,
        label="After PAVA (Stage A4 output)",
        zorder=4,
    )

    ax.annotate(
        "PAVA pool:\nclosest non-decreasing\nfit to this drop",
        xy=(8.5, 6448),
        xytext=(10.5, 5500),
        fontsize=8.5,
        color=V17C_COLOR,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=V17C_COLOR, lw=1.5),
        ha="center",
    )
    ax.annotate(
        "violation:\ndownstream < upstream",
        xy=(8, 6242),
        xytext=(5.5, 4300),
        fontsize=8.5,
        color=V17B_COLOR,
        arrowprops=dict(arrowstyle="->", color=V17B_COLOR, lw=1.5),
        ha="center",
    )

    # Direction arrow
    ax.annotate(
        "",
        xy=(n - 1, 7100),
        xytext=(0, 7100),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1, ls="--"),
    )
    ax.text(
        n / 2,
        7250,
        "facc should increase (downstream \u2192)",
        ha="center",
        fontsize=9,
        color="gray",
        style="italic",
    )

    ax.set_xlabel(
        "Reach position along 1:1 chain (upstream \u2192 downstream)", fontsize=11
    )
    ax.set_ylabel("Flow accumulation (km\u00b2)", fontsize=11)
    ax.set_title(
        "Fig 5: Isotonic Regression (PAVA) on a 1:1 Chain \u2014 South America Example",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"R{i + 1}" for i in x], fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(3500, 7600)
    ax.grid(True, alpha=0.2)

    import matplotlib.patches as mpatches

    viol_patch = mpatches.Patch(
        color=V17B_COLOR, alpha=0.10, label="Violation zone (facc decreases)"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + [viol_patch], fontsize=9, loc="lower right", framealpha=0.9
    )

    fig.tight_layout()
    out = OUTPUT_DIR / "report_fig5.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading v3 correction CSVs...")
    df = load_all_csvs()
    print(
        f"  {len(df):,} total correction rows across {df['region'].nunique()} regions"
    )

    print("Loading summary JSONs...")
    summaries = load_summaries()
    print(f"  {len(summaries)} region summaries loaded")

    # Query DuckDB for before/after evidence
    print("\nQuerying junction conservation from v17b...")
    junc_v17b = _query_junction_conservation(V17B_DB)
    print(f"  {len(junc_v17b):,} junctions")

    print("Querying junction conservation from v17c...")
    junc_v17c = _query_junction_conservation(V17C_DB)
    print(f"  {len(junc_v17c):,} junctions")

    print("Querying bifurcation children from v17b...")
    bifurc_v17b = _query_bifurc_children(V17B_DB)
    print(f"  {len(bifurc_v17b):,} children")

    print("Querying bifurcation children from v17c...")
    bifurc_v17c = _query_bifurc_children(V17C_DB)
    print(f"  {len(bifurc_v17c):,} children")

    print("\nGenerating figures...")
    fig1_before_after(junc_v17b, junc_v17c, bifurc_v17b, bifurc_v17c)
    fig2_stacked_bar(summaries)
    fig3_change_distribution(df)
    fig4_scalability(summaries)
    fig5_pava_example()

    print("\nDone. Figures saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
