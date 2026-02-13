#!/usr/bin/env python3
"""
Compare the CVXPY integrator approach with our v3 denoise pipeline
on the Willamette River basin (basin ID 7822, 55 reaches).

Runs the integrator WITHOUT anchors (no constrain_rids) for a fairer
comparison since v3 doesn't use anchors either.

Outputs:
  - output/facc_detection/willamette_comparison.csv
  - output/facc_detection/figures/report_fig5.png

Usage:
    python scripts/compare_willamette_integrator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path so we can import DrainageAreaFix
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from DrainageAreaFix.fix_drainage_area import fix_drainage_area
from DrainageAreaFix.sfoi_utils import (
    CreateJunctionList,
    extract_sword,
    get_all_sword_reach_in_basin,
    get_junction_downflows,
    table_iteration,
)

SWORD_NC = PROJECT_ROOT / "data" / "netcdf" / "na_sword_v17b.nc"
V17B_DB = PROJECT_ROOT / "data" / "duckdb" / "sword_v17b.duckdb"
V17C_DB = PROJECT_ROOT / "data" / "duckdb" / "sword_v17c.duckdb"
OUTPUT_DIR = PROJECT_ROOT / "output" / "facc_detection"
FIG_DIR = OUTPUT_DIR / "figures"
BASIN = "7822"


def run_integrator() -> pd.DataFrame:
    """Run the CVXPY integrator on basin 7822 without anchors."""
    print("=== Running CVXPY Integrator on Willamette Basin ===")

    # Module 1: Load SWORD NetCDF and identify basin reaches
    sword_dict = extract_sword(SWORD_NC.parent, fname=SWORD_NC.name)
    basin_dict = {"basin_id": BASIN}
    basin_dict = get_all_sword_reach_in_basin(basin_dict, sword_dict)
    basin_dict["reach_ids_all_reorder"] = basin_dict["reach_ids_all"]

    reachids = basin_dict["reach_ids_all"]
    print(f"  Total reaches in basin: {len(reachids)}")

    # Module 2: Create junction list and classify
    junctions = CreateJunctionList(basin_dict, sword_dict, list_order="reorder")
    for i, junction in enumerate(junctions):
        junction["row_num"] = i

    l = len(junctions)  # dependent
    m = len(reachids)  # total
    n = m - l  # independent
    print(f"  Dependent: {l}, Independent: {n}, Total: {m}")

    reachids_reorder = basin_dict["reach_ids_all_reorder"]
    dependent_reachids = reachids_reorder[:l]
    independent_reachids = reachids_reorder[l:]

    # Module 3: Build matrices
    D_reorder = []
    delta_D = []
    for rid in reachids_reorder:
        k = np.where(sword_dict["reach_id"][:] == np.int64(rid))[0][0]
        val = sword_dict["facc"][k]
        D_reorder.append(val)
        delta_D.append(val)

    Adrain = np.zeros((l, l))
    sumDind = np.zeros((l,))
    junction_downflows = get_junction_downflows(junctions)

    for i in range(l):
        reachid = reachids_reorder[i]

        if reachid in junction_downflows:
            start_jdx = junction_downflows.index(reachid)
            updf = pd.DataFrame(
                columns=[
                    "reachid",
                    "upstream reachids",
                    "upstream ids added",
                    "drainage area",
                ]
            )
            updf.at[0, "reachid"] = reachid
            updf.at[0, "upstream reachids"] = [
                str(rid) for rid in junctions[start_jdx]["upflows"]
            ]
            updf.at[0, "upstream ids added"] = False
            updf.at[0, "drainage area"] = delta_D[i]

            indices_not_done = list(
                updf[~updf["upstream ids added"].astype(bool)].index
            )
            while indices_not_done:
                updf, indices_not_done = table_iteration(
                    indices_not_done,
                    updf,
                    junctions,
                    junction_downflows,
                    delta_D,
                    reachids_reorder,
                )
            rids = list(updf["reachid"])
        else:
            rids = [reachid]

        for rid in rids:
            if rid in reachids_reorder:
                idx = reachids_reorder.index(rid)
                if rid in dependent_reachids:
                    if idx < l:
                        Adrain[i, idx] = 1
                elif rid in independent_reachids:
                    sumDind[i] += D_reorder[idx]

    # Module 4: Run solver WITHOUT anchors
    print("  Running CVXPY solver (no anchors)...")
    Dhat, xhat, Ddf = fix_drainage_area(
        l,
        D_reorder,
        sumDind,
        Adrain,
        dependent_reachids,
        independent_reachids,
        constrain_rids=None,  # No anchors for fair comparison
    )

    if Dhat is None:
        print("  ERROR: Solver failed!")
        sys.exit(1)

    # Build result DataFrame with all reaches (dependent + independent)
    results = []
    for rid in reachids_reorder:
        k = np.where(sword_dict["reach_id"][:] == np.int64(rid))[0][0]
        v17b_facc = float(sword_dict["facc"][k])

        if rid in dependent_reachids:
            idx = dependent_reachids.index(rid)
            integrator_facc = float(Dhat[idx])
        else:
            # Independent reaches keep their original value
            integrator_facc = v17b_facc

        results.append(
            {
                "reach_id": int(rid),
                "v17b_facc": v17b_facc,
                "integrator_facc": integrator_facc,
            }
        )

    print(f"  Solver converged. {len(results)} reaches processed.")
    return pd.DataFrame(results)


def get_v3_corrections() -> pd.DataFrame:
    """Get v3 corrected facc from DuckDB for basin 7822 reaches."""
    print("\n=== Querying v3 Corrections from DuckDB ===")
    con = duckdb.connect(str(V17C_DB), read_only=True)
    try:
        df = con.execute("""
            SELECT reach_id, facc as v3_facc
            FROM reaches
            WHERE region = 'NA'
              AND CAST(reach_id AS VARCHAR) LIKE '7822%'
        """).fetchdf()
    finally:
        con.close()
    print(f"  Found {len(df)} reaches in v17c for basin 7822")
    return df


def get_v17b_reference() -> pd.DataFrame:
    """Get v17b original facc from DuckDB for basin 7822 reaches."""
    con = duckdb.connect(str(V17B_DB), read_only=True)
    try:
        df = con.execute("""
            SELECT reach_id, facc as v17b_facc_db, width
            FROM reaches
            WHERE region = 'NA'
              AND CAST(reach_id AS VARCHAR) LIKE '7822%'
        """).fetchdf()
    finally:
        con.close()
    return df


def generate_fig6(comparison: pd.DataFrame) -> None:
    """Generate Fig 6: Integrator vs v3 scatter + per-reach bar chart."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: Scatter plot (integrator vs v3) ---
    ax1.scatter(
        comparison["integrator_facc"],
        comparison["v3_facc"],
        c="#4C72B0",
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
        s=40,
        zorder=3,
    )

    # 1:1 line
    max_val = max(comparison["integrator_facc"].max(), comparison["v3_facc"].max())
    min_val = min(
        comparison["integrator_facc"][comparison["integrator_facc"] > 0].min(),
        comparison["v3_facc"][comparison["v3_facc"] > 0].min(),
    )
    ax1.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        lw=1.5,
        alpha=0.7,
        label="1:1 line",
    )

    ax1.set_xlabel("Integrator facc (km$^2$)", fontsize=11)
    ax1.set_ylabel("v3 Pipeline facc (km$^2$)", fontsize=11)
    ax1.set_title(
        "Integrator vs v3 Pipeline\n(Willamette Basin, 55 reaches)", fontsize=12
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Annotate correlation
    corr = comparison[["integrator_facc", "v3_facc"]].corr().iloc[0, 1]
    ax1.text(
        0.05,
        0.92,
        f"r = {corr:.4f}",
        transform=ax1.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # --- Right: Bar chart of per-reach differences ---
    # Only show reaches where at least one method changed the value
    changed = comparison[
        (comparison["integrator_delta"].abs() > 0.1)
        | (comparison["v3_delta"].abs() > 0.1)
    ].copy()
    changed = changed.sort_values("v17b_facc", ascending=True).reset_index(drop=True)

    if len(changed) > 0:
        x = np.arange(len(changed))
        width = 0.35

        ax2.barh(
            x - width / 2,
            changed["integrator_delta_pct"],
            width,
            label="Integrator",
            color="#C44E52",
            alpha=0.8,
        )
        ax2.barh(
            x + width / 2,
            changed["v3_delta_pct"],
            width,
            label="v3 Pipeline",
            color="#4C72B0",
            alpha=0.8,
        )

        ax2.set_yticks(x)
        reach_labels = [str(rid)[-5:] for rid in changed["reach_id"]]
        ax2.set_yticklabels(reach_labels, fontsize=7)
        ax2.set_xlabel("% change from v17b", fontsize=11)
        ax2.set_ylabel("Reach ID (last 5 digits)", fontsize=11)
        ax2.set_title(
            f"Per-Reach Changes\n({len(changed)} reaches with corrections)", fontsize=12
        )
        ax2.axvline(0, color="k", lw=0.8, alpha=0.5)
        ax2.legend(fontsize=9)
        ax2.grid(True, axis="x", alpha=0.2)

    fig.suptitle(
        "Fig 6: Willamette Basin — Integrator vs v3 Pipeline Comparison",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out = FIG_DIR / "report_fig5.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {out}")


def main():
    # Step 1: Run integrator
    integrator_df = run_integrator()

    # Step 2: Get v3 corrections from DuckDB
    v3_df = get_v3_corrections()

    # Step 3: Get v17b reference + width
    v17b_df = get_v17b_reference()

    # Step 4: Merge everything
    comparison = integrator_df.merge(v3_df, on="reach_id", how="left")
    comparison = comparison.merge(
        v17b_df[["reach_id", "width"]], on="reach_id", how="left"
    )

    # Compute deltas
    comparison["integrator_delta"] = (
        comparison["integrator_facc"] - comparison["v17b_facc"]
    )
    comparison["v3_delta"] = comparison["v3_facc"] - comparison["v17b_facc"]
    comparison["integrator_delta_pct"] = (
        100
        * comparison["integrator_delta"]
        / comparison["v17b_facc"].replace(0, np.nan)
    )
    comparison["v3_delta_pct"] = (
        100 * comparison["v3_delta"] / comparison["v17b_facc"].replace(0, np.nan)
    )
    comparison["method_diff"] = comparison["integrator_facc"] - comparison["v3_facc"]
    comparison["method_diff_pct"] = (
        100 * comparison["method_diff"] / comparison["v17b_facc"].replace(0, np.nan)
    )

    # Sort by v17b facc for readability
    comparison = comparison.sort_values("v17b_facc", ascending=False).reset_index(
        drop=True
    )

    # Step 5: Save CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "willamette_comparison.csv"
    comparison.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path}")

    # Step 6: Summary stats
    n_integrator_changed = (comparison["integrator_delta"].abs() > 0.1).sum()
    n_v3_changed = (comparison["v3_delta"].abs() > 0.1).sum()
    n_both_changed = (
        (comparison["integrator_delta"].abs() > 0.1)
        & (comparison["v3_delta"].abs() > 0.1)
    ).sum()

    print("\n=== Summary ===")
    print(f"  Total reaches: {len(comparison)}")
    print(f"  Integrator changed: {n_integrator_changed}")
    print(f"  v3 changed: {n_v3_changed}")
    print(f"  Both changed: {n_both_changed}")
    print(f"  Mean method diff: {comparison['method_diff'].mean():.2f} km²")
    print(f"  Median method diff: {comparison['method_diff'].median():.2f} km²")
    print(
        f"  Correlation: {comparison[['integrator_facc', 'v3_facc']].corr().iloc[0, 1]:.6f}"
    )

    # Count agreement on direction
    both_changed_mask = (comparison["integrator_delta"].abs() > 0.1) & (
        comparison["v3_delta"].abs() > 0.1
    )
    if both_changed_mask.sum() > 0:
        agree_direction = (
            np.sign(comparison.loc[both_changed_mask, "integrator_delta"])
            == np.sign(comparison.loc[both_changed_mask, "v3_delta"])
        ).sum()
        print(
            f"  Direction agreement (where both changed): "
            f"{agree_direction}/{both_changed_mask.sum()} "
            f"({100 * agree_direction / both_changed_mask.sum():.0f}%)"
        )

    # Step 7: Generate figure
    generate_fig6(comparison)

    print("\nDone.")


if __name__ == "__main__":
    main()
