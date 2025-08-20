#!/usr/bin/env python3
"""
Temporal visualization for WSE drop analysis.

- Loads daily-binned time-series Parquet: data/analysis/master_wse_drop_timeseries_{sample}.parquet
- Loads seasonality Parquet: data/analysis/master_wse_drop_seasonality_{sample}.parquet
- Visualizes:
  - Per-obstruction time series with rolling median
  - Monthly climatology (seasonal index)
  - Year x Month heatmap of median drop
- Can also create a multi-page PDF for Top-K obstructions by coverage

Usage examples:
  python notebooks/visualize_wse_drop_temporal.py --obstruction-id 123456
  python notebooks/visualize_wse_drop_temporal.py --top-k 20
  python notebooks/visualize_wse_drop_temporal.py --top-k 50 --sample 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "data/analysis"

# Plot style
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})


def _find_best_file(pattern: str, prefer_sample: Optional[int]) -> Optional[Path]:
    """Find the best-matching file in ANALYSIS_DIR by sample count then mtime.

    pattern example: "master_wse_drop_timeseries_*.parquet"
    """
    candidates = sorted(ANALYSIS_DIR.glob(pattern))
    if not candidates:
        return None
    def _sample_from_name(p: Path) -> int:
        name = p.stem
        # Expect trailing _{sample}
        try:
            return int(name.split("_")[-1])
        except Exception:
            # Legacy _10k
            if name.endswith("_10k"):
                return 10000
            return 0
    if prefer_sample is not None:
        exact = [p for p in candidates if _sample_from_name(p) == prefer_sample]
        if exact:
            return sorted(exact, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    # Fallback: highest sample then newest
    candidates.sort(key=lambda p: (_sample_from_name(p), p.stat().st_mtime), reverse=True)
    return candidates[0]


def load_time_series(prefer_sample: Optional[int]) -> pd.DataFrame:
    ts_path = _find_best_file("master_wse_drop_timeseries_*.parquet", prefer_sample)
    if ts_path is None:
        raise FileNotFoundError("No time-series Parquet found in data/analysis.")
    print(f"Using time-series: {ts_path}")
    ts = pd.read_parquet(ts_path)
    # Ensure expected columns exist
    required = {"obstruction_node_id", "time_bin", "wse_drop_m_time"}
    if not required.issubset(set(ts.columns)):
        raise ValueError(f"Time-series missing required columns: {required - set(ts.columns)}")
    ts["time_bin"] = pd.to_datetime(ts["time_bin"], utc=True, errors="coerce")
    return ts


def load_seasonality(prefer_sample: Optional[int]) -> Optional[pd.DataFrame]:
    seas_path = _find_best_file("master_wse_drop_seasonality_*.parquet", prefer_sample)
    if seas_path is None:
        print("No seasonality file found (optional).")
        return None
    print(f"Using seasonality: {seas_path}")
    return pd.read_parquet(seas_path)


def plot_obstruction_timeseries(
    ts: pd.DataFrame,
    obstruction_id: str,
    out_dir: Path,
    rolling_days: int = 30,
) -> Tuple[Path, Path, Path]:
    """Create per-obstruction visualizations: time series, monthly climatology, and year-month heatmap.

    Returns tuple of saved image paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = ts[ts["obstruction_node_id"].astype(str) == str(obstruction_id)].copy()
    if sub.empty:
        raise ValueError(f"No time-series rows for obstruction {obstruction_id}")

    sub = sub.sort_values("time_bin")
    # Rolling median (index must be monotonic)
    sub = sub.set_index("time_bin")
    # Daily bins expected; window in days
    roll = sub["wse_drop_m_time"].rolling(f"{rolling_days}D", min_periods=max(5, rolling_days // 6)).median()

    # 1) Time series plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sub.index, sub["wse_drop_m_time"], color="steelblue", alpha=0.7, label="daily drop")
    ax.plot(roll.index, roll.values, color="darkred", linewidth=2, label=f"{rolling_days}-day median")
    ax.axhline(0, color="gray", linewidth=1, alpha=0.5)
    ax.set_title(f"Obstruction {obstruction_id} — Daily WSE drop")
    ax.set_xlabel("Date")
    ax.set_ylabel("WSE drop (m)")
    ax.legend()
    p_ts = out_dir / f"obstruction_{obstruction_id}_timeseries.png"
    fig.savefig(p_ts)
    plt.close(fig)

    # 2) Monthly climatology (seasonal index)
    sub_month = sub.copy()
    sub_month["month"] = sub_month.index.month
    g = sub_month.groupby("month")["wse_drop_m_time"].median()
    overall = sub_month["wse_drop_m_time"].median()
    si = g - overall
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(si.index, si.values, color="teal", alpha=0.8)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xticks(range(1, 13))
    ax.set_title("Seasonal index (median_by_month − overall_median)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Seasonal index (m)")
    p_seas = out_dir / f"obstruction_{obstruction_id}_seasonal_index.png"
    fig.savefig(p_seas)
    plt.close(fig)

    # 3) Year–Month heatmap
    sub_h = sub.copy()
    sub_h["year"] = sub_h.index.year
    sub_h["month"] = sub_h.index.month
    heat = sub_h.pivot_table(index="year", columns="month", values="wse_drop_m_time", aggfunc="median")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(heat, cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Median drop (m)"})
    ax.set_title("Year × Month median WSE drop")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    p_heat = out_dir / f"obstruction_{obstruction_id}_heatmap.png"
    fig.savefig(p_heat)
    plt.close(fig)

    return p_ts, p_seas, p_heat


def select_top_k_obstructions(ts: pd.DataFrame, k: int) -> List[str]:
    cov = ts.groupby("obstruction_node_id").size().sort_values(ascending=False)
    top_ids = cov.head(k).index.astype(str).tolist()
    return top_ids


def build_spotlight_pdf(
    ts: pd.DataFrame,
    out_dir: Path,
    top_k: int = 20,
    rolling_days: int = 30,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"wse_drop_temporal_spotlights_top{top_k}.pdf"

    top_ids = select_top_k_obstructions(ts, top_k)
    with PdfPages(pdf_path) as pdf:
        for oid in top_ids:
            try:
                sub = ts[ts["obstruction_node_id"].astype(str) == oid].copy()
                sub = sub.sort_values("time_bin").set_index("time_bin")
                roll = sub["wse_drop_m_time"].rolling(f"{rolling_days}D", min_periods=max(5, rolling_days // 6)).median()

                fig, axes = plt.subplots(3, 1, figsize=(10, 10))
                # TS
                axes[0].plot(sub.index, sub["wse_drop_m_time"], color="steelblue", alpha=0.7)
                axes[0].plot(roll.index, roll.values, color="darkred", linewidth=2)
                axes[0].axhline(0, color="gray", linewidth=1, alpha=0.5)
                axes[0].set_title(f"Obstruction {oid} — Daily WSE drop (rolling {rolling_days}d)")
                axes[0].set_ylabel("Drop (m)")

                # Monthly climatology
                sub_month = sub.copy()
                sub_month["month"] = sub_month.index.month
                g = sub_month.groupby("month")["wse_drop_m_time"].median()
                overall = sub_month["wse_drop_m_time"].median()
                si = g - overall
                axes[1].bar(si.index, si.values, color="teal", alpha=0.8)
                axes[1].axhline(0, color="gray", linewidth=1)
                axes[1].set_xticks(range(1, 13))
                axes[1].set_title("Seasonal index (median_by_month − overall_median)")
                axes[1].set_xlabel("Month")
                axes[1].set_ylabel("Index (m)")

                # Heatmap
                sub_h = sub.copy()
                sub_h["year"] = sub_h.index.year
                sub_h["month"] = sub_h.index.month
                heat = sub_h.pivot_table(index="year", columns="month", values="wse_drop_m_time", aggfunc="median")
                sns.heatmap(heat, cmap="coolwarm", center=0, ax=axes[2], cbar_kws={"label": "Median (m)"})
                axes[2].set_title("Year × Month median WSE drop")
                axes[2].set_xlabel("Month")
                axes[2].set_ylabel("Year")

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            except Exception as e:
                print(f"Skipping obstruction {oid}: {e}")

    print(f"Saved spotlight PDF: {pdf_path}")
    return pdf_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize temporal WSE drop dynamics")
    p.add_argument("--sample", type=int, default=None, help="Preferred sample size to pick files (e.g., 10000)")
    p.add_argument("--obstruction-id", type=str, default=None, help="Single obstruction_id to visualize")
    p.add_argument("--top-k", type=int, default=0, help="Build a multi-page PDF for top-K obstructions by coverage")
    p.add_argument("--rolling-days", type=int, default=30, help="Rolling median window in days")
    p.add_argument("--out-dir", type=str, default=str(ANALYSIS_DIR / "temporal_viz"), help="Output directory for figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    ts = load_time_series(args.sample)
    # seasonality is optional for now
    _ = load_seasonality(args.sample)

    if args.obstruction_id:
        plot_obstruction_timeseries(ts, args.obstruction_id, out_dir, rolling_days=args.rolling_days)

    if args.top_k and args.top_k > 0:
        build_spotlight_pdf(ts, out_dir, top_k=args.top_k, rolling_days=args.rolling_days)

    if not args.obstruction_id and (not args.top_k or args.top_k <= 0):
        # Default: show a quick coverage summary
        cov = ts.groupby("obstruction_node_id").size().sort_values(ascending=False)
        print("Top 10 obstructions by daily-bin coverage:")
        print(cov.head(10))


if __name__ == "__main__":
    main()
