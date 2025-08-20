#!/usr/bin/env python3
"""
Compute per-obstruction weekly variability metrics and export GeoJSON layers.

Inputs:
- data/analysis/master_wse_drop_timeseries_weekly_{sample}.parquet
- data/analysis/master_wse_drop_crossreach_{sample}.csv (for x,y)

Outputs:
- data/analysis/weekly_variability_{sample}.geojson (all; fields: n_weeks, median_weekly, std_weekly, cv_weekly, q10, q90, range_q90_q10)
- data/analysis/weekly_variability_cv_top{pct}_{sample}.geojson (top by CV)
- data/analysis/weekly_variability_cv_bottom{pct}_{sample}.geojson (bottom by CV)
- data/analysis/weekly_variability_range_top{pct}_{sample}.geojson (top by range)
- data/analysis/weekly_variability_range_bottom{pct}_{sample}.geojson (bottom by range)

Usage:
  python notebooks/build_weekly_variability_metrics_geojson.py --sample 20734 --min-weeks 7 --pct 1
"""

import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd

ANALYSIS_DIR = Path("data/analysis")


def build(sample: int, min_weeks: int, pct: float) -> None:
    ts_path = ANALYSIS_DIR / f"master_wse_drop_timeseries_weekly_{sample}.parquet"
    base_csv = ANALYSIS_DIR / f"master_wse_drop_crossreach_{sample}.csv"
    if not ts_path.exists() or not base_csv.exists():
        raise FileNotFoundError("Required inputs not found in data/analysis/")

    ts = pd.read_parquet(ts_path)
    base = pd.read_csv(base_csv)[['obstruction_node_id','x','y','reach_id']].drop_duplicates('obstruction_node_id')

    # Aggregate weekly metrics per obstruction
    grp = ts.groupby('obstruction_node_id')
    agg = grp['wse_drop_m_time'].agg(['median','std','count']).rename(columns={'median':'median_weekly','std':'std_weekly','count':'n_weeks'}).reset_index()
    q = grp['wse_drop_m_time'].quantile([0.10,0.90]).unstack().rename(columns={0.10:'q10','0.1':'q10','0.90':'q90','0.9':'q90'})
    q.columns = ['q10','q90']
    q = q.reset_index()
    metrics = agg.merge(q, on='obstruction_node_id', how='left')
    metrics['cv_weekly'] = metrics['std_weekly'] / (metrics['median_weekly'].abs() + 1e-6)
    metrics['range_q90_q10'] = metrics['q90'] - metrics['q10']

    # Filter by min weeks and merge coords
    metrics = metrics[metrics['n_weeks'] >= min_weeks].copy()
    metrics = metrics.merge(base, on='obstruction_node_id', how='left')

    # Geo export (all)
    gdf = gpd.GeoDataFrame(metrics.dropna(subset=['x','y']).copy(), geometry=gpd.points_from_xy(metrics.dropna(subset=['x','y'])['x'], metrics.dropna(subset=['x','y'])['y']), crs='EPSG:4326')
    out_all = ANALYSIS_DIR / f"weekly_variability_{sample}.geojson"
    gdf.to_file(out_all, driver='GeoJSON')

    # Percentile layers
    n = len(gdf)
    n_top = max(1, int(n * pct / 100.0))
    # By CV
    gdf_cv_sorted = gdf.sort_values('cv_weekly', ascending=False)
    out_cv_top = ANALYSIS_DIR / f"weekly_variability_cv_top{int(pct)}_{sample}.geojson"
    out_cv_bottom = ANALYSIS_DIR / f"weekly_variability_cv_bottom{int(pct)}_{sample}.geojson"
    gdf_cv_sorted.iloc[:n_top].to_file(out_cv_top, driver='GeoJSON')
    gdf_cv_sorted.iloc[-n_top:].to_file(out_cv_bottom, driver='GeoJSON')
    # By range
    gdf_rg_sorted = gdf.sort_values('range_q90_q10', ascending=False)
    out_rg_top = ANALYSIS_DIR / f"weekly_variability_range_top{int(pct)}_{sample}.geojson"
    out_rg_bottom = ANALYSIS_DIR / f"weekly_variability_range_bottom{int(pct)}_{sample}.geojson"
    gdf_rg_sorted.iloc[:n_top].to_file(out_rg_top, driver='GeoJSON')
    gdf_rg_sorted.iloc[-n_top:].to_file(out_rg_bottom, driver='GeoJSON')

    print(f"Wrote: {out_all}")
    print(f"Wrote: {out_cv_top}")
    print(f"Wrote: {out_cv_bottom}")
    print(f"Wrote: {out_rg_top}")
    print(f"Wrote: {out_rg_bottom}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, required=True)
    ap.add_argument('--min-weeks', type=int, default=7)
    ap.add_argument('--pct', type=float, default=1.0)
    args = ap.parse_args()
    build(args.sample, args.min_weeks, args.pct)


if __name__ == '__main__':
    main()
