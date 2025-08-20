#!/usr/bin/env python3
"""
Build weekly temporal variability GeoJSON layers per obstruction.
- Inputs:
  - data/analysis/master_wse_drop_timeseries_weekly_{sample}.parquet
    (expects columns: obstruction_node_id, week_bin, wse_drop_m_time, n_obs_upstream, n_obs_downstream)
  - data/analysis/master_wse_drop_crossreach_{sample}.csv (for x,y coords)
- Outputs:
  - data/analysis/temporal_variability_weekly_{sample}.geojson (all)
  - data/analysis/temporal_variability_weekly_top{top_pct}_{sample}.geojson
  - data/analysis/temporal_variability_weekly_bottom{bottom_pct}_{sample}.geojson

Usage:
  python notebooks/build_temporal_variability_weekly_geojson.py --sample 20734 --top-pct 1 --bottom-pct 1
"""

import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd

ANALYSIS_DIR = Path("data/analysis")


def build_weekly_variability(sample: int, top_pct: float, bottom_pct: float) -> None:
    ts_path = ANALYSIS_DIR / f"master_wse_drop_timeseries_weekly_{sample}.parquet"
    base_csv = ANALYSIS_DIR / f"master_wse_drop_crossreach_{sample}.csv"

    if not (ts_path.exists() and base_csv.exists()):
        raise FileNotFoundError("Required inputs not found in data/analysis/. Ensure weekly timeseries exists and base CSV present.")

    ts = pd.read_parquet(ts_path)
    base = pd.read_csv(base_csv)

    # Guard columns
    for c in ('obstruction_node_id', 'x', 'y'):
        if c not in base.columns:
            raise KeyError(f"Column '{c}' missing in {base_csv}")
    if 'wse_drop_m_time' not in ts.columns:
        raise KeyError("Column 'wse_drop_m_time' missing in weekly timeseries parquet")

    # Per-obstruction weekly metrics
    grp = ts.groupby('obstruction_node_id')
    stats = grp['wse_drop_m_time'].agg(['std', 'median', 'count']).rename(columns={'std': 'std_weekly', 'median': 'median_weekly', 'count': 'n_weeks'}).reset_index()
    # Weekly amplitude = IQR of weekly drops
    q = grp['wse_drop_m_time'].quantile([0.25, 0.75]).unstack().rename(columns={0.25: 'q25_weekly', 0.75: 'q75_weekly'}).reset_index()
    per_obs = stats.merge(q, on='obstruction_node_id', how='left')
    per_obs['amplitude_weekly'] = per_obs['q75_weekly'] - per_obs['q25_weekly']

    # Merge coords
    obs = base[['obstruction_node_id', 'reach_id', 'x', 'y']].drop_duplicates('obstruction_node_id')
    merged = obs.merge(per_obs, on='obstruction_node_id', how='left')

    # Rankings and flags by amplitude_weekly
    merged['amplitude_weekly'] = pd.to_numeric(merged['amplitude_weekly'], errors='coerce')
    merged['std_weekly'] = pd.to_numeric(merged['std_weekly'], errors='coerce')

    merged['amplitude_weekly_rank'] = merged['amplitude_weekly'].rank(method='min', ascending=False)
    merged['std_weekly_rank'] = merged['std_weekly'].rank(method='min', ascending=False)

    pct_top = float(top_pct)
    pct_bottom = float(bottom_pct)
    n_total = len(merged)
    n_top = max(1, int(n_total * pct_top / 100.0))
    n_bottom = max(1, int(n_total * pct_bottom / 100.0))

    merged = merged.sort_values('amplitude_weekly', ascending=False)
    merged['is_top_variable_weekly'] = False
    merged['is_most_constant_weekly'] = False
    if n_total > 0:
        merged.iloc[:n_top, merged.columns.get_loc('is_top_variable_weekly')] = True
        merged.iloc[-n_bottom:, merged.columns.get_loc('is_most_constant_weekly')] = True

    # Geo export
    gdf_all = gpd.GeoDataFrame(
        merged.dropna(subset=['x', 'y']).copy(),
        geometry=gpd.points_from_xy(merged.dropna(subset=['x', 'y'])['x'], merged.dropna(subset=['x', 'y'])['y']),
        crs='EPSG:4326'
    )

    out_all = ANALYSIS_DIR / f"temporal_variability_weekly_{sample}.geojson"
    out_top = ANALYSIS_DIR / f"temporal_variability_weekly_top{int(pct_top)}_{sample}.geojson"
    out_bottom = ANALYSIS_DIR / f"temporal_variability_weekly_bottom{int(pct_bottom)}_{sample}.geojson"

    gdf_all.to_file(out_all, driver='GeoJSON')
    gdf_all[gdf_all['is_top_variable_weekly']].to_file(out_top, driver='GeoJSON')
    gdf_all[gdf_all['is_most_constant_weekly']].to_file(out_bottom, driver='GeoJSON')

    print(f"Wrote: {out_all}")
    print(f"Wrote: {out_top}")
    print(f"Wrote: {out_bottom}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, required=True)
    ap.add_argument('--top-pct', type=float, default=1.0)
    ap.add_argument('--bottom-pct', type=float, default=1.0)
    args = ap.parse_args()
    build_weekly_variability(args.sample, args.top_pct, args.bottom_pct)


if __name__ == '__main__':
    main()
