#!/usr/bin/env python3
"""
Build GeoJSON layers summarizing temporal variability per obstruction.
- Inputs:
  - data/analysis/master_wse_drop_timeseries_{sample}.parquet
  - data/analysis/master_wse_drop_seasonality_{sample}.parquet
  - data/analysis/master_wse_drop_crossreach_{sample}.csv (for x,y coords)
- Outputs:
  - data/analysis/temporal_variability_{sample}.geojson (all obstructions w/ metrics)
  - data/analysis/temporal_variability_top{top_pct}_{sample}.geojson
  - data/analysis/temporal_variability_bottom{bottom_pct}_{sample}.geojson

Usage:
  python notebooks/build_temporal_variability_geojson.py --sample 20734 --top-pct 1 --bottom-pct 1
"""

import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd

ANALYSIS_DIR = Path("data/analysis")


def build_variability(sample: int, top_pct: float, bottom_pct: float) -> None:
    ts_path = ANALYSIS_DIR / f"master_wse_drop_timeseries_{sample}.parquet"
    seas_path = ANALYSIS_DIR / f"master_wse_drop_seasonality_{sample}.parquet"
    base_csv = ANALYSIS_DIR / f"master_wse_drop_crossreach_{sample}.csv"

    if not (ts_path.exists() and seas_path.exists() and base_csv.exists()):
        raise FileNotFoundError("Required inputs not found in data/analysis/. Run the master analysis first.")

    ts = pd.read_parquet(ts_path)
    seas = pd.read_parquet(seas_path)
    base = pd.read_csv(base_csv)

    # Guard expected columns
    for c in ('obstruction_node_id', 'x', 'y'):
        if c not in base.columns:
            raise KeyError(f"Column '{c}' missing in {base_csv}")

    # Variability metrics per obstruction
    # - amplitude from seasonality table (q75-q25 across months; repeated per row)
    seas_amp = (seas[['obstruction_node_id', 'amplitude']]
                .dropna(subset=['amplitude'])
                .drop_duplicates('obstruction_node_id'))

    # - overall_median (typical drop) and coverage (sum of n_days over months)
    coverage = (seas.groupby('obstruction_node_id')['n_days']
                .sum(min_count=1)
                .rename('n_days_total')
                .reset_index())
    overall = (seas[['obstruction_node_id', 'overall_median']]
               .drop_duplicates('obstruction_node_id'))

    # - std over daily drops from time series
    if 'wse_drop_m_time' in ts.columns:
        std_daily = (ts.groupby('obstruction_node_id')['wse_drop_m_time']
                     .std()
                     .rename('std_daily')
                     .reset_index())
        n_days = (ts.groupby('obstruction_node_id')['time_bin']
                  .nunique()
                  .rename('n_days_ts')
                  .reset_index())
    else:
        std_daily = pd.DataFrame(columns=['obstruction_node_id', 'std_daily'])
        n_days = pd.DataFrame(columns=['obstruction_node_id', 'n_days_ts'])

    # Assemble per-obstruction table
    obs = (base[['obstruction_node_id', 'reach_id', 'x', 'y']]
           .drop_duplicates('obstruction_node_id'))

    merged = (obs
              .merge(seas_amp, on='obstruction_node_id', how='left')
              .merge(overall, on='obstruction_node_id', how='left')
              .merge(coverage, on='obstruction_node_id', how='left')
              .merge(std_daily, on='obstruction_node_id', how='left')
              .merge(n_days, on='obstruction_node_id', how='left'))

    # Rankings and flags
    merged['amplitude'] = pd.to_numeric(merged['amplitude'], errors='coerce')
    merged['std_daily'] = pd.to_numeric(merged['std_daily'], errors='coerce')

    merged['amplitude_rank'] = merged['amplitude'].rank(method='min', ascending=False)
    merged['std_rank'] = merged['std_daily'].rank(method='min', ascending=False)

    # Percentile thresholds by amplitude
    pct_top = float(top_pct)
    pct_bottom = float(bottom_pct)
    n_total = len(merged)
    n_top = max(1, int(n_total * pct_top / 100.0))
    n_bottom = max(1, int(n_total * pct_bottom / 100.0))

    merged = merged.sort_values('amplitude', ascending=False)
    merged['is_top_variable'] = False
    merged['is_most_constant'] = False
    if n_total > 0:
        merged.iloc[:n_top, merged.columns.get_loc('is_top_variable')] = True
        merged.iloc[-n_bottom:, merged.columns.get_loc('is_most_constant')] = True

    # Geo export (all)
    gdf_all = gpd.GeoDataFrame(
        merged.dropna(subset=['x', 'y']).copy(),
        geometry=gpd.points_from_xy(merged.dropna(subset=['x', 'y'])['x'], merged.dropna(subset=['x', 'y'])['y']),
        crs='EPSG:4326'
    )

    out_all = ANALYSIS_DIR / f"temporal_variability_{sample}.geojson"
    gdf_all.to_file(out_all, driver='GeoJSON')

    # Geo export (top variable)
    gdf_top = gdf_all[gdf_all['is_top_variable']]
    out_top = ANALYSIS_DIR / f"temporal_variability_top{int(pct_top)}_{sample}.geojson"
    gdf_top.to_file(out_top, driver='GeoJSON')

    # Geo export (most constant)
    gdf_const = gdf_all[gdf_all['is_most_constant']]
    out_const = ANALYSIS_DIR / f"temporal_variability_bottom{int(pct_bottom)}_{sample}.geojson"
    gdf_const.to_file(out_const, driver='GeoJSON')

    print(f"Wrote: {out_all}")
    print(f"Wrote: {out_top}")
    print(f"Wrote: {out_const}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, required=True, help='Sample size used for file naming (e.g., 20734)')
    ap.add_argument('--top-pct', type=float, default=1.0, help='Top percentile by amplitude to export (default 1%)')
    ap.add_argument('--bottom-pct', type=float, default=1.0, help='Bottom percentile (most constant) to export (default 1%)')
    args = ap.parse_args()
    build_variability(args.sample, args.top_pct, args.bottom_pct)


if __name__ == '__main__':
    main()
