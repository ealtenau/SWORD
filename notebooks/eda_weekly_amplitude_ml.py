#!/usr/bin/env python3
"""
Weekly temporal variability ML and stratified EDA (≥7-week coverage).

Outputs in data/analysis/:
- rf_feature_importances_ge7w.csv
- stratified_spearman_stream_order_ge7w.csv
- stratified_spearman_hemisphere_ge7w.csv
- alternate_targets_correlations_ge7w.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import duckdb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

ANALYSIS_DIR = Path("data/analysis")
DUCKDB_FILE = Path("data/duckdb/sword_global.duckdb")
SAMPLE = 20734
MIN_WEEKS = 7

WEEKLY_GEOJSON = ANALYSIS_DIR / f"temporal_variability_weekly_{SAMPLE}.geojson"
WEEKLY_GEOJSON_KG = ANALYSIS_DIR / f"temporal_variability_weekly_{SAMPLE}_kg.geojson"
BASE_CSV = ANALYSIS_DIR / f"master_wse_drop_crossreach_{SAMPLE}.csv"
WEEKLY_TS = ANALYSIS_DIR / f"master_wse_drop_timeseries_weekly_{SAMPLE}.parquet"


def _load_core() -> pd.DataFrame:
    if not WEEKLY_GEOJSON.exists() or not BASE_CSV.exists():
        raise FileNotFoundError("Missing weekly geojson or base csv. Run prior steps.")
    weekly = gpd.read_file(WEEKLY_GEOJSON)
    base = pd.read_csv(BASE_CSV)
    base_small = base[['obstruction_node_id', 'reach_id', 'dist_out', 'x', 'y', 'median_wse_upstream', 'median_wse_downstream']].drop_duplicates('obstruction_node_id')
    df = weekly.merge(base_small, on='obstruction_node_id', how='left')

    # Ensure reach_id is present (derive from nodes if needed)
    if 'reach_id' not in df.columns or df['reach_id'].isna().all():
        with duckdb.connect(str(DUCKDB_FILE), read_only=True) as con:
            node_ids = df['obstruction_node_id'].dropna().unique().tolist()
            if node_ids:
                node_ids_str = ','.join([str(int(x)) for x in node_ids])
                nid2rid = con.execute(f"""
                    SELECT node_id AS obstruction_node_id, reach_id
                    FROM nodes WHERE node_id IN ({node_ids_str})
                """).df()
                df = df.merge(nid2rid, on='obstruction_node_id', how='left')

    # Node/reach attributes
    with duckdb.connect(str(DUCKDB_FILE), read_only=True) as con:
        node_ids = df['obstruction_node_id'].dropna().unique().tolist()
        reach_ids = df['reach_id'].dropna().unique().tolist() if 'reach_id' in df.columns else []
        if node_ids:
            node_ids_str = ','.join([str(int(x)) for x in node_ids])
            node_attrs = con.execute(f"""
                SELECT node_id,
                       facc,
                       stream_order,
                       main_side,
                       sinuosity,
                       max_width,
                       meander_length,
                       width AS node_width,
                       width_var AS width_var_node
                FROM nodes WHERE node_id IN ({node_ids_str})
            """).df()
        else:
            node_attrs = pd.DataFrame()
        if reach_ids:
            reach_ids_str = ','.join([str(int(x)) for x in reach_ids])
            reach_attrs = con.execute(f"""
                SELECT reach_id,
                       facc AS reach_facc,
                       reach_length,
                       n_nodes,
                       width AS reach_width
                FROM reaches WHERE reach_id IN ({reach_ids_str})
            """).df()
        else:
            reach_attrs = pd.DataFrame()

    if not node_attrs.empty:
        df = df.merge(node_attrs, left_on='obstruction_node_id', right_on='node_id', how='left')
    else:
        for col in ['facc','stream_order','main_side','sinuosity','max_width','meander_length','node_width','width_var_node']:
            df[col] = np.nan
    if not reach_attrs.empty:
        df = df.merge(reach_attrs, on='reach_id', how='left')
    else:
        for col in ['reach_facc','reach_length','n_nodes','reach_width']:
            df[col] = np.nan

    # Merge Köppen–Geiger class if available
    if WEEKLY_GEOJSON_KG.exists():
        try:
            kg = gpd.read_file(WEEKLY_GEOJSON_KG)[['obstruction_node_id', 'kg_class']]
            df = df.merge(kg, on='obstruction_node_id', how='left')
        except Exception:
            pass

    return df


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Coverage filter
    out = out[out.get('n_weeks', 0) >= MIN_WEEKS].copy()

    # Target
    out['amplitude_weekly'] = pd.to_numeric(out['amplitude_weekly'], errors='coerce')

    # Base variables and transforms
    out['facc_col'] = pd.to_numeric(out['facc'], errors='coerce').fillna(pd.to_numeric(out['reach_facc'], errors='coerce'))
    out['dist_out'] = pd.to_numeric(out['dist_out'], errors='coerce')
    out['reach_length'] = pd.to_numeric(out['reach_length'], errors='coerce')
    out['slope'] = np.where(out['dist_out'] > 0, out['reach_length'] / out['dist_out'], np.nan)
    out['wse_mean'] = (pd.to_numeric(out['median_wse_upstream'], errors='coerce') + pd.to_numeric(out['median_wse_downstream'], errors='coerce')) / 2.0
    out['width'] = pd.to_numeric(out['node_width'], errors='coerce')
    out['width_var'] = pd.to_numeric(out['width_var_node'], errors='coerce')
    # Derive lat/lon: prefer explicit y/x, fallback to geometry
    if 'y' in out.columns:
        out['lat'] = pd.to_numeric(out['y'], errors='coerce')
    elif 'geometry' in out.columns:
        try:
            out['lat'] = out.geometry.y
        except Exception:
            out['lat'] = np.nan
    else:
        out['lat'] = np.nan
    out['hemisphere_north'] = (out['lat'] >= 0).astype(int)

    # Logs
    def _log10(s):
        return np.log10(np.clip(pd.to_numeric(s, errors='coerce'), 1.0, None))
    out['log_facc'] = _log10(out['facc_col'])
    out['log_dist_out'] = _log10(out['dist_out'])
    out['log_width'] = _log10(out['width'])
    out['log_width_var'] = _log10(out['width_var'])
    out['log_slope'] = _log10(out['slope'])

    # Alternate targets if available in weekly geojson
    out['std_weekly'] = pd.to_numeric(out.get('std_weekly'), errors='coerce')
    out['median_weekly'] = pd.to_numeric(out.get('median_weekly'), errors='coerce')
    out['cv_weekly'] = out['std_weekly'] / (out['median_weekly'].abs() + 1e-6)

    # q90-q10 from weekly parquet
    if WEEKLY_TS.exists():
        wt = pd.read_parquet(WEEKLY_TS)
        if {'obstruction_node_id','wse_drop_m_time'}.issubset(wt.columns):
            qq = (wt.groupby('obstruction_node_id')['wse_drop_m_time']
                    .quantile([0.10, 0.90]).unstack().rename(columns={0.10:'q10_weekly',0.90:'q90_weekly'}))
            qq['range_q90_q10'] = qq['q90_weekly'] - qq['q10_weekly']
            out = out.merge(qq[['range_q90_q10']].reset_index(), on='obstruction_node_id', how='left')

    # Final drop nulls for target and core features
    core_cols = ['amplitude_weekly','log_facc','log_dist_out','log_slope','log_width','log_width_var','wse_mean','slope','width','width_var','facc_col','dist_out','stream_order','hemisphere_north']
    out = out.dropna(subset=['amplitude_weekly'])
    return out


def _spearman(x, y):
    a = pd.to_numeric(x, errors='coerce')
    b = pd.to_numeric(y, errors='coerce')
    m = a.notna() & b.notna()
    if m.sum() < 10:
        return np.nan
    a_rank = a[m].rank(method='average')
    b_rank = b[m].rank(method='average')
    return float(np.corrcoef(a_rank, b_rank)[0,1])


def run():
    df = _load_core()
    df = _compute_features(df)

    # Select model features
    base_feature_cols = [
        'log_facc','log_dist_out','log_slope','log_width','log_width_var',
        'wse_mean','slope','width','width_var','stream_order','hemisphere_north'
    ]
    feat_df = df.dropna(subset=base_feature_cols + ['amplitude_weekly']).copy()

    X_num = feat_df[base_feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    # Climate dummies if available
    if 'kg_class' in feat_df.columns:
        kg_dum = pd.get_dummies(feat_df['kg_class'].astype('Int64').astype('category'), prefix='kg', drop_first=True)
        X = pd.concat([X_num, kg_dum], axis=1)
    else:
        X = X_num
    y = feat_df['amplitude_weekly']

    # Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # Add aggregated KG importance if present
    if any(importances['feature'].str.startswith('kg_')):
        kg_total = importances[importances['feature'].str.startswith('kg_')]['importance'].sum()
        importances = pd.concat([
            importances,
            pd.DataFrame([{'feature': 'kg_class(total)', 'importance': kg_total}])
        ], ignore_index=True)

    importances.to_csv(ANALYSIS_DIR / 'rf_feature_importances_ge7w.csv', index=False)

    # Stratified Spearman by stream order
    results_so = []
    for so, sub in df.groupby('stream_order'):
        if pd.isna(so):
            continue
        for v in ['log_facc','log_dist_out','log_slope','log_width','log_width_var','wse_mean']:
            rho = _spearman(sub[v], sub['amplitude_weekly'])
            n = int((sub[[v,'amplitude_weekly']].dropna()).shape[0])
            results_so.append({'stream_order': int(so), 'var': v, 'spearman_rho': rho, 'n': n})
    pd.DataFrame(results_so).to_csv(ANALYSIS_DIR / 'stratified_spearman_stream_order_ge7w.csv', index=False)

    # Stratified Spearman by hemisphere
    results_hemi = []
    for hemi, sub in df.groupby('hemisphere_north'):
        for v in ['log_facc','log_dist_out','log_slope','log_width','log_width_var','wse_mean']:
            rho = _spearman(sub[v], sub['amplitude_weekly'])
            n = int((sub[[v,'amplitude_weekly']].dropna()).shape[0])
            results_hemi.append({'hemisphere_north': int(hemi), 'var': v, 'spearman_rho': rho, 'n': n})
    pd.DataFrame(results_hemi).to_csv(ANALYSIS_DIR / 'stratified_spearman_hemisphere_ge7w.csv', index=False)

    # Alternate targets correlations (overall, not stratified)
    alt_rows = []
    for target in ['cv_weekly','range_q90_q10']:
        if target not in df.columns:
            continue
        for v in ['log_facc','log_dist_out','log_slope','log_width','log_width_var','wse_mean']:
            rho = _spearman(df[v], df[target])
            n = int((df[[v,target]].dropna()).shape[0])
            alt_rows.append({'target': target, 'var': v, 'spearman_rho': rho, 'n': n})
    if alt_rows:
        pd.DataFrame(alt_rows).to_csv(ANALYSIS_DIR / 'alternate_targets_correlations_ge7w.csv', index=False)

    # Print brief summary
    print("Random Forest importances (top 8):")
    print(importances.head(8))
    print("Saved outputs to", ANALYSIS_DIR)


if __name__ == '__main__':
    run()
