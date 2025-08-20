#!/usr/bin/env python3
"""
EDA on weekly temporal amplitude per obstruction.

- Loads weekly variability GeoJSON and base analysis CSV
- Merges node/reach attributes from DuckDB
- Computes correlations (Pearson, Spearman) between amplitude_weekly and:
  facc (node, fallback reach_facc), slope, wse_mean, dist_out, dist_out/slope,
  reach_width, p_wid_var (if available)
- Also computes rank-based correlations using ranks of variables
- Saves summary CSV to data/analysis and prints table
"""

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import geopandas as gpd

ANALYSIS_DIR = Path("data/analysis")
DUCKDB_FILE = Path("data/duckdb/sword_global.duckdb")
SAMPLE = 20734
MIN_WEEKS = 7

WEEKLY_GEOJSON = ANALYSIS_DIR / f"temporal_variability_weekly_{SAMPLE}.geojson"
BASE_CSV = ANALYSIS_DIR / f"master_wse_drop_crossreach_{SAMPLE}.csv"


def load_data() -> pd.DataFrame:
    if not WEEKLY_GEOJSON.exists():
        raise FileNotFoundError(WEEKLY_GEOJSON)
    if not BASE_CSV.exists():
        raise FileNotFoundError(BASE_CSV)

    weekly = gpd.read_file(WEEKLY_GEOJSON)
    base = pd.read_csv(BASE_CSV)

    # Keep minimal base columns
    base_small = base[['obstruction_node_id', 'reach_id', 'dist_out', 'median_wse_upstream', 'median_wse_downstream']].drop_duplicates('obstruction_node_id')

    df = weekly.merge(base_small, on='obstruction_node_id', how='left')

    # Ensure reach_id present; derive from nodes if missing
    if 'reach_id' not in df.columns or df['reach_id'].isna().all():
        with duckdb.connect(database=str(DUCKDB_FILE), read_only=True) as con2:
            node_ids2 = df['obstruction_node_id'].dropna().unique().tolist()
            if node_ids2:
                node_ids2_str = ','.join([str(int(x)) for x in node_ids2 if pd.notna(x)])
                nid_to_rid = con2.execute(f"""
                    SELECT node_id AS obstruction_node_id, reach_id
                    FROM nodes
                    WHERE node_id IN ({node_ids2_str})
                """).df()
                df = df.merge(nid_to_rid, on='obstruction_node_id', how='left')

    # Merge attributes from DuckDB
    with duckdb.connect(database=str(DUCKDB_FILE), read_only=True) as con:
        node_ids = df['obstruction_node_id'].dropna().unique().tolist()
        reach_ids = df['reach_id'].dropna().unique().tolist() if 'reach_id' in df.columns else []
        if node_ids:
            node_ids_str = ','.join([str(int(x)) for x in node_ids if pd.notna(x)])
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
                FROM nodes
                WHERE node_id IN ({node_ids_str})
            """).df()
        else:
            node_attrs = pd.DataFrame()
        if reach_ids:
            reach_ids_str = ','.join([str(int(x)) for x in reach_ids if pd.notna(x)])
            # Try to pull p_wid_var if present; otherwise ignore
            reach_attrs = con.execute(f"""
                SELECT reach_id,
                       facc AS reach_facc,
                       reach_length,
                       n_nodes,
                       width AS reach_width
                FROM reaches
                WHERE reach_id IN ({reach_ids_str})
            """).df()
        else:
            reach_attrs = pd.DataFrame()

    if not node_attrs.empty:
        df = df.merge(node_attrs, left_on='obstruction_node_id', right_on='node_id', how='left')
    else:
        for col in ['facc', 'stream_order', 'main_side', 'sinuosity', 'max_width', 'meander_length', 'node_width', 'width_var_node']:
            df[col] = np.nan
    if not reach_attrs.empty:
        df = df.merge(reach_attrs, on='reach_id', how='left')
    else:
        for col in ['reach_facc', 'reach_length', 'n_nodes', 'reach_width']:
            df[col] = np.nan

    return df


def compute_variables(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # FACC with fallback
    out['facc_col'] = out['facc'].fillna(out['reach_facc'])
    # Slope proxy (avoid divide by zero)
    out['slope'] = np.where(pd.to_numeric(out['dist_out'], errors='coerce') > 0,
                            pd.to_numeric(out['reach_length'], errors='coerce') / pd.to_numeric(out['dist_out'], errors='coerce'),
                            np.nan)
    # Mean WSE from side medians
    out['wse_mean'] = (pd.to_numeric(out['median_wse_upstream'], errors='coerce') + pd.to_numeric(out['median_wse_downstream'], errors='coerce')) / 2.0
    # dist_out/slope
    out['dist_out_slope_ratio'] = np.where(pd.to_numeric(out['slope'], errors='coerce') > 0,
                                           pd.to_numeric(out['dist_out'], errors='coerce') / pd.to_numeric(out['slope'], errors='coerce'),
                                           np.nan)
    # Width vars
    # Use node-level width and variance (per obstruction)
    out['width'] = pd.to_numeric(out.get('node_width'), errors='coerce')
    out['width_var'] = pd.to_numeric(out.get('width_var_node'), errors='coerce')

    # Ensure amplitude present
    out['amplitude_weekly'] = pd.to_numeric(out['amplitude_weekly'], errors='coerce')
    out['amplitude_weekly_rank'] = pd.to_numeric(out['amplitude_weekly_rank'], errors='coerce')
    return out


def corr_table(df: pd.DataFrame, y: str, xs: list[str]) -> pd.DataFrame:
    rows = []
    for x in xs:
        a = pd.to_numeric(df[y], errors='coerce')
        b = pd.to_numeric(df[x], errors='coerce')
        mask = a.notna() & b.notna()
        if mask.sum() < 10:
            rows.append({'y': y, 'x': x, 'n': int(mask.sum()), 'pearson_r': np.nan, 'spearman_r': np.nan})
            continue
        a1 = a[mask]
        b1 = b[mask]
        pearson_r = float(np.corrcoef(a1, b1)[0, 1])
        # Spearman via ranks
        a_rank = a1.rank(method='average')
        b_rank = b1.rank(method='average')
        spearman_r = float(np.corrcoef(a_rank, b_rank)[0, 1])
        rows.append({'y': y, 'x': x, 'n': int(mask.sum()), 'pearson_r': pearson_r, 'spearman_r': spearman_r})
    return pd.DataFrame(rows).sort_values('pearson_r', ascending=False)


def main():
    print("Loading weekly variability and attributes...")
    df = load_data()
    df = compute_variables(df)

    variables = ['facc_col', 'slope', 'wse_mean', 'dist_out', 'dist_out_slope_ratio', 'width', 'width_var']
    # Filter: at least MIN_WEEKS coverage and no nulls
    needed = ['amplitude_weekly'] + variables
    df = df[df.get('n_weeks', 0) >= MIN_WEEKS].copy()
    df = df.dropna(subset=[c for c in needed if c in df.columns])
    print("Computing correlations vs amplitude_weekly...")
    t1 = corr_table(df, 'amplitude_weekly', variables)
    print(t1)

    print("\nComputing rank-based correlations (amplitude_weekly_rank vs variable ranks)...")
    # Build rank versions of variables
    df_ranks = df.copy()
    for v in variables:
        df_ranks[f'{v}_rank'] = pd.to_numeric(df_ranks[v], errors='coerce').rank(method='average')
    t2 = corr_table(df_ranks, 'amplitude_weekly_rank', [f'{v}_rank' for v in variables])
    print(t2)

    out1 = ANALYSIS_DIR / 'weekly_amplitude_correlations_ge7w.csv'
    out2 = ANALYSIS_DIR / 'weekly_amplitude_rank_correlations_ge7w.csv'
    t1.to_csv(out1, index=False)
    t2.to_csv(out2, index=False)
    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


if __name__ == '__main__':
    main()
