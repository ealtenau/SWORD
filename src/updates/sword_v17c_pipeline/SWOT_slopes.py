### Option A: Query the NA subset outputs (fastest)
import duckdb
import pandas as pd
import geopandas as gpd
import numpy as np
import statsmodels.api as sm

import warnings
import os
import glob
import numpy as np
from math import hypot
from tqdm import tqdm
import argparse

import statsmodels.formula.api as smf

# collapse_multigraph.py
import pickle
import math
from collections import defaultdict

import networkx as nx


from SWORD_graph import load_sword_data
# from robustLME import parallel_lmme

# -----------------------------
# open SWOT files
# -----------------------------
def open_SWOT_files(node_ids, directory, continent, uncertainty_threshold = None):
    con = duckdb.connect(":memory:")
    # --- Find by node_id ---
    node_df = pd.DataFrame({"node_id": node_ids})
    node_df['node_id'] = node_df['node_id'].astype('str')
    con.register("node_ids", con.from_df(node_df))  # <-- FIXED

    # SWOT data is loaded from external volume
    # Filter out macOS dotfiles (._*) that get created on external volumes
    swot_data_dir = '/Volumes/SWORD_DATA/data/swot/RiverSP_D_parq/node'
    parquet_files = [f for f in glob.glob(os.path.join(swot_data_dir, '*.parquet')) 
                     if not os.path.basename(f).startswith('._')]
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {swot_data_dir} (excluding dotfiles)")
    
    # DuckDB's read_parquet accepts a list of files - construct SQL array syntax
    # Escape single quotes in file paths and create SQL array literal
    escaped_files = [f.replace("'", "''") for f in parquet_files]
    files_array = "[" + ", ".join(f"'{f}'" for f in escaped_files) + "]"

    # Sentinel value for invalid WSE (from extract script)
    SENTINEL = -999_999_999_999.0
    
    # First, detect available columns to build dynamic filters
    # Read a sample to check column availability
    sample_query = f"""
        SELECT * FROM read_parquet({files_array}) LIMIT 1
    """
    try:
        sample_df = con.execute(sample_query).df()
        colnames = set(c.lower() for c in sample_df.columns.tolist())
    except:
        colnames = set()
    
    # Determine which column name convention is used (old: wse_sm, new: wse)
    # Support both old and new column names
    has_wse_sm = "wse_sm" in colnames
    has_wse = "wse" in colnames
    has_wse_sm_q = "wse_sm_q" in colnames
    has_wse_q = "wse_q" in colnames
    has_wse_sm_u = "wse_sm_u" in colnames
    has_wse_u = "wse_u" in colnames
    
    # Build column mapping expressions (prefer new names, fallback to old)
    # Use COALESCE for SELECT, but determine actual column names for WHERE/ORDER BY
    if has_wse:
        wse_col = "wse"
        wse_expr = "wse"
    elif has_wse_sm:
        wse_col = "wse_sm"
        wse_expr = "wse_sm"
    else:
        wse_col = "wse"
        wse_expr = "wse"
    
    if has_wse_q:
        wse_q_col = "wse_q"
        wse_q_expr = "wse_q"
    elif has_wse_sm_q:
        wse_q_col = "wse_sm_q"
        wse_q_expr = "wse_sm_q"
    else:
        wse_q_col = None
        wse_q_expr = None
    
    if has_wse_u:
        wse_u_col = "wse_u"
        wse_u_expr = "wse_u"
    elif has_wse_sm_u:
        wse_u_col = "wse_sm_u"
        wse_u_expr = "wse_sm_u"
    else:
        wse_u_col = None
        wse_u_expr = None
    
    # For SELECT, use COALESCE to map old to new names
    wse_select_expr = f"COALESCE(wse, wse_sm) AS wse" if (has_wse or has_wse_sm) else f"{wse_col} AS wse"
    wse_q_select_expr = f"COALESCE(wse_q, wse_sm_q) AS wse_q" if (has_wse_q or has_wse_sm_q) else None
    wse_u_select_expr = f"COALESCE(wse_u, wse_sm_u) AS wse_u" if (has_wse_u or has_wse_sm_u) else None
    
    # Build dynamic filtering conditions based on available columns
    conditions = []
    
    # Filter out sentinel values (use the actual column name)
    conditions.append(f"NULLIF({wse_col}, {SENTINEL}) IS NOT NULL")
    
    # Summary quality (required threshold when present): COALESCE to 3 (poor)
    if wse_q_col:
        conditions.append(f"COALESCE({wse_q_col}, 3) <= 1")
    
    # Dark water fraction (optional; nulls pass). Harmonize names dynamically
    dark_expr = None
    if ("dark_frac" in colnames) and ("dark_water_frac" in colnames):
        dark_expr = "COALESCE(dark_frac, dark_water_frac)"
    elif "dark_frac" in colnames:
        dark_expr = "dark_frac"
    elif "dark_water_frac" in colnames:
        dark_expr = "dark_water_frac"
    
    if dark_expr:
        conditions.append(f"(({dark_expr} <= 0.5) OR ({dark_expr} IS NULL))")
    
    # Cross-track distance (optional; nulls pass). Harmonize names dynamically
    xtrk_expr = None
    if ("xtrk_dist" in colnames) and ("cross_track_dist" in colnames):
        xtrk_expr = "COALESCE(xtrk_dist, cross_track_dist)"
    elif "xtrk_dist" in colnames:
        xtrk_expr = "xtrk_dist"
    elif "cross_track_dist" in colnames:
        xtrk_expr = "cross_track_dist"
    
    if xtrk_expr:
        conditions.append(f"((ABS({xtrk_expr}) BETWEEN 10000 AND 60000) OR ({xtrk_expr} IS NULL))")
    
    # Crossover calibration quality (optional; nulls pass)
    if "xovr_cal_q" in colnames:
        conditions.append("(xovr_cal_q <= 1 OR xovr_cal_q IS NULL)")
    
    # WSE prior difference (optional; requires threshold and non-null p_wse)
    if "p_wse" in colnames:
        conditions.append(f"(p_wse IS NOT NULL AND ABS({wse_col} - p_wse) < 10)")
    
    # Node ID filter
    conditions.append("node_id IN (SELECT node_id FROM node_ids)")
    
    # Filter out invalid time_str values (NULL or empty)
    conditions.append("time_str IS NOT NULL AND time_str != ''")
    
    # Optional uncertainty threshold
    if uncertainty_threshold is not None and wse_u_col:
        conditions.append(f"{wse_u_col} <= {uncertainty_threshold}")
    
    where_clause = " AND ".join(conditions)
    
    # Check if reach_id column exists (for conditional inclusion)
    include_reach_id = "reach_id" in colnames and uncertainty_threshold is not None
    
    # Build deduplication query with quality-based ranking
    # Extract cycle_id and pass_id from filename if needed, or use cycle/pass columns
    # Map old column names to new ones in SELECT
    wse_select = wse_select_expr
    wse_u_select = wse_u_select_expr if wse_u_select_expr else "NULL AS wse_u"
    
    ranked_select_fields = [
        "CAST(node_id AS BIGINT) AS node_id",
        "time_str",
        wse_select,
        wse_u_select,
        "COALESCE(cycle, cycle_id) AS cycle",
        "COALESCE(pass, pass_id) AS pass",
        "width",
        "width_u"
    ]
    
    if include_reach_id:
        ranked_select_fields.insert(1, "CAST(reach_id AS BIGINT) AS reach_id")
    
    # Add t_start to ranked select for ordering (but don't include in final output)
    ranked_select_fields_with_tstart = ranked_select_fields + ["t_start"]
    ranked_select_clause = ",\n                ".join(ranked_select_fields_with_tstart)
    
    # Final select fields (after deduplication, use cycle/pass directly)
    final_select_fields = [
        "node_id",
        "time_str",
        "wse",
        "wse_u",
        "cycle",
        "pass",
        "width",
        "width_u"
    ]
    
    if include_reach_id:
        final_select_fields.insert(1, "reach_id")
    
    final_select_clause = ",\n            ".join(final_select_fields)
    
    # Build ORDER BY clause with actual column names (reference columns from raw_data)
    # In raw_data CTE, we select *, so we reference the actual column names from parquet
    order_by_wse_q = f"COALESCE({wse_q_col}, 3)" if wse_q_col else "3"
    order_by_wse_u = f"COALESCE({wse_u_col}, 1e18)" if wse_u_col else "1e18"
    
    dedup_query = f"""
        WITH raw_data AS (
            SELECT 
                *,
                COALESCE(try_cast(cycle AS INT), 
                         CAST(regexp_extract(filename, '_(\\d{{3}})_(\\d{{3}})_', 1) AS INT)) AS cycle_id,
                CAST(regexp_extract(filename, '_(\\d{{3}})_(\\d{{3}})_', 2) AS INT) AS pass_id,
                try_strptime(replace(time_str, 'Z', '+00:00'), '%Y-%m-%dT%H:%M:%S%z') AS t_start
            FROM read_parquet({files_array}, filename=true)
            WHERE {where_clause}
        ),
        ranked AS (
            SELECT 
                {ranked_select_clause},
                ROW_NUMBER() OVER (
                    PARTITION BY node_id, t_start, cycle_id, pass_id
                    ORDER BY {order_by_wse_q} ASC,
                             COALESCE(xovr_cal_q, 2) ASC,
                             {order_by_wse_u} ASC,
                             filename ASC
                ) AS rn
            FROM raw_data
        )
        SELECT 
            {final_select_clause}
        FROM ranked
        WHERE rn = 1
        ORDER BY node_id, t_start
    """
    
    # Fallback to simpler query if deduplication fails (e.g., missing columns)
    try:
        df_nodes_in = con.execute(dedup_query).df()
    except Exception as e:
        # Fallback: use DISTINCT with basic filters
        warnings.warn(f"Deduplication query failed, using DISTINCT fallback: {e}")
        base_conditions = [
            f"NULLIF({wse_col}, {SENTINEL}) IS NOT NULL",
            "node_id IN (SELECT node_id FROM node_ids)",
            "time_str IS NOT NULL AND time_str != ''"
        ]
        if uncertainty_threshold is not None and wse_u_col:
            base_conditions.append(f"{wse_u_col} <= {uncertainty_threshold}")
        
        fallback_select = [
            "CAST(node_id AS BIGINT) AS node_id",
            "time_str",
            wse_select,
            wse_u_select,
            "cycle",
            "pass",
            "width",
            "width_u"
        ]
        
        if include_reach_id:
            fallback_select.insert(1, "CAST(reach_id AS BIGINT) AS reach_id")
        
        fallback_query = f"""
            SELECT DISTINCT 
                {', '.join(fallback_select)}
            FROM read_parquet({files_array})
            WHERE {' AND '.join(base_conditions)}
            ORDER BY node_id, time_str
        """
        df_nodes_in = con.execute(fallback_query).df()
    
    df_nodes_in['cyclePass'] = df_nodes_in.groupby(['cycle', 'pass']).ngroup()
    return df_nodes_in


def df_to_parquet_duckdb_snappy(df, path):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    duckdb.query(f"""
        COPY (
            SELECT
                node_id,
                try_strptime(replace(time_str, 'Z', '+00:00'),
                            '%Y-%m-%dT%H:%M:%S%z') AS time_ts,
                wse, wse_u, cycle, pass, width, width_u,
                cyclePass, junction_id,
                section_id, 
                CAST(reach_id AS BIGINT) AS reach_id,  -- Explicitly cast to BIGINT (int64) to prevent int32 overflow
                distance, section_node_size, pass_size
            FROM df
            WHERE time_str IS NOT NULL AND time_str != ''
                AND reach_id IS NOT NULL  -- Only save rows with valid reach_id
        ) 
        TO '{path}' (FORMAT PARQUET, COMPRESSION 'SNAPPY');
    """)
# -----------------------------
# Calculate relative distance from junction to SWORD nodes
# -----------------------------
def transform_to_single_junction_value(d):
    # Rows where both distances exist → we will duplicate
    both_mask = d["dist_to_start_junction"].notna() & d["dist_to_end_junction"].notna()

    # --- Create "start" version
    start_df = d.copy()
    start_df["junction_id"] = start_df["section_start_junction"]
    start_df["distance"] = start_df["dist_to_start_junction"]

    # --- Create "end" version
    end_df = d.copy()
    end_df["junction_id"] = end_df["section_end_junction"]
    end_df["distance"]    = end_df["dist_to_end_junction"]

    # Keep only the relevant subset of columns
    cols_to_keep = [
        "junction_id",
        "section_id",
        'reach_id',
        "node_id",
        "distance",
    ]

    # Combine:
    # - both start and end rows for sections that have both
    # - only one side for those that have NaN on the other
    combined = pd.concat([
        start_df.loc[d["dist_to_start_junction"].notna(), cols_to_keep],
        end_df.loc[d["dist_to_end_junction"].notna(), cols_to_keep]
    ], ignore_index=True)

    return combined

def node_junction_distance(dfNode, dfEdge, nodes_gdf):
    
    # pre-processing
    dn = dfNode.copy()
    dn = dn[['reach_id', 'node_id', 'node_len', 'dist_out', 'x', 'y']]

    de = dfEdge.copy()
    de = de[['reach_id', 'path_seg', 'path_start_node', 'path_end_node']]
    de = de.rename(columns = {'path_seg': 'section_id', 'path_start_node':'section_start_junction', 'path_end_node':'section_end_junction'})
    de = de[['reach_id', 'section_id', 'section_start_junction', 'section_end_junction']]

    dj = nodes_gdf.copy()
    dj = dj.rename(columns = {'node_id':'junction_id', 'node_type':'junction_type'})
    dj = dj[['junction_id', 'junction_type', 'geometry']].copy()
    dj['x_j'] = dj['geometry'].x
    dj['y_j'] = dj['geometry'].y
    dj = dj.drop('geometry', axis = 1)
    
    

    d = dn.merge(de, on = 'reach_id', how = 'left')
    d = d.merge(dj, left_on = 'section_start_junction', right_on = 'junction_id', how = 'left')\
        .drop('junction_id', axis = 1).rename(columns = {'junction_type':'section_start_junction_type', 'x_j':'section_start_junction_x', 'y_j':'section_start_junction_y'})
    d = d.merge(dj, left_on = 'section_end_junction', right_on = 'junction_id', how = 'left')\
        .drop('junction_id', axis = 1).rename(columns = {'junction_type':'section_end_junction_type', 'x_j':'section_end_junction_x', 'y_j':'section_end_junction_y'})
    

    # get distance to junction for start and end junction

    # Initialize result columns
    d["dist_to_start_junction"] = np.nan
    d["dist_to_end_junction"] = np.nan


    for section_id, grp in d.groupby("section_id"):
        idx = grp.index

        # Skip if the group has no variation in dist_out (e.g., single node)
        if grp["dist_out"].nunique() == 0:
            continue

        # Endpoints in this section (by dist_out)
        min_idx = grp["dist_out"].idxmin()
        max_idx = grp["dist_out"].idxmax()
        min_row = grp.loc[min_idx]
        max_row = grp.loc[max_idx]

        # Junction metadata
        start_type = grp["section_start_junction_type"].iloc[0]
        end_type   = grp["section_end_junction_type"].iloc[0]
        sx, sy = grp["section_start_junction_x"].iloc[0], grp["section_start_junction_y"].iloc[0]
        ex, ey = grp["section_end_junction_x"].iloc[0],   grp["section_end_junction_y"].iloc[0]

        start_base = None
        end_base = None
        start_node = None
        end_node = None

        # If both are Junctions: decide mapping once using the start junction
        if start_type == "Junction" and end_type == "Junction":
            d_start_min = hypot(min_row["x"] - sx, min_row["y"] - sy)
            d_start_max = hypot(max_row["x"] - sx, max_row["y"] - sy)

            if d_start_min <= d_start_max:
                # min endpoint is closer to START → max endpoint must be END
                start_base = min_row["dist_out"]
                end_base   = max_row["dist_out"]
                start_node = min_row["node_id"]
                end_node   = max_row["node_id"]
            else:
                # max endpoint is closer to START → min endpoint must be END
                start_base = max_row["dist_out"]
                end_base   = min_row["dist_out"]
                start_node = max_row["node_id"]
                end_node   = min_row["node_id"]

        # Only START is a Junction → pick the nearer endpoint to START
        elif start_type == "Junction" and end_type != "Junction":
            d_start_min = hypot(min_row["x"] - sx, min_row["y"] - sy)
            d_start_max = hypot(max_row["x"] - sx, max_row["y"] - sy)
            if d_start_min <= d_start_max:
                start_base = min_row["dist_out"]; start_node = min_row["node_id"]
            else:
                start_base = max_row["dist_out"]; start_node = max_row["node_id"]

        # Only END is a Junction → pick the nearer endpoint to END
        elif end_type == "Junction" and start_type != "Junction":
            d_end_min = hypot(min_row["x"] - ex, min_row["y"] - ey)
            d_end_max = hypot(max_row["x"] - ex, max_row["y"] - ey)
            if d_end_min <= d_end_max:
                end_base = min_row["dist_out"]; end_node = min_row["node_id"]
            else:
                end_base = max_row["dist_out"]; end_node = max_row["node_id"]

        # Neither is a Junction → nothing to do for this section
        elif end_type == "Outlet":
            end_base = min_row["dist_out"]; end_node = min_row["node_id"]

        # Write relative distances (only where applicable)
        if start_base is not None:
            d.loc[idx, "dist_to_start_junction"] = abs(grp["dist_out"] - start_base)
            # d.loc[idx, "closest_start_node_id"] = start_node
        if end_base is not None:
            d.loc[idx, "dist_to_end_junction"]   = abs(grp["dist_out"] - end_base)
            # d.loc[idx, "closest_end_node_id"] = end_node
    
    d = transform_to_single_junction_value(d)
    d['section_node_size'] = d.groupby(["junction_id", "section_id"])["node_id"].transform("size")

    return d

# -----------------------------
# Transform SWOT slopes
# -----------------------------
def add_end_rows(df):  
    newGraph_expanded = df.copy()
    # Count how many rows exist per section_id
    section_counts = df['section_id'].value_counts()

    # Identify section_ids that appear only once
    single_sections = section_counts[section_counts == 1].index
    for endType in ['Head_water', 'Outlet']:
        if endType == 'Head_water':
            pathType = 'path_start_type'
            pathNode = 'path_start_node'
        else:
            pathType = 'path_end_type'
            pathNode = 'path_end_node'
        # 3. Filter only Head_water or Outlet rows for those sections
        rows_to_mirror = df[
            (df['section_id'].isin(single_sections)) &
            (df[pathType] == endType)
            ].copy()

        # 2. Create their "mirror" version
        mirrored = rows_to_mirror.copy()

        # Reverse the direction and slope values
        mirrored['slope']  = -mirrored['slope']
        mirrored['slopeF'] = -mirrored['slopeF']

        # 6️⃣ Update junction_id logic
        # If original is Head_water, mirrored junction_id = path_start_node
        mirrored.loc[rows_to_mirror[pathType] == endType, 'junction_id'] = rows_to_mirror[pathNode].values

        # # If original is Outlet, mirrored junction_id = path_end_node
        # mirrored.loc[rows_to_mirror['path_end_type'] == 'Outlet', 'junction_id'] = rows_to_mirror['path_end_node'].values

        # Optional: mark mirrored rows
        mirrored['mirrored'] = True

        # 3. Append these new rows back to the original DataFrame
        newGraph_expanded = pd.concat([newGraph_expanded, mirrored], ignore_index=True)

        newGraph_expanded = newGraph_expanded.sort_values('mirrored').drop_duplicates(subset = ['junction_id', 'section_id'], keep = 'last')

    return newGraph_expanded

def identify_up_down_network_nodes(df):
    # Split into two temporary DataFrames based on slope sign
    df_up   = df.loc[(df['slope'] < 0)]
    df_down = df.loc[(df['slope'] > 0), ['section_id', 'junction_id', 'slope']]

    # Rename columns to clarify meaning
    df_up   = df_up.rename(columns={'junction_id': 'upstream_node', 'slope': 'UD_slope'})
    df_down = df_down.rename(columns={'junction_id': 'downstream_node', 'slope': 'DU_slope'})

    # Merge them back on section_id
    merged = pd.merge(df_up, df_down, on='section_id', how='outer')

    # Handle NaN slope sections
    df_nan = df.loc[df['slope'].isna(), ['section_id', 'junction_id', 'confidence', 'area','distance','slopeF', 'SE']] 
    
    # For NaN slopes: assume first is upstream, second is downstream
    df_nan_up   = df_nan.groupby('section_id').nth(0).rename(columns={'junction_id': 'upstream_node'})

    df_nan_down = df_nan.groupby('section_id').nth(1).rename(columns={'junction_id': 'downstream_node'})
    df_nan_merged = pd.merge(df_nan_up, df_nan_down[['section_id', 'downstream_node']], on='section_id', how='outer').reset_index()

    # Combine with NaN-based merged data
    merged = pd.concat([merged, df_nan_merged], ignore_index=True)

    # Optional: Reorder columns
    # Columns to move to the front, in desired order
    cols_to_front = ['section_id', 'upstream_node', 'downstream_node']

    # Reorder columns: bring selected ones to the front, keep the rest as is
    new_order = cols_to_front + [col for col in merged.columns if col not in cols_to_front]
    merged = merged[new_order]

    merged = merged.drop(merged[(merged['downstream_node'] == merged['upstream_node']) & (merged['mirrored'] == True)].index)
    return merged

# -----------------------------
# SWOT Linear Mixed Effects slope calculation
# -----------------------------
def rescale_fixed_effects(res, x_scaled_name, x_mean, x_std, z=1.96):
    """
    Rescale fixed effects from a MixedLM fitted on x_scaled = (x - mean)/std
    and compute SE and 95% CI for intercept and slope in original x units.
    
    Handles tiny negative variances due to numerical issues.
    
    Parameters:
        res: fitted MixedLMResults
        x_scaled_name: name of the scaled predictor
        x_mean: mean of original predictor
        x_std: std of original predictor
        z: z-score for confidence interval (default 1.96 for 95% CI)
        
    Returns:
        dict with rescaled beta, intercept, SEs, and 95% CIs
    """
    
    def safe_sqrt(var, tol=1e-12, name=""):
        if var < 0:
            if abs(var) < tol:
                var = 0.0
            else:
                # warnings.warn(f"Variance for {name} is negative ({var}), results may be unreliable.")
                var = np.nan
        return np.sqrt(var)
    
    fe = res.fe_params
    cov_fe = res.cov_params()  # covariance matrix of fixed effects
    
    # --- slope ---
    b_s = fe[x_scaled_name]
    var_b_s = cov_fe.loc[x_scaled_name, x_scaled_name]
    se_b_s = safe_sqrt(var_b_s, name=f"slope {x_scaled_name}")
    
    beta_orig = b_s / x_std
    se_beta_orig = se_b_s / x_std
    ci_beta_orig = (beta_orig - z*se_beta_orig, beta_orig + z*se_beta_orig)
    
    # --- intercept ---
    intercept_name = 'Intercept' if 'Intercept' in fe.index else fe.index[0]
    alpha_s = fe[intercept_name]
    
    Var_alpha_s = cov_fe.loc[intercept_name, intercept_name]
    Cov_alpha_beta = cov_fe.loc[intercept_name, x_scaled_name]
    Var_beta_s = cov_fe.loc[x_scaled_name, x_scaled_name]
    
    alpha_orig = alpha_s - b_s * x_mean / x_std
    Var_alpha_orig = Var_alpha_s + (x_mean / x_std)**2 * Var_beta_s - 2*(x_mean / x_std)*Cov_alpha_beta
    se_alpha_orig = safe_sqrt(Var_alpha_orig, name="intercept")
    ci_alpha_orig = (alpha_orig - z*se_alpha_orig, alpha_orig + z*se_alpha_orig)
    
    return {
        'beta_orig': beta_orig,
        'se_beta_orig': se_beta_orig,
        'ci_beta_orig': ci_beta_orig,
        'intercept_orig': alpha_orig,
        'se_intercept_orig': se_alpha_orig,
        'ci_intercept_orig': ci_alpha_orig
    }

def lme_tuning(D, center = False, scale = False, ignore_warnigs = True):


    data = D.copy()
    if center == True:
        mu        = data['mu'].iloc[0]
        data['x'] = (data['x'] - mu)
    if scale == True:
        s = data['s'].iloc[0]
        data['x'] = data['x'] / s

    # --- model tuning ---
    modelSetup = [[1, '~1'], [1, '~x'], [2, '~1'], [2, '~x'], [2, '~x2'], [2, '~x+x2']]
    complexity = np.array([0,1,2,3,3,4])
    modelSetup = [[1, '~1'], [1, '~x']]
    complexity = np.array([0,1])
    aics, conv, modeldesign = np.zeros(len(modelSetup)), np.zeros(len(modelSetup), dtype = bool), np.zeros(len(modelSetup), dtype = object)



    if ignore_warnigs:  # your condition here
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Fit selected model with reml True
            model = smf.mixedlm('z~x', data, groups=data["time"], re_formula='~x')# remove if using tuning
            fit = model.fit(reml = True)
            # fit = modeldesign[best_idx].fit(reml=True)
    else:
        model = smf.mixedlm('z~x', data, groups=data["time"], re_formula='~x') # remove if using tuning
        fit = model.fit(reml = True)
        # fit   = modeldesign[best_idx].fit(reml=True)

    conv = getattr(fit, "converged", None)

    return 1, '~x', conv, fit
    # return int(modelSetup[best_idx][0]), str(modelSetup[best_idx][1]),convergence, fit

def lme_results(degree,random_setting,fit, D, print_results = False):
    xmean = D['distance'].mean()
    xstd  = D['distance'].std()
    re_dict = fit.random_effects

    # fixed slope
    beta_fixed = fit.fe_params["x"]
    rescaled = rescale_fixed_effects(fit, x_scaled_name='x', x_mean=xmean, x_std=xstd)
    pred_orig = (rescaled['beta_orig']*D['distance'].unique()) + rescaled['intercept_orig']

    if random_setting != '~1':
        # extract slope per group
        group_slopes = {g: beta_fixed + re.iloc[1] for g, re in re_dict.items()}
        group_slopes = pd.Series(group_slopes, name="slope")

        # optionally sort
        group_slopes = group_slopes.sort_values() / xstd
        num_positive = (group_slopes > 0).sum()
        num_negative = (group_slopes < 0).sum()
        num_zero = (group_slopes == 0).sum()
        num_frac = num_positive / (num_positive + num_negative)
    else:
        num_frac = np.nan
    
    
    if print_results == True:
        print(f'Positive slope fraction: {np.round(num_frac, 2)}, (Zero: {num_zero})')
        print("Number of groups with positive slopes:", num_positive)
        print("Number of groups with negative slopes:", num_negative)
        print("Number of groups with slope zero:", num_zero)


        print('Model Convergence: ', getattr(fit, "converged", None))
        print("P-Values:")
        print(fit.pvalues['x'])

        # Rescale values:
        

        print(f"{'Slope (original units)':26s}: {rescaled['beta_orig']:<10.3e}| {'SE':7s}: {rescaled['se_beta_orig']:<10.3e}")
        print(f"{'Slope CI 2.5%:':26s}: {rescaled['ci_beta_orig'][0]:<10.3e}| {'97.5%':7s}: {rescaled['ci_beta_orig'][1]:<10.3e}")

        print(f"{'Intercept (original units)':26s}: {rescaled['intercept_orig']:<10.3e}| {'SE':7s}: {rescaled['se_intercept_orig']:<10.3e}")
        print(f"{'Intercept CI 2.5%':26s}: {rescaled['ci_intercept_orig'][0]:<10.3e}| {'97.5%':7s}: {rescaled['ci_intercept_orig'][1]:<10.3e}")


    # only returning linear slope and certainties
    return rescaled['beta_orig'], rescaled['se_beta_orig'],fit.pvalues['x'], num_frac, pred_orig

def junction_slope_calc(juns, df, directory, returndf = True, save = True):
    grouped = df.groupby('junction_id')
    
    results = pd.DataFrame(columns=['junction_id', 'section_id', 'slope', 'SE', 'slopeP', 'slopeF'])
    sections_seen = {}  # dictionary to map section_id -> row index in results

    for jun in tqdm(juns):
        # dt1 = dt.now()
        A = grouped.get_group(jun)
        
        for s in A['section_id'].unique():

            if s in sections_seen:
                # reuse previous values
                prev_idx       = sections_seen[s]
                slope          = results.at[prev_idx, 'slope'] * -1
                slopeSE        = results.at[prev_idx, 'SE']
                slopeP         = results.at[prev_idx, 'slopeP']
                slopeF         = 1 - results.at[prev_idx, 'slopeF']
                distance       = results.at[prev_idx, 'distance']
                convergence    = results.at[prev_idx, 'convergence']
                random_effects = results.at[prev_idx, 'random_effects']
            else:
                B = A[A['section_id'] == s]
                distance = B['distance'].max() 
                # B =B[B['wse_u'] < 0.25]
                B = B[(B['pass_size'] > 3) &
                    ~((B['pass_size'] <= 5) & (B['section_node_size'] > 30))]
                
                if B.shape[0] == 0:
                    slope = slopeSE = slopeP = slopeF = convergence = random_effects = np.nan
                else:
                    B = B[~B['wse'].isna()] # final check to make sure no Na values get checked
                    degree, random_effects,convergence, fit  = lme_tuning(B, center = True, scale = True, ignore_warnigs=True)
                    slope, slopeSE, slopeP, slopeF, pred     = lme_results(degree, random_effects, fit, B)
                    
            
            # append results
            new_row = pd.DataFrame([{
                'junction_id': jun,
                'section_id': s,
                'slope': slope,
                'SE': slopeSE,
                'slopeP': slopeP,
                'slopeF': slopeF,
                'convergence': convergence,
                'random_effects': random_effects,
                'distance':distance
            }])
            if results.shape[0] == 0:
                results = new_row
            else:
                results = pd.concat([results, new_row], ignore_index=True)

            if s not in sections_seen:
                sections_seen[s] = results.index[-1]
    
    if save == True:
        output_path = directory + f'/output/{continent}_junction_slope_values.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results.to_csv(output_path)
    
    if returndf == True:
        return results

def slope_OLS(data,section, **kwargs):

    #   X = sm.add_constant(data["distance"])
    #   if data['cyclePass'].nunique() == 1:
    #         ols = sm.OLS(data["wse"], X).fit(cov_type="HC0")
    #   else:
    #         ols = sm.OLS(data["wse"], X).fit(cov_type="cluster", cov_kwds={"groups": data["cyclePass"]})

      results = []
      for g, df_g in data.groupby("cyclePass"):
            if len(df_g) > 2:  # avoid degenerate groups
                  Xg = sm.add_constant(df_g["distance"])
                  fit_g = sm.OLS(df_g["wse"], Xg).fit()
                  results.append({
                        "cyclePass": g,
                        "slope": fit_g.params["distance"],
                        "se": fit_g.bse["distance"],
                        "r2": fit_g.rsquared,
                        "n": len(df_g),
                        'intercept':fit_g.params["const"]
                  })
      group_slopes      = pd.DataFrame(results)
      

      # overall summary
      if len(results) > 0:
        # frac_positive = np.mean(group_slopes["slope"] > 0)
        # median_slope = np.median(group_slopes["slope"])

        weighted_slope    = np.average(group_slopes['slope'], weights = group_slopes['n'])
        weighted_fraction = np.average(np.sign(group_slopes['slope']), weights = group_slopes['n'])
        weighted_se       = np.average(np.sign(group_slopes['se']), weights = group_slopes['n'])
        weighted_int       = np.average(np.sign(group_slopes['intercept']), weights = group_slopes['n'])
      else:
            weighted_slope = weighted_fraction = weighted_se =weighted_int = np.nan

      if 'print' in kwargs:
            if kwargs['print'] == True:
                  print("Global (clustered) slope:")
                  print(f"β = {ols.params['distance']:.6f} ± {ols.bse['distance']:.6f}")
                  print(f"95% CI = ({ols.conf_int().loc['distance',0]:.6f}, "
                        f"{ols.conf_int().loc['distance',1]:.6f})")
                  print(f"t = {ols.tvalues['distance']:.2f},  p = {ols.pvalues['distance']:.3g}")

                  print(f"Fraction of positive slopes: {frac_positive:.2%}")
                  print(f"Median slope: {median_slope:.6f}")
      if 'intercept' in kwargs:
          resultDict = {
            'section_id': section,
            'junction_id': data['junction_id'].iloc[0],
            'slope':weighted_slope,
            'SE':weighted_se,
            'slopeF':weighted_fraction, 
            'slopeP':0,
            'convergence':True,
            'intercept': weighted_int
            }
      else:
          resultDict = {
            'section_id': section,
            'junction_id': data['junction_id'].iloc[0],
            'slope':weighted_slope,
            'SE':weighted_se,
            'slopeF':weighted_fraction, 
            'slopeP':0,
            'convergence':True,}
    #   print(resultDict)
      return resultDict

def run_OLS(data, **kwargs):
      unique_sections = (
            data.drop_duplicates(["section_id", "cyclePass", "node_id"])
            .query("pass_size > 2")[["junction_id", "section_id", "distance", "wse", 'cyclePass', 'pass_size']]
      )
      # print('parallel_lmme 3')
      groups = unique_sections.groupby("section_id")
      results = []
      for s, g in tqdm(groups):
            if 'intercept' in kwargs:
                results.append(slope_OLS(g,s, intercept = kwargs['intercept']))
            else:
                results.append(slope_OLS(g,s))
      results_df = pd.DataFrame(results)


      jun_sections = (
            data.sort_values(["junction_id", "section_id", "distance"], ascending=False)
            [["junction_id", "section_id", "distance"]]
            .drop_duplicates(["junction_id", "section_id"])
            )

      merged = jun_sections.merge(
      results_df.rename(columns={"junction_id": "junction_id_src"}), on="section_id", how="left"
      )
      mask = merged["slope"].notna() & (merged["junction_id"] != merged["junction_id_src"])
      merged.loc[mask, "slope"] *= -1
      merged.loc[mask, "slopeF"] *= -1
      merged.drop(columns=["junction_id_src"], inplace=True)
      return merged


def compute_clean_slopes(df):
    con = duckdb.connect()
    con.register("df", df)

    # STEP 1 — initial section-level stats (unchanged)
    con.execute("""
        CREATE OR REPLACE TABLE section_stats AS
        WITH cycle_groups AS (
            SELECT
                junction_id,
                section_id,
                cyclePass,
                COUNT(*) AS n_obs,
                MAX(distance) AS max_dist,
                    CASE WHEN COUNT(*) > 2
                     THEN regr_slope(wse, distance)
                     ELSE NULL
                END AS slope,
                CASE WHEN COUNT(*) > 2
                     THEN regr_intercept(wse, distance)
                     ELSE NULL
                END AS intercept
            FROM df
            GROUP BY junction_id, section_id, cyclePass
        ),
        section_agg AS (
            SELECT
                junction_id,
                section_id,
                MAX(max_dist) AS max_distance,
                SUM(n_obs) AS sum_w,
                SUM(n_obs * slope) AS sum_wx,
                SUM(n_obs * slope * slope) AS sum_wx2,
                SUM(n_obs * intercept) AS sum_wb0,
                SUM(
                    n_obs *
                    CASE
                        WHEN slope > 0 THEN 1
                        WHEN slope < 0 THEN -1
                        ELSE 0
                    END
                ) / SUM(n_obs) AS slopeF
            FROM cycle_groups
            GROUP BY junction_id, section_id
        )
        SELECT
            junction_id,
            section_id,
            max_distance AS distance,
            (sum_wx / sum_w) AS slope,
            sqrt(
                GREATEST(
                    sum_wx2 / sum_w - POWER(sum_wx / sum_w, 2),
                    0
                )
            ) AS SE,
            (sum_wb0 / sum_w) AS intercept,
            slopeF
        FROM section_agg
    """)

    # STEP 2 — per-cyclePass slopes + per-section MAD
    con.execute("""
        CREATE OR REPLACE TABLE cycle_slopes AS
        WITH slopes AS (
            SELECT
                junction_id,
                section_id,
                cyclePass,
                COUNT(*) AS n_obs,
                CASE WHEN COUNT(*) > 2
                     THEN regr_slope(wse, distance)
                     ELSE NULL
                END AS slope
            FROM df
            GROUP BY junction_id, section_id, cyclePass
        ),
        m AS (
            SELECT
                *,
                median(slope) FILTER (WHERE slope IS NOT NULL)
                    OVER (PARTITION BY junction_id, section_id) AS med_slope
            FROM slopes
        ),
        mad_calc AS (
            SELECT
                *,
                abs(slope - med_slope) AS abs_dev,
                median(abs(slope - med_slope))
                    FILTER (WHERE abs(slope - med_slope) IS NOT NULL)
                    OVER (PARTITION BY junction_id, section_id) AS mad_slope
            FROM m
        )
        SELECT
            *,
            CASE WHEN mad_slope IS NULL OR mad_slope = 0 THEN NULL
                 ELSE abs_dev / (mad_slope * 1.4826)
            END AS mad_score
        FROM mad_calc
    """)

    # STEP 3 — outlier cyclePasses (section-local MAD + sign/slopeF rules)
    con.execute("""
        CREATE OR REPLACE TABLE outliers AS
        SELECT
            cs.junction_id,
            cs.section_id,
            cs.cyclePass,
            cs.slope,
            cs.mad_score,
            ss.slopeF
        FROM cycle_slopes cs
        JOIN section_stats ss USING (junction_id, section_id)
        WHERE cs.slope IS NOT NULL
          AND cs.mad_score > 3.5
          AND SIGN(cs.slope) <> SIGN(ss.slopeF)
          AND ABS(ss.slopeF) > 0.3
    """)

    # STEP 4 — filter df by removing only those outlier cyclePasses
    con.execute("""
        CREATE OR REPLACE TABLE df_filtered AS
        SELECT *
        FROM df
        WHERE (junction_id, section_id, cyclePass) NOT IN (
            SELECT junction_id, section_id, cyclePass FROM outliers
        )
    """)

    # STEP 5 — recompute section-level stats on filtered data, preserving all sections
    con.execute("""
            CREATE OR REPLACE TABLE section_stats_clean AS
            WITH
            section_ids AS (
                SELECT DISTINCT junction_id, section_id FROM df
            ),

            -- 1) TRUE max distance from raw (unfiltered) data
            true_max_dist AS (
                SELECT
                    junction_id,
                    section_id,
                    MAX(distance) AS true_max_distance
                FROM df
                GROUP BY junction_id, section_id
            ),

            cycle_groups AS (
                SELECT
                    junction_id,
                    section_id,
                    cyclePass,
                    COUNT(*) AS n_obs,
                    MAX(distance) AS max_dist,
                    CASE WHEN COUNT(*) > 2
                     THEN regr_slope(wse, distance)
                     ELSE NULL
                END AS slope,
                CASE WHEN COUNT(*) > 2
                     THEN regr_intercept(wse, distance)
                     ELSE NULL
                END AS intercept
                FROM df_filtered
                GROUP BY junction_id, section_id, cyclePass
            ),

            cycle_agg AS (
                SELECT
                    s.junction_id,
                    s.section_id,
                    MAX(c.max_dist) AS max_distance,
                    SUM(c.n_obs) AS sum_w,
                    SUM(c.n_obs * c.slope) AS sum_wx,
                    SUM(c.n_obs * c.slope * c.slope) AS sum_wx2,
                    SUM(c.n_obs * c.intercept) AS sum_wb0,
                    SUM(
                        c.n_obs *
                        CASE
                            WHEN c.slope > 0 THEN 1
                            WHEN c.slope < 0 THEN -1
                            ELSE 0
                        END
                    ) AS signed_sum,
                    SUM(c.n_obs) AS total_w
                FROM section_ids s
                LEFT JOIN cycle_groups c
                    ON s.junction_id = c.junction_id
                AND s.section_id = c.section_id
                GROUP BY s.junction_id, s.section_id
            )

            SELECT
                ca.junction_id,
                ca.section_id,

                -- use TRUE max distance before filtering
                tmd.true_max_distance AS distance,

                CASE WHEN sum_w IS NULL OR sum_w = 0 THEN NULL
                    ELSE (sum_wx / sum_w)
                END AS slope,

                CASE WHEN sum_w IS NULL OR sum_w = 0 THEN NULL
                    ELSE sqrt(GREATEST(sum_wx2 / sum_w - POWER(sum_wx / sum_w, 2), 0))
                END AS SE,

                CASE WHEN sum_w IS NULL OR sum_w = 0 THEN NULL
                    ELSE (sum_wb0 / sum_w)
                END AS intercept,

                CASE WHEN total_w IS NULL OR total_w = 0 THEN 0
                    ELSE signed_sum / total_w
                END AS slopeF,

                CASE WHEN EXISTS (
                    SELECT 1 FROM outliers o
                    WHERE o.junction_id = ca.junction_id
                    AND o.section_id = ca.section_id
                )
                THEN TRUE ELSE FALSE END AS outlier_removed_flag

            FROM cycle_agg ca
            LEFT JOIN true_max_dist tmd
                ON ca.junction_id = tmd.junction_id
            AND ca.section_id = tmd.section_id;

    """)

    return {
        # "section_stats_original": con.table("section_stats").fetchdf(),
        # "cycle_slopes": con.table("cycle_slopes").fetchdf(),
        # "outliers": con.table("outliers").fetchdf(),
        "section_stats_clean": con.table("section_stats_clean").fetchdf(),
        # "df_filtered": con.table("df_filtered").fetchdf()
    }




def duckdb_ols_slope(df):    
    result = duckdb.query("""
    WITH all_groups AS (
        SELECT
            junction_id,
            section_id,
            cyclePass,
            MAX(distance) AS max_dist,
            COUNT(*) AS n_obs
        FROM df
        GROUP BY junction_id, section_id, cyclePass
    ),
    regression_groups AS (
        SELECT
            junction_id,
            section_id,
            cyclePass,
            regr_slope(wse, distance) AS slope,
            regr_intercept(wse, distance) AS intercept,
            COUNT(*) AS n_obs
        FROM df
        GROUP BY junction_id, section_id, cyclePass
        HAVING COUNT(*) > 2
    ),
    group_stats AS (
        SELECT
            a.junction_id,
            a.section_id,
            a.cyclePass,
            a.max_dist,
            COALESCE(r.slope, NULL) AS slope,
            COALESCE(r.intercept, NULL) AS intercept,
            a.n_obs
        FROM all_groups a
        LEFT JOIN regression_groups r
        USING (junction_id, section_id, cyclePass)
    ),
    section_agg AS (
        SELECT
            junction_id,
            section_id,
            MAX(max_dist) AS max_distance,
            SUM(n_obs) AS sum_w,
            SUM(n_obs * slope) AS sum_wx,
            SUM(n_obs * slope * slope) AS sum_wx2,
            SUM(n_obs * intercept) AS sum_wb0,
            SUM(n_obs * CASE WHEN slope > 0 THEN 1
                            WHEN slope < 0 THEN -1
                            ELSE 0 END) / SUM(n_obs) AS weighted_signed_fraction
        FROM group_stats
        GROUP BY junction_id, section_id
    )
    SELECT
        junction_id,
        section_id,
        max_distance AS distance,
        (sum_wx / sum_w) AS slope,
        sqrt(GREATEST(sum_wx2 / sum_w - (sum_wx / sum_w) * (sum_wx / sum_w), 0)) AS SE,
        (sum_wb0 / sum_w) AS intercept,
        weighted_signed_fraction AS slopeF,
        0 AS slopeP,
        CASE WHEN sum_wx / sum_w IS NULL THEN FALSE ELSE TRUE END AS convergence
    FROM section_agg
    ORDER BY junction_id, section_id
    """).to_df()


    result["opposite_sign"] = ((result["slope"] * result["slopeF"]) < 0) & (result["slopeF"].abs() > 0.3)
    outlier_candidates = result[result['opposite_sign'] == True]



    return result
# -----------------------------
# Change from MultiGraph to Graph
# -----------------------------
def detect_ud_keys(G):
    """
    Try to detect the upstream/downstream attribute keys from the graph's edges.
    Fallbacks to 'upstream_node' / 'downstream_node' if not found.
    """
    cand_up   = ["upstream_node", "upstream", "up_node", "u_node"]
    cand_down = ["downstream_node", "downstream", "down_node", "d_node"]

    up_key   = None
    down_key = None
    for _, _, data in G.edges(data=True):
        if up_key is None:
            for k in cand_up:
                if k in data:
                    up_key = k
                    break
        if down_key is None:
            for k in cand_down:
                if k in data:
                    down_key = k
                    break
        if up_key and down_key:
            break

    return up_key or "upstream_node", down_key or "downstream_node"

def as_number(x):
    """Best effort to turn x into a float; fallback to +inf if missing/invalid."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float("inf")
        return float(x)
    except Exception:
        return float("inf")

def choose_min_distance(edge_dicts):
    """Pick the edge dict with the smallest numeric 'distance' (inf if missing)."""
    return min(edge_dicts, key=lambda e: as_number(e.get("distance", float("inf"))))

def edges_agree_on_updown(edge_dicts, up_key, down_key):
    """True if all edges share the same upstream & downstream values."""
    ups = [e.get(up_key) for e in edge_dicts]
    downs = [e.get(down_key) for e in edge_dicts]
    return (len(set(ups)) == 1) and (len(set(downs)) == 1)

def collapse_group(u, v, edge_datas, up_key, down_key):
    """
    Apply your rules to a list of parallel edges (attribute dicts).
    Returns (attrs, case_label, chosen_key).
    Rules:
      - if only one of the two paths is confidence R: keep attributes of that edge
      - if both U: keep attributes of edge with smallest 'distance'
      - if both are R:
          - if upstream_node and downstream_node are the same for both: keep smallest 'distance'
          - else: set confidence U, upstream/downstream NaN, area 0.0, keep smallest 'distance'
    """
    # Normalize fields
    for e in edge_datas:
        # Ensure 'confidence' is present, default 'U'
        conf = e.get("confidence", None)
        e["confidence"] = "U" if conf is None else str(conf).upper()
        # Ensure 'distance' exists for comparison
        if "distance" not in e:
            e["distance"] = float("inf")

    Rs = [e for e in edge_datas if e["confidence"] == "R"]
    Us = [e for e in edge_datas if e["confidence"] == "U"]

    # Case 1: exactly one R
    if len(Rs) == 1:
        chosen = Rs[0]
        case = "one_R_kept"
        attrs = {k: v for k, v in chosen.items() if k != "_key"}
        return attrs, case, chosen.get("_key")

    # Case 2: all U (no R)
    if len(Rs) == 0:
        chosen = choose_min_distance(Us if Us else edge_datas)
        case = "both_U_pick_min_distance" if len(edge_datas) == 2 else "all_U_pick_min_distance"
        attrs = {k: v for k, v in chosen.items() if k != "_key"}
        return attrs, case, chosen.get("_key")

    # Case 3: two or more R
    if edges_agree_on_updown(Rs, up_key, down_key):
        chosen = choose_min_distance(Rs)
        case = "both_R_agree_pick_min_distance" if len(edge_datas) == 2 else "multi_R_agree_pick_min_distance"
        attrs = {k: v for k, v in chosen.items() if k != "_key"}
        attrs["confidence"] = "R"  # keep R
        return attrs, case, chosen.get("_key")
    else:
        # Disagree on upstream/downstream → demote to U, zero area, NaN up/down
        chosen = choose_min_distance(edge_datas)
        case = "both_R_disagree_demote_to_U"
        attrs = {k: v for k, v in chosen.items() if k != "_key"}
        attrs["confidence"] = "U"
        attrs[up_key] = math.nan
        attrs[down_key] = math.nan
        attrs["area"] = 0.0
        return attrs, case, chosen.get("_key")

def multigraph_to_collapsed_graph(G, directory, continent):
    """
    Collapse a MultiGraph/MultiDiGraph into a simple Graph/DiGraph using the rules above.
    Returns (H, summary_rows, up_key, down_key).
    """
    if not G.is_multigraph():
        # Nothing to do; just convert to simple Graph/DiGraph
        H = nx.DiGraph() if G.is_directed() else nx.Graph()
        H.graph.update(G.graph)
        H.add_nodes_from((n, G.nodes[n]) for n in G.nodes)
        H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True))
        return H, [], "upstream_node", "downstream_node"

    up_key, down_key = detect_ud_keys(G)

    H = nx.DiGraph() if G.is_directed() else nx.Graph()
    H.graph.update(G.graph)
    H.add_nodes_from((n, G.nodes[n]) for n in G.nodes)

    # Group edges by node pair (respecting direction for DiGraph)
    groups = defaultdict(list)
    if G.is_directed():
        for u, v, k, d in G.edges(keys=True, data=True):
            groups[(u, v)].append({**d, "_key": k})
    else:
        for u, v, k, d in G.edges(keys=True, data=True):
            if isinstance(u, float):
                print('U', u, v)
            if isinstance(v, float):
                print('V', v, u)
            a, b = (u, v) if u <= v else (v, u)
            groups[(a, b)].append({**d, "_key": k})

    summary_rows = []
    for (u, v), ds in groups.items():
        if len(ds) == 1:
            chosen = ds[0]
            H.add_edge(u, v, **{k: v for k, v in chosen.items() if k != "_key"})
            summary_rows.append({
                "u": u,
                "v": v,
                "parallel_count": 1,
                "case": "single_edge_copy",
                "chosen_key": chosen.get("_key"),
                "final_confidence": chosen.get("confidence"),
                "final_distance": chosen.get("distance"),
            })
            continue

        attrs, case, chosen_key = collapse_group(u, v, ds, up_key, down_key)
        H.add_edge(u, v, **attrs)
        summary_rows.append({
            "u": u,
            "v": v,
            "parallel_count": len(ds),
            "case": case,
            "chosen_key": chosen_key,
            "final_confidence": attrs.get("confidence"),
            "final_distance": attrs.get("distance"),
        })

    output_path = directory + f"/output/{continent}_slope_single.pkl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(H, f)
    
# -----------------------------
# Get relative node junction distances
# -----------------------------
def merge_SWOT_node_distance(df_swot_nodes, dfNode, dfNetworkEdge, dfNetworkNode):
    ################################################################
    # compute distance from junction to node
    dfNodeJunctionDistance = node_junction_distance(dfNode, dfNetworkEdge, dfNetworkNode)
    
    ################################################################
    # Merge SWOT observations with SWORD nodes and 
    df_swot_nodes = df_swot_nodes.merge(dfNodeJunctionDistance, how = 'outer', on = 'node_id')

    df_swot_nodes['pass_size'] = (
        df_swot_nodes
        .groupby(['junction_id', 'section_id', 'cyclePass'])['junction_id']
        .transform('size')
    )

    df_swot_nodes.loc[df_swot_nodes['pass_size'].isna(), 'pass_size'] = 0


    return df_swot_nodes
# -----------------------------
# Create Graph from slope SWORD Graph with SWOT slope information
# -----------------------------
def create_slopeGraph(dfTSingle, directory, continent):
    
    with open(directory + f'/output/{continent}_MultiDirected.pkl', 'rb') as f:
        DG = pickle.load(f)

    
    # Create a new MultiGraph
    slopeGraph = nx.MultiGraph()
    # Select which attributes you want to keep (for example 'attr1' and 'attr2')
    attributes_to_keep = ['x', 'y', 'subnetwork_id', 'node_type']
    for node, data in DG.nodes(data=True):
        if data['node_type'] in ['Head_water', 'Outlet', 'Junction']:
            filtered_data = {k: data[k] for k in attributes_to_keep if k in data}
            slopeGraph.add_node(node, **filtered_data)


    # Add edges with key = (junction_id, section_id) and other attributes
    for _, row in dfTSingle.iterrows():
        edge_key = row["section_id"]
        edge_attr = row[['section_id', 'upstream_node', 'downstream_node', 'confidence', 'area', 'distance',
                         'UD_slope', 'DU_slope','slopeF', 'SE']].to_dict() # add distance
        if edge_attr['confidence'] == 'U':
            edge_attr['upstream_node'] = np.nan
            edge_attr['downstream_node'] = np.nan
        if edge_attr['distance'] == 0:
            edge_attr['distance'] = 1
        # else:
        #     edge_attr['classification'] = 'R'
        slopeGraph.add_edge(row["upstream_node"], row["downstream_node"], key=edge_key, **edge_attr)
    return slopeGraph
# -----------------------------
# Merge SWOT slopes and SWORD dataframes
# -----------------------------
def merge_SWOT_SWORD(dfNetworkNode, dfNetworkEdge, dfSwotSlope, slopeFractionThreshold):
        dfNetworkEdgeF = dfNetworkEdge[['reach_id', 'path_seg', 'path_start_node', 'path_end_node']] 
        dfNetworkNodeF = dfNetworkNode[['node_id', 'node_type']]
        
        # Double merge Edges and Junctions to get start and end junctions
        dfNetworkEdgeF = dfNetworkEdgeF.merge(dfNetworkNodeF[['node_id', 'node_type']], 
                                            left_on = 'path_start_node', 
                                            right_on = 'node_id', how = 'left').drop('node_id', axis = 1)
        dfNetworkEdgeF = dfNetworkEdgeF.rename(columns = {'node_type':'path_start_type'})
        
        dfNetworkEdgeF = dfNetworkEdgeF.merge(dfNetworkNode, 
                                            left_on = 'path_end_node', 
                                            right_on = 'node_id', how = 'left').drop('node_id', axis = 1)
        dfNetworkEdgeF = dfNetworkEdgeF.rename(columns = {'node_type':'path_end_type'})
        dfNetworkEdgeF = dfNetworkEdgeF[['path_seg','path_start_type', 'path_end_type', 'path_end_node', 'path_start_node']].drop_duplicates()

        # Merge SWORD Graph information with SWOT slopes
        dfT = dfSwotSlope.merge(dfNetworkEdgeF[['path_seg','path_start_type', 'path_end_type', 'path_end_node', 'path_start_node']]
                                , left_on = 'section_id', right_on = 'path_seg', how = 'left').drop('path_seg', axis = 1)


        # conditions = [
        #     (dfT['slopeF'] < slopeFractionThreshold[0]) | (dfT['slopeF'] > slopeFractionThreshold[1])
        #     ]
        # choices = ['U', 'R']
        # dfT['confidence'] = np.select(conditions, choices, default='U')
        
        dfT['confidence'] = 'U'
        dfT.loc[(dfT['slopeF'] < slopeFractionThreshold[0]) | (dfT['slopeF'] > slopeFractionThreshold[1]), 'confidence'] = 'R'


        dfT['area'] = np.where(
            dfT['confidence'] == 'U',
            0,  # if unreliable, area = 0
            0.5 * np.maximum(abs(dfT['slope']), 0) * dfT['distance']**2
        )
        return dfT

# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",required=True)
    ap.add_argument("--continent",required=True)
    ap.add_argument("--fraction_low",type=float,default=0.2)
    ap.add_argument("--fraction_high",type=float,default=8)

    args = ap.parse_args()


    directory = args.dir
    continent = args.continent

    slopeFractionThreshold = [args.fraction_low, args.fraction_high]
    # Check if external SWOT data directory exists
    swot_data_dir = '/Volumes/SWORD_DATA/data/swot/RiverSP_D_parq/node'
    if not os.path.isdir(swot_data_dir):
        raise FileNotFoundError(f"SWOT data directory '{swot_data_dir}' does not exist.")

    ################################################################
    # Open files
    df, dfNode = load_sword_data(data_dir=directory + '/data/', continent = continent, include_nodes = True)

    dfNetworkNode = gpd.read_file(directory + f'/output/{continent}_network_nodes.gpkg')
    dfNetworkEdge = gpd.read_file(directory + f'/output/{continent}_network_edges.gpkg')
    

    ################################################################
    # Transform SWOT files
    # Count raw observations before filtering (for comparison)
    con_temp = duckdb.connect(":memory:")
    swot_data_dir = '/Volumes/SWORD_DATA/data/swot/RiverSP_D_parq/node'
    parquet_files = [f for f in glob.glob(os.path.join(swot_data_dir, '*.parquet')) 
                     if not os.path.basename(f).startswith('._')]
    escaped_files = [f.replace("'", "''") for f in parquet_files]
    files_array = "[" + ", ".join(f"'{f}'" for f in escaped_files) + "]"
    node_df_temp = pd.DataFrame({"node_id": dfNode['node_id'].to_list()})
    node_df_temp['node_id'] = node_df_temp['node_id'].astype('str')
    con_temp.register("node_ids_temp", con_temp.from_df(node_df_temp))
    
    raw_count = con_temp.execute(f"""
        SELECT COUNT(*) FROM read_parquet({files_array})
        WHERE node_id IN (SELECT node_id FROM node_ids_temp)
    """).fetchone()[0]
    print(f"Raw SWOT observations (before quality filters): {raw_count:,}")
    
    df_swot_nodes = open_SWOT_files(dfNode['node_id'].to_list(), directory,continent=continent, uncertainty_threshold = None)
    print(f"Number of swot nodes after quality filters & deduplication: {df_swot_nodes.shape[0]:,}")
    print(f"Filtered out: {raw_count - df_swot_nodes.shape[0]:,} observations ({100*(raw_count - df_swot_nodes.shape[0])/raw_count:.1f}%)")
    df_swot_nodes = merge_SWOT_node_distance(df_swot_nodes, dfNode, dfNetworkEdge, dfNetworkNode)
    print(f"Number of swot nodes in file after merge: {df_swot_nodes.shape[0]}")
    df_to_parquet_duckdb_snappy(df_swot_nodes, directory + f"/output/{continent}/{continent}_swot_nodes.parquet")

    ################################################################
    # Calculate SWOT Junction slopes
    # juns        = df_swot_nodes['junction_id'].unique()
    # dfSwotSlope = junction_slope_calc(juns, df_swot_nodes, directory, returndf = True, save = True)
    # print('start parallel lmme')
    # dfSwotSlope = parallel_lmme(df_swot_nodes, returndf=True, save = True, continent = continent, directory = directory)   
    # dfSwotSlope = run_OLS(df_swot_nodes)
    
    #  dfSwotSlope = duckdb_ols_slope(df_swot_nodes)
    
    result = compute_clean_slopes(df_swot_nodes)
    dfSwotSlope = result["section_stats_clean"]
    output_path = directory + f'/output/{continent}/{continent}_swot_slopes.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dfSwotSlope.to_csv(output_path)

    ################################################################
    # Calculate SWOT Junction slopes
    dfT = merge_SWOT_SWORD(dfNetworkNode, dfNetworkEdge, dfSwotSlope, slopeFractionThreshold)

    dfT       = add_end_rows(dfT)
    dfTSingle = identify_up_down_network_nodes(dfT)

    slopeGraph = create_slopeGraph(dfTSingle, directory, continent)
    
    output_path = directory + f"/output/{continent}_slope.pkl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(slopeGraph, f)

    # Turn MultiGraph into Graph
    multigraph_to_collapsed_graph(slopeGraph, directory, continent)

