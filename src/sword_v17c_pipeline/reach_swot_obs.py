"""
Reach-level SWOT observation aggregation (slope, wse, width).

Computes slope_obs, wse_obs, and width_obs statistics for each reach by:
1. Loading SWOT node observations with quality filters
2. For slope: OLS regression (wse ~ p_dist_out) per cycle/pass group
3. For wse/width: direct aggregation across all valid observations
4. Weighted aggregation for slope (weight = n_nodes in group)

Output columns added to reaches table:
- slope_obs_mean/std/median/range: slope statistics (m/km)
- slope_obs_n: total observations used for slope
- slope_obs_n_passes: number of cycle/pass groups
- slope_obs_q: quality flag (bit flags)
- wse_obs_mean/std/median/range: WSE statistics (m)
- width_obs_mean/std/median/range: width statistics (m)
- n_obs: total node observations
- swot_obs_source: 'node' (computed from SWOT node data)

Quality flag bits (slope_obs_q):
- 0: good (no issues)
- 1: negative slope (possible backwater/tidal)
- 2: low n_passes (<10)
- 4: high variability (std > 2*|mean|)
- 8: extreme slope (|slope| > 50 m/km)
- 16: below noise floor (clipped to positive)
"""

import argparse
import glob
import os

import duckdb

from sword_duckdb.swot_filters import (
    DARK_FRAC_MAX,
    QUALITY_MAX,
    SENTINEL,
    WIDTH_MAX,
    WSE_MAX,
    WSE_MIN,
    XTRACK_MAX,
    XTRACK_MIN,
)


def compute_reach_swot_obs(
    swot_node_dir: str,
    sword_db_path: str,
    region: str,
    min_nodes_per_pass: int = 3,
    noise_floor: float = 0.1,
    noise_clip_value: float = 1e-5,
) -> dict:
    """
    Compute reach-level SWOT observations (slope, wse, width) from node data.

    Quality filters (from swot_filters module):
    - WSE sentinel removal (wse != SENTINEL)
    - WSE quality <= QUALITY_MAX
    - Dark water fraction <= DARK_FRAC_MAX or NULL
    - Cross-track distance XTRACK_MIN-XTRACK_MAX or NULL
    - Crossover calibration quality <= QUALITY_MAX or NULL
    - Ice climatology = 0 (likely not ice covered)
    - Valid time_str

    Args:
        swot_node_dir: Path to SWOT node parquet files
        sword_db_path: Path to SWORD DuckDB database
        region: Region code (NA, SA, EU, AF, AS, OC)
        min_nodes_per_pass: Min nodes per cycle/pass for valid slope (default 3)
        noise_floor: Slopes with |slope| < noise_floor (m/km) are clipped (default 0.1)
        noise_clip_value: Value to clip noise floor slopes to (default 1e-5 m/km)

    Returns:
        dict with 'stats' DataFrame and 'n_reaches' count
    """

    # Find parquet files
    parquet_files = [
        f
        for f in glob.glob(os.path.join(swot_node_dir, "*.parquet"))
        if not os.path.basename(f).startswith("._")
    ]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {swot_node_dir}")

    # Build file list for DuckDB
    escaped = [f.replace("'", "''") for f in parquet_files]
    files_sql = "[" + ", ".join(f"'{f}'" for f in escaped) + "]"

    con = duckdb.connect(":memory:")

    # Get reach_ids for this region from SWORD
    sword_con = duckdb.connect(sword_db_path, read_only=True)
    reach_ids = sword_con.execute(f"""
        SELECT DISTINCT reach_id FROM reaches WHERE region = '{region}'
    """).fetchdf()
    sword_con.close()

    con.register("sword_reaches", reach_ids)

    # Detect available columns for dynamic filtering
    # Use union_by_name to handle schema differences between old/new parquet files
    sample_query = (
        f"SELECT * FROM read_parquet({files_sql}, union_by_name=true) LIMIT 1"
    )
    try:
        sample_df = con.execute(sample_query).df()
        colnames = set(c.lower() for c in sample_df.columns.tolist())
    except Exception:
        colnames = set()

    # Build dynamic filtering conditions using swot_filters constants
    conditions = []

    # WSE column detection (prefer new names, fallback to old)
    has_wse = "wse" in colnames
    has_wse_sm = "wse_sm" in colnames
    wse_col = "wse" if has_wse else ("wse_sm" if has_wse_sm else "wse")

    # WSE sentinel filter
    conditions.append(f"NULLIF({wse_col}, {SENTINEL}) IS NOT NULL")

    # WSE quality filter
    if "wse_q" in colnames:
        conditions.append(f"COALESCE(wse_q, 3) <= {QUALITY_MAX}")
    elif "wse_sm_q" in colnames:
        conditions.append(f"COALESCE(wse_sm_q, 3) <= {QUALITY_MAX}")

    # Dark water fraction filter
    if "dark_frac" in colnames and "dark_water_frac" in colnames:
        conditions.append(
            f"((COALESCE(dark_frac, dark_water_frac) <= {DARK_FRAC_MAX})"
            " OR (dark_frac IS NULL AND dark_water_frac IS NULL))"
        )
    elif "dark_frac" in colnames:
        conditions.append(f"((dark_frac <= {DARK_FRAC_MAX}) OR (dark_frac IS NULL))")
    elif "dark_water_frac" in colnames:
        conditions.append(
            f"((dark_water_frac <= {DARK_FRAC_MAX}) OR (dark_water_frac IS NULL))"
        )

    # Cross-track distance filter
    if "xtrk_dist" in colnames and "cross_track_dist" in colnames:
        conditions.append(
            f"((ABS(COALESCE(xtrk_dist, cross_track_dist)) BETWEEN {XTRACK_MIN} AND {XTRACK_MAX})"
            " OR (xtrk_dist IS NULL AND cross_track_dist IS NULL))"
        )
    elif "xtrk_dist" in colnames:
        conditions.append(
            f"((ABS(xtrk_dist) BETWEEN {XTRACK_MIN} AND {XTRACK_MAX}) OR (xtrk_dist IS NULL))"
        )
    elif "cross_track_dist" in colnames:
        conditions.append(
            f"((ABS(cross_track_dist) BETWEEN {XTRACK_MIN} AND {XTRACK_MAX})"
            " OR (cross_track_dist IS NULL))"
        )

    # Crossover calibration quality filter
    if "xovr_cal_q" in colnames:
        conditions.append(f"(xovr_cal_q <= {QUALITY_MAX} OR xovr_cal_q IS NULL)")

    # Ice climatology filter
    if "ice_clim_f" in colnames:
        conditions.append("ice_clim_f = 0")

    # Valid time_str filter
    if "time_str" in colnames:
        conditions.append("time_str IS NOT NULL AND time_str != ''")

    # Reach filter (no continent filter - rely on reach_id matching to SWORD region)
    conditions.append(
        "CAST(reach_id AS BIGINT) IN (SELECT reach_id FROM sword_reaches)"
    )

    where_clause = " AND ".join(conditions)

    # Step 1: Load filtered SWOT node data and compute per-pass slopes
    query = f"""
    WITH filtered_nodes AS (
        SELECT
            CAST(reach_id AS BIGINT) AS reach_id,
            CAST(node_id AS BIGINT) AS node_id,
            COALESCE(wse, wse_sm) AS wse,
            width,
            p_dist_out,
            cycle,
            pass
        FROM read_parquet({files_sql}, union_by_name=true)
        WHERE {where_clause}
          AND width IS NOT NULL AND width > 0
    ),

    -- Filter for valid bounds (prevents STDDEV_SAMP overflow)
    bounded_nodes AS (
        SELECT * FROM filtered_nodes
        WHERE width > 0 AND width < {WIDTH_MAX}
          AND wse > {WSE_MIN} AND wse < {WSE_MAX}
    ),

    -- Aggregate WSE/width per reach (all observations)
    reach_wse_width AS (
        SELECT
            reach_id,
            AVG(wse) AS wse_obs_mean,
            MEDIAN(wse) AS wse_obs_median,
            CASE WHEN COUNT(*) > 1 THEN STDDEV_SAMP(wse) ELSE NULL END AS wse_obs_std,
            MAX(wse) - MIN(wse) AS wse_obs_range,
            AVG(width) AS width_obs_mean,
            MEDIAN(width) AS width_obs_median,
            CASE WHEN COUNT(*) > 1 THEN STDDEV_SAMP(width) ELSE NULL END AS width_obs_std,
            MAX(width) - MIN(width) AS width_obs_range,
            COUNT(*) AS n_obs
        FROM bounded_nodes
        GROUP BY reach_id
    ),

    -- Step 2: Compute slope per reach + cycle/pass using OLS regression
    pass_slopes AS (
        SELECT
            reach_id,
            cycle,
            pass,
            COUNT(*) AS n_nodes,
            -- OLS slope: Cov(x,y) / Var(x) where x=p_dist_out, y=wse
            CASE
                WHEN COUNT(*) >= {min_nodes_per_pass}
                THEN regr_slope(wse, p_dist_out)
                ELSE NULL
            END AS slope
        FROM filtered_nodes
        GROUP BY reach_id, cycle, pass
    ),

    -- Keep only valid slopes
    valid_slopes AS (
        SELECT * FROM pass_slopes
        WHERE slope IS NOT NULL
          AND ABS(slope) < 0.1  -- sanity check: slope < 10% grade
    ),

    -- Step 3: Weighted aggregation per reach (convert m/m to m/km: *1000)
    reach_stats AS (
        SELECT
            reach_id,

            -- Weighted mean: sum(w * x) / sum(w) where w = n_nodes
            -- Multiply by 1000 to convert m/m to m/km
            (SUM(n_nodes * slope) / SUM(n_nodes)) * 1000 AS slope_obs_mean,

            -- Weighted variance for std (also convert to m/km)
            SQRT(
                GREATEST(
                    SUM(n_nodes * slope * slope) / SUM(n_nodes)
                    - POWER(SUM(n_nodes * slope) / SUM(n_nodes), 2),
                    0
                )
            ) * 1000 AS slope_obs_std,

            -- Median (unweighted, convert to m/km)
            MEDIAN(slope) * 1000 AS slope_obs_median,

            -- Range (convert to m/km)
            (MAX(slope) - MIN(slope)) * 1000 AS slope_obs_range,

            -- Counts
            SUM(n_nodes) AS slope_obs_n,
            COUNT(*) AS slope_obs_n_passes

        FROM valid_slopes
        GROUP BY reach_id
    ),

    -- Step 4: Apply noise floor clipping
    -- If |slope| < noise_floor, clip to small positive value (assumes downstream flow)
    reach_stats_clipped AS (
        SELECT
            reach_id,
            CASE
                WHEN ABS(slope_obs_mean) < {noise_floor} THEN {noise_clip_value}
                ELSE slope_obs_mean
            END AS slope_obs_mean,
            slope_obs_std,
            slope_obs_median,
            slope_obs_range,
            slope_obs_n,
            slope_obs_n_passes,
            -- Track if clipped
            ABS(slope_obs_mean) < {noise_floor} AS was_clipped
        FROM reach_stats
    ),

    -- Step 5: Add quality flags
    reach_stats_with_flags AS (
        SELECT
            reach_id,
            slope_obs_mean,
            slope_obs_std,
            slope_obs_median,
            slope_obs_range,
            slope_obs_n,
            slope_obs_n_passes,
            -- Quality flag (bit flags):
            -- 1: negative slope
            -- 2: low n_passes (<10)
            -- 4: high variability (std > 2*|mean|)
            -- 8: extreme slope (|slope| > 50 m/km)
            -- 16: below noise floor (clipped to positive)
            (CASE WHEN slope_obs_mean < 0 THEN 1 ELSE 0 END)
            + (CASE WHEN slope_obs_n_passes < 10 THEN 2 ELSE 0 END)
            + (CASE WHEN slope_obs_std > 2 * ABS(slope_obs_mean) THEN 4 ELSE 0 END)
            + (CASE WHEN ABS(slope_obs_mean) > 50 THEN 8 ELSE 0 END)
            + (CASE WHEN was_clipped THEN 16 ELSE 0 END)
            AS slope_obs_q
        FROM reach_stats_clipped
    )

    -- Final: join slope stats with wse/width stats, add source flag
    SELECT
        COALESCE(s.reach_id, w.reach_id) AS reach_id,
        -- Slope stats
        s.slope_obs_mean,
        s.slope_obs_std,
        s.slope_obs_median,
        s.slope_obs_range,
        s.slope_obs_n,
        s.slope_obs_n_passes,
        s.slope_obs_q,
        -- WSE/width stats from node aggregation
        w.wse_obs_mean,
        w.wse_obs_median,
        w.wse_obs_std,
        w.wse_obs_range,
        w.width_obs_mean,
        w.width_obs_median,
        w.width_obs_std,
        w.width_obs_range,
        w.n_obs,
        -- Source flag: 'node' indicates computed from SWOT node data (not PODAAC reach product)
        'node' AS swot_obs_source
    FROM reach_stats_with_flags s
    FULL OUTER JOIN reach_wse_width w ON s.reach_id = w.reach_id
    ORDER BY reach_id
    """

    result = con.execute(query).fetchdf()

    return {"stats": result, "n_reaches": len(result), "region": region}


def update_sword_db(sword_db_path: str, stats_df, region: str):
    """
    Update SWORD database with SWOT observation columns (slope, wse, width).

    All stats computed from SWOT node data (not PODAAC reach products).

    Args:
        sword_db_path: Path to SWORD DuckDB database
        stats_df: DataFrame with reach_id and *_obs_* columns
        region: Region code
    """

    con = duckdb.connect(sword_db_path)

    # Load spatial extension for RTREE index support
    con.execute("INSTALL spatial; LOAD spatial;")

    # Check if columns exist, add if not
    existing_cols = con.execute("SELECT * FROM reaches LIMIT 0").df().columns.tolist()

    new_cols = [
        # Slope stats
        ("slope_obs_mean", "DOUBLE"),
        ("slope_obs_std", "DOUBLE"),
        ("slope_obs_median", "DOUBLE"),
        ("slope_obs_range", "DOUBLE"),
        ("slope_obs_n", "INTEGER"),
        ("slope_obs_n_passes", "INTEGER"),
        ("slope_obs_q", "INTEGER"),
        # WSE stats
        ("wse_obs_mean", "DOUBLE"),
        ("wse_obs_median", "DOUBLE"),
        ("wse_obs_std", "DOUBLE"),
        ("wse_obs_range", "DOUBLE"),
        # Width stats
        ("width_obs_mean", "DOUBLE"),
        ("width_obs_median", "DOUBLE"),
        ("width_obs_std", "DOUBLE"),
        ("width_obs_range", "DOUBLE"),
        # Observation count and source
        ("n_obs", "INTEGER"),
        ("swot_obs_source", "VARCHAR"),
    ]

    for col_name, col_type in new_cols:
        if col_name not in existing_cols:
            con.execute(f"ALTER TABLE reaches ADD COLUMN {col_name} {col_type}")
            print(f"  Added column: {col_name}")

    # Register stats DataFrame
    con.register("swot_stats", stats_df)

    # Update reaches table
    con.execute(f"""
        UPDATE reaches
        SET
            slope_obs_mean = s.slope_obs_mean,
            slope_obs_std = s.slope_obs_std,
            slope_obs_median = s.slope_obs_median,
            slope_obs_range = s.slope_obs_range,
            slope_obs_n = s.slope_obs_n,
            slope_obs_n_passes = s.slope_obs_n_passes,
            slope_obs_q = s.slope_obs_q,
            wse_obs_mean = s.wse_obs_mean,
            wse_obs_median = s.wse_obs_median,
            wse_obs_std = s.wse_obs_std,
            wse_obs_range = s.wse_obs_range,
            width_obs_mean = s.width_obs_mean,
            width_obs_median = s.width_obs_median,
            width_obs_std = s.width_obs_std,
            width_obs_range = s.width_obs_range,
            n_obs = s.n_obs,
            swot_obs_source = s.swot_obs_source
        FROM swot_stats s
        WHERE reaches.reach_id = s.reach_id
          AND reaches.region = '{region}'
    """)

    # Verify update
    updated = con.execute(f"""
        SELECT COUNT(*) FROM reaches
        WHERE region = '{region}' AND slope_obs_n IS NOT NULL
    """).fetchone()[0]

    print(f"  Updated {updated} reaches with slope_obs data")

    con.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compute reach-level SWOT observations (slope, wse, width)"
    )
    parser.add_argument("--db", required=True, help="Path to SWORD DuckDB")
    parser.add_argument(
        "--swot-dir",
        default="/Volumes/SWORD_DATA/data/swot/parquet_lake_D/nodes",
        help="Path to SWOT node parquet directory",
    )
    parser.add_argument("--region", help="Region code (NA, SA, EU, AF, AS, OC)")
    parser.add_argument("--all", action="store_true", help="Process all regions")
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=3,
        help="Min nodes per pass for valid slope (default: 3)",
    )
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=0.1,
        help="Noise floor threshold in m/km (default: 0.1)",
    )
    parser.add_argument(
        "--noise-clip",
        type=float,
        default=1e-5,
        help="Value to clip noise floor slopes to (default: 1e-5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute stats but do not update DB"
    )

    args = parser.parse_args()

    if not args.all and not args.region:
        parser.error("Either --region or --all is required")

    regions = ["NA", "SA", "EU", "AF", "AS", "OC"] if args.all else [args.region]

    for region in regions:
        print(f"\nProcessing region: {region}")

        result = compute_reach_swot_obs(
            swot_node_dir=args.swot_dir,
            sword_db_path=args.db,
            region=region,
            min_nodes_per_pass=args.min_nodes,
            noise_floor=args.noise_floor,
            noise_clip_value=args.noise_clip,
        )

        print(f"  Computed SWOT obs for {result['n_reaches']} reaches")

        if result["n_reaches"] > 0:
            stats = result["stats"]
            slope_valid = stats["slope_obs_mean"].notna().sum()
            wse_valid = stats["wse_obs_mean"].notna().sum()
            width_valid = stats["width_obs_mean"].notna().sum()
            print(
                f"  slope: {slope_valid}, wse: {wse_valid}, width: {width_valid} reaches"
            )
            if slope_valid > 0:
                print(
                    f"  slope_obs_mean range: [{stats['slope_obs_mean'].min():.6f}, {stats['slope_obs_mean'].max():.6f}]"
                )

        if not args.dry_run and result["n_reaches"] > 0:
            update_sword_db(args.db, result["stats"], region)
        elif args.dry_run:
            print("  (dry-run, DB not updated)")


if __name__ == "__main__":
    main()
