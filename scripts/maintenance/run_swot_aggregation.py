"""Run SWOT percentile-based observation aggregation, one region at a time.

Usage:
    python scripts/maintenance/run_swot_aggregation.py [--region NA]

Strategy: For each SWORD region, collect matching observations from each SWOT
continent file batch into a small staging table, then aggregate. This keeps
peak memory proportional to one region's data (~10M rows) not all regions.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb

from sword_duckdb.schema import add_swot_obs_columns
from sword_duckdb.swot_filters import (
    SLOPE_REF_UNCERTAINTY,
    build_node_filter_sql,
    build_reach_filter_sql,
)

DB_PATH = "data/duckdb/sword_v17c.duckdb"
SWOT_PATH = "/Volumes/SWORD_DATA/data/swot/parquet_lake_D"
SWOT_CONTINENTS = ["AF", "AR", "AS", "AU", "EU", "GR", "NA", "SA", "SI"]

NODE_RANGES = {
    "AF": (11000000000000, 19999999999999),
    "EU": (21000000000000, 29999999999999),
    "AS": (31000000000000, 49999999999999),
    "OC": (51000000000000, 59999999999999),
    "SA": (61000000000000, 69999999999999),
    "NA": (71000000000000, 99999999999999),
}
REACH_RANGES = {
    "AF": (11000000000, 19999999999),
    "EU": (21000000000, 29999999999),
    "AS": (31000000000, 49999999999),
    "OC": (51000000000, 59999999999),
    "SA": (61000000000, 69999999999),
    "NA": (71000000000, 99999999999),
}


def detect_columns(con, sample_dir, table_type):
    """Detect columns from one parquet file."""
    sample = next(
        (
            f
            for f in sample_dir.iterdir()
            if f.suffix == ".parquet" and not f.name.startswith("._")
        ),
        None,
    )
    if not sample:
        return set()
    return set(
        c.lower()
        for c in con.execute(f"SELECT * FROM read_parquet('{sample}') LIMIT 1")
        .df()
        .columns.tolist()
    )


def _agg_sql(var: str, alias: str | None = None):
    """Build APPROX_QUANTILE + range aggregation SQL for one variable.

    Parameters
    ----------
    var : str
        Column name in the source data.
    alias : str, optional
        Prefix for output column names. Defaults to var.
    """
    out = alias or var
    lines = [
        f"APPROX_QUANTILE({var}, 0.{p}) as {out}_obs_p{p}" for p in range(10, 100, 10)
    ]
    lines.append(f"MAX({var}) - MIN({var}) as {out}_obs_range")
    return ",\n            ".join(lines)


def process_nodes(con, swot_path, region, id_min, id_max):
    """Aggregate node observations directly from parquet, one continent at a time.

    No staging table — aggregates per-continent into a union view, then
    re-aggregates. This keeps peak memory proportional to one continent's
    data rather than the entire region.
    """
    nodes_dir = swot_path / "nodes"
    colnames = detect_columns(con, nodes_dir, "nodes")
    if not colnames:
        return 0, 0
    where_clause, wse_col = build_node_filter_sql(colnames)
    range_filter = f"CAST(node_id AS BIGINT) BETWEEN {id_min} AND {id_max}"

    con.execute("DROP TABLE IF EXISTS _node_agg")
    con.execute("""
        CREATE TEMP TABLE _node_agg (
            node_id BIGINT,
            wse_obs_p10 DOUBLE, wse_obs_p20 DOUBLE, wse_obs_p30 DOUBLE,
            wse_obs_p40 DOUBLE, wse_obs_p50 DOUBLE, wse_obs_p60 DOUBLE,
            wse_obs_p70 DOUBLE, wse_obs_p80 DOUBLE, wse_obs_p90 DOUBLE,
            wse_obs_range DOUBLE, wse_obs_mad DOUBLE,
            width_obs_p10 DOUBLE, width_obs_p20 DOUBLE, width_obs_p30 DOUBLE,
            width_obs_p40 DOUBLE, width_obs_p50 DOUBLE, width_obs_p60 DOUBLE,
            width_obs_p70 DOUBLE, width_obs_p80 DOUBLE, width_obs_p90 DOUBLE,
            width_obs_range DOUBLE, width_obs_mad DOUBLE,
            n_obs INTEGER
        )
    """)

    # Sub-chunk the ID range to limit t-digest memory (18 digests × ~3KB each per group)
    # With 27 sub-chunks, each has ~70K groups → ~3.6GB hash table state
    n_subchunks = 27
    sub_size = (id_max - id_min + 1) // n_subchunks

    import time

    total_obs = 0
    t0_all = time.monotonic()
    for chunk_idx in range(n_subchunks):
        t0 = time.monotonic()
        sub_min = id_min + chunk_idx * sub_size
        sub_max = (
            id_min + (chunk_idx + 1) * sub_size - 1
            if chunk_idx < n_subchunks - 1
            else id_max
        )
        sub_filter = f"CAST(node_id AS BIGINT) BETWEEN {sub_min} AND {sub_max}"

        for sc in SWOT_CONTINENTS:
            glob = str(nodes_dir / f"SWOT*_{sc}_*.parquet")
            try:
                con.execute(f"""
                    INSERT INTO _node_agg
                    SELECT CAST(node_id AS BIGINT) as node_id,
                        {_agg_sql("wse_val", "wse")},
                        NULL as wse_obs_mad,
                        {_agg_sql("width_val", "width")},
                        NULL as width_obs_mad,
                        COUNT(*) as n_obs
                    FROM (
                        SELECT CAST(node_id AS BIGINT) as node_id,
                               {wse_col} as wse_val, width as width_val
                        FROM read_parquet('{glob}', union_by_name=true)
                        WHERE {sub_filter} AND {where_clause}
                    ) filtered
                    GROUP BY node_id
                """)
            except Exception:
                pass  # No files for this continent

        elapsed = time.monotonic() - t0
        cnt = con.execute("SELECT SUM(n_obs) FROM _node_agg").fetchone()[0] or 0
        print(
            f"    chunk {chunk_idx + 1}/{n_subchunks}: {cnt} cumulative obs ({elapsed:.0f}s)",
            flush=True,
        )
        total_obs = cnt
    print(f"    all chunks: {time.monotonic() - t0_all:.0f}s total", flush=True)

    if total_obs == 0:
        con.execute("DROP TABLE IF EXISTS _node_agg")
        return 0, 0

    # Re-aggregate: merge continent-level partial results into final per-node stats
    # Since APPROX_QUANTILE can't merge t-digests, we need a weighted approach.
    # But we already grouped by node_id per continent, so we may have multiple rows
    # per node (one per continent). We need to re-aggregate these.
    # The simple fix: the partial percentiles can't be re-merged, so we just
    # take the weighted average for each. For nodes with data from only one continent
    # (the vast majority), this is exact. For the rare multi-continent nodes,
    # the n_obs-weighted average of percentiles is a reasonable approximation.
    n_dupes = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT node_id FROM _node_agg GROUP BY node_id HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    if n_dupes > 0:
        print(f"    Merging {n_dupes} nodes with multi-continent data", flush=True)
        con.execute("""
            CREATE OR REPLACE TEMP TABLE _node_merged AS
            SELECT node_id,
                SUM(wse_obs_p10 * n_obs) / SUM(n_obs) as wse_obs_p10,
                SUM(wse_obs_p20 * n_obs) / SUM(n_obs) as wse_obs_p20,
                SUM(wse_obs_p30 * n_obs) / SUM(n_obs) as wse_obs_p30,
                SUM(wse_obs_p40 * n_obs) / SUM(n_obs) as wse_obs_p40,
                SUM(wse_obs_p50 * n_obs) / SUM(n_obs) as wse_obs_p50,
                SUM(wse_obs_p60 * n_obs) / SUM(n_obs) as wse_obs_p60,
                SUM(wse_obs_p70 * n_obs) / SUM(n_obs) as wse_obs_p70,
                SUM(wse_obs_p80 * n_obs) / SUM(n_obs) as wse_obs_p80,
                SUM(wse_obs_p90 * n_obs) / SUM(n_obs) as wse_obs_p90,
                MAX(wse_obs_range) as wse_obs_range,
                NULL as wse_obs_mad,
                SUM(width_obs_p10 * n_obs) / SUM(n_obs) as width_obs_p10,
                SUM(width_obs_p20 * n_obs) / SUM(n_obs) as width_obs_p20,
                SUM(width_obs_p30 * n_obs) / SUM(n_obs) as width_obs_p30,
                SUM(width_obs_p40 * n_obs) / SUM(n_obs) as width_obs_p40,
                SUM(width_obs_p50 * n_obs) / SUM(n_obs) as width_obs_p50,
                SUM(width_obs_p60 * n_obs) / SUM(n_obs) as width_obs_p60,
                SUM(width_obs_p70 * n_obs) / SUM(n_obs) as width_obs_p70,
                SUM(width_obs_p80 * n_obs) / SUM(n_obs) as width_obs_p80,
                SUM(width_obs_p90 * n_obs) / SUM(n_obs) as width_obs_p90,
                MAX(width_obs_range) as width_obs_range,
                NULL as width_obs_mad,
                CAST(SUM(n_obs) AS INTEGER) as n_obs
            FROM _node_agg
            GROUP BY node_id
        """)
        con.execute("DROP TABLE _node_agg")
        con.execute("ALTER TABLE _node_merged RENAME TO _node_agg")

    # Derive MAD from percentiles
    con.execute("""
        UPDATE _node_agg SET
            wse_obs_mad = (wse_obs_p80 - wse_obs_p20) * 0.7413,
            width_obs_mad = (width_obs_p80 - width_obs_p20) * 0.7413
    """)
    print(f"    {total_obs} obs total, MAD derived", flush=True)

    # Update nodes table
    con.execute("INSTALL spatial; LOAD spatial;")
    rtrees = con.execute(
        "SELECT index_name, sql FROM duckdb_indexes() "
        "WHERE sql LIKE '%RTREE%' AND table_name = 'nodes'"
    ).fetchall()
    for name, _ in rtrees:
        con.execute(f'DROP INDEX "{name}"')

    result = con.execute(f"""
        UPDATE nodes SET
            wse_obs_p10 = a.wse_obs_p10, wse_obs_p20 = a.wse_obs_p20,
            wse_obs_p30 = a.wse_obs_p30, wse_obs_p40 = a.wse_obs_p40,
            wse_obs_p50 = a.wse_obs_p50, wse_obs_p60 = a.wse_obs_p60,
            wse_obs_p70 = a.wse_obs_p70, wse_obs_p80 = a.wse_obs_p80,
            wse_obs_p90 = a.wse_obs_p90, wse_obs_range = a.wse_obs_range,
            wse_obs_mad = a.wse_obs_mad,
            width_obs_p10 = a.width_obs_p10, width_obs_p20 = a.width_obs_p20,
            width_obs_p30 = a.width_obs_p30, width_obs_p40 = a.width_obs_p40,
            width_obs_p50 = a.width_obs_p50, width_obs_p60 = a.width_obs_p60,
            width_obs_p70 = a.width_obs_p70, width_obs_p80 = a.width_obs_p80,
            width_obs_p90 = a.width_obs_p90, width_obs_range = a.width_obs_range,
            width_obs_mad = a.width_obs_mad,
            n_obs = a.n_obs
        FROM _node_agg a
        WHERE nodes.node_id = a.node_id AND nodes.region = '{region}'
    """)
    updated = result.fetchone()[0] if result else 0

    for name, sql in rtrees:
        con.execute(sql)

    con.execute("DROP TABLE IF EXISTS _node_agg")
    return updated, total_obs


def process_reaches(con, swot_path, region, id_min, id_max):
    """Aggregate reach observations directly from parquet, one continent at a time."""
    reaches_dir = swot_path / "reaches"
    colnames = detect_columns(con, reaches_dir, "reaches")
    if not colnames:
        return 0, 0
    where_clause = build_reach_filter_sql(colnames)
    ref_u = SLOPE_REF_UNCERTAINTY
    range_filter = f"CAST(reach_id AS BIGINT) BETWEEN {id_min} AND {id_max}"

    con.execute("DROP TABLE IF EXISTS _reach_pct")
    con.execute("""
        CREATE TEMP TABLE _reach_pct (
            reach_id BIGINT,
            wse_obs_p10 DOUBLE, wse_obs_p20 DOUBLE, wse_obs_p30 DOUBLE,
            wse_obs_p40 DOUBLE, wse_obs_p50 DOUBLE, wse_obs_p60 DOUBLE,
            wse_obs_p70 DOUBLE, wse_obs_p80 DOUBLE, wse_obs_p90 DOUBLE,
            wse_obs_range DOUBLE,
            width_obs_p10 DOUBLE, width_obs_p20 DOUBLE, width_obs_p30 DOUBLE,
            width_obs_p40 DOUBLE, width_obs_p50 DOUBLE, width_obs_p60 DOUBLE,
            width_obs_p70 DOUBLE, width_obs_p80 DOUBLE, width_obs_p90 DOUBLE,
            width_obs_range DOUBLE,
            slope_obs_p10 DOUBLE, slope_obs_p20 DOUBLE, slope_obs_p30 DOUBLE,
            slope_obs_p40 DOUBLE, slope_obs_p50 DOUBLE, slope_obs_p60 DOUBLE,
            slope_obs_p70 DOUBLE, slope_obs_p80 DOUBLE, slope_obs_p90 DOUBLE,
            slope_obs_range DOUBLE,
            sum_w DOUBLE, signed_sum DOUBLE, n_obs INTEGER
        )
    """)

    # Sub-chunk reach IDs to limit t-digest memory
    n_subchunks = 27
    sub_size = (id_max - id_min + 1) // n_subchunks

    import time

    total_obs = 0
    t0_all = time.monotonic()
    for chunk_idx in range(n_subchunks):
        t0 = time.monotonic()
        sub_min = id_min + chunk_idx * sub_size
        sub_max = (
            id_min + (chunk_idx + 1) * sub_size - 1
            if chunk_idx < n_subchunks - 1
            else id_max
        )
        sub_filter = f"CAST(reach_id AS BIGINT) BETWEEN {sub_min} AND {sub_max}"

        for sc in SWOT_CONTINENTS:
            glob = str(reaches_dir / f"SWOT*_{sc}_*.parquet")
            try:
                con.execute(f"""
                    INSERT INTO _reach_pct
                    SELECT CAST(reach_id AS BIGINT) as reach_id,
                        {_agg_sql("wse")},
                        {_agg_sql("width")},
                        {_agg_sql("slope")},
                        SUM(COALESCE(n_good_nod, 1)) as sum_w,
                        SUM(COALESCE(n_good_nod, 1) * CASE WHEN slope > 0 THEN 1
                                          WHEN slope < 0 THEN -1 ELSE 0 END) as signed_sum,
                        COUNT(*) as n_obs
                    FROM read_parquet('{glob}', union_by_name=true)
                    WHERE {sub_filter} AND {where_clause}
                    GROUP BY reach_id
                """)
            except Exception:
                pass

        elapsed = time.monotonic() - t0
        cnt = con.execute("SELECT SUM(n_obs) FROM _reach_pct").fetchone()[0] or 0
        print(
            f"    chunk {chunk_idx + 1}/{n_subchunks}: {cnt} cumulative obs ({elapsed:.0f}s)",
            flush=True,
        )
        total_obs = cnt
    print(f"    all chunks: {time.monotonic() - t0_all:.0f}s total", flush=True)

    if total_obs == 0:
        con.execute("DROP TABLE IF EXISTS _reach_pct")
        return 0, 0

    # Merge multi-continent reaches (weighted average of percentiles)
    n_dupes = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT reach_id FROM _reach_pct GROUP BY reach_id HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    if n_dupes > 0:
        print(f"    Merging {n_dupes} reaches with multi-continent data", flush=True)
        con.execute("""
            CREATE OR REPLACE TEMP TABLE _reach_merged AS
            SELECT reach_id,
                SUM(wse_obs_p10 * n_obs) / SUM(n_obs) as wse_obs_p10,
                SUM(wse_obs_p20 * n_obs) / SUM(n_obs) as wse_obs_p20,
                SUM(wse_obs_p30 * n_obs) / SUM(n_obs) as wse_obs_p30,
                SUM(wse_obs_p40 * n_obs) / SUM(n_obs) as wse_obs_p40,
                SUM(wse_obs_p50 * n_obs) / SUM(n_obs) as wse_obs_p50,
                SUM(wse_obs_p60 * n_obs) / SUM(n_obs) as wse_obs_p60,
                SUM(wse_obs_p70 * n_obs) / SUM(n_obs) as wse_obs_p70,
                SUM(wse_obs_p80 * n_obs) / SUM(n_obs) as wse_obs_p80,
                SUM(wse_obs_p90 * n_obs) / SUM(n_obs) as wse_obs_p90,
                MAX(wse_obs_range) as wse_obs_range,
                SUM(width_obs_p10 * n_obs) / SUM(n_obs) as width_obs_p10,
                SUM(width_obs_p20 * n_obs) / SUM(n_obs) as width_obs_p20,
                SUM(width_obs_p30 * n_obs) / SUM(n_obs) as width_obs_p30,
                SUM(width_obs_p40 * n_obs) / SUM(n_obs) as width_obs_p40,
                SUM(width_obs_p50 * n_obs) / SUM(n_obs) as width_obs_p50,
                SUM(width_obs_p60 * n_obs) / SUM(n_obs) as width_obs_p60,
                SUM(width_obs_p70 * n_obs) / SUM(n_obs) as width_obs_p70,
                SUM(width_obs_p80 * n_obs) / SUM(n_obs) as width_obs_p80,
                SUM(width_obs_p90 * n_obs) / SUM(n_obs) as width_obs_p90,
                MAX(width_obs_range) as width_obs_range,
                SUM(slope_obs_p10 * n_obs) / SUM(n_obs) as slope_obs_p10,
                SUM(slope_obs_p20 * n_obs) / SUM(n_obs) as slope_obs_p20,
                SUM(slope_obs_p30 * n_obs) / SUM(n_obs) as slope_obs_p30,
                SUM(slope_obs_p40 * n_obs) / SUM(n_obs) as slope_obs_p40,
                SUM(slope_obs_p50 * n_obs) / SUM(n_obs) as slope_obs_p50,
                SUM(slope_obs_p60 * n_obs) / SUM(n_obs) as slope_obs_p60,
                SUM(slope_obs_p70 * n_obs) / SUM(n_obs) as slope_obs_p70,
                SUM(slope_obs_p80 * n_obs) / SUM(n_obs) as slope_obs_p80,
                SUM(slope_obs_p90 * n_obs) / SUM(n_obs) as slope_obs_p90,
                MAX(slope_obs_range) as slope_obs_range,
                SUM(sum_w) as sum_w,
                SUM(signed_sum) as signed_sum,
                CAST(SUM(n_obs) AS INTEGER) as n_obs
            FROM _reach_pct
            GROUP BY reach_id
        """)
        con.execute("DROP TABLE _reach_pct")
        con.execute("ALTER TABLE _reach_merged RENAME TO _reach_pct")

    # Derive MAD + slope quality from percentiles
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _reach_agg AS
        SELECT reach_id,
            wse_obs_p10, wse_obs_p20, wse_obs_p30, wse_obs_p40, wse_obs_p50,
            wse_obs_p60, wse_obs_p70, wse_obs_p80, wse_obs_p90,
            wse_obs_range,
            (wse_obs_p80 - wse_obs_p20) * 0.7413 as wse_obs_mad,
            width_obs_p10, width_obs_p20, width_obs_p30, width_obs_p40, width_obs_p50,
            width_obs_p60, width_obs_p70, width_obs_p80, width_obs_p90,
            width_obs_range,
            (width_obs_p80 - width_obs_p20) * 0.7413 as width_obs_mad,
            slope_obs_p10, slope_obs_p20, slope_obs_p30, slope_obs_p40, slope_obs_p50,
            slope_obs_p60, slope_obs_p70, slope_obs_p80, slope_obs_p90,
            slope_obs_range,
            (slope_obs_p80 - slope_obs_p20) * 0.7413 as slope_obs_mad,
            GREATEST(slope_obs_p50, 0.0) as slope_obs_adj,
            CASE WHEN sum_w > 0 THEN signed_sum / sum_w ELSE 0 END as slope_obs_slopeF,
            CASE WHEN sum_w > 0 AND ABS(signed_sum / sum_w) > 0.5
                 AND ABS(slope_obs_p50) > {ref_u}
                 THEN TRUE ELSE FALSE END as slope_obs_reliable,
            CASE
                WHEN slope_obs_p50 < -{ref_u} THEN 'negative'
                WHEN ABS(slope_obs_p50) <= {ref_u} THEN 'below_ref_uncertainty'
                WHEN sum_w > 0 AND ABS(signed_sum / sum_w) <= 0.5 THEN 'high_uncertainty'
                ELSE 'reliable'
            END as slope_obs_quality,
            n_obs
        FROM _reach_pct
    """)

    try:
        con.execute("INSTALL spatial; LOAD spatial;")
    except Exception:
        pass
    rtrees = con.execute(
        "SELECT index_name, sql FROM duckdb_indexes() "
        "WHERE sql LIKE '%RTREE%' AND table_name = 'reaches'"
    ).fetchall()
    for name, _ in rtrees:
        con.execute(f'DROP INDEX "{name}"')

    result = con.execute(f"""
        UPDATE reaches SET
            wse_obs_p10 = a.wse_obs_p10, wse_obs_p20 = a.wse_obs_p20,
            wse_obs_p30 = a.wse_obs_p30, wse_obs_p40 = a.wse_obs_p40,
            wse_obs_p50 = a.wse_obs_p50, wse_obs_p60 = a.wse_obs_p60,
            wse_obs_p70 = a.wse_obs_p70, wse_obs_p80 = a.wse_obs_p80,
            wse_obs_p90 = a.wse_obs_p90, wse_obs_range = a.wse_obs_range,
            wse_obs_mad = a.wse_obs_mad,
            width_obs_p10 = a.width_obs_p10, width_obs_p20 = a.width_obs_p20,
            width_obs_p30 = a.width_obs_p30, width_obs_p40 = a.width_obs_p40,
            width_obs_p50 = a.width_obs_p50, width_obs_p60 = a.width_obs_p60,
            width_obs_p70 = a.width_obs_p70, width_obs_p80 = a.width_obs_p80,
            width_obs_p90 = a.width_obs_p90, width_obs_range = a.width_obs_range,
            width_obs_mad = a.width_obs_mad,
            slope_obs_p10 = a.slope_obs_p10, slope_obs_p20 = a.slope_obs_p20,
            slope_obs_p30 = a.slope_obs_p30, slope_obs_p40 = a.slope_obs_p40,
            slope_obs_p50 = a.slope_obs_p50, slope_obs_p60 = a.slope_obs_p60,
            slope_obs_p70 = a.slope_obs_p70, slope_obs_p80 = a.slope_obs_p80,
            slope_obs_p90 = a.slope_obs_p90, slope_obs_range = a.slope_obs_range,
            slope_obs_mad = a.slope_obs_mad,
            slope_obs_adj = a.slope_obs_adj,
            slope_obs_slopeF = a.slope_obs_slopeF,
            slope_obs_reliable = a.slope_obs_reliable,
            slope_obs_quality = a.slope_obs_quality,
            n_obs = a.n_obs
        FROM _reach_agg a
        WHERE reaches.reach_id = a.reach_id AND reaches.region = '{region}'
    """)
    updated = result.fetchone()[0] if result else 0

    for name, sql in rtrees:
        con.execute(sql)

    con.execute("DROP TABLE IF EXISTS _reach_pct")
    con.execute("DROP TABLE IF EXISTS _reach_agg")
    return updated, total_obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", help="Single region (e.g., NA)")
    args = parser.parse_args()

    swot_path = Path(SWOT_PATH)
    regions = [args.region.upper()] if args.region else list(REACH_RANGES.keys())

    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("SET threads=2")
        con.execute("SET memory_limit='32GB'")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET temp_directory='/Volumes/SWORD_DATA/tmp'")
        con.execute("SET max_temp_directory_size='60GiB'")

        if add_swot_obs_columns(con):
            print("Schema migrated to percentile columns", flush=True)

        # Clean up any leftover materialized tables from previous failed runs
        con.execute("DROP TABLE IF EXISTS swot_nodes_raw")
        con.execute("DROP TABLE IF EXISTS swot_reaches_raw")

        for region in regions:
            n_min, n_max = NODE_RANGES[region]
            r_min, r_max = REACH_RANGES[region]
            print(f"\n=== {region} ===", flush=True)

            print(f"  Nodes (IDs {n_min}-{n_max})...", flush=True)
            n_upd, n_obs = process_nodes(con, swot_path, region, n_min, n_max)
            print(f"  {n_upd} nodes updated, {n_obs} obs total", flush=True)

            print(f"  Reaches (IDs {r_min}-{r_max})...", flush=True)
            r_upd, r_obs = process_reaches(con, swot_path, region, r_min, r_max)
            print(f"  {r_upd} reaches updated, {r_obs} obs total", flush=True)

            con.execute("CHECKPOINT")

        print("\nDone - all regions processed.", flush=True)
    finally:
        con.close()


if __name__ == "__main__":
    main()
