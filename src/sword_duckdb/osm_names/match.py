"""Spatial matching of OSM waterways to SWORD reaches."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def match_osm_names(
    conn: duckdb.DuckDBPyConnection,
    gpkg_path: str | Path,
    region: str,
    buffer_deg: float = 0.001,
) -> pd.DataFrame:
    """
    Spatial join OSM waterways to SWORD reaches.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active connection to SWORD DuckDB.
    gpkg_path : str | Path
        Path to GeoPackage with OSM waterway lines.
    region : str
        SWORD region code (NA, SA, EU, AF, AS, OC).
    buffer_deg : float
        Buffer in degrees for ST_DWithin pre-filter (~111m at equator).

    Returns
    -------
    pd.DataFrame
        Columns: [reach_id, river_name_local, river_name_en]
    """
    gpkg_path = str(Path(gpkg_path).resolve())
    region = region.upper()

    conn.execute("INSTALL spatial; LOAD spatial;")

    # Load OSM waterways into temp table.
    # Real ogr2ogr output stores extended tags (including name:en) in the
    # hstore-format "other_tags" column. Synthetic test fixtures may have
    # "name:en" as a direct column instead.
    cols = [
        d[0]
        for d in conn.execute(
            f"SELECT * FROM ST_Read('{gpkg_path}') LIMIT 0"
        ).description
    ]
    if "other_tags" in cols:
        name_en_expr = """regexp_extract(other_tags, '"name:en"=>"([^"]*)"', 1)"""
    else:
        name_en_expr = '"name:en"'

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE osm_waterways AS
        SELECT
            name,
            {name_en_expr} AS name_en,
            geom
        FROM ST_Read('{gpkg_path}')
        WHERE name IS NOT NULL
    """)

    # Spatial join: pre-filter with ST_DWithin, score by intersection length
    query = f"""
    WITH spatial_hits AS (
        SELECT
            r.reach_id,
            o.name,
            o.name_en,
            ST_Length(ST_Intersection(r.geom, ST_Buffer(o.geom, {buffer_deg}))) AS overlap_len
        FROM reaches r
        JOIN osm_waterways o
            ON ST_DWithin(r.geom, o.geom, {buffer_deg})
        WHERE r.region = '{region}'
            AND r.geom IS NOT NULL
    ),
    ranked AS (
        SELECT
            reach_id,
            name,
            name_en,
            overlap_len,
            ROW_NUMBER() OVER (PARTITION BY reach_id, name ORDER BY overlap_len DESC) AS rn
        FROM spatial_hits
    ),
    deduped AS (
        SELECT reach_id, name, name_en, overlap_len
        FROM ranked
        WHERE rn = 1
    )
    SELECT
        reach_id,
        STRING_AGG(name, '; ' ORDER BY overlap_len DESC) AS river_name_local,
        STRING_AGG(name_en, '; ' ORDER BY overlap_len DESC) AS river_name_en
    FROM deduped
    GROUP BY reach_id
    """

    result = conn.execute(query).fetchdf()

    # Clean up
    conn.execute("DROP TABLE IF EXISTS osm_waterways")

    return result


def save_osm_names(
    conn: duckdb.DuckDBPyConnection,
    region: str,
    matches_df: pd.DataFrame,
) -> int:
    """
    Write OSM name matches to reaches table.

    Uses RTREE drop/recreate pattern per CLAUDE.md.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active connection to SWORD DuckDB.
    region : str
        SWORD region code.
    matches_df : pd.DataFrame
        From match_osm_names(): [reach_id, river_name_local, river_name_en]

    Returns
    -------
    int
        Number of reaches updated.
    """
    if matches_df.empty:
        return 0

    region = region.upper()
    conn.execute("INSTALL spatial; LOAD spatial;")

    # Register DataFrame as temp table
    conn.register("_osm_matches", matches_df)

    # RTREE drop/recreate pattern
    indexes = conn.execute(
        "SELECT index_name, table_name, sql FROM duckdb_indexes() WHERE sql LIKE '%RTREE%'"
    ).fetchall()

    for idx_name, _tbl, _sql in indexes:
        conn.execute(f'DROP INDEX "{idx_name}"')

    # Clear existing OSM names for this region, then update
    conn.execute(f"""
        UPDATE reaches
        SET river_name_local = NULL, river_name_en = NULL
        WHERE region = '{region}'
    """)

    conn.execute(f"""
        UPDATE reaches
        SET
            river_name_local = m.river_name_local,
            river_name_en = m.river_name_en
        FROM _osm_matches m
        WHERE reaches.reach_id = m.reach_id
            AND reaches.region = '{region}'
    """)

    updated = conn.execute(f"""
        SELECT COUNT(*) FROM reaches
        WHERE region = '{region}' AND river_name_local IS NOT NULL
    """).fetchone()[0]

    # Recreate RTREE indexes
    for _idx_name, _tbl, sql in indexes:
        conn.execute(sql)

    conn.unregister("_osm_matches")

    return updated


def infill_name_gaps(
    conn: duckdb.DuckDBPyConnection,
    max_passes: int = 10,
) -> dict[str, int]:
    """
    Fill unnamed 1:1 reaches where upstream and downstream names agree.

    Iteratively propagates names inward from both ends of gaps on 1:1 chains
    (no junctions). Converges when no more fills are possible or max_passes
    is reached, whichever comes first.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active connection to SWORD DuckDB.
    max_passes : int
        Maximum iterations. Gap length N requires ceil(N/2) passes.

    Returns
    -------
    dict
        {"river_name_local": int, "river_name_en": int} — counts filled.
    """
    for col in ("river_name_local", "river_name_en"):
        for _ in range(max_passes):
            prev = conn.execute(
                f"SELECT COUNT(*) FROM reaches WHERE {col} IS NOT NULL"
            ).fetchone()[0]

            conn.execute(f"""
                UPDATE reaches r
                SET {col} = up.{col}
                FROM reaches dn, reaches up
                WHERE r.{col} IS NULL
                    AND r.n_rch_down = 1
                    AND r.n_rch_up = 1
                    AND r.rch_id_dn_main = dn.reach_id
                    AND r.rch_id_up_main = up.reach_id
                    AND dn.{col} IS NOT NULL
                    AND up.{col} IS NOT NULL
                    AND dn.{col} = up.{col}
            """)

            cur = conn.execute(
                f"SELECT COUNT(*) FROM reaches WHERE {col} IS NOT NULL"
            ).fetchone()[0]
            if cur == prev:
                break


def infill_name_gaps_with_rtree(
    conn: duckdb.DuckDBPyConnection,
    max_passes: int = 10,
) -> dict[str, int]:
    """
    infill_name_gaps wrapped with RTREE drop/recreate pattern.

    Returns counts of reaches filled per column.
    """
    conn.execute("INSTALL spatial; LOAD spatial;")

    indexes = conn.execute(
        "SELECT index_name, table_name, sql FROM duckdb_indexes() WHERE sql LIKE '%RTREE%'"
    ).fetchall()
    for idx_name, _tbl, _sql in indexes:
        conn.execute(f'DROP INDEX "{idx_name}"')

    before = {}
    for col in ("river_name_local", "river_name_en"):
        before[col] = conn.execute(
            f"SELECT COUNT(*) FROM reaches WHERE {col} IS NOT NULL"
        ).fetchone()[0]

    infill_name_gaps(conn, max_passes)

    after = {}
    for col in ("river_name_local", "river_name_en"):
        after[col] = conn.execute(
            f"SELECT COUNT(*) FROM reaches WHERE {col} IS NOT NULL"
        ).fetchone()[0]

    for _idx_name, _tbl, sql in indexes:
        conn.execute(sql)

    return {col: after[col] - before[col] for col in before}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match OSM waterways to SWORD reaches")
    parser.add_argument("--db", required=True, help="Path to SWORD DuckDB")
    parser.add_argument("--gpkg", required=True, help="Path to OSM waterway GPKG")
    parser.add_argument(
        "--region", required=True, help="SWORD region (NA/SA/EU/AF/AS/OC)"
    )
    parser.add_argument(
        "--buffer", type=float, default=0.001, help="Buffer in degrees (default: 0.001)"
    )
    args = parser.parse_args()

    conn = duckdb.connect(args.db)
    print(f"Matching OSM names for region {args.region.upper()}...")
    matches = match_osm_names(conn, args.gpkg, args.region, args.buffer)
    print(f"Found {len(matches)} reach matches")

    from src.sword_duckdb.schema import add_osm_name_columns

    # Ensure columns exist
    add_osm_name_columns(conn)

    n = save_osm_names(conn, args.region, matches)
    print(f"Updated {n} reaches with OSM names")
    conn.close()
