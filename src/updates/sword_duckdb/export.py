# -*- coding: utf-8 -*-
"""
SWORD Export Module
===================

Export DuckDB SWORD data to various formats:
- PostgreSQL/PostGIS for QGIS editing
- GeoParquet for data exchange
- GeoPackage for desktop GIS

Example Usage:
    from sword_duckdb.export import export_to_postgres, export_to_geoparquet

    # Export to PostgreSQL for QGIS editing
    export_to_postgres(
        sword=sword,
        pg_connection_string="postgresql://user:pass@localhost/sword_edit",
        region="NA",
        tables=["reaches", "nodes"]
    )

    # Export to GeoParquet
    export_to_geoparquet(sword, "/data/exports/na_reaches.parquet", table="reaches")

TODO(LOW): Add error handling for network failures during PostgreSQL operations.
Current implementation assumes stable connection. Should add:
1. Connection retry logic with exponential backoff
2. Transaction rollback on partial failure
3. Progress checkpointing for large exports
4. Informative error messages for common failures (auth, network, disk)
See MIGRATION_STATUS.md "Immediate TODOs" section.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .sword_class import SWORD


# PostgreSQL/PostGIS schema definitions
# These map DuckDB columns to PostgreSQL columns with proper types

PG_CENTERLINES_SCHEMA = """
CREATE TABLE IF NOT EXISTS {table_name} (
    cl_id BIGINT NOT NULL,
    region VARCHAR(2) NOT NULL,
    x DOUBLE PRECISION NOT NULL,
    y DOUBLE PRECISION NOT NULL,
    geom GEOMETRY(Point, 4326),
    reach_id BIGINT NOT NULL,
    node_id BIGINT NOT NULL,
    version VARCHAR(10) NOT NULL,
    PRIMARY KEY (cl_id, region)
);
"""

PG_NODES_SCHEMA = """
CREATE TABLE IF NOT EXISTS {table_name} (
    node_id BIGINT NOT NULL,
    region VARCHAR(2) NOT NULL,
    x DOUBLE PRECISION NOT NULL,
    y DOUBLE PRECISION NOT NULL,
    geom GEOMETRY(Point, 4326),
    cl_id_min BIGINT,
    cl_id_max BIGINT,
    reach_id BIGINT NOT NULL,
    node_length DOUBLE PRECISION,
    wse DOUBLE PRECISION,
    wse_var DOUBLE PRECISION,
    width DOUBLE PRECISION,
    width_var DOUBLE PRECISION,
    max_width DOUBLE PRECISION,
    facc DOUBLE PRECISION,
    dist_out DOUBLE PRECISION,
    lakeflag INTEGER,
    obstr_type INTEGER,
    grod_id BIGINT,
    hfalls_id BIGINT,
    n_chan_max INTEGER,
    n_chan_mod INTEGER,
    wth_coef DOUBLE PRECISION,
    ext_dist_coef DOUBLE PRECISION,
    meander_length DOUBLE PRECISION,
    sinuosity DOUBLE PRECISION,
    river_name VARCHAR,
    manual_add INTEGER,
    edit_flag VARCHAR,
    trib_flag INTEGER,
    path_freq BIGINT,
    path_order BIGINT,
    path_segs BIGINT,
    stream_order INTEGER,
    main_side INTEGER,
    end_reach INTEGER,
    network INTEGER,
    add_flag INTEGER,
    version VARCHAR(10) NOT NULL,
    PRIMARY KEY (node_id, region)
);
"""

PG_REACHES_SCHEMA = """
CREATE TABLE IF NOT EXISTS {table_name} (
    reach_id BIGINT NOT NULL,
    region VARCHAR(2) NOT NULL,
    x DOUBLE PRECISION,
    y DOUBLE PRECISION,
    x_min DOUBLE PRECISION,
    x_max DOUBLE PRECISION,
    y_min DOUBLE PRECISION,
    y_max DOUBLE PRECISION,
    geom GEOMETRY(LineString, 4326),
    cl_id_min BIGINT,
    cl_id_max BIGINT,
    reach_length DOUBLE PRECISION,
    n_nodes INTEGER,
    wse DOUBLE PRECISION,
    wse_var DOUBLE PRECISION,
    width DOUBLE PRECISION,
    width_var DOUBLE PRECISION,
    slope DOUBLE PRECISION,
    max_width DOUBLE PRECISION,
    facc DOUBLE PRECISION,
    dist_out DOUBLE PRECISION,
    lakeflag INTEGER,
    obstr_type INTEGER,
    grod_id BIGINT,
    hfalls_id BIGINT,
    n_chan_max INTEGER,
    n_chan_mod INTEGER,
    n_rch_up INTEGER,
    n_rch_down INTEGER,
    swot_obs INTEGER,
    iceflag INTEGER,
    low_slope_flag INTEGER,
    river_name VARCHAR,
    edit_flag VARCHAR,
    trib_flag INTEGER,
    path_freq BIGINT,
    path_order BIGINT,
    path_segs BIGINT,
    stream_order INTEGER,
    main_side INTEGER,
    end_reach INTEGER,
    network INTEGER,
    add_flag INTEGER,
    version VARCHAR(10) NOT NULL,
    PRIMARY KEY (reach_id, region)
);
"""

PG_REACH_TOPOLOGY_SCHEMA = """
CREATE TABLE IF NOT EXISTS {table_name} (
    reach_id BIGINT NOT NULL,
    region VARCHAR(2) NOT NULL,
    direction VARCHAR(4) NOT NULL,
    neighbor_rank SMALLINT NOT NULL,
    neighbor_reach_id BIGINT NOT NULL,
    PRIMARY KEY (reach_id, region, direction, neighbor_rank)
);
"""


def _get_pg_connection(connection_string: str):
    """
    Get a PostgreSQL connection using psycopg2.

    Parameters
    ----------
    connection_string : str
        PostgreSQL connection string (e.g., "postgresql://user:pass@localhost/db")

    Returns
    -------
    psycopg2.connection
        Active database connection
    """
    try:
        import psycopg2
    except ImportError:
        raise ImportError(
            "psycopg2 is required for PostgreSQL export. "
            "Install with: pip install psycopg2-binary"
        )

    return psycopg2.connect(connection_string)


def _ensure_postgis(conn) -> bool:
    """
    Ensure PostGIS extension is enabled.

    Returns True if PostGIS is available, False otherwise.
    """
    cursor = conn.cursor()
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Warning: Could not enable PostGIS: {e}")
        return False
    finally:
        cursor.close()


def export_to_postgres(
    sword: 'SWORD',
    connection_string: str,
    tables: List[str] = None,
    schema: str = "public",
    prefix: str = "",
    drop_existing: bool = False,
    batch_size: int = 10000,
    include_topology: bool = True,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Export SWORD data to PostgreSQL/PostGIS for QGIS editing.

    Parameters
    ----------
    sword : SWORD
        Loaded SWORD instance
    connection_string : str
        PostgreSQL connection string (e.g., "postgresql://user:pass@localhost/sword_edit")
    tables : list of str, optional
        Tables to export: ['reaches', 'nodes', 'centerlines']
        Default is ['reaches', 'nodes'] (centerlines are rarely edited directly)
    schema : str, optional
        PostgreSQL schema name. Default is "public"
    prefix : str, optional
        Prefix for table names (e.g., "na_" for "na_reaches")
    drop_existing : bool, optional
        If True, drop and recreate tables. Default is False (append/update)
    batch_size : int, optional
        Number of rows to insert per batch. Default is 10000
    include_topology : bool, optional
        If True, also export reach_topology table. Default is True
    verbose : bool, optional
        If True, print progress messages. Default is True

    Returns
    -------
    dict
        Dictionary with table names and row counts exported

    Raises
    ------
    ImportError
        If psycopg2 is not installed

    Example
    -------
    >>> from sword_duckdb import SWORD
    >>> sword = SWORD('data/duckdb/sword_v17b.duckdb', 'NA', 'v17b')
    >>> export_to_postgres(
    ...     sword,
    ...     "postgresql://postgres:password@localhost/sword_edit",
    ...     tables=["reaches", "nodes"],
    ...     prefix="na_"
    ... )
    {'na_reaches': 38696, 'na_nodes': 1705705}
    """
    if tables is None:
        tables = ['reaches', 'nodes']

    conn = _get_pg_connection(connection_string)
    has_postgis = _ensure_postgis(conn)

    results = {}
    region = sword.region
    version = sword.version

    cursor = conn.cursor()

    try:
        # Set schema
        if schema != "public":
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
            cursor.execute(f"SET search_path TO {schema}, public;")
            conn.commit()

        for table in tables:
            table_name = f"{prefix}{table}" if prefix else table

            if verbose:
                print(f"Exporting {table} to {table_name}...")

            if table == 'reaches':
                count = _export_reaches_to_pg(
                    sword, cursor, conn, table_name,
                    drop_existing, batch_size, has_postgis, verbose
                )
            elif table == 'nodes':
                count = _export_nodes_to_pg(
                    sword, cursor, conn, table_name,
                    drop_existing, batch_size, has_postgis, verbose
                )
            elif table == 'centerlines':
                count = _export_centerlines_to_pg(
                    sword, cursor, conn, table_name,
                    drop_existing, batch_size, has_postgis, verbose
                )
            else:
                print(f"Unknown table: {table}")
                continue

            results[table_name] = count

        # Export topology if requested
        if include_topology and 'reaches' in tables:
            topo_name = f"{prefix}reach_topology" if prefix else "reach_topology"
            if verbose:
                print(f"Exporting topology to {topo_name}...")
            count = _export_topology_to_pg(
                sword, cursor, conn, topo_name,
                drop_existing, batch_size, verbose
            )
            results[topo_name] = count

        # Create spatial indexes if PostGIS available
        if has_postgis:
            for table in tables:
                table_name = f"{prefix}{table}" if prefix else table
                _create_pg_spatial_index(cursor, conn, table_name, verbose)

        conn.commit()

    finally:
        cursor.close()
        conn.close()

    if verbose:
        print(f"Export complete: {results}")

    return results


def _export_reaches_to_pg(
    sword: 'SWORD',
    cursor,
    conn,
    table_name: str,
    drop_existing: bool,
    batch_size: int,
    has_postgis: bool,
    verbose: bool
) -> int:
    """Export reaches table to PostgreSQL."""

    if drop_existing:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
        conn.commit()

    # Create table
    cursor.execute(PG_REACHES_SCHEMA.format(table_name=table_name))
    conn.commit()

    # Get data from SWORD
    reaches = sword.reaches
    n_reaches = len(reaches)

    # Build DataFrame for export
    df = pd.DataFrame({
        'reach_id': reaches.id,
        'region': sword.region,
        'x': reaches.x,
        'y': reaches.y,
        'x_min': reaches.x_min,
        'x_max': reaches.x_max,
        'y_min': reaches.y_min,
        'y_max': reaches.y_max,
        'cl_id_min': reaches.cl_id[0],
        'cl_id_max': reaches.cl_id[1],
        'reach_length': reaches.len,
        'n_nodes': reaches.n_nodes,
        'wse': reaches.wse,
        'wse_var': reaches.wse_var,
        'width': reaches.wth,
        'width_var': reaches.wth_var,
        'slope': reaches.slope,
        'max_width': reaches.max_wth,
        'facc': reaches.facc,
        'dist_out': reaches.dist_out,
        'lakeflag': reaches.lakeflag,
        'obstr_type': reaches.grod,
        'grod_id': reaches.grod_id,
        'hfalls_id': reaches.hfalls_id,
        'n_chan_max': reaches.n_chan_max,
        'n_chan_mod': reaches.n_chan_mod,
        'n_rch_up': reaches.n_rch_up,
        'n_rch_down': reaches.n_rch_down,
        'swot_obs': reaches.swot_obs,
        'iceflag': reaches.iceflag,
        'low_slope_flag': reaches.low_slope,
        'river_name': reaches.river_name,
        'edit_flag': reaches.edit_flag,
        'trib_flag': reaches.trib_flag,
        'path_freq': reaches.p_freq,
        'path_order': reaches.p_ord,
        'path_segs': reaches.p_seg,
        'stream_order': reaches.strm_order,
        'main_side': reaches.main_side,
        'end_reach': reaches.end_rch,
        'network': reaches.network,
        'add_flag': getattr(reaches, 'add_flag', np.zeros(n_reaches, dtype=np.int32)),
        'version': sword.version
    })

    # Replace NaN with None for PostgreSQL
    df = df.replace({np.nan: None})

    # Insert in batches
    columns = df.columns.tolist()
    placeholders = ', '.join(['%s'] * len(columns))
    insert_sql = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT (reach_id, region) DO UPDATE SET
            {', '.join(f'{c} = EXCLUDED.{c}' for c in columns if c not in ['reach_id', 'region'])}
    """

    count = 0
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        values = [tuple(row) for row in batch.values]
        cursor.executemany(insert_sql, values)
        count += len(batch)
        if verbose and count % 50000 == 0:
            print(f"  Inserted {count}/{n_reaches} reaches...")

    conn.commit()

    # Update geometry from x,y if PostGIS available
    if has_postgis:
        cursor.execute(f"""
            UPDATE {table_name}
            SET geom = ST_SetSRID(ST_MakePoint(x, y), 4326)
            WHERE geom IS NULL
        """)
        conn.commit()

    return count


def _export_nodes_to_pg(
    sword: 'SWORD',
    cursor,
    conn,
    table_name: str,
    drop_existing: bool,
    batch_size: int,
    has_postgis: bool,
    verbose: bool
) -> int:
    """Export nodes table to PostgreSQL."""

    if drop_existing:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
        conn.commit()

    # Create table
    cursor.execute(PG_NODES_SCHEMA.format(table_name=table_name))
    conn.commit()

    # Get data from SWORD
    nodes = sword.nodes
    n_nodes = len(nodes)

    # Build DataFrame for export
    df = pd.DataFrame({
        'node_id': nodes.id,
        'region': sword.region,
        'x': nodes.x,
        'y': nodes.y,
        'cl_id_min': nodes.cl_id[0],
        'cl_id_max': nodes.cl_id[1],
        'reach_id': nodes.reach_id,
        'node_length': nodes.len,
        'wse': nodes.wse,
        'wse_var': nodes.wse_var,
        'width': nodes.wth,
        'width_var': nodes.wth_var,
        'max_width': nodes.max_wth,
        'facc': nodes.facc,
        'dist_out': nodes.dist_out,
        'lakeflag': nodes.lakeflag,
        'obstr_type': nodes.grod,
        'grod_id': nodes.grod_id,
        'hfalls_id': nodes.hfalls_id,
        'n_chan_max': nodes.n_chan_max,
        'n_chan_mod': nodes.n_chan_mod,
        'wth_coef': nodes.wth_coef,
        'ext_dist_coef': nodes.ext_dist_coef,
        'meander_length': nodes.meand_len,
        'sinuosity': nodes.sinuosity,
        'river_name': nodes.river_name,
        'manual_add': nodes.manual_add,
        'edit_flag': nodes.edit_flag,
        'trib_flag': nodes.trib_flag,
        'path_freq': nodes.p_freq,
        'path_order': nodes.p_ord,
        'path_segs': nodes.p_seg,
        'stream_order': nodes.strm_order,
        'main_side': nodes.main_side,
        'end_reach': nodes.end_rch,
        'network': nodes.network,
        'add_flag': getattr(nodes, 'add_flag', np.zeros(n_nodes, dtype=np.int32)),
        'version': sword.version
    })

    # Replace NaN with None for PostgreSQL
    df = df.replace({np.nan: None})

    # Insert in batches
    columns = df.columns.tolist()
    placeholders = ', '.join(['%s'] * len(columns))
    insert_sql = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT (node_id, region) DO UPDATE SET
            {', '.join(f'{c} = EXCLUDED.{c}' for c in columns if c not in ['node_id', 'region'])}
    """

    # Disable GC during batch inserts (same pattern as SWORD class)
    gc_was_enabled = gc.isenabled()
    gc.disable()

    try:
        count = 0
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            values = [tuple(row) for row in batch.values]
            cursor.executemany(insert_sql, values)
            count += len(batch)
            if verbose and count % 100000 == 0:
                print(f"  Inserted {count}/{n_nodes} nodes...")

        conn.commit()
    finally:
        if gc_was_enabled:
            gc.enable()

    # Update geometry from x,y if PostGIS available
    if has_postgis:
        cursor.execute(f"""
            UPDATE {table_name}
            SET geom = ST_SetSRID(ST_MakePoint(x, y), 4326)
            WHERE geom IS NULL
        """)
        conn.commit()

    return count


def _export_centerlines_to_pg(
    sword: 'SWORD',
    cursor,
    conn,
    table_name: str,
    drop_existing: bool,
    batch_size: int,
    has_postgis: bool,
    verbose: bool
) -> int:
    """Export centerlines table to PostgreSQL."""

    if drop_existing:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
        conn.commit()

    # Create table
    cursor.execute(PG_CENTERLINES_SCHEMA.format(table_name=table_name))
    conn.commit()

    # Get data from SWORD
    centerlines = sword.centerlines
    n_cls = len(centerlines)

    # Build DataFrame for export
    df = pd.DataFrame({
        'cl_id': centerlines.cl_id,
        'region': sword.region,
        'x': centerlines.x,
        'y': centerlines.y,
        'reach_id': centerlines.reach_id[0],
        'node_id': centerlines.node_id[0],
        'version': sword.version
    })

    # Insert in batches
    columns = df.columns.tolist()
    placeholders = ', '.join(['%s'] * len(columns))
    insert_sql = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT (cl_id, region) DO UPDATE SET
            {', '.join(f'{c} = EXCLUDED.{c}' for c in columns if c not in ['cl_id', 'region'])}
    """

    # Disable GC during batch inserts
    gc_was_enabled = gc.isenabled()
    gc.disable()

    try:
        count = 0
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            values = [tuple(row) for row in batch.values]
            cursor.executemany(insert_sql, values)
            count += len(batch)
            if verbose and count % 500000 == 0:
                print(f"  Inserted {count}/{n_cls} centerlines...")

        conn.commit()
    finally:
        if gc_was_enabled:
            gc.enable()

    # Update geometry from x,y if PostGIS available
    if has_postgis:
        cursor.execute(f"""
            UPDATE {table_name}
            SET geom = ST_SetSRID(ST_MakePoint(x, y), 4326)
            WHERE geom IS NULL
        """)
        conn.commit()

    return count


def _export_topology_to_pg(
    sword: 'SWORD',
    cursor,
    conn,
    table_name: str,
    drop_existing: bool,
    batch_size: int,
    verbose: bool
) -> int:
    """Export reach topology table to PostgreSQL."""

    if drop_existing:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
        conn.commit()

    # Create table
    cursor.execute(PG_REACH_TOPOLOGY_SCHEMA.format(table_name=table_name))
    conn.commit()

    # Get data from SWORD
    reaches = sword.reaches
    n_reaches = len(reaches)

    # Build topology records
    records = []
    for i in range(n_reaches):
        reach_id = reaches.id[i]

        # Upstream neighbors (from rch_id_up[4,N])
        for rank in range(4):
            neighbor_id = reaches.rch_id_up[rank, i]
            if neighbor_id != 0:  # 0 means no neighbor
                records.append((reach_id, sword.region, 'up', rank, neighbor_id))

        # Downstream neighbors (from rch_id_dn[4,N])
        for rank in range(4):
            neighbor_id = reaches.rch_id_dn[rank, i]
            if neighbor_id != 0:
                records.append((reach_id, sword.region, 'down', rank, neighbor_id))

    # Insert in batches
    insert_sql = f"""
        INSERT INTO {table_name} (reach_id, region, direction, neighbor_rank, neighbor_reach_id)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (reach_id, region, direction, neighbor_rank) DO UPDATE SET
            neighbor_reach_id = EXCLUDED.neighbor_reach_id
    """

    count = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        cursor.executemany(insert_sql, batch)
        count += len(batch)
        if verbose and count % 100000 == 0:
            print(f"  Inserted {count}/{len(records)} topology records...")

    conn.commit()
    return count


def _create_pg_spatial_index(cursor, conn, table_name: str, verbose: bool):
    """Create spatial index on PostGIS geometry column."""
    try:
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_geom
            ON {table_name} USING GIST (geom);
        """)
        conn.commit()
        if verbose:
            print(f"  Created spatial index on {table_name}")
    except Exception as e:
        conn.rollback()
        print(f"  Warning: Could not create spatial index: {e}")


def export_to_geoparquet(
    sword: 'SWORD',
    output_path: Union[str, Path],
    table: str = 'reaches',
    compression: str = 'snappy'
) -> int:
    """
    Export SWORD table to GeoParquet format.

    Parameters
    ----------
    sword : SWORD
        Loaded SWORD instance
    output_path : str or Path
        Output file path (.parquet)
    table : str
        Table to export: 'reaches', 'nodes', or 'centerlines'
    compression : str
        Compression algorithm: 'snappy', 'gzip', 'zstd', or None

    Returns
    -------
    int
        Number of rows exported

    Raises
    ------
    ImportError
        If geopandas or pyarrow is not installed
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point, LineString
    except ImportError:
        raise ImportError(
            "geopandas is required for GeoParquet export. "
            "Install with: pip install geopandas pyarrow"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if table == 'reaches':
        data = sword.reaches
        gdf = gpd.GeoDataFrame({
            'reach_id': data.id,
            'region': sword.region,
            'dist_out': data.dist_out,
            'facc': data.facc,
            'wse': data.wse,
            'width': data.wth,
            'slope': data.slope,
            'river_name': data.river_name,
            'stream_order': data.strm_order,
        }, geometry=[Point(x, y) for x, y in zip(data.x, data.y)], crs="EPSG:4326")

    elif table == 'nodes':
        data = sword.nodes
        gdf = gpd.GeoDataFrame({
            'node_id': data.id,
            'reach_id': data.reach_id,
            'region': sword.region,
            'dist_out': data.dist_out,
            'facc': data.facc,
            'wse': data.wse,
            'width': data.wth,
        }, geometry=[Point(x, y) for x, y in zip(data.x, data.y)], crs="EPSG:4326")

    elif table == 'centerlines':
        data = sword.centerlines
        gdf = gpd.GeoDataFrame({
            'cl_id': data.cl_id,
            'reach_id': data.reach_id[0],
            'node_id': data.node_id[0],
            'region': sword.region,
        }, geometry=[Point(x, y) for x, y in zip(data.x, data.y)], crs="EPSG:4326")
    else:
        raise ValueError(f"Unknown table: {table}")

    gdf.to_parquet(str(output_path), compression=compression)
    return len(gdf)


def export_to_geopackage(
    sword: 'SWORD',
    output_path: Union[str, Path],
    tables: List[str] = None,
    layer_prefix: str = ""
) -> Dict[str, int]:
    """
    Export SWORD tables to GeoPackage format.

    Parameters
    ----------
    sword : SWORD
        Loaded SWORD instance
    output_path : str or Path
        Output file path (.gpkg)
    tables : list of str, optional
        Tables to export. Default is ['reaches', 'nodes']
    layer_prefix : str, optional
        Prefix for layer names

    Returns
    -------
    dict
        Dictionary with layer names and row counts
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        raise ImportError(
            "geopandas is required for GeoPackage export. "
            "Install with: pip install geopandas"
        )

    if tables is None:
        tables = ['reaches', 'nodes']

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}

    for table in tables:
        layer_name = f"{layer_prefix}{table}" if layer_prefix else table

        if table == 'reaches':
            data = sword.reaches
            gdf = gpd.GeoDataFrame({
                'reach_id': data.id,
                'region': sword.region,
                'dist_out': data.dist_out,
                'facc': data.facc,
                'wse': data.wse,
                'width': data.wth,
                'slope': data.slope,
                'river_name': data.river_name,
                'stream_order': data.strm_order,
                'main_side': data.main_side,
                'end_rch': data.end_rch,
            }, geometry=[Point(x, y) for x, y in zip(data.x, data.y)], crs="EPSG:4326")

        elif table == 'nodes':
            data = sword.nodes
            gdf = gpd.GeoDataFrame({
                'node_id': data.id,
                'reach_id': data.reach_id,
                'region': sword.region,
                'dist_out': data.dist_out,
                'facc': data.facc,
                'wse': data.wse,
                'width': data.wth,
                'stream_order': data.strm_order,
            }, geometry=[Point(x, y) for x, y in zip(data.x, data.y)], crs="EPSG:4326")
        else:
            continue

        # Always write fresh layers (mode='a' can cause fiona issues)
        # GeoPackage supports multiple layers natively
        gdf.to_file(str(output_path), layer=layer_name, driver='GPKG')
        results[layer_name] = len(gdf)

    return results


def sync_from_postgres(
    sword: 'SWORD',
    connection_string: str,
    table: str,
    prefix: str = "",
    changed_only: bool = True,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Sync changes from PostgreSQL back to DuckDB.

    This function is used after editing in QGIS to pull changes back
    into the main DuckDB database.

    Parameters
    ----------
    sword : SWORD
        Loaded SWORD instance (will be modified)
    connection_string : str
        PostgreSQL connection string
    table : str
        Table to sync: 'reaches' or 'nodes'
    prefix : str, optional
        Table name prefix
    changed_only : bool, optional
        If True, only sync rows marked as changed. Default is True.
        Requires an 'updated_at' or 'changed' column in PostgreSQL.
    verbose : bool, optional
        Print progress messages

    Returns
    -------
    dict
        Statistics about the sync operation
    """
    conn = _get_pg_connection(connection_string)
    cursor = conn.cursor()

    table_name = f"{prefix}{table}" if prefix else table

    try:
        if table == 'reaches':
            return _sync_reaches_from_pg(sword, cursor, table_name, changed_only, verbose)
        elif table == 'nodes':
            return _sync_nodes_from_pg(sword, cursor, table_name, changed_only, verbose)
        else:
            raise ValueError(f"Cannot sync table: {table}")
    finally:
        cursor.close()
        conn.close()


def _sync_reaches_from_pg(
    sword: 'SWORD',
    cursor,
    table_name: str,
    changed_only: bool,
    verbose: bool
) -> Dict[str, int]:
    """Sync reach changes from PostgreSQL to DuckDB."""

    # Get columns that can be synced (editable attributes)
    editable_cols = [
        'dist_out', 'facc', 'wse', 'wse_var', 'width', 'width_var', 'slope',
        'lakeflag', 'river_name', 'edit_flag', 'trib_flag',
        'main_side', 'end_reach', 'stream_order'
    ]

    # Get all rows from PostgreSQL (or only changed ones)
    where_clause = ""
    if changed_only:
        # Check if table has 'updated_at' column
        cursor.execute(f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = '{table_name}' AND column_name = 'updated_at'
        """)
        if cursor.fetchone():
            where_clause = "WHERE updated_at > CURRENT_TIMESTAMP - INTERVAL '1 day'"

    col_list = ', '.join(['reach_id'] + editable_cols)
    cursor.execute(f"SELECT {col_list} FROM {table_name} {where_clause}")

    rows = cursor.fetchall()

    if verbose:
        print(f"Syncing {len(rows)} reaches from {table_name}...")

    updated = 0
    for row in rows:
        reach_id = row[0]
        # Find reach index in SWORD
        idx = np.where(sword.reaches.id == reach_id)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]

        # Update each editable column
        for i, col in enumerate(editable_cols):
            new_value = row[i + 1]
            if new_value is not None:
                attr = getattr(sword.reaches, col, None)
                if attr is not None:
                    attr[idx] = new_value

        updated += 1

    return {'updated': updated, 'total': len(rows)}


def _sync_nodes_from_pg(
    sword: 'SWORD',
    cursor,
    table_name: str,
    changed_only: bool,
    verbose: bool
) -> Dict[str, int]:
    """Sync node changes from PostgreSQL to DuckDB."""

    editable_cols = [
        'dist_out', 'facc', 'wse', 'wse_var', 'width', 'width_var',
        'lakeflag', 'river_name', 'edit_flag', 'trib_flag',
        'main_side', 'end_reach', 'stream_order'
    ]

    where_clause = ""
    if changed_only:
        cursor.execute(f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = '{table_name}' AND column_name = 'updated_at'
        """)
        if cursor.fetchone():
            where_clause = "WHERE updated_at > CURRENT_TIMESTAMP - INTERVAL '1 day'"

    col_list = ', '.join(['node_id'] + editable_cols)
    cursor.execute(f"SELECT {col_list} FROM {table_name} {where_clause}")

    rows = cursor.fetchall()

    if verbose:
        print(f"Syncing {len(rows)} nodes from {table_name}...")

    # Disable GC during bulk updates
    gc_was_enabled = gc.isenabled()
    gc.disable()

    try:
        updated = 0
        for row in rows:
            node_id = row[0]
            idx = np.where(sword.nodes.id == node_id)[0]
            if len(idx) == 0:
                continue
            idx = idx[0]

            for i, col in enumerate(editable_cols):
                new_value = row[i + 1]
                if new_value is not None:
                    attr = getattr(sword.nodes, col, None)
                    if attr is not None:
                        attr[idx] = new_value

            updated += 1
    finally:
        if gc_was_enabled:
            gc.enable()

    return {'updated': updated, 'total': len(rows)}
