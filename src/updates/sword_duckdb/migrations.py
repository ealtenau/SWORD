# -*- coding: utf-8 -*-
"""
SWORD NetCDF to DuckDB Migration
================================

This module provides utilities for migrating SWORD data from NetCDF format
to DuckDB. It handles the transformation from NumPy arrays to relational tables,
including normalization of multi-dimensional arrays.

Example Usage:
    from migrations import migrate_region, migrate_all_regions

    # Migrate a single region
    db = SWORDDatabase('sword_v18.duckdb')
    migrate_region(
        nc_path='/data/netcdf/na_sword_v18.nc',
        db=db,
        region='NA',
        version='v18'
    )

    # Migrate all regions
    migrate_all_regions(
        nc_dir='/data/netcdf/',
        db=db,
        version='v18'
    )
"""

from __future__ import annotations

import logging
import sys
import os
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

# Add parent directory to path for importing sword_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from .sword_db import SWORDDatabase

logger = logging.getLogger(__name__)


# Region codes and their NetCDF file prefixes
REGIONS = ['NA', 'SA', 'EU', 'AF', 'OC', 'AS']


def migrate_region(
    nc_path: str,
    db: SWORDDatabase,
    region: str,
    version: str,
    batch_size: int = 25000,
    verbose: bool = True,
    build_geometry: bool = False
) -> dict:
    """
    Migrate a single NetCDF file to DuckDB.

    Parameters
    ----------
    nc_path : str
        Path to the NetCDF file to migrate.
    db : SWORDDatabase
        Target database connection.
    region : str
        Two-letter region code (e.g., 'NA', 'SA').
    version : str
        SWORD version (e.g., 'v18').
    batch_size : int, optional
        Number of records to insert per batch. Default is 25000.
    verbose : bool, optional
        If True, print progress messages. Default is True.
    build_geometry : bool, optional
        If True, build geometry columns after data migration. Default is False
        to avoid memory issues during large migrations. Geometry can be built
        separately using build_all_geometry() after migration completes.

    Returns
    -------
    dict
        Migration statistics including row counts and timing.

    Raises
    ------
    FileNotFoundError
        If the NetCDF file does not exist.
    """
    from src.updates.sword_utils import read_nc

    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    stats = {
        'region': region,
        'version': version,
        'start_time': datetime.now(),
        'row_counts': {}
    }

    if verbose:
        logger.info(f"Starting migration for {region} from {nc_path}")

    # Read NetCDF data
    if verbose:
        logger.info("Reading NetCDF file...")
    centerlines, nodes, reaches = read_nc(nc_path)

    conn = db.connect()

    # Migrate centerlines
    if verbose:
        logger.info(f"Migrating centerlines ({len(centerlines.cl_id):,} records)...")
    cl_count = _migrate_centerlines(conn, centerlines, region, version, batch_size)
    stats['row_counts']['centerlines'] = cl_count

    # Migrate centerline neighbors
    if verbose:
        logger.info("Migrating centerline neighbors...")
    cl_neighbor_count = _migrate_centerline_neighbors(conn, centerlines, region, batch_size)
    stats['row_counts']['centerline_neighbors'] = cl_neighbor_count

    # Migrate nodes
    if verbose:
        logger.info(f"Migrating nodes ({len(nodes.id):,} records)...")
    node_count = _migrate_nodes(conn, nodes, region, version, batch_size)
    stats['row_counts']['nodes'] = node_count

    # Migrate reaches
    if verbose:
        logger.info(f"Migrating reaches ({len(reaches.id):,} records)...")
    reach_count = _migrate_reaches(conn, reaches, region, version, batch_size)
    stats['row_counts']['reaches'] = reach_count

    # Migrate reach topology
    if verbose:
        logger.info("Migrating reach topology...")
    topo_count = _migrate_reach_topology(conn, reaches, region, batch_size)
    stats['row_counts']['reach_topology'] = topo_count

    # Migrate SWOT orbits
    if verbose:
        logger.info("Migrating SWOT orbits...")
    orbit_count = _migrate_swot_orbits(conn, reaches, region, batch_size)
    stats['row_counts']['reach_swot_orbits'] = orbit_count

    # Build geometry columns (optional - can cause memory issues on large datasets)
    if build_geometry and db.spatial_available():
        if verbose:
            logger.info("Building geometry columns...")
        _build_geometry(conn, region)

    stats['end_time'] = datetime.now()
    stats['duration'] = stats['end_time'] - stats['start_time']

    if verbose:
        logger.info(f"Migration complete in {stats['duration']}")
        logger.info(f"Row counts: {stats['row_counts']}")

    return stats


def _migrate_centerlines(
    conn,
    centerlines,
    region: str,
    version: str,
    batch_size: int
) -> int:
    """
    Migrate centerlines data to DuckDB.

    NOTE: Schema uses composite primary key (cl_id, region) because cl_id
    is only unique within a region, not globally across all SWORD data.
    """
    n = len(centerlines.cl_id)

    # Create DataFrame with primary associations (row 0)
    # Column order matches schema: cl_id, region, x, y, reach_id, node_id, version
    df = pd.DataFrame({
        'cl_id': centerlines.cl_id,
        'region': region,
        'x': centerlines.x,
        'y': centerlines.y,
        'reach_id': centerlines.reach_id[0, :],  # Primary reach
        'node_id': centerlines.node_id[0, :],    # Primary node
        'version': version
    })

    # Insert in batches
    for i in range(0, n, batch_size):
        batch = df.iloc[i:i + batch_size]
        conn.execute("""
            INSERT INTO centerlines (cl_id, region, x, y, reach_id, node_id, version)
            SELECT * FROM batch
        """)

    return n


def _migrate_centerline_neighbors(conn, centerlines, region: str, batch_size: int) -> int:
    """
    Migrate centerline neighbor relationships.

    NOTE: Region is part of the composite key since cl_id is only unique within a region.
    """
    n = len(centerlines.cl_id)
    count = 0

    # Process neighbor ranks 1, 2, 3 (row 0 is primary, in main table)
    for rank in [1, 2, 3]:
        # Filter to non-zero neighbors
        mask = centerlines.reach_id[rank, :] > 0
        if not np.any(mask):
            continue

        indices = np.where(mask)[0]

        df = pd.DataFrame({
            'cl_id': centerlines.cl_id[indices],
            'region': region,
            'neighbor_rank': rank,
            'reach_id': centerlines.reach_id[rank, indices],
            'node_id': centerlines.node_id[rank, indices]
        })

        # Insert in batches
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            conn.execute("""
                INSERT INTO centerline_neighbors (cl_id, region, neighbor_rank, reach_id, node_id)
                SELECT * FROM batch
            """)

        count += len(df)

    return count


def _migrate_nodes(conn, nodes, region: str, version: str, batch_size: int) -> int:
    """
    Migrate nodes data to DuckDB.

    NOTE: Schema uses composite primary key (node_id, region) because node_id
    is only unique within a region.
    """
    n = len(nodes.id)

    # Build DataFrame - column order matches schema with (node_id, region) as PK
    df_dict = {
        'node_id': nodes.id,
        'region': region,  # Part of composite PK
        'x': nodes.x,
        'y': nodes.y,
        'cl_id_min': nodes.cl_id[0, :],
        'cl_id_max': nodes.cl_id[1, :],
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
        'grod_id': nodes.grod_fid,
        'hfalls_id': nodes.hfalls_fid,
        'n_chan_max': nodes.nchan_max,
        'n_chan_mod': nodes.nchan_mod,
        'wth_coef': nodes.wth_coef,
        'ext_dist_coef': nodes.ext_dist_coef,
        'meander_length': nodes.meand_len,
        'sinuosity': nodes.sinuosity,
        'river_name': _decode_strings(nodes.river_name),
        'manual_add': nodes.manual_add,
        'edit_flag': _decode_strings(nodes.edit_flag),
        'trib_flag': nodes.trib_flag,
        'path_freq': nodes.path_freq,
        'path_order': nodes.path_order,
        'path_segs': nodes.path_segs,
        'stream_order': nodes.strm_order,
        'main_side': nodes.main_side,
        'end_reach': nodes.end_rch,
        'network': nodes.network,
        'version': version
    }

    # Add optional add_flag if present
    if hasattr(nodes, 'add_flag'):
        df_dict['add_flag'] = nodes.add_flag

    df = pd.DataFrame(df_dict)

    # Get column list for insert
    columns = list(df_dict.keys())
    placeholders = ', '.join(columns)

    # Insert in batches
    for i in range(0, n, batch_size):
        batch = df.iloc[i:i + batch_size]
        conn.execute(f"""
            INSERT INTO nodes ({placeholders})
            SELECT {placeholders} FROM batch
        """)

    return n


def _migrate_reaches(conn, reaches, region: str, version: str, batch_size: int) -> int:
    """
    Migrate reaches data to DuckDB.

    NOTE: reach_id IS globally unique in SWORD (first digits encode region/basin),
    but we include region in the composite PK for consistency and query efficiency.
    """
    n = len(reaches.id)

    # Build DataFrame - column order matches schema with (reach_id, region) as PK
    df_dict = {
        'reach_id': reaches.id,
        'region': region,  # Part of composite PK for consistency
        'x': reaches.x,
        'y': reaches.y,
        'x_min': reaches.x_min,
        'x_max': reaches.x_max,
        'y_min': reaches.y_min,
        'y_max': reaches.y_max,
        'cl_id_min': reaches.cl_id[0, :],
        'cl_id_max': reaches.cl_id[1, :],
        'reach_length': reaches.len,
        'n_nodes': reaches.rch_n_nodes,
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
        'grod_id': reaches.grod_fid,
        'hfalls_id': reaches.hfalls_fid,
        'n_chan_max': reaches.nchan_max,
        'n_chan_mod': reaches.nchan_mod,
        'n_rch_up': reaches.n_rch_up,
        'n_rch_down': reaches.n_rch_down,
        'swot_obs': reaches.max_obs,
        'iceflag': _get_iceflag_scalar(reaches),
        'low_slope_flag': reaches.low_slope,
        'river_name': _decode_strings(reaches.river_name),
        'edit_flag': _decode_strings(reaches.edit_flag),
        'trib_flag': reaches.trib_flag,
        'path_freq': reaches.path_freq,
        'path_order': reaches.path_order,
        'path_segs': reaches.path_segs,
        'stream_order': reaches.strm_order,
        'main_side': reaches.main_side,
        'end_reach': reaches.end_rch,
        'network': reaches.network,
        'version': version
    }

    # Add optional add_flag if present
    if hasattr(reaches, 'add_flag'):
        df_dict['add_flag'] = reaches.add_flag

    df = pd.DataFrame(df_dict)

    # Get column list for insert
    columns = list(df_dict.keys())
    placeholders = ', '.join(columns)

    # Insert in batches
    for i in range(0, n, batch_size):
        batch = df.iloc[i:i + batch_size]
        conn.execute(f"""
            INSERT INTO reaches ({placeholders})
            SELECT {placeholders} FROM batch
        """)

    return n


def _migrate_reach_topology(conn, reaches, region: str, batch_size: int) -> int:
    """
    Migrate reach topology (upstream/downstream neighbors).

    NOTE: Region is included in the table for efficient filtering even though
    reach_id is globally unique (encodes region in first digits).
    """
    count = 0

    # Upstream neighbors
    for rank in range(4):
        mask = reaches.rch_id_up[rank, :] > 0
        if not np.any(mask):
            continue

        indices = np.where(mask)[0]
        df = pd.DataFrame({
            'reach_id': reaches.id[indices],
            'region': region,
            'direction': 'up',
            'neighbor_rank': rank,
            'neighbor_reach_id': reaches.rch_id_up[rank, indices]
        })

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            conn.execute("""
                INSERT INTO reach_topology
                SELECT * FROM batch
            """)

        count += len(df)

    # Downstream neighbors
    for rank in range(4):
        mask = reaches.rch_id_down[rank, :] > 0
        if not np.any(mask):
            continue

        indices = np.where(mask)[0]
        df = pd.DataFrame({
            'reach_id': reaches.id[indices],
            'region': region,
            'direction': 'down',
            'neighbor_rank': rank,
            'neighbor_reach_id': reaches.rch_id_down[rank, indices]
        })

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            conn.execute("""
                INSERT INTO reach_topology
                SELECT * FROM batch
            """)

        count += len(df)

    return count


def _migrate_swot_orbits(conn, reaches, region: str, batch_size: int) -> int:
    """
    Migrate SWOT orbit data.

    NOTE: Region is included for efficient filtering.
    """
    count = 0

    for rank in range(75):  # Max 75 orbits
        mask = reaches.orbits[rank, :] > 0
        if not np.any(mask):
            continue

        indices = np.where(mask)[0]
        df = pd.DataFrame({
            'reach_id': reaches.id[indices],
            'region': region,
            'orbit_rank': rank,
            'orbit_id': reaches.orbits[rank, indices]
        })

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            conn.execute("""
                INSERT INTO reach_swot_orbits
                SELECT * FROM batch
            """)

        count += len(df)

    return count


def _build_geometry(conn, region: str) -> None:
    """Build geometry columns from x,y coordinates."""
    # Build point geometry for centerlines
    conn.execute("""
        UPDATE centerlines
        SET geom = ST_Point(x, y)
        WHERE region = ? AND geom IS NULL
    """, [region])

    # Build point geometry for nodes
    conn.execute("""
        UPDATE nodes
        SET geom = ST_Point(x, y)
        WHERE region = ? AND geom IS NULL
    """, [region])

    # Note: Reach geometry (LINESTRING) needs to be built separately
    # from centerline points, which is more complex and done in a later step


def build_all_geometry(db: SWORDDatabase, regions: list = None, verbose: bool = True) -> None:
    """
    Build geometry columns for all migrated data.

    This function should be called after all data migrations are complete
    to avoid memory issues during the migration process.

    Parameters
    ----------
    db : SWORDDatabase
        Database connection.
    regions : list, optional
        List of regions to build geometry for. If None, builds for all regions.
    verbose : bool, optional
        If True, print progress messages.
    """
    if not db.spatial_available():
        if verbose:
            logger.warning("Spatial extension not available, skipping geometry build")
        return

    conn = db.connect()

    if regions is None:
        # Get all regions present in database
        result = conn.execute("SELECT DISTINCT region FROM reaches ORDER BY region").fetchall()
        regions = [r[0] for r in result]

    for region in regions:
        if verbose:
            logger.info(f"Building geometry for {region}...")
        _build_geometry(conn, region)

    if verbose:
        logger.info("Geometry build complete")


def _decode_strings(arr: np.ndarray) -> list:
    """Decode byte strings from NetCDF to Python strings."""
    if arr.dtype.kind == 'S':  # byte string
        return [s.decode('utf-8', errors='replace') if isinstance(s, bytes) else str(s) for s in arr]
    return arr.tolist()


def _get_iceflag_scalar(reaches) -> np.ndarray:
    """
    Get scalar iceflag from potentially multi-dimensional array.

    The iceflag in NetCDF may be [366, N] for daily values.
    We reduce it to a single value per reach (max ice coverage).
    """
    if reaches.iceflag.ndim == 1:
        return reaches.iceflag
    # For multi-dimensional, take max across days
    return np.max(reaches.iceflag, axis=0)


def migrate_all_regions(
    nc_dir: str,
    db: SWORDDatabase,
    version: str,
    regions: list = None,
    verbose: bool = True
) -> dict:
    """
    Migrate all regions from NetCDF to DuckDB.

    Parameters
    ----------
    nc_dir : str
        Directory containing NetCDF files.
    db : SWORDDatabase
        Target database connection.
    version : str
        SWORD version (e.g., 'v18').
    regions : list, optional
        List of regions to migrate. If None, migrates all regions.
    verbose : bool, optional
        If True, print progress messages.

    Returns
    -------
    dict
        Migration statistics for all regions.
    """
    if regions is None:
        regions = REGIONS

    all_stats = {}

    for region in regions:
        nc_path = os.path.join(nc_dir, f"{region.lower()}_sword_{version}.nc")
        if os.path.exists(nc_path):
            stats = migrate_region(nc_path, db, region, version, verbose=verbose)
            all_stats[region] = stats
        else:
            if verbose:
                logger.warning(f"NetCDF file not found for {region}: {nc_path}")

    return all_stats


def validate_migration(
    nc_path: str,
    db: SWORDDatabase,
    region: str,
    sample_size: int = 1000
) -> dict:
    """
    Validate migrated data against source NetCDF.

    Parameters
    ----------
    nc_path : str
        Path to source NetCDF file.
    db : SWORDDatabase
        Database to validate.
    region : str
        Region code to validate.
    sample_size : int, optional
        Number of records to sample for attribute validation.

    Returns
    -------
    dict
        Validation results including row count matches and sample checks.
    """
    from src.updates.sword_utils import read_nc

    centerlines, nodes, reaches = read_nc(nc_path)

    validation = {
        'region': region,
        'row_counts_match': {},
        'sample_checks': {},
        'passed': True
    }

    conn = db.connect()

    # Check row counts
    nc_counts = {
        'centerlines': len(centerlines.cl_id),
        'nodes': len(nodes.id),
        'reaches': len(reaches.id)
    }

    for table, nc_count in nc_counts.items():
        db_count = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE region = ?",
            [region]
        ).fetchone()[0]

        match = nc_count == db_count
        validation['row_counts_match'][table] = {
            'netcdf': nc_count,
            'duckdb': db_count,
            'match': match
        }
        if not match:
            validation['passed'] = False

    # Sample attribute validation
    # Sample reach IDs
    sample_indices = np.random.choice(
        len(reaches.id),
        min(sample_size, len(reaches.id)),
        replace=False
    )

    for idx in sample_indices[:10]:  # Detailed check on 10 samples
        reach_id = reaches.id[idx]
        db_row = conn.execute(
            "SELECT * FROM reaches WHERE reach_id = ?",
            [int(reach_id)]
        ).fetchdf()

        if len(db_row) == 0:
            validation['sample_checks'][str(reach_id)] = 'MISSING'
            validation['passed'] = False
            continue

        # Check key attributes match
        checks = {
            'x': abs(reaches.x[idx] - db_row['x'].iloc[0]) < 1e-10,
            'wse': abs(reaches.wse[idx] - db_row['wse'].iloc[0]) < 1e-6,
            'facc': abs(reaches.facc[idx] - db_row['facc'].iloc[0]) < 1e-6,
        }

        validation['sample_checks'][str(reach_id)] = checks

        if not all(checks.values()):
            validation['passed'] = False

    return validation
