# -*- coding: utf-8 -*-
"""
SWORD DuckDB Class
==================

DuckDB-backed SWORD class - drop-in replacement for the original NetCDF-based class.
Provides the same interface for backward compatibility with existing pipelines.

Example Usage:
    from sword_duckdb import SWORD

    # Load from DuckDB
    sword = SWORD('data/duckdb/sword_v17b.duckdb', 'NA')

    # Access data (same interface as original)
    print(sword.reaches.wse[:5])
    print(sword.nodes.facc[:5])
    print(sword.centerlines.x[:5])

    # Topology arrays
    print(sword.reaches.rch_id_up[:, :5])
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List
import numpy as np

from .sword_db import SWORDDatabase
from .views import CenterlinesView, NodesView, ReachesView


class SWORD:
    """
    DuckDB-backed SWORD class.

    Drop-in replacement for the original NetCDF-based SWORD class.
    Loads data from DuckDB and provides numpy-array-style attribute access.

    Parameters
    ----------
    db_path : str or Path
        Path to the DuckDB database file.
    region : str
        Two-letter region code (NA, SA, EU, AF, OC, AS).
    version : str, optional
        SWORD version. Default is 'v17b'.

    Attributes
    ----------
    region : str
        The region code.
    version : str
        The SWORD version.
    centerlines : CenterlinesView
        Centerline data with numpy-array access.
    nodes : NodesView
        Node data with numpy-array access.
    reaches : ReachesView
        Reach data with numpy-array access.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        region: str,
        version: str = 'v17b',
        spatial: bool = True
    ):
        self.region = region.upper()
        self.version = version
        self._db_path = Path(db_path)
        # Always load spatial for write operations (needed for RTREE indexes)
        self._db = SWORDDatabase(db_path, read_only=False, spatial=spatial)

        # Load data
        self._load_data()

    def _load_data(self) -> None:
        """Load all data for the region from DuckDB."""
        # Load main DataFrames
        self._centerlines_df = self._db.query(
            "SELECT * FROM centerlines WHERE region = ? ORDER BY cl_id",
            [self.region]
        )
        self._nodes_df = self._db.query(
            "SELECT * FROM nodes WHERE region = ? ORDER BY node_id",
            [self.region]
        )
        self._reaches_df = self._db.query(
            "SELECT * FROM reaches WHERE region = ? ORDER BY reach_id",
            [self.region]
        )

        # Reconstruct multi-dimensional arrays
        cl_reach_id, cl_node_id = self._reconstruct_centerline_neighbors()
        rch_id_up = self._reconstruct_reach_topology('up')
        rch_id_down = self._reconstruct_reach_topology('down')
        orbits = self._reconstruct_orbits()
        iceflag = self._reconstruct_iceflag()

        # Create view objects
        self._centerlines = CenterlinesView(
            self._centerlines_df, cl_reach_id, cl_node_id
        )
        self._nodes = NodesView(self._nodes_df)
        self._reaches = ReachesView(
            self._reaches_df, rch_id_up, rch_id_down, orbits, iceflag
        )

    def _reconstruct_centerline_neighbors(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct [4,N] reach_id and node_id arrays from centerlines + neighbors.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (reach_id[4,N], node_id[4,N]) arrays
        """
        n = len(self._centerlines_df)
        reach_id = np.zeros((4, n), dtype=np.int64)
        node_id = np.zeros((4, n), dtype=np.int64)

        # Row 0: primary from main table
        reach_id[0, :] = self._centerlines_df['reach_id'].values
        node_id[0, :] = self._centerlines_df['node_id'].values

        # Rows 1-3: from centerline_neighbors table
        neighbors_df = self._db.query("""
            SELECT cl_id, neighbor_rank, reach_id, node_id
            FROM centerline_neighbors
            WHERE region = ?
            ORDER BY cl_id, neighbor_rank
        """, [self.region])

        if len(neighbors_df) > 0:
            # Build lookup from cl_id to array index
            cl_ids = self._centerlines_df['cl_id'].values
            cl_idx = {cid: i for i, cid in enumerate(cl_ids)}

            for _, row in neighbors_df.iterrows():
                idx = cl_idx.get(row['cl_id'])
                if idx is not None:
                    rank = row['neighbor_rank']
                    if 1 <= rank <= 3:
                        reach_id[rank, idx] = row['reach_id'] if row['reach_id'] else 0
                        node_id[rank, idx] = row['node_id'] if row['node_id'] else 0

        return reach_id, node_id

    def _reconstruct_reach_topology(self, direction: str) -> np.ndarray:
        """
        Reconstruct [4,N] topology array from reach_topology table.

        Parameters
        ----------
        direction : str
            'up' for upstream, 'down' for downstream.

        Returns
        -------
        np.ndarray
            [4,N] array of neighbor reach IDs.
        """
        n = len(self._reaches_df)
        arr = np.zeros((4, n), dtype=np.int64)

        # Build lookup from reach_id to array index
        reach_ids = self._reaches_df['reach_id'].values
        reach_idx = {rid: i for i, rid in enumerate(reach_ids)}

        # Query topology
        topo_df = self._db.query("""
            SELECT reach_id, neighbor_rank, neighbor_reach_id
            FROM reach_topology
            WHERE direction = ? AND region = ?
        """, [direction, self.region])

        for _, row in topo_df.iterrows():
            idx = reach_idx.get(row['reach_id'])
            if idx is not None:
                rank = row['neighbor_rank']
                if 0 <= rank <= 3:
                    arr[rank, idx] = row['neighbor_reach_id']

        return arr

    def _reconstruct_orbits(self) -> np.ndarray:
        """
        Reconstruct [75,N] SWOT orbits array from reach_swot_orbits table.

        Returns
        -------
        np.ndarray
            [75,N] array of orbit IDs.
        """
        n = len(self._reaches_df)
        arr = np.zeros((75, n), dtype=np.int64)

        reach_ids = self._reaches_df['reach_id'].values
        reach_idx = {rid: i for i, rid in enumerate(reach_ids)}

        orbits_df = self._db.query("""
            SELECT reach_id, orbit_rank, orbit_id
            FROM reach_swot_orbits
            WHERE region = ?
        """, [self.region])

        for _, row in orbits_df.iterrows():
            idx = reach_idx.get(row['reach_id'])
            if idx is not None:
                rank = row['orbit_rank']
                if 0 <= rank < 75:
                    arr[rank, idx] = row['orbit_id']

        return arr

    def _reconstruct_iceflag(self) -> np.ndarray:
        """
        Reconstruct [366,N] ice flag array from reach_ice_flags table.

        Returns
        -------
        np.ndarray
            [366,N] array of daily ice flags.
        """
        n = len(self._reaches_df)
        arr = np.zeros((366, n), dtype=np.int32)

        reach_ids = self._reaches_df['reach_id'].values
        reach_idx = {rid: i for i, rid in enumerate(reach_ids)}

        # Ice flags may be large, query in batches if needed
        ice_df = self._db.query("""
            SELECT reach_id, julian_day, iceflag
            FROM reach_ice_flags
            WHERE reach_id IN (SELECT reach_id FROM reaches WHERE region = ?)
        """, [self.region])

        for _, row in ice_df.iterrows():
            idx = reach_idx.get(row['reach_id'])
            if idx is not None:
                day = row['julian_day'] - 1  # Convert 1-366 to 0-365
                if 0 <= day < 366:
                    arr[day, idx] = row['iceflag']

        return arr

    @property
    def centerlines(self) -> CenterlinesView:
        """Centerline data with numpy-array access."""
        return self._centerlines

    @property
    def nodes(self) -> NodesView:
        """Node data with numpy-array access."""
        return self._nodes

    @property
    def reaches(self) -> ReachesView:
        """Reach data with numpy-array access."""
        return self._reaches

    def copy(self) -> None:
        """
        Create a backup of the current state.

        For DuckDB, this creates a checkpoint and stores a backup timestamp.
        The database file can be copied externally if needed.
        """
        current_datetime = datetime.now()
        backup_note = f"Backup created at {current_datetime.strftime('%Y-%m-%d_%H-%M-%S')}"

        # Create a checkpoint to ensure data is flushed
        self._db.execute("CHECKPOINT")

        # Record backup in versions table
        self._db.execute("""
            INSERT OR REPLACE INTO sword_versions (version, notes)
            VALUES (?, ?)
        """, [f'backup_{current_datetime.strftime("%Y%m%d_%H%M%S")}', backup_note])

        print(f"Checkpoint created: {backup_note}")

    def delete_data(self, rm_rch: Union[List[int], np.ndarray]) -> None:
        """
        Delete reaches and associated data across all dimensions.

        Parameters
        ----------
        rm_rch : list or np.ndarray
            Reach IDs to delete.
        """
        if len(rm_rch) == 0:
            return

        rm_rch = np.array(rm_rch)

        # Start transaction
        conn = self._db.connect()
        conn.execute("BEGIN TRANSACTION")

        try:
            # Convert to list for SQL IN clause
            rm_list = rm_rch.tolist()

            # Delete from centerlines
            conn.execute(f"""
                DELETE FROM centerlines
                WHERE reach_id IN ({','.join('?' * len(rm_list))})
                AND region = ?
            """, rm_list + [self.region])

            # Delete from centerline_neighbors (may have references)
            conn.execute(f"""
                DELETE FROM centerline_neighbors
                WHERE cl_id IN (
                    SELECT cl_id FROM centerlines
                    WHERE reach_id IN ({','.join('?' * len(rm_list))})
                    AND region = ?
                )
                AND region = ?
            """, rm_list + [self.region, self.region])

            # Delete from nodes
            conn.execute(f"""
                DELETE FROM nodes
                WHERE reach_id IN ({','.join('?' * len(rm_list))})
                AND region = ?
            """, rm_list + [self.region])

            # Delete from reach_topology (both as source and neighbor)
            conn.execute(f"""
                DELETE FROM reach_topology
                WHERE reach_id IN ({','.join('?' * len(rm_list))})
                AND region = ?
            """, rm_list + [self.region])

            # Update topology: remove deleted reaches from neighbor lists
            conn.execute(f"""
                DELETE FROM reach_topology
                WHERE neighbor_reach_id IN ({','.join('?' * len(rm_list))})
                AND region = ?
            """, rm_list + [self.region])

            # Delete from reach_swot_orbits
            conn.execute(f"""
                DELETE FROM reach_swot_orbits
                WHERE reach_id IN ({','.join('?' * len(rm_list))})
                AND region = ?
            """, rm_list + [self.region])

            # Delete from reach_ice_flags
            conn.execute(f"""
                DELETE FROM reach_ice_flags
                WHERE reach_id IN ({','.join('?' * len(rm_list))})
            """, rm_list)

            # Delete from reaches (main table)
            conn.execute(f"""
                DELETE FROM reaches
                WHERE reach_id IN ({','.join('?' * len(rm_list))})
                AND region = ?
            """, rm_list + [self.region])

            # Update n_rch_up and n_rch_down counts for affected reaches
            conn.execute(f"""
                UPDATE reaches
                SET n_rch_up = (
                    SELECT COUNT(*) FROM reach_topology t
                    WHERE t.reach_id = reaches.reach_id
                    AND t.region = reaches.region
                    AND t.direction = 'up'
                )
                WHERE region = ?
            """, [self.region])

            conn.execute(f"""
                UPDATE reaches
                SET n_rch_down = (
                    SELECT COUNT(*) FROM reach_topology t
                    WHERE t.reach_id = reaches.reach_id
                    AND t.region = reaches.region
                    AND t.direction = 'down'
                )
                WHERE region = ?
            """, [self.region])

            conn.execute("COMMIT")

            # Reload data
            self._load_data()

            print(f"Deleted {len(rm_rch)} reaches and associated data")

        except Exception as e:
            conn.execute("ROLLBACK")
            raise RuntimeError(f"Failed to delete data: {e}") from e

    def save_nc(self, output_path: Optional[str] = None) -> None:
        """
        Export data to NetCDF format for backward compatibility.

        Parameters
        ----------
        output_path : str, optional
            Output file path. If not provided, uses default naming.
        """
        try:
            import netCDF4 as nc
        except ImportError:
            raise ImportError("netCDF4 is required for save_nc(). Install with: pip install netCDF4")

        if output_path is None:
            output_path = f"sword_{self.region}_{self.version}.nc"

        # Create NetCDF file
        with nc.Dataset(output_path, 'w', format='NETCDF4') as ds:
            # Global attributes
            ds.title = f"SWORD {self.version} - {self.region}"
            ds.created = datetime.now().isoformat()
            ds.source = "Exported from DuckDB"

            # Centerlines group
            cl_grp = ds.createGroup('centerlines')
            n_cl = len(self._centerlines)
            cl_grp.createDimension('num_centerlines', n_cl)
            cl_grp.createDimension('num_neighbors', 4)

            cl_grp.createVariable('cl_id', 'i8', ('num_centerlines',))[:] = self._centerlines.cl_id
            cl_grp.createVariable('x', 'f8', ('num_centerlines',))[:] = self._centerlines.x
            cl_grp.createVariable('y', 'f8', ('num_centerlines',))[:] = self._centerlines.y
            cl_grp.createVariable('reach_id', 'i8', ('num_neighbors', 'num_centerlines'))[:] = self._centerlines.reach_id
            cl_grp.createVariable('node_id', 'i8', ('num_neighbors', 'num_centerlines'))[:] = self._centerlines.node_id

            # Nodes group
            nd_grp = ds.createGroup('nodes')
            n_nd = len(self._nodes)
            nd_grp.createDimension('num_nodes', n_nd)
            nd_grp.createDimension('num_cl_bounds', 2)

            nd_grp.createVariable('node_id', 'i8', ('num_nodes',))[:] = self._nodes.id
            nd_grp.createVariable('x', 'f8', ('num_nodes',))[:] = self._nodes.x
            nd_grp.createVariable('y', 'f8', ('num_nodes',))[:] = self._nodes.y
            nd_grp.createVariable('reach_id', 'i8', ('num_nodes',))[:] = self._nodes.reach_id
            nd_grp.createVariable('cl_id', 'i8', ('num_cl_bounds', 'num_nodes'))[:] = self._nodes.cl_id
            nd_grp.createVariable('len', 'f8', ('num_nodes',))[:] = self._nodes.len
            nd_grp.createVariable('wse', 'f8', ('num_nodes',))[:] = self._nodes.wse
            nd_grp.createVariable('wse_var', 'f8', ('num_nodes',))[:] = self._nodes.wse_var
            nd_grp.createVariable('wth', 'f8', ('num_nodes',))[:] = self._nodes.wth
            nd_grp.createVariable('wth_var', 'f8', ('num_nodes',))[:] = self._nodes.wth_var
            nd_grp.createVariable('max_wth', 'f8', ('num_nodes',))[:] = self._nodes.max_wth
            nd_grp.createVariable('facc', 'f8', ('num_nodes',))[:] = self._nodes.facc
            nd_grp.createVariable('dist_out', 'f8', ('num_nodes',))[:] = self._nodes.dist_out
            nd_grp.createVariable('grod', 'i4', ('num_nodes',))[:] = self._nodes.grod
            nd_grp.createVariable('lakeflag', 'i4', ('num_nodes',))[:] = self._nodes.lakeflag
            nd_grp.createVariable('strm_order', 'i4', ('num_nodes',))[:] = self._nodes.strm_order
            nd_grp.createVariable('end_rch', 'i4', ('num_nodes',))[:] = self._nodes.end_rch

            # Reaches group
            rch_grp = ds.createGroup('reaches')
            n_rch = len(self._reaches)
            rch_grp.createDimension('num_reaches', n_rch)
            rch_grp.createDimension('num_rch_neighbors', 4)
            rch_grp.createDimension('num_orbits', 75)
            rch_grp.createDimension('num_days', 366)

            rch_grp.createVariable('reach_id', 'i8', ('num_reaches',))[:] = self._reaches.id
            rch_grp.createVariable('x', 'f8', ('num_reaches',))[:] = self._reaches.x
            rch_grp.createVariable('y', 'f8', ('num_reaches',))[:] = self._reaches.y
            rch_grp.createVariable('cl_id', 'i8', ('num_cl_bounds', 'num_reaches'))[:] = self._reaches.cl_id
            rch_grp.createVariable('len', 'f8', ('num_reaches',))[:] = self._reaches.len
            rch_grp.createVariable('wse', 'f8', ('num_reaches',))[:] = self._reaches.wse
            rch_grp.createVariable('wth', 'f8', ('num_reaches',))[:] = self._reaches.wth
            rch_grp.createVariable('slope', 'f8', ('num_reaches',))[:] = self._reaches.slope
            rch_grp.createVariable('facc', 'f8', ('num_reaches',))[:] = self._reaches.facc
            rch_grp.createVariable('dist_out', 'f8', ('num_reaches',))[:] = self._reaches.dist_out
            rch_grp.createVariable('n_rch_up', 'i4', ('num_reaches',))[:] = self._reaches.n_rch_up
            rch_grp.createVariable('n_rch_down', 'i4', ('num_reaches',))[:] = self._reaches.n_rch_down
            rch_grp.createVariable('rch_id_up', 'i8', ('num_rch_neighbors', 'num_reaches'))[:] = self._reaches.rch_id_up
            rch_grp.createVariable('rch_id_down', 'i8', ('num_rch_neighbors', 'num_reaches'))[:] = self._reaches.rch_id_down
            rch_grp.createVariable('orbits', 'i8', ('num_orbits', 'num_reaches'))[:] = self._reaches.orbits
            rch_grp.createVariable('iceflag', 'i4', ('num_days', 'num_reaches'))[:] = self._reaches.iceflag
            rch_grp.createVariable('strm_order', 'i4', ('num_reaches',))[:] = self._reaches.strm_order
            rch_grp.createVariable('end_rch', 'i4', ('num_reaches',))[:] = self._reaches.end_rch

        print(f"Exported to {output_path}")

    def append_data(self, subcls, subnodes, subreaches) -> None:
        """
        Append new centerlines, nodes, and reaches to the database.

        Parameters
        ----------
        subcls : object
            Object containing centerlines data with attributes:
            cl_id, lon/x, lat/y, reach_id[4,N], node_id[4,N]
        subnodes : object
            Object containing nodes data with attributes matching NodesView.
        subreaches : object
            Object containing reaches data with attributes matching ReachesView.
        """
        import gc

        if len(subcls.cl_id) == 0 and len(subnodes.id) == 0 and len(subreaches.id) == 0:
            return

        conn = self._db.connect()
        conn.execute("BEGIN TRANSACTION")

        # Disable gc during bulk insert to avoid DuckDB segfaults
        gc_was_enabled = gc.isenabled()
        gc.disable()

        try:
            # Insert centerlines
            if len(subcls.cl_id) > 0:
                self._insert_centerlines(conn, subcls)

            # Insert nodes
            if len(subnodes.id) > 0:
                self._insert_nodes(conn, subnodes)

            # Insert reaches (includes topology, orbits, ice flags)
            if len(subreaches.id) > 0:
                self._insert_reaches(conn, subreaches)

            conn.execute("COMMIT")

            # Reload data to refresh views
            self._load_data()

            print(f"Appended {len(subcls.cl_id)} centerlines, "
                  f"{len(subnodes.id)} nodes, {len(subreaches.id)} reaches")

        except Exception as e:
            conn.execute("ROLLBACK")
            raise RuntimeError(f"Failed to append data: {e}") from e
        finally:
            # Re-enable gc if it was enabled before
            if gc_was_enabled:
                gc.enable()

    def _to_python(self, val):
        """Convert numpy types to Python types for DuckDB."""
        if hasattr(val, 'item'):
            return val.item()
        return val

    def _insert_centerlines(self, conn, subcls) -> None:
        """Insert centerlines and neighbors into database."""
        n = len(subcls.cl_id)

        # Handle both lon/lat and x/y attribute names
        x_vals = getattr(subcls, 'x', None)
        if x_vals is None:
            x_vals = getattr(subcls, 'lon', None)
        y_vals = getattr(subcls, 'y', None)
        if y_vals is None:
            y_vals = getattr(subcls, 'lat', None)

        # Build rows for insertion
        rows = []
        for i in range(n):
            rows.append((
                int(subcls.cl_id[i]),
                self.region,
                float(x_vals[i]),
                float(y_vals[i]),
                int(subcls.reach_id[0, i]),
                int(subcls.node_id[0, i]),
                self.version,
            ))

        # Insert main centerlines using executemany
        conn.executemany("""
            INSERT INTO centerlines (cl_id, region, x, y, reach_id, node_id, version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rows)

        # Insert neighbors (rows 1-3 of reach_id/node_id arrays)
        neighbor_rows = []
        for i in range(n):
            for rank in range(1, 4):
                rid = int(subcls.reach_id[rank, i]) if subcls.reach_id.shape[0] > rank else 0
                nid = int(subcls.node_id[rank, i]) if subcls.node_id.shape[0] > rank else 0
                if rid != 0 or nid != 0:  # Only store non-zero neighbors
                    neighbor_rows.append((
                        int(subcls.cl_id[i]),
                        self.region,
                        rank,
                        rid,
                        nid,
                    ))

        if neighbor_rows:
            conn.executemany("""
                INSERT INTO centerline_neighbors (cl_id, region, neighbor_rank, reach_id, node_id)
                VALUES (?, ?, ?, ?, ?)
            """, neighbor_rows)

        # Update geometry only for newly inserted rows (skip if spatial extension unavailable)
        try:
            cl_ids = [int(cid) for cid in subcls.cl_id]
            placeholders = ', '.join(['?'] * len(cl_ids))
            conn.execute(f"""
                UPDATE centerlines
                SET geom = ST_Point(x, y)
                WHERE region = ? AND cl_id IN ({placeholders}) AND geom IS NULL
            """, [self.region] + cl_ids)
        except Exception:
            pass  # Spatial extension may not be available

    def _insert_nodes(self, conn, subnodes) -> None:
        """Insert nodes into database."""
        import pandas as pd

        n = len(subnodes.id)

        # Build cl_id bounds from [2,N] array
        cl_id_min = subnodes.cl_id[0, :] if subnodes.cl_id.ndim == 2 else subnodes.cl_id
        cl_id_max = subnodes.cl_id[1, :] if subnodes.cl_id.ndim == 2 else subnodes.cl_id

        # Helper to safely get attribute with fallback
        def get_attr(obj, names, default=None):
            """Get attribute by trying multiple names."""
            if isinstance(names, str):
                names = [names]
            for name in names:
                val = getattr(obj, name, None)
                if val is not None:
                    return val
            if default is not None:
                return [default] * n
            return [None] * n

        nodes_data = {
            'node_id': subnodes.id,
            'region': [self.region] * n,
            'x': subnodes.x,
            'y': subnodes.y,
            'cl_id_min': cl_id_min,
            'cl_id_max': cl_id_max,
            'reach_id': subnodes.reach_id,
            'node_length': get_attr(subnodes, 'len', 0),
            'wse': get_attr(subnodes, 'wse', 0),
            'wse_var': get_attr(subnodes, 'wse_var', 0),
            'width': get_attr(subnodes, ['wth', 'width'], 0),
            'width_var': get_attr(subnodes, ['wth_var', 'width_var'], 0),
            'max_width': get_attr(subnodes, ['max_wth', 'max_width'], 0),
            'facc': get_attr(subnodes, 'facc', 0),
            'dist_out': get_attr(subnodes, 'dist_out', 0),
            'lakeflag': get_attr(subnodes, 'lakeflag', 0),
            'obstr_type': get_attr(subnodes, ['grod', 'obstr_type'], 0),
            'grod_id': get_attr(subnodes, ['grod_fid', 'grod_id'], 0),
            'hfalls_id': get_attr(subnodes, ['hfalls_fid', 'hfalls_id'], 0),
            'n_chan_max': get_attr(subnodes, ['nchan_max', 'n_chan_max'], 0),
            'n_chan_mod': get_attr(subnodes, ['nchan_mod', 'n_chan_mod'], 0),
            'wth_coef': get_attr(subnodes, 'wth_coef', 0),
            'ext_dist_coef': get_attr(subnodes, 'ext_dist_coef', 0),
            'meander_length': get_attr(subnodes, ['meand_len', 'meander_length'], 0),
            'sinuosity': get_attr(subnodes, 'sinuosity', 0),
            'river_name': get_attr(subnodes, 'river_name', ''),
            'manual_add': get_attr(subnodes, 'manual_add', 0),
            'edit_flag': get_attr(subnodes, 'edit_flag', ''),
            'trib_flag': get_attr(subnodes, 'trib_flag', 0),
            'path_freq': get_attr(subnodes, 'path_freq', 0),
            'path_order': get_attr(subnodes, 'path_order', 0),
            'path_segs': get_attr(subnodes, 'path_segs', 0),
            'stream_order': get_attr(subnodes, ['strm_order', 'stream_order'], 0),
            'main_side': get_attr(subnodes, 'main_side', 0),
            'end_reach': get_attr(subnodes, ['end_rch', 'end_reach'], 0),
            'network': get_attr(subnodes, 'network', 0),
            'add_flag': get_attr(subnodes, 'add_flag', 0),
            'version': [self.version] * n,
        }

        nodes_df = pd.DataFrame(nodes_data)

        # Reorder columns to match INSERT statement
        cols = [
            'node_id', 'region', 'x', 'y', 'cl_id_min', 'cl_id_max', 'reach_id',
            'node_length', 'wse', 'wse_var', 'width', 'width_var', 'max_width',
            'facc', 'dist_out', 'lakeflag', 'obstr_type', 'grod_id', 'hfalls_id',
            'n_chan_max', 'n_chan_mod', 'wth_coef', 'ext_dist_coef',
            'meander_length', 'sinuosity', 'river_name', 'manual_add',
            'edit_flag', 'trib_flag', 'path_freq', 'path_order', 'path_segs',
            'stream_order', 'main_side', 'end_reach', 'network', 'add_flag', 'version'
        ]
        nodes_df = nodes_df[cols]

        # Insert nodes using executemany for reliability
        placeholders = ', '.join(['?'] * len(cols))
        conn.executemany(f"""
            INSERT INTO nodes ({', '.join(cols)})
            VALUES ({placeholders})
        """, nodes_df.values.tolist())

        # Update geometry only for newly inserted rows (skip if spatial extension unavailable)
        try:
            node_ids = [int(nid) for nid in subnodes.id]
            placeholders = ', '.join(['?'] * len(node_ids))
            conn.execute(f"""
                UPDATE nodes
                SET geom = ST_Point(x, y)
                WHERE region = ? AND node_id IN ({placeholders}) AND geom IS NULL
            """, [self.region] + node_ids)
        except Exception:
            pass  # Spatial extension may not be available

    def _insert_reaches(self, conn, subreaches) -> None:
        """Insert reaches and related normalized tables."""
        import pandas as pd

        n = len(subreaches.id)

        # Build cl_id bounds from [2,N] array
        cl_id_min = subreaches.cl_id[0, :] if subreaches.cl_id.ndim == 2 else subreaches.cl_id
        cl_id_max = subreaches.cl_id[1, :] if subreaches.cl_id.ndim == 2 else subreaches.cl_id

        # Helper to safely get attribute with fallback
        def get_attr(obj, names, default=None):
            if isinstance(names, str):
                names = [names]
            for name in names:
                val = getattr(obj, name, None)
                if val is not None:
                    return val
            if default is not None:
                return [default] * n
            return [None] * n

        reaches_data = {
            'reach_id': subreaches.id,
            'region': [self.region] * n,
            'x': subreaches.x,
            'y': subreaches.y,
            'x_min': get_attr(subreaches, 'x_min', 0),
            'x_max': get_attr(subreaches, 'x_max', 0),
            'y_min': get_attr(subreaches, 'y_min', 0),
            'y_max': get_attr(subreaches, 'y_max', 0),
            'cl_id_min': cl_id_min,
            'cl_id_max': cl_id_max,
            'reach_length': get_attr(subreaches, 'len', 0),
            'n_nodes': get_attr(subreaches, ['rch_n_nodes', 'n_nodes'], 0),
            'wse': get_attr(subreaches, 'wse', 0),
            'wse_var': get_attr(subreaches, 'wse_var', 0),
            'width': get_attr(subreaches, ['wth', 'width'], 0),
            'width_var': get_attr(subreaches, ['wth_var', 'width_var'], 0),
            'slope': get_attr(subreaches, 'slope', 0),
            'max_width': get_attr(subreaches, ['max_wth', 'max_width'], 0),
            'facc': get_attr(subreaches, 'facc', 0),
            'dist_out': get_attr(subreaches, 'dist_out', 0),
            'lakeflag': get_attr(subreaches, 'lakeflag', 0),
            'obstr_type': get_attr(subreaches, ['grod', 'obstr_type'], 0),
            'grod_id': get_attr(subreaches, ['grod_fid', 'grod_id'], 0),
            'hfalls_id': get_attr(subreaches, ['hfalls_fid', 'hfalls_id'], 0),
            'n_chan_max': get_attr(subreaches, ['nchan_max', 'n_chan_max'], 0),
            'n_chan_mod': get_attr(subreaches, ['nchan_mod', 'n_chan_mod'], 0),
            'n_rch_up': get_attr(subreaches, 'n_rch_up', 0),
            'n_rch_down': get_attr(subreaches, 'n_rch_down', 0),
            'swot_obs': get_attr(subreaches, ['max_obs', 'swot_obs'], 0),
            'iceflag': get_attr(subreaches, 'iceflag_scalar', 0),  # Scalar version
            'low_slope_flag': get_attr(subreaches, ['low_slope', 'low_slope_flag'], 0),
            'river_name': get_attr(subreaches, 'river_name', ''),
            'edit_flag': get_attr(subreaches, 'edit_flag', ''),
            'trib_flag': get_attr(subreaches, 'trib_flag', 0),
            'path_freq': get_attr(subreaches, 'path_freq', 0),
            'path_order': get_attr(subreaches, 'path_order', 0),
            'path_segs': get_attr(subreaches, 'path_segs', 0),
            'stream_order': get_attr(subreaches, ['strm_order', 'stream_order'], 0),
            'main_side': get_attr(subreaches, 'main_side', 0),
            'end_reach': get_attr(subreaches, ['end_rch', 'end_reach'], 0),
            'network': get_attr(subreaches, 'network', 0),
            'add_flag': get_attr(subreaches, 'add_flag', 0),
            'version': [self.version] * n,
        }

        reaches_df = pd.DataFrame(reaches_data)

        # Reorder columns to match INSERT statement
        cols = [
            'reach_id', 'region', 'x', 'y', 'x_min', 'x_max', 'y_min', 'y_max',
            'cl_id_min', 'cl_id_max', 'reach_length', 'n_nodes',
            'wse', 'wse_var', 'width', 'width_var', 'slope', 'max_width',
            'facc', 'dist_out', 'lakeflag', 'obstr_type', 'grod_id', 'hfalls_id',
            'n_chan_max', 'n_chan_mod', 'n_rch_up', 'n_rch_down',
            'swot_obs', 'iceflag', 'low_slope_flag', 'river_name',
            'edit_flag', 'trib_flag', 'path_freq', 'path_order', 'path_segs',
            'stream_order', 'main_side', 'end_reach', 'network', 'add_flag', 'version'
        ]
        reaches_df = reaches_df[cols]

        # Insert main reaches using executemany for reliability
        placeholders = ', '.join(['?'] * len(cols))
        conn.executemany(f"""
            INSERT INTO reaches ({', '.join(cols)})
            VALUES ({placeholders})
        """, reaches_df.values.tolist())

        # Insert reach topology (rch_id_up and rch_id_down [4,N] arrays)
        self._insert_reach_topology(conn, subreaches)

        # Insert SWOT orbits if present
        if hasattr(subreaches, 'orbits') and subreaches.orbits is not None:
            self._insert_reach_orbits(conn, subreaches)

        # Insert ice flags if present (daily [366,N] array)
        if hasattr(subreaches, 'iceflag') and subreaches.iceflag is not None:
            if subreaches.iceflag.ndim == 2:
                self._insert_reach_ice_flags(conn, subreaches)

    def _insert_reach_topology(self, conn, subreaches) -> None:
        """Insert reach topology (upstream/downstream neighbors)."""
        import pandas as pd

        n = len(subreaches.id)
        topology_data = []

        # Process upstream neighbors
        if hasattr(subreaches, 'rch_id_up') and subreaches.rch_id_up is not None:
            for i in range(n):
                for rank in range(4):
                    if rank < subreaches.rch_id_up.shape[0]:
                        neighbor_id = subreaches.rch_id_up[rank, i]
                        if neighbor_id != 0:
                            topology_data.append({
                                'reach_id': subreaches.id[i],
                                'region': self.region,
                                'direction': 'up',
                                'neighbor_rank': rank,
                                'neighbor_reach_id': neighbor_id,
                            })

        # Process downstream neighbors
        if hasattr(subreaches, 'rch_id_down') and subreaches.rch_id_down is not None:
            for i in range(n):
                for rank in range(4):
                    if rank < subreaches.rch_id_down.shape[0]:
                        neighbor_id = subreaches.rch_id_down[rank, i]
                        if neighbor_id != 0:
                            topology_data.append({
                                'reach_id': subreaches.id[i],
                                'region': self.region,
                                'direction': 'down',
                                'neighbor_rank': rank,
                                'neighbor_reach_id': neighbor_id,
                            })

        if topology_data:
            topo_df = pd.DataFrame(topology_data)
            cols = ['reach_id', 'region', 'direction', 'neighbor_rank', 'neighbor_reach_id']
            topo_df = topo_df[cols]
            conn.executemany("""
                INSERT INTO reach_topology (reach_id, region, direction, neighbor_rank, neighbor_reach_id)
                VALUES (?, ?, ?, ?, ?)
            """, topo_df.values.tolist())

    def _insert_reach_orbits(self, conn, subreaches) -> None:
        """Insert SWOT orbit data."""
        import pandas as pd

        n = len(subreaches.id)
        orbits_data = []

        for i in range(n):
            for rank in range(min(75, subreaches.orbits.shape[0])):
                orbit_id = subreaches.orbits[rank, i]
                if orbit_id != 0:
                    orbits_data.append({
                        'reach_id': subreaches.id[i],
                        'region': self.region,
                        'orbit_rank': rank,
                        'orbit_id': orbit_id,
                    })

        if orbits_data:
            orbits_df = pd.DataFrame(orbits_data)
            cols = ['reach_id', 'region', 'orbit_rank', 'orbit_id']
            orbits_df = orbits_df[cols]
            conn.executemany("""
                INSERT INTO reach_swot_orbits (reach_id, region, orbit_rank, orbit_id)
                VALUES (?, ?, ?, ?)
            """, orbits_df.values.tolist())

    def _insert_reach_ice_flags(self, conn, subreaches) -> None:
        """Insert daily ice flag data."""
        import pandas as pd

        n = len(subreaches.id)
        ice_data = []

        for i in range(n):
            for day in range(min(366, subreaches.iceflag.shape[0])):
                flag = subreaches.iceflag[day, i]
                if flag != 0:  # Only store non-zero flags
                    ice_data.append({
                        'reach_id': subreaches.id[i],
                        'julian_day': day + 1,  # Convert 0-365 to 1-366
                        'iceflag': flag,
                    })

        if ice_data:
            ice_df = pd.DataFrame(ice_data)
            cols = ['reach_id', 'julian_day', 'iceflag']
            ice_df = ice_df[cols]
            conn.executemany("""
                INSERT INTO reach_ice_flags (reach_id, julian_day, iceflag)
                VALUES (?, ?, ?)
            """, ice_df.values.tolist())

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def __enter__(self) -> 'SWORD':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
