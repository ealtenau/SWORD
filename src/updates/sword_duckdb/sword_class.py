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

import gc
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

        # Create view objects with db and region for write support
        self._centerlines = CenterlinesView(
            self._centerlines_df, cl_reach_id, cl_node_id,
            db=self._db, region=self.region
        )
        self._nodes = NodesView(self._nodes_df, db=self._db, region=self.region)
        self._reaches = ReachesView(
            self._reaches_df, rch_id_up, rch_id_down, orbits, iceflag,
            db=self._db, region=self.region
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

    @property
    def paths(self) -> dict:
        """
        Backward-compatible paths dictionary for file I/O operations.

        Returns the same directory structure as the original SWORD class,
        derived from the DuckDB database path.

        Returns
        -------
        dict
            Dictionary containing all SWORD file paths and filenames.
        """
        # Derive main_dir from db_path: data/duckdb/sword.duckdb -> project root
        main_dir = str(self._db_path.parent.parent.parent)
        region = self.region
        version = self.version

        paths = {}

        # Directory paths
        paths['shp_dir'] = f"{main_dir}/data/outputs/Reaches_Nodes/{version}/shp/{region}/"
        paths['gpkg_dir'] = f"{main_dir}/data/outputs/Reaches_Nodes/{version}/gpkg/"
        paths['nc_dir'] = f"{main_dir}/data/outputs/Reaches_Nodes/{version}/netcdf/"
        paths['geom_dir'] = f"{main_dir}/data/outputs/Reaches_Nodes/{version}/reach_geometry/"
        paths['update_dir'] = f"{main_dir}/data/update_requests/{version}/{region}/"
        paths['topo_dir'] = f"{main_dir}/data/outputs/Topology/{version}/{region}/"
        paths['version_dir'] = f"{main_dir}/data/outputs/Version_Differences/{version}/"
        paths['pts_gpkg_dir'] = f"{main_dir}/data/outputs/Reaches_Nodes/{version}/gpkg_30m/{region}/"

        # Filenames
        paths['nc_fn'] = f"{region.lower()}_sword_{version}.nc"
        paths['gpkg_rch_fn'] = f"{region.lower()}_sword_reaches_{version}.gpkg"
        paths['gpkg_node_fn'] = f"{region.lower()}_sword_nodes_{version}.gpkg"
        paths['shp_rch_fn'] = f"{region.lower()}_sword_reaches_hbXX_{version}.shp"
        paths['shp_node_fn'] = f"{region.lower()}_sword_nodes_hbXX_{version}.shp"
        paths['geom_fn'] = f"{region.lower()}_sword_{version}_connectivity.nc"

        return paths

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

        # Disable GC during database operations to avoid DuckDB segfaults
        gc_was_enabled = gc.isenabled()
        gc.disable()

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
        finally:
            if gc_was_enabled:
                gc.enable()

    def delete_rchs(self, rm_rch: Union[List[int], np.ndarray]) -> None:
        """
        Delete reaches only (without cascading to centerlines/nodes).

        Parameters
        ----------
        rm_rch : list or numpy.ndarray
            List of reach IDs to delete.
        """
        import gc

        rm_rch = np.array(rm_rch)
        if len(rm_rch) == 0:
            return

        gc_was_enabled = gc.isenabled()
        gc.disable()

        conn = self._db.connect()
        try:
            conn.execute("BEGIN TRANSACTION")

            for rid in rm_rch:
                rid = int(rid)
                # Delete from normalized tables
                conn.execute(
                    "DELETE FROM reach_topology WHERE reach_id = ? AND region = ?",
                    [rid, self.region]
                )
                conn.execute(
                    "DELETE FROM reach_swot_orbits WHERE reach_id = ? AND region = ?",
                    [rid, self.region]
                )
                conn.execute(
                    "DELETE FROM reach_ice_flags WHERE reach_id = ?",
                    [rid]
                )
                # Delete the reach
                conn.execute(
                    "DELETE FROM reaches WHERE reach_id = ? AND region = ?",
                    [rid, self.region]
                )

            conn.execute("COMMIT")
            self._load_data()
            print(f"Deleted {len(rm_rch)} reaches")

        except Exception as e:
            conn.execute("ROLLBACK")
            raise RuntimeError(f"Failed to delete reaches: {e}") from e
        finally:
            if gc_was_enabled:
                gc.enable()

    def delete_nodes(self, node_ids: Union[List[int], np.ndarray]) -> None:
        """
        Delete nodes by node ID.

        Parameters
        ----------
        node_ids : list or numpy.ndarray
            List of node IDs to delete.
        """
        import gc

        node_ids = np.array(node_ids)
        if len(node_ids) == 0:
            return

        gc_was_enabled = gc.isenabled()
        gc.disable()

        conn = self._db.connect()
        try:
            conn.execute("BEGIN TRANSACTION")

            for nid in node_ids:
                conn.execute(
                    "DELETE FROM nodes WHERE node_id = ? AND region = ?",
                    [int(nid), self.region]
                )

            conn.execute("COMMIT")
            self._load_data()
            print(f"Deleted {len(node_ids)} nodes")

        except Exception as e:
            conn.execute("ROLLBACK")
            raise RuntimeError(f"Failed to delete nodes: {e}") from e
        finally:
            if gc_was_enabled:
                gc.enable()

    def append_nodes(self, subnodes) -> None:
        """
        Append nodes to the database.

        Parameters
        ----------
        subnodes : object
            Object containing node data with attributes matching NodesView.
        """
        import gc

        if len(subnodes.id) == 0:
            return

        gc_was_enabled = gc.isenabled()
        gc.disable()

        conn = self._db.connect()
        try:
            conn.execute("BEGIN TRANSACTION")
            self._insert_nodes(conn, subnodes)
            conn.execute("COMMIT")
            self._load_data()
            print(f"Appended {len(subnodes.id)} nodes")

        except Exception as e:
            conn.execute("ROLLBACK")
            raise RuntimeError(f"Failed to append nodes: {e}") from e
        finally:
            if gc_was_enabled:
                gc.enable()

    def save_vectors(self, export: str = 'All', output_dir: Optional[str] = None) -> None:
        """
        Save SWORD data to vector formats (GeoPackage and/or Shapefile).

        Parameters
        ----------
        export : str
            'All' - writes both reach and node files.
            'nodes' - writes node files only.
            'reaches' - writes reach files only.
        output_dir : str, optional
            Output directory. If not provided, uses current directory.
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Point, LineString
        except ImportError:
            raise ImportError("geopandas and shapely are required for save_vectors()")

        if output_dir is None:
            output_dir = Path('.')
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if export in ('All', 'reaches'):
            print('Creating reach geometries...')
            self._save_reach_vectors(output_dir)

        if export in ('All', 'nodes'):
            print('Creating node geometries...')
            self._save_node_vectors(output_dir)

    def _save_reach_vectors(self, output_dir: Path) -> None:
        """Save reach data to GeoPackage and Shapefile."""
        import geopandas as gpd
        from shapely.geometry import LineString

        # Build reach geometries from centerlines
        reach_geoms = {}
        for rid in self.reaches.id:
            cl_idx = np.where(self.centerlines.reach_id[0, :] == rid)[0]
            if len(cl_idx) > 1:
                # Sort by cl_id
                sorted_idx = cl_idx[np.argsort(self.centerlines.cl_id[cl_idx])]
                coords = list(zip(
                    self.centerlines.x[sorted_idx],
                    self.centerlines.y[sorted_idx]
                ))
                if len(coords) >= 2:
                    reach_geoms[rid] = LineString(coords)
                else:
                    reach_geoms[rid] = None
            else:
                reach_geoms[rid] = None

        # Build GeoDataFrame
        data = {
            'reach_id': self.reaches.id,
            'x': self.reaches.x,
            'y': self.reaches.y,
            'reach_len': self.reaches.len,
            'wse': self.reaches.wse,
            'width': self.reaches.wth,
            'slope': self.reaches.slope,
            'n_nodes': self.reaches.n_nodes,
            'facc': self.reaches.facc,
            'dist_out': self.reaches.dist_out,
            'river_name': self.reaches.river_name,
        }
        gdf = gpd.GeoDataFrame(
            data,
            geometry=[reach_geoms.get(rid) for rid in self.reaches.id],
            crs='EPSG:4326'
        )
        # Remove rows with null geometry
        gdf = gdf[gdf.geometry.notna()]

        # Save
        gpkg_path = output_dir / f'sword_{self.region}_reaches.gpkg'
        shp_path = output_dir / f'sword_{self.region}_reaches.shp'
        gdf.to_file(gpkg_path, driver='GPKG')
        gdf.to_file(shp_path, driver='ESRI Shapefile')
        print(f'Saved reaches to {gpkg_path} and {shp_path}')

    def _save_node_vectors(self, output_dir: Path) -> None:
        """Save node data to GeoPackage and Shapefile."""
        import geopandas as gpd
        from shapely.geometry import Point

        # Build GeoDataFrame
        data = {
            'node_id': self.nodes.id,
            'x': self.nodes.x,
            'y': self.nodes.y,
            'reach_id': self.nodes.reach_id,
            'node_len': self.nodes.len,
            'wse': self.nodes.wse,
            'width': self.nodes.wth,
            'facc': self.nodes.facc,
            'dist_out': self.nodes.dist_out,
            'river_name': self.nodes.river_name,
        }
        geometry = [Point(x, y) for x, y in zip(self.nodes.x, self.nodes.y)]
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:4326')

        # Save
        gpkg_path = output_dir / f'sword_{self.region}_nodes.gpkg'
        shp_path = output_dir / f'sword_{self.region}_nodes.shp'
        gdf.to_file(gpkg_path, driver='GPKG')
        gdf.to_file(shp_path, driver='ESRI Shapefile')
        print(f'Saved nodes to {gpkg_path} and {shp_path}')

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
            rch_grp.createDimension('num_cl_bounds', 2)
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

    def break_reaches(self, reach_id, break_cl_id, verbose=False) -> None:
        """
        Break and create new SWORD reaches at specified centerline locations.

        This method splits existing reaches at specified centerline points,
        creating new reach and node IDs for the resulting segments. It updates
        all related data including centerlines, nodes, reaches, and topology.

        Parameters
        ----------
        reach_id : numpy.ndarray or list
            Reach IDs of SWORD reaches to break.
        break_cl_id : numpy.ndarray or list
            Centerline IDs along the reach indicating where to break.
        verbose : bool, optional
            If True, print progress information. Default is False.

        Notes
        -----
        - New reach IDs are generated based on the basin and existing max reach number
        - New node IDs follow the SWORD ID convention
        - Topology (upstream/downstream neighbors) is automatically updated
        - Edit flag '6' is added to indicate broken reaches
        """
        import gc
        from geopy import distance

        def get_distances(lon, lat):
            """Calculate geodesic distances along coordinate arrays."""
            traces = len(lon) - 1
            distances = np.zeros(traces)
            for i in range(traces):
                start = (lat[i], lon[i])
                finish = (lat[i + 1], lon[i + 1])
                distances[i] = distance.geodesic(start, finish).m
            return np.append(0, distances)

        # Extract level6 basin info and node numbers from centerline node IDs
        cl_level6 = np.array([str(ind)[0:6] for ind in self.centerlines.node_id[0, :]])
        cl_node_num_int = np.array([int(str(ind)[10:13]) for ind in self.centerlines.node_id[0, :]])
        cl_rch_type = np.array([str(ind)[-1] for ind in self.centerlines.node_id[0, :]])

        # Format input
        reach = np.array(reach_id)
        break_id = np.array(break_cl_id)

        # Disable gc during bulk operations
        gc_was_enabled = gc.isenabled()
        gc.disable()

        conn = self._db.connect()

        try:
            # Loop through unique reaches to break
            unq_rchs = np.unique(reach)
            for r in range(len(unq_rchs)):
                if verbose:
                    print(f"Processing reach {r}: {unq_rchs[r]} ({r}/{len(unq_rchs)-1})")

                # Find centerline points for this reach and sort by cl_id
                cl_r = np.where(self.centerlines.reach_id[0, :] == unq_rchs[r])[0]
                order_ids = np.argsort(self.centerlines.cl_id[cl_r])
                old_dist = self.reaches.dist_out[np.where(self.reaches.id == unq_rchs[r])[0]]
                old_len = self.reaches.len[np.where(self.reaches.id == unq_rchs[r])[0]]
                base_val = old_dist - old_len

                # Find break points
                breaks = break_id[np.where(reach == unq_rchs[r])[0]]
                break_pts = np.array([
                    np.where(self.centerlines.cl_id[cl_r[order_ids]] == b)[0][0]
                    for b in breaks
                ])

                # Build boundary array with start, break points, and end
                bounds = np.append(0, break_pts)
                bounds = np.append(bounds, len(cl_r))
                bounds = np.sort(bounds)
                bounds = np.unique(bounds)

                # Create temporary array marking new reach divisions at centerline scale
                new_divs = np.zeros(len(cl_r))
                count = 1
                for b in range(len(bounds) - 1):
                    update_nds = cl_r[order_ids[bounds[b]:bounds[b + 1]]]
                    nds = np.unique(self.centerlines.node_id[0, update_nds])
                    fill = np.where(np.in1d(self.centerlines.node_id[0, cl_r[order_ids]], nds))[0]
                    if np.max(new_divs[fill]) == 0:
                        new_divs[fill] = count
                        count += 1
                    else:
                        z = np.where(new_divs[fill] == 0)[0]
                        new_divs[fill[z]] = count
                        count += 1

                # Process each new division
                unq_divs = np.unique(new_divs)
                if len(unq_divs) == 1:
                    continue

                for d in range(len(unq_divs)):
                    if d == 0:
                        # First division keeps original reach ID
                        update_ids = cl_r[order_ids[np.where(new_divs == unq_divs[d])]]
                        new_cl_rch_id = self.centerlines.reach_id[0, update_ids]
                        new_cl_node_ids = self.centerlines.node_id[0, update_ids]
                        new_rch_id = np.unique(self.centerlines.reach_id[0, update_ids])[0]
                    else:
                        # Create new reach ID
                        update_ids = cl_r[order_ids[np.where(new_divs == unq_divs[d])]]
                        old_nodes = np.unique(self.centerlines.node_id[0, update_ids])
                        old_rch = np.unique(self.centerlines.reach_id[0, update_ids])[0]
                        l6_basin = np.where(cl_level6 == np.unique(cl_level6[update_ids]))[0]
                        cl_rch_num_int = np.array([
                            int(str(ind)[6:10])
                            for ind in self.centerlines.node_id[0, l6_basin]
                        ])
                        new_rch_num = np.max(cl_rch_num_int) + 1

                        # Format new reach ID with proper zero-padding
                        level6 = str(np.unique(cl_level6[update_ids])[0])
                        rch_type = str(np.unique(cl_rch_type[update_ids])[0])
                        new_rch_id = int(f"{level6}{new_rch_num:04d}{rch_type}")

                        new_cl_rch_id = np.repeat(new_rch_id, len(update_ids))

                        # Create new node IDs
                        new_cl_node_ids = np.zeros(len(update_ids), dtype=np.int64)
                        new_cl_node_nums = cl_node_num_int[update_ids] - np.min(cl_node_num_int[update_ids]) + 1
                        for n in range(len(new_cl_node_nums)):
                            new_cl_node_ids[n] = int(f"{str(new_rch_id)[:-1]}{new_cl_node_nums[n]:03d}{str(new_rch_id)[-1]}")

                    # Calculate new geometry for this division
                    x_coords = self.centerlines.x[update_ids]
                    y_coords = self.centerlines.y[update_ids]
                    diff = get_distances(x_coords, y_coords)
                    dist = np.cumsum(diff)

                    new_rch_len = np.max(dist)
                    new_rch_x = np.median(self.centerlines.x[update_ids])
                    new_rch_y = np.median(self.centerlines.y[update_ids])
                    new_rch_x_max = np.max(self.centerlines.x[update_ids])
                    new_rch_x_min = np.min(self.centerlines.x[update_ids])
                    new_rch_y_max = np.max(self.centerlines.y[update_ids])
                    new_rch_y_min = np.min(self.centerlines.y[update_ids])

                    # Calculate node-level attributes
                    unq_nodes = np.unique(new_cl_node_ids)
                    new_node_len = np.zeros(len(unq_nodes))
                    new_node_x = np.zeros(len(unq_nodes))
                    new_node_y = np.zeros(len(unq_nodes))
                    new_node_id = np.zeros(len(unq_nodes), dtype=np.int64)
                    new_node_cl_ids = np.zeros((2, len(unq_nodes)), dtype=np.int64)

                    for n2 in range(len(unq_nodes)):
                        pts = np.where(new_cl_node_ids == unq_nodes[n2])[0]
                        new_node_x[n2] = np.median(self.centerlines.x[update_ids[pts]])
                        new_node_y[n2] = np.median(self.centerlines.y[update_ids[pts]])
                        new_node_len[n2] = max(np.cumsum(diff[pts])) if len(pts) > 1 else 30
                        new_node_id[n2] = unq_nodes[n2]
                        new_node_cl_ids[0, n2] = np.min(self.centerlines.cl_id[update_ids[pts]])
                        new_node_cl_ids[1, n2] = np.max(self.centerlines.cl_id[update_ids[pts]])

                    # Determine edit flag
                    rch_idx = np.where(self.reaches.id == (new_rch_id if d == 0 else old_rch))[0]
                    current_edit = self.reaches.edit_flag[rch_idx][0] if len(rch_idx) > 0 else ''
                    if current_edit == 'NaN' or current_edit == '':
                        edit_val = '6'
                    elif '6' not in current_edit.split(','):
                        edit_val = f"{current_edit},6"
                    else:
                        edit_val = current_edit

                    if new_rch_id in self.reaches.id:
                        # Update existing reach in database
                        self._update_existing_reach_break(
                            conn, new_rch_id, new_node_id, new_node_len,
                            new_node_cl_ids, new_node_x, new_node_y,
                            update_ids, new_rch_x, new_rch_y,
                            new_rch_x_min, new_rch_x_max, new_rch_y_min, new_rch_y_max,
                            new_rch_len, edit_val
                        )
                    else:
                        # Insert new reach
                        self._insert_new_reach_break(
                            conn, new_rch_id, old_rch, new_cl_rch_id, new_cl_node_ids,
                            update_ids, old_nodes, new_node_id, new_node_len,
                            new_node_cl_ids, new_node_x, new_node_y,
                            new_rch_x, new_rch_y, new_rch_x_min, new_rch_x_max,
                            new_rch_y_min, new_rch_y_max, new_rch_len, edit_val
                        )

                # Update topology for all new reaches
                self._update_break_topology(conn, cl_r, order_ids, unq_rchs[r])

                # Update distance from outlet
                nrchs = np.unique(self.centerlines.reach_id[0, cl_r[order_ids]])
                self._update_dist_out(conn, nrchs, cl_r, order_ids, base_val)

            # Reload data to refresh views
            self._load_data()

            if verbose:
                print(f"Break reaches complete. Processed {len(unq_rchs)} reaches.")

        except Exception as e:
            raise RuntimeError(f"Failed to break reaches: {e}") from e
        finally:
            if gc_was_enabled:
                gc.enable()

    def _update_existing_reach_break(
        self, conn, reach_id, node_ids, node_lens,
        node_cl_ids, node_x, node_y, cl_update_ids,
        rch_x, rch_y, rch_x_min, rch_x_max, rch_y_min, rch_y_max,
        rch_len, edit_val
    ) -> None:
        """Update an existing reach after break operation."""
        # Update nodes
        for i, nid in enumerate(node_ids):
            conn.execute("""
                UPDATE nodes SET
                    node_length = ?,
                    cl_id_min = ?,
                    cl_id_max = ?,
                    x = ?,
                    y = ?,
                    edit_flag = ?
                WHERE node_id = ? AND region = ?
            """, [
                float(node_lens[i]),
                int(node_cl_ids[0, i]),
                int(node_cl_ids[1, i]),
                float(node_x[i]),
                float(node_y[i]),
                edit_val,
                int(nid),
                self.region
            ])

        # Update reach
        conn.execute("""
            UPDATE reaches SET
                cl_id_min = ?,
                cl_id_max = ?,
                x = ?,
                y = ?,
                x_min = ?,
                x_max = ?,
                y_min = ?,
                y_max = ?,
                reach_length = ?,
                n_nodes = ?,
                edit_flag = ?
            WHERE reach_id = ? AND region = ?
        """, [
            int(np.min(self.centerlines.cl_id[cl_update_ids])),
            int(np.max(self.centerlines.cl_id[cl_update_ids])),
            float(rch_x),
            float(rch_y),
            float(rch_x_min),
            float(rch_x_max),
            float(rch_y_min),
            float(rch_y_max),
            float(rch_len),
            len(node_ids),
            edit_val,
            int(reach_id),
            self.region
        ])

    def _insert_new_reach_break(
        self, conn, new_rch_id, old_rch, new_cl_rch_id, new_cl_node_ids,
        update_ids, old_nodes, new_node_id, new_node_len,
        new_node_cl_ids, new_node_x, new_node_y,
        rch_x, rch_y, rch_x_min, rch_x_max, rch_y_min, rch_y_max,
        rch_len, edit_val
    ) -> None:
        """Insert a new reach created from break operation."""
        # Update centerlines with new reach/node IDs
        for i, cl_idx in enumerate(update_ids):
            conn.execute("""
                UPDATE centerlines SET
                    reach_id = ?,
                    node_id = ?
                WHERE cl_id = ? AND region = ?
            """, [
                int(new_cl_rch_id[i]) if hasattr(new_cl_rch_id, '__len__') else int(new_cl_rch_id),
                int(new_cl_node_ids[i]),
                int(self.centerlines.cl_id[cl_idx]),
                self.region
            ])

        # Update nodes with new IDs and attributes
        old_ind = np.where(np.in1d(self.nodes.id, old_nodes))[0]
        for i, old_idx in enumerate(old_ind):
            if i < len(new_node_id):
                conn.execute("""
                    UPDATE nodes SET
                        node_id = ?,
                        node_length = ?,
                        cl_id_min = ?,
                        cl_id_max = ?,
                        x = ?,
                        y = ?,
                        reach_id = ?,
                        edit_flag = ?
                    WHERE node_id = ? AND region = ?
                """, [
                    int(new_node_id[i]),
                    float(new_node_len[i]),
                    int(new_node_cl_ids[0, i]),
                    int(new_node_cl_ids[1, i]),
                    float(new_node_x[i]),
                    float(new_node_y[i]),
                    int(new_rch_id),
                    edit_val,
                    int(self.nodes.id[old_idx]),
                    self.region
                ])

        # Get attributes from old reach to copy to new reach
        old_rch_idx = np.where(self.reaches.id == old_rch)[0]
        if len(old_rch_idx) == 0:
            return

        old_rch_idx = old_rch_idx[0]

        # Insert new reach with copied attributes
        conn.execute("""
            INSERT INTO reaches (
                reach_id, region, x, y, x_min, x_max, y_min, y_max,
                cl_id_min, cl_id_max, reach_length, n_nodes, wse, wse_var,
                width, width_var, slope, max_width, facc, dist_out, lakeflag,
                obstr_type, grod_id, hfalls_id, n_chan_max, n_chan_mod,
                swot_obs, low_slope_flag, river_name, edit_flag,
                trib_flag, path_freq, path_order, path_segs, stream_order,
                main_side, end_reach, network, add_flag, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            int(new_rch_id),
            self.region,
            float(rch_x),
            float(rch_y),
            float(rch_x_min),
            float(rch_x_max),
            float(rch_y_min),
            float(rch_y_max),
            int(np.min(self.centerlines.cl_id[update_ids])),
            int(np.max(self.centerlines.cl_id[update_ids])),
            float(rch_len),
            len(new_node_id),
            float(self.reaches.wse[old_rch_idx]),
            float(self.reaches.wse_var[old_rch_idx]),
            float(self.reaches.wth[old_rch_idx]),
            float(self.reaches.wth_var[old_rch_idx]),
            float(self.reaches.slope[old_rch_idx]),
            float(self.reaches.max_wth[old_rch_idx]),
            float(self.reaches.facc[old_rch_idx]),
            float(self.reaches.dist_out[old_rch_idx]),
            int(self.reaches.lakeflag[old_rch_idx]),
            int(self.reaches.grod[old_rch_idx]),
            int(self.reaches.grod_fid[old_rch_idx]),
            int(self.reaches.hfalls_fid[old_rch_idx]),
            int(self.reaches.nchan_max[old_rch_idx]),
            int(self.reaches.nchan_mod[old_rch_idx]),
            int(self.reaches.max_obs[old_rch_idx]),
            int(self.reaches.low_slope[old_rch_idx]),
            str(self.reaches.river_name[old_rch_idx]),
            edit_val,
            int(self.reaches.trib_flag[old_rch_idx]),
            int(self.reaches.path_freq[old_rch_idx]),
            int(self.reaches.path_order[old_rch_idx]),
            int(self.reaches.path_segs[old_rch_idx]),
            int(self.reaches.strm_order[old_rch_idx]),
            int(self.reaches.main_side[old_rch_idx]),
            int(self.reaches.end_rch[old_rch_idx]),
            int(self.reaches.network[old_rch_idx]),
            int(self.reaches.add_flag[old_rch_idx]) if hasattr(self.reaches, 'add_flag') and not (self.reaches.add_flag[old_rch_idx] is None or str(self.reaches.add_flag[old_rch_idx]) == '<NA>') else 0,
            self.version
        ])

        # Copy topology from old reach
        self._copy_reach_topology(conn, old_rch, new_rch_id)

        # Copy orbits from old reach
        self._copy_reach_orbits(conn, old_rch, new_rch_id)

        # Copy ice flags from old reach
        self._copy_reach_ice_flags(conn, old_rch, new_rch_id)

    def _copy_reach_topology(self, conn, old_rch_id, new_rch_id) -> None:
        """Copy topology entries from old reach to new reach."""
        # Get existing topology
        result = conn.execute("""
            SELECT direction, neighbor_rank, neighbor_reach_id
            FROM reach_topology
            WHERE reach_id = ? AND region = ?
        """, [int(old_rch_id), self.region]).fetchall()

        for row in result:
            conn.execute("""
                INSERT INTO reach_topology (reach_id, region, direction, neighbor_rank, neighbor_reach_id)
                VALUES (?, ?, ?, ?, ?)
            """, [int(new_rch_id), self.region, row[0], row[1], row[2]])

    def _copy_reach_orbits(self, conn, old_rch_id, new_rch_id) -> None:
        """Copy orbit entries from old reach to new reach."""
        result = conn.execute("""
            SELECT orbit_rank, orbit_id
            FROM reach_swot_orbits
            WHERE reach_id = ? AND region = ?
        """, [int(old_rch_id), self.region]).fetchall()

        for row in result:
            conn.execute("""
                INSERT INTO reach_swot_orbits (reach_id, region, orbit_rank, orbit_id)
                VALUES (?, ?, ?, ?)
            """, [int(new_rch_id), self.region, row[0], row[1]])

    def _copy_reach_ice_flags(self, conn, old_rch_id, new_rch_id) -> None:
        """Copy ice flag entries from old reach to new reach."""
        result = conn.execute("""
            SELECT julian_day, iceflag
            FROM reach_ice_flags
            WHERE reach_id = ?
        """, [int(old_rch_id)]).fetchall()

        for row in result:
            conn.execute("""
                INSERT INTO reach_ice_flags (reach_id, julian_day, iceflag)
                VALUES (?, ?, ?)
            """, [int(new_rch_id), row[0], row[1]])

    def _update_break_topology(self, conn, cl_r, order_ids, original_rch) -> None:
        """Update topology after breaking reaches."""
        nrchs = np.unique(self.centerlines.reach_id[0, cl_r[order_ids]])
        max_id = [
            max(self.centerlines.cl_id[cl_r[order_ids[
                np.where(self.centerlines.reach_id[0, cl_r[order_ids]] == n)[0]
            ]]])
            for n in nrchs
        ]
        id_sort = np.argsort(max_id)
        nrchs = nrchs[id_sort]

        for idx in range(len(nrchs)):
            pts = np.where(self.centerlines.reach_id[0, cl_r[order_ids]] == nrchs[idx])[0]
            binary = np.copy(self.centerlines.reach_id[1:, cl_r[order_ids[pts]]])
            binary[np.where(binary > 0)] = 1
            binary_sum = np.sum(binary, axis=0)
            existing_nghs = np.where(binary_sum > 0)[0]

            if len(existing_nghs) > 0:
                mn = np.where(
                    self.centerlines.cl_id[cl_r[order_ids[pts]]] ==
                    min(self.centerlines.cl_id[cl_r[order_ids[pts]]])
                )[0]
                mx = np.where(
                    self.centerlines.cl_id[cl_r[order_ids[pts]]] ==
                    max(self.centerlines.cl_id[cl_r[order_ids[pts]]])
                )[0]

                current_rch_id = int(self.centerlines.reach_id[0, cl_r[order_ids[pts[0]]]])

                if mn[0] in existing_nghs and mx[0] not in existing_nghs:
                    # Update upstream neighbor relationship
                    if pts[mx[0]] + 1 < len(order_ids):
                        neighbor_rch_id = int(self.centerlines.reach_id[0, cl_r[order_ids[pts[mx[0]] + 1]]])

                        # Update centerline neighbors
                        self._update_centerline_neighbor(
                            conn, cl_r[order_ids[pts[mx[0]]]], neighbor_rch_id
                        )

                        # Update reach topology
                        self._update_reach_topology_entry(
                            conn, current_rch_id, 'up', neighbor_rch_id
                        )

                        if idx > 0:
                            self._update_reach_topology_entry(
                                conn, neighbor_rch_id, 'down', current_rch_id
                            )

                elif mx[0] in existing_nghs and mn[0] not in existing_nghs:
                    # Update downstream neighbor relationship
                    if pts[mn[0]] > 0:
                        neighbor_rch_id = int(self.centerlines.reach_id[0, cl_r[order_ids[pts[mn[0]] - 1]]])

                        self._update_centerline_neighbor(
                            conn, cl_r[order_ids[pts[mn[0]]]], neighbor_rch_id
                        )

                        self._update_reach_topology_entry(
                            conn, current_rch_id, 'down', neighbor_rch_id
                        )

                        if idx > 0:
                            self._update_reach_topology_entry(
                                conn, neighbor_rch_id, 'up', current_rch_id
                            )

    def _update_centerline_neighbor(self, conn, cl_idx, neighbor_rch_id) -> None:
        """Update centerline neighbor reach ID."""
        cl_id = int(self.centerlines.cl_id[cl_idx])
        # Clear existing neighbors and set new one
        conn.execute("""
            DELETE FROM centerline_neighbors
            WHERE cl_id = ? AND region = ?
        """, [cl_id, self.region])

        conn.execute("""
            INSERT INTO centerline_neighbors (cl_id, region, neighbor_rank, reach_id, node_id)
            VALUES (?, ?, 1, ?, 0)
        """, [cl_id, self.region, neighbor_rch_id])

    def _update_reach_topology_entry(self, conn, reach_id, direction, neighbor_id) -> None:
        """Update or insert a reach topology entry."""
        # Clear existing entries for this direction
        conn.execute("""
            DELETE FROM reach_topology
            WHERE reach_id = ? AND region = ? AND direction = ?
        """, [reach_id, self.region, direction])

        # Insert new entry
        conn.execute("""
            INSERT INTO reach_topology (reach_id, region, direction, neighbor_rank, neighbor_reach_id)
            VALUES (?, ?, ?, 0, ?)
        """, [reach_id, self.region, direction, neighbor_id])

    def _update_dist_out(self, conn, nrchs, cl_r, order_ids, base_val) -> None:
        """Update distance from outlet for broken reaches."""
        # Get reach lengths and calculate cumulative distance
        rch_lens = []
        for rch in nrchs:
            idx = np.where(self.reaches.id == rch)[0]
            if len(idx) > 0:
                rch_lens.append(self.reaches.len[idx[0]])
            else:
                # Calculate from centerlines
                cl_pts = np.where(self.centerlines.reach_id[0, cl_r[order_ids]] == rch)[0]
                if len(cl_pts) > 0:
                    from geopy import distance
                    x = self.centerlines.x[cl_r[order_ids[cl_pts]]]
                    y = self.centerlines.y[cl_r[order_ids[cl_pts]]]
                    dists = []
                    for i in range(len(x) - 1):
                        d = distance.geodesic((y[i], x[i]), (y[i+1], x[i+1])).m
                        dists.append(d)
                    rch_lens.append(sum(dists))
                else:
                    rch_lens.append(0)

        rch_cs = np.cumsum(rch_lens) + base_val

        for i, rch in enumerate(nrchs):
            conn.execute("""
                UPDATE reaches SET dist_out = ?
                WHERE reach_id = ? AND region = ?
            """, [float(rch_cs[i]), int(rch), self.region])

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def __enter__(self) -> 'SWORD':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
