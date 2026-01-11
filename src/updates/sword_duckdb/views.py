# -*- coding: utf-8 -*-
"""
SWORD DuckDB View Classes
=========================

Wrapper classes that provide numpy-array-style attribute access to DataFrame columns.
These classes enable backward compatibility with the original SWORD class interface.

The original SWORD class uses attribute access like:
    sword.reaches.wse  -> numpy array
    sword.nodes.facc   -> numpy array

These view classes wrap DataFrames and provide the same interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class CenterlinesView:
    """
    Wrapper for centerlines data providing numpy-array-style attribute access.

    Maps original SWORD attribute names to DuckDB column names.
    Handles [4,N] arrays for reach_id and node_id by reconstruction.

    Original attributes:
        cl_id: [N] centerline IDs
        x: [N] longitude coordinates
        y: [N] latitude coordinates
        reach_id: [4,N] reach IDs (primary + 3 neighbors)
        node_id: [4,N] node IDs (primary + 3 neighbors)
    """

    def __init__(self, df: 'pd.DataFrame', reach_id_array: np.ndarray, node_id_array: np.ndarray):
        """
        Initialize CenterlinesView.

        Parameters
        ----------
        df : pd.DataFrame
            Centerlines data from DuckDB query.
        reach_id_array : np.ndarray
            Reconstructed [4,N] reach_id array.
        node_id_array : np.ndarray
            Reconstructed [4,N] node_id array.
        """
        self._df = df
        self._reach_id = reach_id_array
        self._node_id = node_id_array

    @property
    def cl_id(self) -> np.ndarray:
        """Centerline IDs [N]."""
        return self._df['cl_id'].values

    @property
    def x(self) -> np.ndarray:
        """Longitude coordinates [N]."""
        return self._df['x'].values

    @property
    def y(self) -> np.ndarray:
        """Latitude coordinates [N]."""
        return self._df['y'].values

    @property
    def reach_id(self) -> np.ndarray:
        """Reach IDs [4,N] - primary + 3 neighbors."""
        return self._reach_id

    @reach_id.setter
    def reach_id(self, value: np.ndarray):
        """Set reach_id array."""
        self._reach_id = value

    @property
    def node_id(self) -> np.ndarray:
        """Node IDs [4,N] - primary + 3 neighbors."""
        return self._node_id

    @node_id.setter
    def node_id(self, value: np.ndarray):
        """Set node_id array."""
        self._node_id = value

    def __len__(self) -> int:
        return len(self._df)


class NodesView:
    """
    Wrapper for nodes data providing numpy-array-style attribute access.

    Maps original SWORD attribute names to DuckDB column names.

    Original -> DuckDB mapping:
        id -> node_id
        len -> node_length
        wth -> width
        wth_var -> width_var
        max_wth -> max_width
        grod -> obstr_type
        grod_fid -> grod_id
        hfalls_fid -> hfalls_id
        nchan_max -> n_chan_max
        nchan_mod -> n_chan_mod
        meand_len -> meander_length
        strm_order -> stream_order
        end_rch -> end_reach
        cl_id[2,N] -> (cl_id_min, cl_id_max)
    """

    def __init__(self, df: 'pd.DataFrame'):
        """
        Initialize NodesView.

        Parameters
        ----------
        df : pd.DataFrame
            Nodes data from DuckDB query.
        """
        self._df = df
        # Build cl_id [2,N] array from min/max columns
        self._cl_id = np.vstack([
            df['cl_id_min'].values,
            df['cl_id_max'].values
        ])

    # Direct mappings (same name)
    @property
    def x(self) -> np.ndarray:
        return self._df['x'].values

    @property
    def y(self) -> np.ndarray:
        return self._df['y'].values

    @property
    def wse(self) -> np.ndarray:
        return self._df['wse'].values

    @property
    def wse_var(self) -> np.ndarray:
        return self._df['wse_var'].values

    @property
    def facc(self) -> np.ndarray:
        return self._df['facc'].values

    @property
    def dist_out(self) -> np.ndarray:
        return self._df['dist_out'].values

    @property
    def lakeflag(self) -> np.ndarray:
        return self._df['lakeflag'].values

    @property
    def wth_coef(self) -> np.ndarray:
        return self._df['wth_coef'].values

    @property
    def ext_dist_coef(self) -> np.ndarray:
        return self._df['ext_dist_coef'].values

    @property
    def sinuosity(self) -> np.ndarray:
        return self._df['sinuosity'].values

    @property
    def river_name(self) -> np.ndarray:
        return self._df['river_name'].values

    @property
    def manual_add(self) -> np.ndarray:
        return self._df['manual_add'].values

    @property
    def edit_flag(self) -> np.ndarray:
        return self._df['edit_flag'].values

    @property
    def trib_flag(self) -> np.ndarray:
        return self._df['trib_flag'].values

    @property
    def path_freq(self) -> np.ndarray:
        return self._df['path_freq'].values

    @property
    def path_order(self) -> np.ndarray:
        return self._df['path_order'].values

    @property
    def path_segs(self) -> np.ndarray:
        return self._df['path_segs'].values

    @property
    def main_side(self) -> np.ndarray:
        return self._df['main_side'].values

    @property
    def network(self) -> np.ndarray:
        return self._df['network'].values

    @property
    def add_flag(self) -> np.ndarray:
        return self._df['add_flag'].values

    # Renamed mappings
    @property
    def id(self) -> np.ndarray:
        """Node IDs [N] (DuckDB: node_id)."""
        return self._df['node_id'].values

    @property
    def len(self) -> np.ndarray:
        """Node length [N] (DuckDB: node_length)."""
        return self._df['node_length'].values

    @property
    def wth(self) -> np.ndarray:
        """Width [N] (DuckDB: width)."""
        return self._df['width'].values

    @property
    def wth_var(self) -> np.ndarray:
        """Width variance [N] (DuckDB: width_var)."""
        return self._df['width_var'].values

    @property
    def max_wth(self) -> np.ndarray:
        """Max width [N] (DuckDB: max_width)."""
        return self._df['max_width'].values

    @property
    def grod(self) -> np.ndarray:
        """Obstruction type [N] (DuckDB: obstr_type)."""
        return self._df['obstr_type'].values

    @property
    def grod_fid(self) -> np.ndarray:
        """GROD feature ID [N] (DuckDB: grod_id)."""
        return self._df['grod_id'].values

    @property
    def hfalls_fid(self) -> np.ndarray:
        """HydroFALLS feature ID [N] (DuckDB: hfalls_id)."""
        return self._df['hfalls_id'].values

    @property
    def nchan_max(self) -> np.ndarray:
        """Max channels [N] (DuckDB: n_chan_max)."""
        return self._df['n_chan_max'].values

    @property
    def nchan_mod(self) -> np.ndarray:
        """Modal channels [N] (DuckDB: n_chan_mod)."""
        return self._df['n_chan_mod'].values

    @property
    def meand_len(self) -> np.ndarray:
        """Meander length [N] (DuckDB: meander_length)."""
        return self._df['meander_length'].values

    @property
    def strm_order(self) -> np.ndarray:
        """Stream order [N] (DuckDB: stream_order)."""
        return self._df['stream_order'].values

    @property
    def end_rch(self) -> np.ndarray:
        """End reach flag [N] (DuckDB: end_reach)."""
        return self._df['end_reach'].values

    @property
    def reach_id(self) -> np.ndarray:
        """Parent reach ID [N]."""
        return self._df['reach_id'].values

    # [2,N] array
    @property
    def cl_id(self) -> np.ndarray:
        """Centerline ID range [2,N] - (min, max)."""
        return self._cl_id

    @cl_id.setter
    def cl_id(self, value: np.ndarray):
        """Set cl_id array."""
        self._cl_id = value

    def __len__(self) -> int:
        return len(self._df)


class ReachesView:
    """
    Wrapper for reaches data providing numpy-array-style attribute access.

    Maps original SWORD attribute names to DuckDB column names.
    Handles multi-dimensional arrays by reconstruction.

    Original -> DuckDB mapping:
        id -> reach_id
        len -> reach_length
        rch_n_nodes -> n_nodes
        grod -> obstr_type
        grod_fid -> grod_id
        hfalls_fid -> hfalls_id
        nchan_max -> n_chan_max
        nchan_mod -> n_chan_mod
        max_obs -> swot_obs
        low_slope -> low_slope_flag
        strm_order -> stream_order
        end_rch -> end_reach
        wth -> width
        wth_var -> width_var
        max_wth -> max_width
        cl_id[2,N] -> (cl_id_min, cl_id_max)
        rch_id_up[4,N] -> reach_topology table
        rch_id_down[4,N] -> reach_topology table
        orbits[75,N] -> reach_swot_orbits table
        iceflag[366,N] -> reach_ice_flags table
    """

    def __init__(
        self,
        df: 'pd.DataFrame',
        rch_id_up: np.ndarray,
        rch_id_down: np.ndarray,
        orbits: np.ndarray = None,
        iceflag: np.ndarray = None
    ):
        """
        Initialize ReachesView.

        Parameters
        ----------
        df : pd.DataFrame
            Reaches data from DuckDB query.
        rch_id_up : np.ndarray
            Reconstructed [4,N] upstream neighbor array.
        rch_id_down : np.ndarray
            Reconstructed [4,N] downstream neighbor array.
        orbits : np.ndarray, optional
            Reconstructed [75,N] SWOT orbits array.
        iceflag : np.ndarray, optional
            Reconstructed [366,N] ice flag array.
        """
        self._df = df
        self._rch_id_up = rch_id_up
        self._rch_id_down = rch_id_down
        self._orbits = orbits
        self._iceflag = iceflag

        # Build cl_id [2,N] array from min/max columns
        self._cl_id = np.vstack([
            df['cl_id_min'].values,
            df['cl_id_max'].values
        ])

    # Direct mappings (same name)
    @property
    def x(self) -> np.ndarray:
        return self._df['x'].values

    @property
    def y(self) -> np.ndarray:
        return self._df['y'].values

    @property
    def x_min(self) -> np.ndarray:
        return self._df['x_min'].values

    @property
    def x_max(self) -> np.ndarray:
        return self._df['x_max'].values

    @property
    def y_min(self) -> np.ndarray:
        return self._df['y_min'].values

    @property
    def y_max(self) -> np.ndarray:
        return self._df['y_max'].values

    @property
    def wse(self) -> np.ndarray:
        return self._df['wse'].values

    @property
    def wse_var(self) -> np.ndarray:
        return self._df['wse_var'].values

    @property
    def slope(self) -> np.ndarray:
        return self._df['slope'].values

    @property
    def facc(self) -> np.ndarray:
        return self._df['facc'].values

    @property
    def dist_out(self) -> np.ndarray:
        return self._df['dist_out'].values

    @property
    def lakeflag(self) -> np.ndarray:
        return self._df['lakeflag'].values

    @property
    def n_rch_up(self) -> np.ndarray:
        return self._df['n_rch_up'].values

    @property
    def n_rch_down(self) -> np.ndarray:
        return self._df['n_rch_down'].values

    @property
    def river_name(self) -> np.ndarray:
        return self._df['river_name'].values

    @property
    def edit_flag(self) -> np.ndarray:
        return self._df['edit_flag'].values

    @property
    def trib_flag(self) -> np.ndarray:
        return self._df['trib_flag'].values

    @property
    def path_freq(self) -> np.ndarray:
        return self._df['path_freq'].values

    @property
    def path_order(self) -> np.ndarray:
        return self._df['path_order'].values

    @property
    def path_segs(self) -> np.ndarray:
        return self._df['path_segs'].values

    @property
    def main_side(self) -> np.ndarray:
        return self._df['main_side'].values

    @property
    def network(self) -> np.ndarray:
        return self._df['network'].values

    @property
    def add_flag(self) -> np.ndarray:
        return self._df['add_flag'].values

    # Renamed mappings
    @property
    def id(self) -> np.ndarray:
        """Reach IDs [N] (DuckDB: reach_id)."""
        return self._df['reach_id'].values

    @property
    def len(self) -> np.ndarray:
        """Reach length [N] (DuckDB: reach_length)."""
        return self._df['reach_length'].values

    @property
    def rch_n_nodes(self) -> np.ndarray:
        """Number of nodes per reach [N] (DuckDB: n_nodes)."""
        return self._df['n_nodes'].values

    @property
    def grod(self) -> np.ndarray:
        """Obstruction type [N] (DuckDB: obstr_type)."""
        return self._df['obstr_type'].values

    @property
    def grod_fid(self) -> np.ndarray:
        """GROD feature ID [N] (DuckDB: grod_id)."""
        return self._df['grod_id'].values

    @property
    def hfalls_fid(self) -> np.ndarray:
        """HydroFALLS feature ID [N] (DuckDB: hfalls_id)."""
        return self._df['hfalls_id'].values

    @property
    def nchan_max(self) -> np.ndarray:
        """Max channels [N] (DuckDB: n_chan_max)."""
        return self._df['n_chan_max'].values

    @property
    def nchan_mod(self) -> np.ndarray:
        """Modal channels [N] (DuckDB: n_chan_mod)."""
        return self._df['n_chan_mod'].values

    @property
    def max_obs(self) -> np.ndarray:
        """Max SWOT observations [N] (DuckDB: swot_obs)."""
        return self._df['swot_obs'].values

    @property
    def low_slope(self) -> np.ndarray:
        """Low slope flag [N] (DuckDB: low_slope_flag)."""
        return self._df['low_slope_flag'].values

    @property
    def strm_order(self) -> np.ndarray:
        """Stream order [N] (DuckDB: stream_order)."""
        return self._df['stream_order'].values

    @property
    def end_rch(self) -> np.ndarray:
        """End reach flag [N] (DuckDB: end_reach)."""
        return self._df['end_reach'].values

    @property
    def wth(self) -> np.ndarray:
        """Width [N] (DuckDB: width)."""
        return self._df['width'].values

    @property
    def wth_var(self) -> np.ndarray:
        """Width variance [N] (DuckDB: width_var)."""
        return self._df['width_var'].values

    @property
    def max_wth(self) -> np.ndarray:
        """Max width [N] (DuckDB: max_width)."""
        return self._df['max_width'].values

    # Multi-dimensional arrays
    @property
    def cl_id(self) -> np.ndarray:
        """Centerline ID range [2,N] - (min, max)."""
        return self._cl_id

    @cl_id.setter
    def cl_id(self, value: np.ndarray):
        """Set cl_id array."""
        self._cl_id = value

    @property
    def rch_id_up(self) -> np.ndarray:
        """Upstream reach IDs [4,N]."""
        return self._rch_id_up

    @rch_id_up.setter
    def rch_id_up(self, value: np.ndarray):
        """Set upstream reach IDs array."""
        self._rch_id_up = value

    @property
    def rch_id_down(self) -> np.ndarray:
        """Downstream reach IDs [4,N]."""
        return self._rch_id_down

    @rch_id_down.setter
    def rch_id_down(self, value: np.ndarray):
        """Set downstream reach IDs array."""
        self._rch_id_down = value

    @property
    def orbits(self) -> np.ndarray:
        """SWOT orbits [75,N]."""
        return self._orbits

    @orbits.setter
    def orbits(self, value: np.ndarray):
        """Set orbits array."""
        self._orbits = value

    @property
    def iceflag(self) -> np.ndarray:
        """Daily ice flags [366,N]."""
        return self._iceflag

    @iceflag.setter
    def iceflag(self, value: np.ndarray):
        """Set iceflag array."""
        self._iceflag = value

    def __len__(self) -> int:
        return len(self._df)
