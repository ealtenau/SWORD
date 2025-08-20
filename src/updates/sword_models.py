# -*- coding: utf-8 -*-
"""
SWORD Data Models (sword_models.py)
=====================================
This module defines the core data structures for the SWORD dataset using 
Python's dataclasses. These classes provide a clear, type-hinted schema 
for the different components of the river database (reaches, nodes, 
and centerlines), mirroring the logical structure of the original
NetCDF files for downstream compatibility.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class CenterlineData:
    """
    Data container for a SWORD centerline, mirroring original NetCDF structure.
    """
    cl_id: np.ndarray = field(default_factory=lambda: np.array([]))
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    node_id: np.ndarray = field(default_factory=lambda: np.array([]))
    rch_id: np.ndarray = field(default_factory=lambda: np.array([]))
    continent: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class NodeData:
    """
    Data container for SWORD nodes, mirroring original NetCDF structure.
    """
    node_id: np.ndarray = field(default_factory=lambda: np.array([]))
    reach_id: np.ndarray = field(default_factory=lambda: np.array([]))
    cl_id_min: np.ndarray = field(default_factory=lambda: np.array([]))
    cl_id_max: np.ndarray = field(default_factory=lambda: np.array([]))
    wse: np.ndarray = field(default_factory=lambda: np.array([]))
    wse_var: np.ndarray = field(default_factory=lambda: np.array([]))
    width: np.ndarray = field(default_factory=lambda: np.array([]))
    width_var: np.ndarray = field(default_factory=lambda: np.array([]))
    len: np.ndarray = field(default_factory=lambda: np.array([])) # From original, now node_length
    len_var: np.ndarray = field(default_factory=lambda: np.array([])) # Not in DB, for compatibility
    q: np.ndarray = field(default_factory=lambda: np.array([])) # Not in DB, for compatibility
    q_var: np.ndarray = field(default_factory=lambda: np.array([])) # Not in DB, for compatibility
    facc: np.ndarray = field(default_factory=lambda: np.array([]))
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    continent: np.ndarray = field(default_factory=lambda: np.array([]))
    # Optional field for compatibility
    iceflag: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))


@dataclass
class ReachData:
    """
    Data container for a single SWORD reach, mirroring original NetCDF structure.
    """
    reach_id: int
    # Core attributes from DB
    grod_id: int
    hfalls_id: int
    n_nodes: int
    width: float
    wse: float
    wse_var: float
    reach_length: float
    facc: float
    dist_out: float
    x: float
    y: float
    river_name: str
    continent: str
    
    # Re-assembled list attributes
    rch_id_up: List[int] = field(default_factory=list)
    rch_id_down: List[int] = field(default_factory=list)
    
    # Attributes that existed in some original docs/code but not in current DB
    # We include them for compatibility and provide default values.
    type: int = -9999
    slope: float = np.nan
    slope_var: float = np.nan
    sinuosity: float = np.nan
    lake_id: int = -9999
    cl_ids: List[int] = field(default_factory=list)
    
    # A selection of other attributes from the DB for completeness
    x_min: float = np.nan
    x_max: float = np.nan
    y_min: float = np.nan
    y_max: float = np.nan
    stream_order: int = -9999
    
    # We are omitting the ~60 discharge model fields for clarity for now,
    # as they represent a complex nested structure. They can be added later if needed. 