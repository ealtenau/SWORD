"""
Canonical column ordering for SWORD tables.

Single source of truth. All code that needs column order imports from here:
- schema.py (DDL generation)
- sword_class.py (insert reordering)
- export.py (export format output)
- migrations.py (NetCDF to DuckDB migration)

Grouping: variable-group ordering — related measurements adjacent.
"""

from __future__ import annotations

from typing import TypeVar

import pandas as pd

_DF = TypeVar("_DF", bound=pd.DataFrame)

# ── Reaches ──────────────────────────────────────────────────────────────────

REACHES_COLUMN_ORDER: tuple[str, ...] = (
    # Identity
    "reach_id",
    "region",
    # Geometry
    "x",
    "y",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "geom",
    # Structure
    "reach_length",
    "n_nodes",
    "cl_id_min",
    "cl_id_max",
    "dn_node_id",
    "up_node_id",
    # WSE (prior + observed)
    "wse",
    "wse_var",
    "wse_obs_mean",
    "wse_obs_median",
    "wse_obs_std",
    "wse_obs_range",
    # Width (prior + observed)
    "width",
    "width_var",
    "max_width",
    "width_obs_mean",
    "width_obs_median",
    "width_obs_std",
    "width_obs_range",
    # Slope (prior + observed)
    "slope",
    "slope_obs_mean",
    "slope_obs_median",
    "slope_obs_std",
    "slope_obs_range",
    "slope_obs_adj",
    "slope_obs_slopeF",
    "slope_obs_reliable",
    "slope_obs_quality",
    # Hydrology & Distance
    "facc",
    "dist_out",
    "hydro_dist_out",
    "hydro_dist_hw",
    "dist_out_short",
    # Topology
    "n_rch_up",
    "n_rch_down",
    "rch_id_up_main",
    "rch_id_dn_main",
    "end_reach",
    "trib_flag",
    # Network & Path Analysis
    "network",
    "stream_order",
    "path_freq",
    "path_order",
    "path_segs",
    "main_side",
    "main_path_id",
    "is_mainstem_edge",
    "best_headwater",
    "best_outlet",
    "pathlen_hw",
    "pathlen_out",
    # Classification & Flags
    "lakeflag",
    "n_chan_max",
    "n_chan_mod",
    "obstr_type",
    "grod_id",
    "hfalls_id",
    "swot_obs",
    "iceflag",
    "low_slope_flag",
    "edit_flag",
    "add_flag",
    # Names
    "river_name",
    "river_name_en",
    "river_name_local",
    # Observation count
    "n_obs",
    # Metadata
    "version",
)

# ── Nodes ────────────────────────────────────────────────────────────────────

NODES_COLUMN_ORDER: tuple[str, ...] = (
    # Identity
    "node_id",
    "region",
    # Geometry
    "x",
    "y",
    "geom",
    # Structure
    "cl_id_min",
    "cl_id_max",
    "reach_id",
    "node_order",
    "node_length",
    # WSE (prior + observed)
    "wse",
    "wse_var",
    "wse_obs_mean",
    "wse_obs_median",
    "wse_obs_std",
    "wse_obs_range",
    # Width (prior + observed)
    "width",
    "width_var",
    "max_width",
    "width_obs_mean",
    "width_obs_median",
    "width_obs_std",
    "width_obs_range",
    # Hydrology & Distance
    "facc",
    "dist_out",
    # SWOT Search Parameters
    "wth_coef",
    "ext_dist_coef",
    # Morphology
    "meander_length",
    "sinuosity",
    # Network & Path Analysis
    "network",
    "stream_order",
    "path_freq",
    "path_order",
    "path_segs",
    "main_side",
    "end_reach",
    "best_headwater",
    "best_outlet",
    "pathlen_hw",
    "pathlen_out",
    # Classification & Flags
    "lakeflag",
    "n_chan_max",
    "n_chan_mod",
    "obstr_type",
    "grod_id",
    "hfalls_id",
    "trib_flag",
    "manual_add",
    "edit_flag",
    "add_flag",
    # Names
    "river_name",
    # Observation count
    "n_obs",
    # Metadata
    "version",
)

# ── Centerlines ──────────────────────────────────────────────────────────────

CENTERLINES_COLUMN_ORDER: tuple[str, ...] = (
    "cl_id",
    "region",
    "x",
    "y",
    "geom",
    "reach_id",
    "node_id",
    "version",
)

# ── Table name → order mapping ───────────────────────────────────────────────

_TABLE_ORDERS: dict[str, tuple[str, ...]] = {
    "reaches": REACHES_COLUMN_ORDER,
    "nodes": NODES_COLUMN_ORDER,
    "centerlines": CENTERLINES_COLUMN_ORDER,
}


def get_column_order(table_name: str) -> tuple[str, ...]:
    """Return the canonical column order for a table.

    Raises ValueError for unknown tables.
    """
    try:
        return _TABLE_ORDERS[table_name]
    except KeyError:
        raise ValueError(
            f"Unknown table '{table_name}'. Valid tables: {sorted(_TABLE_ORDERS)}"
        ) from None


def reorder_columns(df: _DF, table_name: str) -> _DF:
    """Reorder DataFrame columns to match canonical order.

    - Columns in the canonical list come first, in canonical order.
    - Columns present in df but NOT in canonical list are appended at end.
    - Columns in canonical list but NOT in df are silently skipped.
    """
    canonical = get_column_order(table_name)
    df_cols = set(df.columns)
    ordered = [c for c in canonical if c in df_cols]
    extra = [c for c in df.columns if c not in set(canonical)]
    return df[ordered + extra]
