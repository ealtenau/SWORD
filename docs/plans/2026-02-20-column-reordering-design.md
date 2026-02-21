# Column Reordering Design

Date: 2026-02-20
Status: Approved

## Goal

Single source of truth for column ordering across all SWORD tables, consumed by schema, migrations, exports, and runtime code. Variable-group ordering: related measurements adjacent (e.g. wse + wse_obs_*).

## New Module: `src/sword_duckdb/column_order.py`

Central definitions. All other code imports from here.

### REACHES_COLUMN_ORDER (70 columns)

| Group | Columns |
|-------|---------|
| Identity | reach_id, region |
| Geometry | x, y, x_min, x_max, y_min, y_max, geom |
| Structure | reach_length, n_nodes, cl_id_min, cl_id_max, dn_node_id, up_node_id |
| WSE | wse, wse_var, wse_obs_mean, wse_obs_median, wse_obs_std, wse_obs_range |
| Width | width, width_var, max_width, width_obs_mean, width_obs_median, width_obs_std, width_obs_range |
| Slope | slope, slope_obs_mean, slope_obs_median, slope_obs_std, slope_obs_range, slope_obs_adj, slope_obs_slopeF, slope_obs_reliable, slope_obs_quality |
| Hydrology | facc, dist_out, hydro_dist_out, hydro_dist_hw, dist_out_short |
| Topology | n_rch_up, n_rch_down, rch_id_up_main, rch_id_dn_main, end_reach, trib_flag |
| Network | network, stream_order, path_freq, path_order, path_segs, main_side, main_path_id, is_mainstem_edge, best_headwater, best_outlet, pathlen_hw, pathlen_out |
| Classification | lakeflag, n_chan_max, n_chan_mod, obstr_type, grod_id, hfalls_id, swot_obs, iceflag, low_slope_flag, edit_flag, add_flag |
| Names | river_name, river_name_en, river_name_local |
| Obs count | n_obs |
| Metadata | version |

### NODES_COLUMN_ORDER (50 columns)

| Group | Columns |
|-------|---------|
| Identity | node_id, region |
| Geometry | x, y, geom |
| Structure | cl_id_min, cl_id_max, reach_id, node_order, node_length |
| WSE | wse, wse_var, wse_obs_mean, wse_obs_median, wse_obs_std, wse_obs_range |
| Width | width, width_var, max_width, width_obs_mean, width_obs_median, width_obs_std, width_obs_range |
| Hydrology | facc, dist_out |
| SWOT params | wth_coef, ext_dist_coef |
| Morphology | meander_length, sinuosity |
| Network | network, stream_order, path_freq, path_order, path_segs, main_side, end_reach, best_headwater, best_outlet, pathlen_hw, pathlen_out |
| Classification | lakeflag, n_chan_max, n_chan_mod, obstr_type, grod_id, hfalls_id, trib_flag, manual_add, edit_flag, add_flag |
| Names | river_name |
| Obs count | n_obs |
| Metadata | version |

### CENTERLINES_COLUMN_ORDER (8 columns)

cl_id, region, x, y, geom, reach_id, node_id, version

## Helpers

- `reorder_columns(df, table_name)` — reorder DataFrame columns. Unknown columns appended at end. Missing columns silently skipped.
- `get_column_order(table_name)` — return canonical tuple.
- `validate_column_order(conn, table_name)` — compare actual DB order to canonical. Return list of mismatches.

## Integration Points

1. **schema.py** — DDL generation uses column_order tuples to emit CREATE TABLE in correct order
2. **sword_class.py** — `_reorder_nodes_columns()` and `_reorder_reaches_columns()` import lists instead of hardcoding
3. **export.py** — all export functions (PG, Parquet, GPKG, NetCDF) use `reorder_columns()` before writing
4. **migrations.py** — dictionary construction follows canonical order

## Non-Goals

- Not changing column names (that's a separate task)
- Not dropping unused columns
- Not changing DuckDB physical storage order (only logical/export order)
