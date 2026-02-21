# Legacy Code Port: Data Quality Fixes & Spatial Utilities

**Date:** 2026-02-20
**Branch:** legacy-code
**Source:** `src/_legacy/updates/formatting_scripts/`, `src/_legacy/updates/geo_utils.py`

## Context

Audit of `src/_legacy/` identified routines with no current equivalent. This design ports the useful ones into the DuckDB-backed pipeline.

Deferred to issues: MERIT Hydro raster ingestion (#1), GRanD/GROD dam assignment (#2), SWOT orbit track assignment (#3), HydroBASINS code assignment (#12), version diff/ID translation (#10), GPKG-to-DuckDB sync (#11).

## Part A: `src/sword_duckdb/spatial_utils.py`

New module. Four pure functions, no class.

### `meters_to_degrees(meters, latitude) -> float`
- Formula: `meters / (111320 * cos(lat * pi / 180))`
- Source: `geo_utils.py:69`

### `reproject_utm(latitudes, longitudes) -> tuple[ndarray, ndarray, int, str]`
- Uses `utm` library. Picks most-common UTM zone across input points.
- Returns (easting, northing, zone_num, zone_letter).
- Source: `geo_utils.py:293`

### `get_all_upstream(con, reach_id, region=None) -> set[int]`
- BFS on `reach_topology` table where `direction='up'`.
- Pure DuckDB query, no NetworkX.
- Returns set of all upstream reach IDs (excluding start).

### `get_all_downstream(con, reach_id, region=None) -> set[int]`
- Same as above, `direction='down'`.

## Part B: `SWORDWorkflow` Data Quality Methods

Each method: provenance logging, dry_run mode, corresponding lint check.

### B1: `fill_zero_width_nodes(region, dry_run=False)`

**Source:** `formatting_scripts/fill_zero_node_wths.py`

**Algorithm:**
1. `SELECT node_id, reach_id FROM nodes WHERE width <= 0 AND region = ?`
2. Per reach: `median(width)` of non-zero nodes in same reach
3. `UPDATE nodes SET width = median_val WHERE node_id IN (...)`
4. Skip reaches where ALL nodes are zero-width (log warning)

**Lint check:** A030 `zero_node_width` (WARNING) — count of nodes with width <= 0.

### B2: `rederive_nodes(reach_ids, region, dry_run=False)`

**Source:** `formatting_scripts/fix_problem_node_order_length.py`

**Algorithm per reach:**
1. Query centerlines sorted by cl_id
2. Divide into N groups (N = current n_nodes for reach)
3. Assign new node_id per group (SWORD encoding: reach_id_prefix + 3-digit node number + type digit)
4. Per new node: x = median(cl.x), y = median(cl.y), length = geodesic sum, cl_id_min/max
5. node dist_out = reach.dist_out - reach.reach_length + cumsum(node_lengths)
6. Delete old nodes, insert new, update centerline.node_id references
7. Trigger reactive recalc for affected reaches

**Lint check:** N011 `node_ordering_problems` (WARNING) — detects non-sequential node numbering (jumps in node number along cl_id order) or node length = 0 or node length > 1000m.

### B3: `find_and_merge_single_node_reaches(region, dry_run=False)`

**Source:** `formatting_scripts/aggregate_1node_rchs.py`

**Algorithm:**
1. Find reaches: `n_nodes == 1 AND type NOT IN (4, 6)` (exclude dam, ghost)
2. For each:
   - If 1 downstream neighbor whose n_rch_up == 1: merge into downstream
   - Else if 1 upstream neighbor whose n_rch_down == 1: merge into upstream
   - Else: skip (log as unresolvable)
3. Call `workflow.merge_reach(target, source)` for each
4. Return summary DataFrame

**Lint check:** G013 `single_node_reaches` (INFO) — count of non-ghost non-dam reaches with n_nodes == 1.

### B4: `remove_duplicate_centerline_points(region, dry_run=False)`

**Source:** `formatting_scripts/remove_duplicate_pts.py`

**Algorithm:**
1. `SELECT cl_id, reach_id, x, y, ROW_NUMBER() OVER (PARTITION BY reach_id, x, y ORDER BY cl_id) as rn FROM centerlines WHERE region = ?`
2. Delete rows where rn > 1
3. For affected reaches: trigger reactive recalc (reach_length, node_xy, node_lengths, dist_out, bbox)

**Lint check:** G014 `duplicate_centerline_points` (INFO) — count of duplicate (x, y) pairs within same reach.

## New Lint Checks Summary

| ID | Name | Severity | Category |
|----|------|----------|----------|
| A030 | zero_node_width | WARNING | Attributes |
| N011 | node_ordering_problems | WARNING | Network |
| G013 | single_node_reaches | INFO | Geometry |
| G014 | duplicate_centerline_points | INFO | Geometry |

## Files to Create/Modify

| Action | File | What |
|--------|------|------|
| CREATE | `src/sword_duckdb/spatial_utils.py` | 4 functions |
| MODIFY | `src/sword_duckdb/workflow.py` | 4 new methods |
| CREATE | `src/sword_duckdb/lint/checks/a030_zero_node_width.py` | Lint check |
| CREATE | `src/sword_duckdb/lint/checks/n011_node_ordering.py` | Lint check |
| CREATE | `src/sword_duckdb/lint/checks/g013_single_node_reaches.py` | Lint check |
| CREATE | `src/sword_duckdb/lint/checks/g014_duplicate_centerlines.py` | Lint check |
| MODIFY | `src/sword_duckdb/lint/registry.py` | Register 4 new checks |
| CREATE | `tests/sword_duckdb/test_spatial_utils.py` | Tests for spatial_utils |
| CREATE | `tests/sword_duckdb/test_data_quality.py` | Tests for workflow methods |

## Deferred Work (GitHub Issues)

- MERIT Hydro raster ingestion (`auxillary_utils.mh_vals` + `attach_mh`) — blocked on volume mount
- GRanD/GROD dam spatial assignment (`auxillary_utils.add_dams`)
- SWOT orbit track assignment (`auxillary_utils.add_swot_tracks`)
