# Reconstruction Engine Integration Design

## Goal

Make `reconstruction.py` capable of fully reconstructing all SWORD attributes from source data. Fix known bugs, implement all external-data stubs, validate against legacy algorithms.

## Scope

- Attributes only (geometry/topology tables assumed to exist in DuckDB)
- All external source datasets available on disk

## Bugs to Fix

### 1. `end_reach` reconstructor (line ~2508-2519)

Current logic only checks `n_up > 1` for junction classification. Misses bifurcations where `n_down > 1`.

**Fix:** Change `classify_reach` to:
```python
if n_up == 0:
    return 1  # headwater
elif n_down == 0:
    return 2  # outlet
elif n_up > 1 or n_down > 1:
    return 3  # junction (confluence OR bifurcation)
else:
    return 0  # main
```

Matches `reactive.py` logic. ~30k reaches affected (v17b "phantom junctions" already documented).

### 2. `path_freq` for 4,952 broken reaches

Connected non-ghost reaches with path_freq=0 or -9999. 91% are 1:1 links (propagation fix), 9% are junctions (need traversal).

**Fix:** After main path_freq computation, detect any connected reach with invalid path_freq. For 1:1 links: propagate from neighbor. For junctions: sum upstream path_freqs.

### 3. `main_side` reconstructor

Currently a stub preserving existing values. Needs path-traversal logic.

**Algorithm** (from legacy + validation spec):
- main_side=0 (main channel, 95%): highest path_freq branch at each junction
- main_side=1 (side channel, 3%): lower path_freq branch
- main_side=2 (secondary outlet, 2%): reach with n_down=0 that isn't the primary outlet of its network

## External Data Stubs to Implement

### 4. `trib_flag` (reach + node)

**Source:** MHV (MERIT Hydro Vector) point files in `data/inputs/MHV_SWORD/`

**Algorithm** (from `Add_Trib_Flag.py`):
1. Load MHV points where `sword_flag == 0` and `strmorder >= 3`
2. Build KDTree from MHV points
3. Query each SWORD node against KDTree
4. If nearest MHV point <= 0.003 degrees (~333m): node trib_flag=1
5. Reach trib_flag=1 if any node in reach has trib_flag=1

### 5. `grod_id` + `obstr_type` (reach + node)

**Source:** GROD database (Global River Obstruction Database)

**Algorithm:** Spatial join — match GROD points to nearest SWORD reach/node within threshold. Set `obstr_type` from GROD classification, `grod_id` from GROD identifier.

### 6. `hfalls_id` (reach + node)

**Source:** HydroFALLS database

**Algorithm:** Spatial join — match waterfall points to nearest SWORD reach/node within threshold.

### 7. `river_name` (reach + node)

**Source:** River names shapefile

**Algorithm:** Spatial join — match named river features to SWORD reaches. Already partially done via OSM enrichment; this provides the original GRWL-sourced names.

### 8. `iceflag` (reach)

**Source:** External ice flag CSV (366-day binary array per reach)

**Algorithm:** Read CSV with reach_id + 366 columns. Join to reaches table. Store as 366-element array or individual columns per existing schema.

## Architecture

### Source Data Configuration

Add `source_data_paths` dict to `ReconstructionEngine.__init__`:

```python
source_data_paths: Dict[str, Path] = {
    "mhv": Path("data/inputs/MHV_SWORD/"),
    "grod": Path("data/inputs/GROD/"),
    "hydrofalls": Path("data/inputs/HydroFALLS/"),
    "river_names": Path("data/inputs/river_names/"),
    "ice_flags": Path("data/inputs/ice_flags/"),
}
```

Each reconstructor validates its source path exists before proceeding.

### Method Pattern

All follow existing signature:
```python
def _reconstruct_{table}_{attr}(
    self, reach_ids=None, force=False, dry_run=False
) -> Dict[str, Any]
```

### Testing

Each new/fixed reconstructor gets a test against `sword_test_minimal.duckdb`. External data tests use small fixtures (10-20 rows) in `tests/sword_duckdb/fixtures/`.

## Team Structure (3 agents)

| Agent | Work Items | Region of `reconstruction.py` |
|-------|-----------|-------------------------------|
| bug-fixer | #1 end_reach, #2 path_freq, #3 main_side | Existing methods (~lines 2375-2525, 3735-3757) |
| topology-algo | Validate/improve dist_out, path_freq, stream_order, path_segs, path_order against legacy | Topology methods (~lines 2200-2600) |
| external-data | #4 trib_flag, #5 grod_id/obstr_type, #6 hfalls_id, #7 river_name, #8 iceflag | Stub section (~lines 3939-4170) |

Agents work in isolated worktrees. Results merged sequentially.

## Unresolved Questions

None — scope and data availability confirmed.
