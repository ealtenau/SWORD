# SWORD DuckDB Migration Status

## Project Overview
Migrating SWORD (SWOT River Database) from NetCDF4 to DuckDB for SQL query capabilities, better performance, and simpler tooling.

## Current State: Phase 3 COMPLETE

### Completed Phases

**Phase 1: Schema & Infrastructure** ✅
- Created `src/updates/sword_duckdb/` package
- Files: `__init__.py`, `schema.py`, `sword_db.py`, `migrations.py`, `sword_class.py`, `views.py`
- Schema v1.1.0 with composite primary keys

**Phase 2: Data Migration** ✅
- All 6 regions migrated and validated
- Database: `data/duckdb/sword_v17b.duckdb` (9.9 GB)

| Region | Centerlines | Nodes | Reaches |
|--------|-------------|-------|---------|
| NA | 10,349,756 | 1,705,705 | 38,696 |
| SA | 12,861,259 | 2,135,054 | 42,159 |
| EU | 7,048,975 | 1,174,236 | 31,103 |
| AF | 6,725,494 | 1,116,775 | 21,441 |
| OC | 4,125,498 | 686,595 | 15,089 |
| AS | 25,820,450 | 4,294,089 | 100,185 |
| **Total** | **66,931,432** | **11,112,454** | **248,673** |

**Phase 3: SWORD Class & Script Migration** ✅
- Full DuckDB-backed SWORD class with all methods
- View wrapper classes (CenterlinesView, NodesView, ReachesView)
- WritableArray class for in-place array modifications with DB persistence
- **40+ scripts migrated** to use `from src.updates.sword_duckdb import SWORD`

### Key Technical Decisions

1. **Composite Primary Keys** - cl_id and node_id are NOT globally unique, only unique within region:
   - centerlines: (cl_id, region)
   - nodes: (node_id, region)
   - reaches: (reach_id, region)

2. **Package Name** - `sword_duckdb` not `duckdb` to avoid conflict with duckdb library

3. **WritableArray** - Intercepts `__setitem__` and syncs changes to DuckDB immediately

4. **View Classes** - Provide numpy-array-style attribute access to database columns

5. **gc.disable() Workaround** - Required during DuckDB write operations to avoid segfaults

### Git Status
- Branch: `gearon_dev3`
- Latest commits:
  - `bf72ed5` - Add WritableArray for in-place array modifications with DB persistence
  - `0846265` - Migrate all scripts to use DuckDB SWORD backend

---

## NEXT GOALS

### Priority 1: Testing & Validation (HIGH) - IN PROGRESS

Unit test suite created with **40 passing tests**.

- [x] Create `tests/` directory structure
- [x] Unit tests for SWORD class methods (load, views, paths)
- [x] Unit tests for WritableArray persistence
- [x] Unit tests for SWORD methods (delete_data, delete_rchs, delete_nodes, save_nc)
- [ ] Integration tests for full workflow (load → modify → save)
- [ ] Backward compatibility tests comparing outputs with original NetCDF implementation

### Priority 2: Reactive Update System (HIGH) - COMPLETE

Created `reactive.py` with dependency graph and recalculation engine.

- [x] **Dependency graph** - Maps which attributes depend on others
- [x] **Topological sort** - Ensures recalculation in correct order
- [x] **Change tracking** - DirtySet for marking changed entities
- [x] **Implement recalc functions** - All core recalculation functions implemented:
  - `_recalc_reach_lengths()` - From centerline geodesic distances
  - `_recalc_reach_bounds()` - Centroid and bounding box
  - `_recalc_reach_dist_out()` - Graph traversal from outlets upstream
  - `_recalc_reach_end_rch()` - From topology counts
  - `_recalc_node_lengths()` - From centerline geometry
  - `_recalc_node_xy()` - Centroid of centerlines
  - `_recalc_node_dist_out()` - From reach dist_out + cumulative
  - `_recalc_node_end_rch()` - Propagate from reach
  - `_recalc_node_main_side()` - Propagate from reach
- [ ] **Hook into WritableArray** - Auto-mark dirty on `__setitem__`

**Attribute Dependency Chains:**
```
centerline.geometry → reach.len, reach.bounds, node.len, node.xy
                    → reach.dist_out → node.dist_out
                    → centerline.node_id_neighbors (KDTree)

reach.topology → reach.n_rch_up/down → reach.end_rch → node.end_rch
              → reach.dist_out → node.dist_out
              → reach.main_side → node.main_side
```

### Priority 3: QGIS/PostgreSQL Integration (HIGH) - IN PROGRESS

Goal: Edit SWORD in QGIS with automatic recalculation and versioning.

- [x] **PostgreSQL/PostGIS export** - `export_to_postgres()` function in `export.py`
- [x] **Sync back to DuckDB** - `sync_from_postgres()` function in `export.py`
- [x] **Database triggers** - `install_triggers()` in `triggers.py`
- [x] **Change detection** - `get_pending_changes()`, `get_changed_entities()` in `triggers.py`
- [ ] **Git-like versioning** - Track edit history with before/after states

**Complete QGIS Workflow:**
```python
from sword_duckdb import (
    SWORD, export_to_postgres, install_triggers,
    get_pending_changes, sync_from_postgres, mark_changes_synced
)

sword = SWORD('data/duckdb/sword_v17b.duckdb', 'NA', 'v17b')
pg_conn = "postgresql://user:pass@localhost/sword_edit"

# Step 1: Export to PostgreSQL for QGIS editing
export_to_postgres(sword, pg_conn, tables=["reaches", "nodes"], prefix="na_")

# Step 2: Install change tracking triggers
install_triggers(pg_conn, prefix="na_")

# Step 3: User edits data in QGIS...

# Step 4: Check what changed
changes = get_pending_changes(pg_conn, prefix="na_")
print(f"Changed: {changes}")  # {'reaches': 5, 'nodes': 23}

# Step 5: Sync changes back to DuckDB
sync_from_postgres(sword, pg_conn, table="reaches", prefix="na_")
sync_from_postgres(sword, pg_conn, table="nodes", prefix="na_")

# Step 6: Mark changes as synced
mark_changes_synced(pg_conn, prefix="na_", all_changes=True)
```

### Priority 4: Performance Optimization (MEDIUM)

- [ ] **Vectorize WritableArray updates** - Batch UPDATE with CASE/WHEN
- [ ] **Build spatial indexes** on geometry columns
- [ ] **Connection pooling** for thread safety
- [ ] **Memory optimization** for large regions (AS has 25M+ centerlines)

### Priority 5: Additional Modules (MEDIUM)

- [ ] **Create `queries.py`** - Common SQL query patterns:
  - Find upstream/downstream reaches
  - Spatial bounding box queries
  - Aggregate statistics
  - Cross-region queries

- [x] **Create `export.py`** - Export functions:
  - `export_to_geoparquet()` - GeoParquet export
  - `export_to_postgres()` - PostgreSQL/PostGIS export
  - `export_to_geopackage()` - GeoPackage export
  - `sync_from_postgres()` - Sync QGIS edits back to DuckDB

### Priority 6: Documentation (LOW)

- [ ] API documentation for SWORD class methods
- [ ] Migration guide for updating scripts
- [ ] Example Jupyter notebooks demonstrating common workflows
- [ ] Architecture decision records (ADRs)

### Priority 7: Technical Debt (LOW)

- [ ] **2D array WritableArray** - `centerlines.reach_id[4,N]` and `centerlines.node_id[4,N]` are read-only
- [ ] **Error handling improvements** - Better error messages in WritableArray
- [ ] **Type hints** - Add comprehensive type hints throughout
- [ ] **Logging** - Add structured logging for debugging

---

## Architecture Overview

### Current Package Structure
```
src/updates/sword_duckdb/
├── __init__.py         # Exports: SWORD, SWORDReactive, etc.
├── schema.py           # Table definitions (v1.1.0)
├── sword_db.py         # Connection manager (SWORDDatabase class)
├── migrations.py       # NetCDF → DuckDB migration
├── sword_class.py      # Main SWORD class (backward compatible)
├── views.py            # View classes + WritableArray
├── reactive.py         # Dependency graph + recalculation engine
├── export.py           # Export to PostgreSQL, GeoParquet, GeoPackage
├── triggers.py         # PostgreSQL triggers for change tracking
└── queries.py          # TODO: Common SQL patterns
```

### SWORD Class Methods
```python
class SWORD:
    # Data access (read)
    .centerlines      # CenterlinesView - numpy-style access to centerline data
    .nodes            # NodesView - numpy-style access to node data
    .reaches          # ReachesView - numpy-style access to reach data
    .paths            # dict - backward-compatible file paths

    # Data modification (write)
    .append_data()    # Add new centerlines, nodes, reaches
    .append_nodes()   # Add nodes to existing reaches
    .delete_rchs()    # Delete reaches (and associated nodes/centerlines)
    .delete_nodes()   # Delete specific nodes
    .break_reaches()  # Split reach at node position

    # Export
    .save_vectors()   # Export to GeoPackage/Shapefile
```

### WritableArray Pattern
```python
# Array modifications automatically persist to database
sword.reaches.dist_out[idx] = 1234.5  # Single value
sword.reaches.dist_out[mask] = new_values  # Multiple values

# Under the hood:
# 1. Updates local numpy array
# 2. Executes UPDATE SQL statement
# 3. Changes are immediately committed
```

---

## Testing the Current State

```python
from src.updates.sword_duckdb import SWORD
import os

main_dir = os.getcwd()
db_path = os.path.join(main_dir, 'data/duckdb/sword_v17b.duckdb')
sword = SWORD(db_path, 'NA', 'v17b')

# Test attribute access
print('Reaches:', len(sword.reaches.id))
print('Nodes:', len(sword.nodes.id))
print('Centerlines:', len(sword.centerlines.cl_id))

# Test paths property
print('GeoPackage dir:', sword.paths['gpkg_dir'])

# Test WritableArray persistence
idx = 0
original = float(sword.reaches.dist_out[idx])
sword.reaches.dist_out[idx] = 99999.0
assert sword.reaches.dist_out[idx] == 99999.0

# Reload and verify persistence
sword2 = SWORD(db_path, 'NA', 'v17b')
assert sword2.reaches.dist_out[idx] == 99999.0

# Restore original
sword2.reaches.dist_out[idx] = original
print('All tests passed!')
```

---

## Migrated Scripts (40+)

All scripts now use:
```python
from src.updates.sword_duckdb import SWORD
db_path = os.path.join(main_dir, f'data/duckdb/sword_{version}.duckdb')
sword = SWORD(db_path, region, version)
```

**Directories migrated:**
- `src/updates/formatting_scripts/` (22 files)
- `src/updates/delta_updates/` (2 files)
- `src/updates/channel_additions/coastal/` (1 file)
- `src/updates/channel_additions/interior/` (4 files)
- `src/updates/channel_additions/tools/` (1 file)
- `src/updates/centerline_shifting/` (2 files)
- `src/updates/quality_checking/` (3 files)
- `src/updates/network_analysis/` (4 files)
- `src/updates/mhv_sword/` (1 file)
- `src/updates/sword_vectors.py` (1 file)

---

## Known Issues

1. **gc.disable() Required** - DuckDB segfaults during certain write operations if garbage collection runs. WritableArray and append methods disable GC during writes.

2. **2D Arrays Read-Only** - `centerlines.reach_id` and `centerlines.node_id` return standard numpy arrays, not WritableArray. These 2D arrays require special handling if modification is needed.

3. **No Spatial Indexing Yet** - Geometry columns exist but lack spatial indexes, making spatial queries slow on large datasets.
