# SWORD Database Development Status

## Project Overview

Re-engineering the SWOT River Database (SWORD) from NetCDF to a modern DuckDB backend with:
- Full provenance tracking (who, what, when, why)
- Reactive recalculation system
- Reconstruction capability from source datasets
- QGIS/PostgreSQL integration for visual editing

**Website**: https://www.swordexplorer.com/

---

## Current Architecture

```
                     User Code / Scripts
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │           SWORDWorkflow               │
        │  (SINGLE ENTRY POINT)                 │
        │                                       │
        │  .load() .modify() .commit()          │
        │  .transaction() .rollback()           │
        │  .reconstruct() .get_history()        │
        │  .export_for_qgis() .sync_from_qgis() │
        └───────────────┬───────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
    ┌─────────┐   ┌──────────┐   ┌───────────────┐
    │Provenance│   │ SWORD    │   │Reconstruction │
    │ Logger  │   │ Class    │   │    Engine     │
    │         │   │(internal)│   │               │
    └─────────┘   └──────────┘   └───────────────┘
         │              │              │
         └──────────────┼──────────────┘
                        │
                        ▼
                    DuckDB
        ┌───────────────────────────────────────┐
        │ reaches │ nodes │ centerlines         │
        │ sword_operations (provenance)         │
        │ sword_value_snapshots (rollback)      │
        │ sword_source_lineage (reconstruction) │
        └───────────────────────────────────────┘
```

---

## Data State Machine

```
                         ┌────────────────┐
                         │   NETCDF       │
                         │   (Legacy)     │
                         └───────┬────────┘
                                 │ migrate_region()
                                 ▼
                         ┌────────────────┐
                         │   MIGRATED     │
                         │   (in DuckDB)  │
                         └───────┬────────┘
                                 │ SWORDWorkflow.load()
                                 ▼
        ┌────────────────────────────────────────────────┐
        │                    LOADED_CLEAN                │
        │  - DataFrames in memory                        │
        │  - Views created                               │
        │  - No pending changes                          │
        └───────┬────────────────────────────┬───────────┘
                │                            │
   ┌────────────▼────────────┐    ┌──────────▼──────────────┐
   │ LOCAL MODIFY            │    │ EXPORT TO POSTGRES      │
   │ workflow.modify_reach() │    │ workflow.export_for_qgis│
   └────────────┬────────────┘    └──────────┬──────────────┘
                │                            │
                ▼                            ▼
        ┌────────────────┐           ┌────────────────┐
        │   MODIFIED     │           │  QGIS_EDITING  │
        │  (logged to    │           │  (triggers     │
        │   provenance)  │           │   tracking)    │
        └───────┬────────┘           └───────┬────────┘
                │                            │
                │ workflow.commit()          │ workflow.sync_from_qgis()
                ▼                            ▼
        ┌────────────────────────────────────────────────┐
        │                DIRTY / NEEDS_RECALC            │
        │  - Reactive system knows what changed          │
        │  - Dependencies need cascade update            │
        └───────────────────────┬────────────────────────┘
                                │ recalculate()
                                ▼
        ┌────────────────────────────────────────────────┐
        │                   RECALCULATED                 │
        │  - All derived attributes updated              │
        │  - Database consistent                         │
        └───────────────────────┬────────────────────────┘
                                │ workflow.export()
                                ▼
        ┌────────────────────────────────────────────────┐
        │                    EXPORTED                    │
        │  - NetCDF / GeoPackage / Shapefile created     │
        └────────────────────────────────────────────────┘
```

---

## Implementation Progress

### Phase 1: Provenance Infrastructure - COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| Provenance tables in schema.py | COMPLETE | sword_operations, sword_value_snapshots, sword_source_lineage, sword_reconstruction_recipes |
| ProvenanceLogger class | COMPLETE | Operation context manager, value change logging, rollback capability |
| Transaction support in SWORDWorkflow | COMPLETE | transaction() context manager with auto-rollback |
| Wire provenance into WritableArray | Pending | Optional - logging at workflow level works |

### Phase 2: Enhanced SWORDWorkflow - MOSTLY COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| modify_reach/node methods | COMPLETE | With provenance logging |
| bulk_modify method | COMPLETE | Efficient batch modifications |
| get_history/get_lineage | COMPLETE | Query provenance records |
| rollback method | COMPLETE | Restore previous values |
| delete_reaches with cascade | COMPLETE | Cascade bug fixed |
| split_reach (break_reach) | COMPLETE | break_reach in workflow, break_reaches in SWORD class |
| merge_reaches | COMPLETE | merge_reach/merge_reaches in workflow, merge_reaches in SWORD class |

### Phase 3: Reconstruction Engine - COMPLETE

| Task | Status | Notes |
|------|--------|-------|
| ReconstructionEngine class | COMPLETE | Full implementation with attribute mappings |
| Attribute source mappings | COMPLETE | **50+ attributes** mapped to sources (GRWL, MERIT, GROD, etc.) |
| Recipe system | COMPLETE | register_recipe, get_recipe, list_recipes |
| Reconstruction methods | COMPLETE | **35+ reconstructors** - see list below |
| Validation system | COMPLETE | Compare reconstructed vs existing values |
| Integration with workflow | COMPLETE | reconstruct(), reconstruct_from_centerlines(), validate_reconstruction() |

**Implemented Reconstructors:**
- **Reach hydrology**: dist_out, wse, wse_var, slope, facc
- **Reach geometry**: x, y, x_min, x_max, y_min, y_max, reach_length
- **Reach width**: width, width_var
- **Reach counts**: n_nodes, n_rch_up, n_rch_down
- **Reach categorical (mode)**: lakeflag, n_chan_max, n_chan_mod
- **Reach network**: stream_order, path_freq, end_reach, network
- **Node basic**: x, y, dist_out, wse, facc
- **Node inherited**: stream_order, path_freq, path_order, path_segs, main_side, end_reach, network

**Source Datasets Supported:**
- GRWL, MERIT_HYDRO, HYDROBASINS, HYDROLAKES, HYDROFALLS
- GRAND, GROD, SWOT_TRACKS, ICE_FLAGS, RIVER_NAMES
- MAX_WIDTH_RASTER, COMPUTED, MANUAL, INHERITED

**Derivation Methods:**
- DIRECT, INTERPOLATED, AGGREGATED, MEAN, MEDIAN, MODE
- MAX, MIN, VARIANCE, COUNT, SUM, LINEAR_REGRESSION
- LOG_TRANSFORM, SPATIAL_JOIN, SPATIAL_PROXIMITY
- GRAPH_TRAVERSAL, PATH_ACCUMULATION, COMPUTED, INHERITED

### Existing Components (Complete)

| Component | Status | Description |
|-----------|--------|-------------|
| DuckDB schema | COMPLETE | Schema v1.2.0 with composite keys |
| Data migration | COMPLETE | All 6 regions migrated (9.9 GB) |
| SWORD class | COMPLETE | Drop-in replacement for NetCDF version |
| View classes | COMPLETE | CenterlinesView, NodesView, ReachesView |
| WritableArray | COMPLETE | numpy-style access with DB persistence |
| Reactive system | COMPLETE | Dependency graph, recalculation engine |
| PostgreSQL export | COMPLETE | export_to_postgres, sync_from_postgres |
| Change tracking triggers | COMPLETE | install_triggers, get_pending_changes |
| SWORDWorkflow (basic) | COMPLETE | load, batch_modify, commit, export |

---

## Database Statistics

| Region | Centerlines | Nodes | Reaches |
|--------|-------------|-------|---------|
| NA | 10,349,756 | 1,705,705 | 38,696 |
| SA | 12,861,259 | 2,135,054 | 42,159 |
| EU | 7,048,975 | 1,174,236 | 31,103 |
| AF | 6,725,494 | 1,116,775 | 21,441 |
| OC | 4,125,498 | 686,595 | 15,089 |
| AS | 25,820,450 | 4,294,089 | 100,185 |
| **Total** | **66,931,432** | **11,112,454** | **248,673** |

Database: `data/duckdb/sword_v17b.duckdb` (9.9 GB)

---

## Package Structure

```
src/updates/sword_duckdb/
├── __init__.py         # Exports: SWORDWorkflow, SWORD, ProvenanceLogger, etc.
├── schema.py           # Table definitions (v1.2.0 with provenance)
├── sword_db.py         # Connection manager (SWORDDatabase class)
├── migrations.py       # NetCDF → DuckDB migration
├── sword_class.py      # Main SWORD class (internal, use SWORDWorkflow)
├── views.py            # View classes + WritableArray
├── reactive.py         # Dependency graph + recalculation engine
├── provenance.py       # ProvenanceLogger for operation tracking
├── export.py           # Export to PostgreSQL, GeoParquet, GeoPackage
├── triggers.py         # PostgreSQL triggers for QGIS change tracking
├── workflow.py         # SWORDWorkflow orchestration (MAIN ENTRY POINT)
└── reconstruction.py   # ReconstructionEngine for attribute reconstruction
```

---

## Usage Example

```python
from src.updates.sword_duckdb import SWORDWorkflow

# Initialize workflow (recommended entry point)
workflow = SWORDWorkflow(user_id="jake")

# Load database
sword = workflow.load('data/duckdb/sword_v17b.duckdb', 'NA')

# Modify with provenance tracking
with workflow.transaction("Fix elevation errors"):
    workflow.modify_reach(reach_id=123, wse=45.5, reason="Corrected from field data")

# Check history
history = workflow.get_history(entity_type='reach', entity_id=123)

# Export
workflow.export(['geopackage'], output_dir='outputs/', version='v18')

workflow.close()
```

---

## Key Technical Decisions

1. **SWORDWorkflow as single entry point** - All operations should go through SWORDWorkflow for provenance tracking

2. **Composite Primary Keys** - cl_id and node_id are only unique within a region

3. **Provenance tables** - Track all operations with full context (who, what, when, why)

4. **Value snapshots** - Store old/new values for rollback capability

5. **Source lineage** - Track which source datasets (GRWL, MERIT, etc.) contributed to each attribute

---

## Source Datasets

SWORD was constructed from:

- **GRWL** - River centerlines and width (Allen & Pavelsky, 2018)
- **MERIT Hydro** - Elevation and flow accumulation (Yamazaki et al., 2019)
- **HydroBASINS** - Basin boundaries (Lehner & Grill, 2013)
- **GRanD** - Dam locations (Lehner et al., 2011)
- **GROD** - River obstructions (Yang et al., 2021)
- **SWOT Tracks** - Satellite orbit coverage

---

## Usage Example: Reconstruction

```python
from src.updates.sword_duckdb import SWORDWorkflow

workflow = SWORDWorkflow(user_id="jake")
sword = workflow.load('data/duckdb/sword_v17b.duckdb', 'NA')

# List what can be reconstructed
attrs = workflow.list_reconstructable_attributes()
# ['reach.dist_out', 'reach.wse', 'reach.slope', 'reach.facc', 'reach.reach_length', ...]

# Get source info for an attribute
info = workflow.get_source_info('reach.wse')
# {'source': 'MERIT_HYDRO', 'method': 'median', 'description': '...'}

# Reconstruct dist_out for specific reaches
result = workflow.reconstruct('reach.dist_out', entity_ids=[123, 456])

# Validate reconstruction matches existing values
report = workflow.validate_reconstruction('reach.wse', tolerance=0.01)
if report['passed']:
    print(f"All {report['total']} values within 1% tolerance")

workflow.close()
```

---

## Git Status

- Branch: `gearon_dev3`
- Recent work:
  - ReconstructionEngine with attribute source mappings
  - Provenance infrastructure (schema + ProvenanceLogger)
  - Transaction support in SWORDWorkflow
  - Reactive system completion
  - PostgreSQL export and change tracking
  - Unit test suite (110+ tests)

---

## Legacy Code Exploration Documentation

**COMPLETE** - All legacy construction algorithms have been documented.

**Location:** `/Users/jakegearon/.claude/plans/tingly-kindling-kite.md` (~4,800 lines)

### What's Documented
- **76 legacy Python files** fully analyzed
- **45+ algorithms** with code snippets and thresholds
- **35+ reconstruction recipes** for all attributes
- **40+ critical thresholds** cataloged
- **7 edge case categories** (deltas, ghosts, dams, lakes, high-lat, multi-channel, single-node)
- **Full attribute-to-source mapping** for all centerline/node/reach attributes

### Next Implementation Priorities

| Priority | Task | Documentation Reference |
|----------|------|------------------------|
| 1 | **Implement `break_reaches`** | Phase 5 Task 5.1 |
| 1 | **Implement `delete_reaches` cascade** | Phase 5 Task 5.1 |
| 1 | **Implement `append_data`** | Phase 7 Roadmap |
| 2 | `check_topo_consistency` | Phase 6 Task 6.1 |
| 2 | `rch_node_length_check` | Phase 6 Task 6.1 |
| 3 | `create_ghost_reach` | Phase 5 Task 5.4 |
| 4 | `aggregate_1node_rchs` | Phase 5 Task 5.1 |
| 5 | `dist_out` from topology (BFS) | Phase 4 Task 4.2 |

### Key Legacy File References

| Algorithm | Legacy File | Lines |
|-----------|-------------|-------|
| Reach splitting | `break_reaches_post_topo.py` | 104 |
| 1-node aggregation | `aggregate_1node_rchs.py` | 457 |
| Tributary detection | `find_tributary_breaks.py` | 174 |
| Ghost reach creation | `create_missing_ghost_reach.py` | 717 |
| Topology validation | `check_topo_consistency.py` | 202 |
| Delta handling | `2_add_deltas_to_sword.py` | 201 |

---
