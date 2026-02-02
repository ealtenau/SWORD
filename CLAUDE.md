# SWORD Project Instructions

## Project Overview

**SWORD (SWOT River Database)** - Global hydrological river network database re-engineered from NetCDF to DuckDB.

- **66.9M centerlines**, **11.1M nodes**, **248.7K reaches**
- **6 regions**: NA, SA, EU, AF, AS, OC
- **Database**: ~10 GB (v17b), ~11 GB (v17c)
- **Website**: https://www.swordexplorer.com/

## Architecture

```
SWORDWorkflow (ALWAYS use this - main entry point)
    ├── SWORD class (DuckDB-backed data access)
    ├── SWORDReactive (auto-recalculation of derived attrs)
    ├── ProvenanceLogger (audit trail + rollback)
    ├── ReconstructionEngine (rebuild from source data)
    └── ImageryPipeline (satellite water mask integration)
```

## ⚠️ CRITICAL: Database Handling Rules

| Database | Purpose | Editable? |
|----------|---------|-----------|
| `sword_v17b.duckdb` | **READ-ONLY reference baseline** | **NEVER modify** |
| `sword_v17c.duckdb` | Working database for all edits | Yes |

**v17b is the pristine reference for comparison.** If v17b gets corrupted, rebuild from NetCDF:
```bash
python rebuild_v17b.py  # Rebuilds from data/netcdf/*.nc
```

**All topology fixes, facc corrections, and experimental changes go to v17c only.**

## Key Directories

```
src/updates/
  sword_duckdb/           # Core module - workflow, schema, validation
    imagery/              # Satellite water detection (NDWI, ML4Floods, OPERA)
  sword_v17c_pipeline/    # v17b→v17c topology enhancement (phi algorithm)
  delta_updates/          # Delta region processing
  mhv_sword/              # MERIT Hydro Vector integration

data/
  duckdb/
    sword_v17b.duckdb     # ⚠️ READ-ONLY REFERENCE - never modify! (9.9 GB)
    sword_v17c.duckdb     # Working database for edits (11 GB)
  netcdf/                 # Legacy source files (rebuild v17b from these)

tests/sword_duckdb/
  fixtures/sword_test_minimal.duckdb  # Test DB (8.26 MB, 100 reaches)
```

## Usage

**ALWAYS use SWORDWorkflow:**

```python
from updates.sword_duckdb import SWORDWorkflow

workflow = SWORDWorkflow(user_id="jake")
# IMPORTANT: Use v17c for modifications, v17b is READ-ONLY reference
sword = workflow.load('data/duckdb/sword_v17c.duckdb', 'NA')

# Modify with provenance
workflow.modify_reach(reach_id, wse=45.5, reason="field correction")

# Recalculate topology
workflow.calculate_dist_out()
workflow.recalculate_stream_order()
workflow.recalculate_path_segs()

# Query history
workflow.get_history(entity_type='reach', entity_id=123)

# Export
workflow.export(formats=['geopackage'], output_dir='outputs/')
workflow.close()
```

## Database Schema

**Core Tables:**
- **centerlines** - PK: (cl_id, region) - river path points
- **nodes** - PK: (node_id, region) - measurement points at ~200m intervals
- **reaches** - PK: reach_id - river segments between junctions

**Topology:**
- **reach_topology** - upstream/downstream neighbors (direction: 'up'/'down', neighbor_rank: 0-3)
- **reach_swot_orbits** - SWOT satellite coverage
- **reach_ice_flags** - daily ice presence (366 days)

**Provenance:**
- **sword_operations** - audit trail
- **sword_value_snapshots** - old/new values for rollback

## Key Attributes

| Attribute | Description |
|-----------|-------------|
| dist_out | Distance to outlet (m) - decreases downstream |
| facc | Flow accumulation (km²) |
| stream_order | Log scale of path_freq |
| path_freq | Traversal count - increases toward outlets |
| path_segs | Unique ID for (path_order, path_freq) combo |
| lakeflag | 0=river, 1=lake, 2=canal, 3=tidal |
| trib_flag | 0=no tributary, 1=has tributary |
| n_rch_up/down | Count of upstream/downstream neighbors |

## v17c Pipeline

**Location:** `src/updates/sword_v17c_pipeline/`

**Steps:**
1. Load v17b topology from DuckDB
2. Build reach-level directed graph
3. Compute v17c attributes (hydro_dist_out, best_headwater, is_mainstem)
4. Save sections + validation to DuckDB
5. (Optional) Apply SWOT-derived slopes

**New columns:** hydro_dist_out, hydro_dist_hw, best_headwater, best_outlet, pathlen_hw, pathlen_out, is_mainstem_edge, swot_slope, swot_slope_se, swot_slope_confidence

**New tables:** v17c_sections, v17c_section_slope_validation

**Input:** sword_v17c.duckdb (reads topology, writes attributes)

**Run:**
```bash
# All regions
python -m src.updates.sword_v17c_pipeline.v17c_pipeline --db data/duckdb/sword_v17c.duckdb --all

# Single region
python -m src.updates.sword_v17c_pipeline.v17c_pipeline --db data/duckdb/sword_v17c.duckdb --region NA

# Skip SWOT (faster)
python -m src.updates.sword_v17c_pipeline.v17c_pipeline --db data/duckdb/sword_v17c.duckdb --all --skip-swot
```

**Note:** MILP optimization files archived in `_archived/` - v17c uses original v17b topology.

## Known Issues

| Issue | Workaround |
|-------|------------|
| **RTREE index segfault** | Drop index → UPDATE → Recreate index |
| **Region case sensitivity** | DuckDB=uppercase (NA), pipeline=lowercase (na) |
| **Lake sandwiches** | 3,167 reaches (1.55%) - river between lakes |

## Reactive Recalculation

Dependency graph auto-recalculates derived attributes:
- Geometry changes → reach_length, sinuosity
- Topology changes → dist_out, stream_order, path_freq, path_segs
- Node changes → reach aggregates (wse, width)

## Validation Checks

`src/updates/sword_duckdb/validation.py`:
- dist_out decreasing downstream
- path_freq increasing toward outlets
- lake sandwich detection
- topology consistency

## Lint Framework

**Location:** `src/updates/sword_duckdb/lint/`

Comprehensive linting framework with 35 checks across 5 categories (T=Topology, A=Attributes, G=Geometry, C=Classification, V=v17c).

**CLI Usage:**
```bash
# Run all checks
python -m src.updates.sword_duckdb.lint.cli --db sword_v17c.duckdb

# Filter by region
python -m src.updates.sword_duckdb.lint.cli --db sword_v17c.duckdb --region NA

# Specific checks or category
python -m src.updates.sword_duckdb.lint.cli --db sword_v17c.duckdb --checks T001 T002
python -m src.updates.sword_duckdb.lint.cli --db sword_v17c.duckdb --checks T  # all topology

# Output formats
python -m src.updates.sword_duckdb.lint.cli --db sword_v17c.duckdb --format json -o report.json
python -m src.updates.sword_duckdb.lint.cli --db sword_v17c.duckdb --format markdown -o report.md

# CI mode (exit codes)
python -m src.updates.sword_duckdb.lint.cli --db sword_v17c.duckdb --fail-on-error   # exit 2 on errors
python -m src.updates.sword_duckdb.lint.cli --db sword_v17c.duckdb --fail-on-warning  # exit 1 on warnings

# List all checks
python -m src.updates.sword_duckdb.lint.cli --list-checks
```

**Python API:**
```python
from sword_duckdb.lint import LintRunner, Severity

with LintRunner("sword_v17c.duckdb") as runner:
    results = runner.run()  # all checks
    results = runner.run(checks=["T"])  # topology only
    results = runner.run(region="NA", severity=Severity.ERROR)
```

**Check IDs (35 total):**

| ID | Name | Severity | Description |
|----|------|----------|-------------|
| T001 | dist_out_monotonicity | ERROR | dist_out decreases downstream |
| T002 | path_freq_monotonicity | WARNING | path_freq increases to outlets |
| T003 | facc_monotonicity | WARNING | facc increases downstream |
| T004 | orphan_reaches | WARNING | No neighbors |
| T005 | neighbor_count_consistency | ERROR | n_rch_up/down matches topology |
| T006 | connected_components | INFO | Network connectivity |
| T007 | topology_reciprocity | WARNING | A→B implies B→A |
| T008 | dist_out_negative | ERROR | No negative dist_out |
| T009 | dist_out_zero_at_nonoutlet | ERROR | dist_out=0 only at outlets |
| T010 | headwater_path_freq | ERROR | Headwaters have path_freq >= 1 |
| T011 | path_freq_zero | WARNING | path_freq=0 only for disconnected |
| A002 | slope_reasonableness | WARNING | No negative, <100 m/km |
| A003 | width_trend | INFO | Width increases downstream |
| A004 | attribute_completeness | INFO | Required attrs present |
| A005 | trib_flag_distribution | INFO | Unmapped tributary stats |
| A006 | attribute_outliers | INFO | Extreme values |
| A007 | headwater_facc | WARNING | Headwaters have low facc |
| A008 | headwater_width | WARNING | Headwaters have narrow width |
| A009 | outlet_facc | INFO | Outlets have high facc |
| A010 | end_reach_consistency | WARNING | end_reach matches topology |
| G001 | reach_length_bounds | INFO | 100m-50km, excl end_reach |
| G002 | node_length_consistency | WARNING | Node sum ≈ reach length |
| G003 | zero_length_reaches | INFO | Zero/negative length |
| C001 | lake_sandwich | WARNING | River between lakes |
| C002 | lakeflag_distribution | INFO | Lakeflag values |
| C003 | type_distribution | INFO | Type field values |
| C004 | lakeflag_type_consistency | INFO | Lakeflag/type cross-tab (needs investigation) |
| V001 | hydro_dist_out_monotonicity | ERROR | hydro_dist_out decreases downstream |
| V002 | hydro_dist_vs_pathlen | INFO | hydro_dist_out vs pathlen_out diff |
| V003 | pathlen_consistency | WARNING | pathlen_hw + pathlen_out consistency |
| V004 | mainstem_continuity | WARNING | is_mainstem_edge forms continuous path |
| V005 | hydro_dist_out_coverage | ERROR | All connected reaches have hydro_dist_out |
| V006 | mainstem_coverage | INFO | is_mainstem_edge coverage stats |
| V007 | best_headwater_validity | WARNING | best_headwater is actual headwater |
| V008 | best_outlet_validity | WARNING | best_outlet is actual outlet |

**Validation Specs:** Deep documentation for key variables in `docs/validation_specs/`:
- `dist_out` - algorithm, failure modes, proposed checks
- `facc` - MERIT Hydro source, D8 routing limitations
- `path_freq` - traversal algorithm, edge cases
- `wse` - elevation data, slope derivation
- `v17c mainstem` - hydro_dist_out, is_mainstem_edge, best_headwater/outlet

## Testing

```bash
cd /Users/jakegearon/projects/SWORD
python -m pytest tests/sword_duckdb/ -v
```

Test DB: `tests/sword_duckdb/fixtures/sword_test_minimal.duckdb` (100 reaches, 500 nodes)

## Important Files

| File | Purpose |
|------|---------|
| `src/updates/sword_duckdb/workflow.py` | Main entry point (3,511 lines) |
| `src/updates/sword_duckdb/sword_class.py` | SWORD data class (4,623 lines) |
| `src/updates/sword_duckdb/schema.py` | Table definitions |
| `src/updates/sword_duckdb/reactive.py` | Dependency graph |
| `src/updates/sword_duckdb/reconstruction.py` | 35+ attribute reconstructors |
| `src/updates/sword_duckdb/lint/` | Lint framework (35 checks) |
| `run_v17c_topology.py` | Topology recalculation script |
| `rebuild_v17b.py` | Rebuild v17b from NetCDF (if corrupted) |
| `topology_reviewer.py` | Streamlit GUI for facc/topology fixes |

## Git

- **Main branch:** main
- **Dev branch:** gearon_dev3
- **v17c branch:** v17c-updates (for v17c work)
- **v18 branch:** v18-planning (for v18 planning)
- Never force push to main
- **NEVER merge to main** - PRs go to gearon_dev3
- Commit with provenance context

## GitHub Issue Tracking

**All v17c/v18 work is tracked via GitHub Issues.** See: https://github.com/ealtenau/SWORD/issues

### Milestones

| Milestone | Description | Deadline |
|-----------|-------------|----------|
| v17c-verify | Verify pipeline outputs before use | FIRST |
| v17c-topology | Keep dist_out, add hydro_dist_out | 1-2 months |
| v17c-lake-type | Fix lake/type classification | 1-2 months |
| v17c-pipeline | Import 20+ new attrs | 1-2 months |
| v17c-swot | WSE/width/slope stats | 1-2 months |
| v17c-schema | New columns only | 1-2 months |
| v17c-export | DuckDB, GPKG, NetCDF, Parquet | 1-2 months |
| v17c-docs | Release notes, data dict | 1-2 months |
| v18-planning | Scope, ID mapping | 6+ months |
| v18-sources | MERIT Hydro, GROD | 6+ months |
| v18-imagery | Sentinel-2 centerlines | 6+ months |
| v18-reach-mod | Merge/add reaches | 6+ months |
| v18-export | v18 exports | 6+ months |

### Labels

- **Priority:** P0-critical, P1-high, P2-medium, P3-low
- **Type:** type:bug, type:feature, type:docs, type:verify
- **Region:** region:NA/SA/EU/AF/AS/OC, region:all
- **Component:** comp:topology, comp:pipeline, comp:swot, comp:export, comp:lake-type, comp:schema, comp:verify

### Key Issues (v17c)

| # | Title | Milestone |
|---|-------|-----------|
| 4 | Inventory pipeline output files | v17c-verify |
| 14 | Fix facc using MERIT Hydro | v17c-topology |
| 17 | Fix island-in-lake misclassification | v17c-lake-type |
| 31 | Run aggregate_swot_observations | v17c-swot |
| 34 | Export DuckDB (v17c final) | v17c-export |

### Workflow

1. Pick issue from milestone (priority order: v17c-verify → topology → lake-type → pipeline → swot → export → docs)
2. Create branch from v17c-updates: `git checkout -b issue-N-short-desc`
3. Work on issue, reference it in commits: `git commit -m "Fix #N: description"`
4. PR to v17c-updates (NOT main)

## Source Datasets

- **GRWL** - Global River Widths from Landsat
- **MERIT Hydro** - Elevation, flow accumulation
- **HydroBASINS** - Drainage areas
- **GRanD/GROD** - Dams and obstructions
- **SWOT** - Satellite water surface elevation

## Imagery Pipeline

**Location:** `src/updates/sword_duckdb/imagery/`

**Water Detection Ensemble (6 methods):**
- NDWI, MNDWI, AWEI_nsh, AWEI_sh (spectral indices)
- ML4Floods, DeepWaterMap (ML models)
- Voting threshold: ≥4/6 methods agree
- Post-processing: morphological closing, blob removal (200px), relative threshold

**Key Classes:**
- `SentinelSTACClient` - Sentinel-2 imagery search
- `COGReader` - Cloud Optimized GeoTIFF reads
- `WaterEnsemble` - Multi-method water detection
- `RiverTracer` - Patch-based water mask + RivGraph centerline

## Centerline Update Approach

**Goal:** Update SWORD geometries using satellite-derived water masks

**Algorithm (skeleton + SWORD-guided pathfinding):**
1. Get water mask from ensemble
2. Skeletonize → true water center
3. Find start/end on skeleton nearest SWORD start/end
4. Pathfind with cost = `1 + dist_to_sword * 0.1`
5. At junctions, cost naturally picks SWORD's branch
6. Result: true center following SWORD's path

**Key insight:** SWORD defines PATH (which channel), skeleton defines POSITION (center)

**Test results:**
| River | Mean Drift | Notes |
|-------|------------|-------|
| Rhine | 61.5m | Clean single channel |
| Missouri | 85.6m | Correct branch selection |

**Limitations:**
- Narrow rivers (<50m) fail with 4/6 vote threshold
- Braided/anastomosing rivers need manual review
- Wide lake-like sections have noisy skeletons

**Test script:** `test_constrained_centerline.py`
