# SWORD v17c Pipeline

Simplified pipeline for generating SWORD v17c from v17b with topology attributes.

## Overview

The v17c pipeline:
1. **Uses original v17b topology** - No MILP optimization (archived in `_archived/`)
2. **Computes hydrologic attributes** - Distance metrics, headwater/outlet assignments
3. **Validates with SWOT data** - Junction-level slope validation
4. **Writes to DuckDB** - Direct output via SWORDWorkflow with provenance

## Pipeline Steps

```
v17c_pipeline.py
    ├── Load v17b topology from DuckDB
    ├── Build reach-level directed graph (NetworkX)
    ├── Identify junctions (confluences, bifurcations, HW, outlets)
    ├── Compute:
    │   ├── hydro_dist_out / hydro_dist_hw
    │   ├── best_headwater / best_outlet
    │   ├── pathlen_hw / pathlen_out
    │   └── is_mainstem_edge
    ├── Build section graph (junction-to-junction)
    ├── Validate slopes at junctions (if WSE data available)
    └── Write to sword_v17c.duckdb
```

## New Columns Added

| Column | Type | Description |
|--------|------|-------------|
| hydro_dist_out | DOUBLE | Hydrologic distance to outlet (m) |
| hydro_dist_hw | DOUBLE | Hydrologic distance from headwater (m) |
| best_headwater | BIGINT | Upstream headwater reach ID |
| best_outlet | BIGINT | Downstream outlet reach ID |
| pathlen_hw | DOUBLE | Path length to headwater (m) |
| pathlen_out | DOUBLE | Path length to outlet (m) |
| is_mainstem_edge | BOOLEAN | True if on mainstem path |
| swot_slope | DOUBLE | SWOT-derived slope (m/km) |
| swot_slope_se | DOUBLE | Standard error of SWOT slope |
| swot_slope_confidence | VARCHAR | R=reliable, U=unreliable |

## New Tables

### v17c_sections
Junction-to-junction sections with reach lists.

| Column | Type | Description |
|--------|------|-------------|
| section_id | INTEGER | Section identifier |
| region | VARCHAR(2) | Region code |
| upstream_junction | BIGINT | Upstream junction reach ID |
| downstream_junction | BIGINT | Downstream junction reach ID |
| reach_ids | VARCHAR | JSON array of reach IDs |
| distance | DOUBLE | Section length (m) |
| n_reaches | INTEGER | Number of reaches |

### v17c_section_slope_validation
SWOT slope validation results per section.

| Column | Type | Description |
|--------|------|-------------|
| section_id | INTEGER | Section identifier |
| region | VARCHAR(2) | Region code |
| slope_from_upstream | DOUBLE | Slope from upstream junction |
| slope_from_downstream | DOUBLE | Slope from downstream junction |
| direction_valid | BOOLEAN | True if slopes match expected |
| likely_cause | VARCHAR | lake_section, extreme_slope_data_error, potential_topology_error |

## Usage

### Command Line

```bash
# Process all regions
python -m src.updates.sword_v17c_pipeline.v17c_pipeline \
    --db data/duckdb/sword_v17c.duckdb --all

# Process single region
python -m src.updates.sword_v17c_pipeline.v17c_pipeline \
    --db data/duckdb/sword_v17c.duckdb --region NA

# Skip SWOT integration (faster)
python -m src.updates.sword_v17c_pipeline.v17c_pipeline \
    --db data/duckdb/sword_v17c.duckdb --all --skip-swot

# Custom SWOT path
python -m src.updates.sword_v17c_pipeline.v17c_pipeline \
    --db data/duckdb/sword_v17c.duckdb --region NA \
    --swot-path /path/to/swot/data
```

### Shell Script

```bash
cd src/updates/sword_v17c_pipeline

# All regions
./run_pipeline.sh

# Single region
REGION=NA ./run_pipeline.sh

# Skip SWOT
SKIP_SWOT=1 ./run_pipeline.sh

# Custom database
DB=/path/to/sword_v17c.duckdb ./run_pipeline.sh
```

### Python API

```python
from src.updates.sword_v17c_pipeline import run_pipeline, process_region

# Process all regions
stats = run_pipeline(
    db_path="data/duckdb/sword_v17c.duckdb",
    regions=["NA", "SA", "EU", "AF", "AS", "OC"],
    skip_swot=True,
)

# Process single region
stats = process_region(
    db_path="data/duckdb/sword_v17c.duckdb",
    region="NA",
    user_id="my_user",
    skip_swot=False,
    swot_path="/Volumes/SWORD_DATA/data/swot/RiverSP_D_parq/node",
)
```

## Verification

```bash
# Query results
duckdb data/duckdb/sword_v17c.duckdb -c "
SELECT reach_id, hydro_dist_out, best_headwater, is_mainstem_edge
FROM reaches WHERE region='NA' AND hydro_dist_out IS NOT NULL LIMIT 10"

# Check provenance
duckdb data/duckdb/sword_v17c.duckdb -c "
SELECT * FROM sword_operations ORDER BY started_at DESC LIMIT 5"

# Run lint
python -m src.updates.sword_duckdb.lint.cli --db data/duckdb/sword_v17c.duckdb --region NA
```

## Archived Files

The original MILP-based phi optimization has been archived in `_archived/`:
- `phi_only_global.py` - MILP flow direction optimization
- `phi_r_global_refine.py` - Phi refinement pass
- `ocn.py` - Optimal Channel Network research
- `orient_global_edges.py` - Edge orientation utilities

These were archived because v17c now uses the original v17b topology.

## Supporting Files

| File | Purpose |
|------|---------|
| `v17c_pipeline.py` | Main pipeline script |
| `run_pipeline.sh` | Shell wrapper |
| `SWOT_slopes.py` | SWOT slope computation (standalone) |
| `SWORD_graph.py` | Graph utilities |
| `validate_topology.py` | Topology validation |
| `validate_edges.py` | Edge validation |
| `check_dag.py` | DAG property checks |
