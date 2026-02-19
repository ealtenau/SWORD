# Archived Pipeline Files

Files archived during v17c pipeline simplification.

## MILP Optimization Files

These files implemented the phi optimization algorithm for flow direction determination:

- `phi_only_global.py` - MILP-based global flow direction optimization
- `phi_r_global_refine.py` - Refinement pass for phi optimization
- `ocn.py` - Optimal Channel Network research code
- `orient_global_edges.py` - Edge orientation utilities (unused)

## Superseded Files

These files were superseded by `v17c_pipeline.py`:

- `v17c_nophi.py` - Original no-phi pipeline (output to Parquet/Pickle)
- `run_pipeline_nophi.sh` - Shell script for v17c_nophi.py
- `run_all_continents.sh` - Loop over all continents (use `--all` flag instead)
- `README_nophi.md` - Documentation for nophi approach

## Why Archived

The v17c pipeline was simplified to:
1. Use original v17b topology (trusted flow directions)
2. Add SWOT-derived slope validation
3. Compute new hydrologic attributes
4. Output directly to DuckDB via SWORDWorkflow

The MILP optimization added complexity without clear benefit over the manually-curated v17b topology.

## If You Need These Files

These files are preserved for reference. If you need to resurrect them:

### MILP Optimization
1. Move phi files back to parent directory
2. Install pulp/HiGHS solver
3. See `topology_optimizer.py` for standalone MILP experiments

### Old Nophi Pipeline
1. `v17c_nophi.py` outputs to Parquet/Pickle (not DuckDB)
2. Use this if you need the old output format
3. Note: Does not use SWORDWorkflow provenance
