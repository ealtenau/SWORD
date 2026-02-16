# Design: Recompute path_freq, stream_order, path_segs, path_order

**Issue:** #16 — Recalculate path_freq and path_segs
**Date:** 2026-02-16
**Approach:** Integrate into v17c_pipeline.py

## Problem

6,775 non-ghost reaches have invalid path_freq (0 or -9999), cascading to invalid stream_order and path_segs. 91% are 1:1 links, 9% junctions. Additionally, path_segs reconstruction is fundamentally broken (0.1% match rate with v17b).

Rather than patching individual reaches, do a full recompute of all four traversal-derived variables using the current v17c topology.

## Variables

| Variable | Algorithm | Depends on |
|----------|-----------|------------|
| path_freq | BFS from all outlets upstream; count visits per reach | topology |
| stream_order | `round(log(path_freq)) + 1`; -9999 for side channels | path_freq, main_side |
| path_segs | Sequential ID per junction-to-junction segment | topology (junctions) |
| path_order | Rank by dist_out ASC within path_freq groups | path_freq, dist_out |

## Integration into v17c_pipeline.py

### New function: `compute_path_variables(G, sections_df)`

Located in v17c_pipeline.py, called after `build_section_graph()`.

**Inputs:**
- `G` — NetworkX DiGraph (already built by pipeline)
- `sections_df` — junction-to-junction sections (already computed)

**Returns:** `Dict[int, Dict]` mapping reach_id to `{path_freq, stream_order, path_segs, path_order}`

### Algorithm detail

**path_freq (BFS from outlets):**
1. Find outlets: nodes with out-degree 0 in G (no downstream neighbors)
2. For each outlet, BFS upstream (follow predecessors)
3. Each reach accumulates a visit count
4. Ghost reaches (type=6) excluded from graph, get -9999

**stream_order:**
- `round(log(path_freq)) + 1` for valid path_freq > 0
- -9999 for main_side IN (1, 2) OR path_freq <= 0

**path_segs:**
- Reuse existing `sections_df` which maps section_id to reach_ids
- Each reach gets its section's section_id as path_segs
- New sequential IDs (won't match v17b numbering — acceptable)

**path_order:**
- Group reaches by path_freq
- Within each group, rank by dist_out ascending (closest to outlet = rank 1)
- Uses dist_out from reach attributes already on graph nodes

### Modification to process_region()

```python
# After line 1182 (build_section_graph)
junctions = identify_junctions(G)
R, sections_df = build_section_graph(G, junctions)

# NEW: Compute path variables
if not skip_path_freq:
    path_vars = compute_path_variables(G, sections_df)
else:
    path_vars = None

# ... existing code ...

# Modified save call
with workflow.transaction(f"v17c attributes for {region}"):
    n_updated = save_to_duckdb(
        conn, region, hydro_dist, hw_out, is_mainstem, main_neighbors,
        path_vars=path_vars,  # NEW parameter
    )
```

### Modification to save_to_duckdb()

Add optional `path_vars` parameter. When provided, include path_freq, stream_order, path_segs, path_order in the UPDATE statement. Uses existing RTREE drop/recreate pattern.

### CLI flag

Add `--skip-path-freq` to argparse. Default: compute path variables. Allows skipping during development or when only v17c-specific attributes are needed.

## Validation (log-only, non-blocking)

Before writing results:
1. **T002** — path_freq monotonicity: check no downstream decrease
2. **T010** — headwater path_freq >= 1
3. **stream_order consistency** — verify formula matches
4. **path_segs contiguity** — adjacent reaches in same segment share ID
5. **Aggregate comparison** — log max path_freq, mean, distribution percentiles vs v17b

Warnings logged but results written regardless.

## Scope exclusions

- No changes to reactive.py dependency graph (future work)
- No changes to reconstruction.py (deprecated for these variables)
- No attempt to match v17b path_segs numbering
- Ghost reaches (type=6) get -9999 for all four variables
