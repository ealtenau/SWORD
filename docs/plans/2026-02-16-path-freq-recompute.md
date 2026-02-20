# path_freq Recompute — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Full recompute of path_freq, stream_order, path_segs, path_order integrated into v17c_pipeline.py

**Architecture:** New `compute_path_variables(G, sections_df)` in v17c_pipeline.py, called after `build_section_graph()`. BFS from outlets for path_freq, log transform for stream_order, section_id for path_segs, dist_out ranking for path_order.

**Tech Stack:** NetworkX, DuckDB, pandas, math, pytest

**Design doc:** `docs/plans/2026-02-16-path-freq-recompute-design.md`

---

### Task 1: Write failing tests for path_freq

**Files:**
- Modify: `tests/sword_duckdb/test_v17c_pipeline_graph.py`

Add `compute_path_variables` to the existing import block (line 14-20). Add test classes at end of file:

- `TestComputePathFreqLinear`: linear chain → all path_freq=1
- `TestComputePathFreqConfluence`: headwaters=1, downstream of confluence=2
- `TestComputePathFreqBifurcation`: upstream of bifurcation visited by 2 outlet BFS → path_freq=2

Use existing synthetic fixtures (`simple_linear_*`, `confluence_*`, `bifurcation_*`).

Run: `pytest tests/sword_duckdb/test_v17c_pipeline_graph.py -k PathFreq -v`
Expected: ImportError (function doesn't exist yet)

Commit: `"Add failing tests for compute_path_variables path_freq (#16)"`

---

### Task 2: Implement `compute_path_variables`

**Files:**
- Modify: `src/sword_v17c_pipeline/v17c_pipeline.py` (insert after `build_section_graph` ~line 458)

New function `compute_path_variables(G, sections_df) -> Dict[int, Dict]`:

1. **path_freq**: find outlets (out_degree=0), BFS upstream from each, count visits per reach
2. **stream_order**: `round(log(path_freq)) + 1`, -9999 for main_side IN (1,2) or path_freq<=0
3. **path_segs**: map section_id from sections_df to reach_ids (1-based). Junctions not in sections get unique IDs.
4. **path_order**: group by path_freq, rank by dist_out ASC within group
5. **Validation**: log monotonicity violations (T002) and headwater violations (T010) — non-blocking

Returns `{reach_id: {path_freq, stream_order, path_segs, path_order}}`.

Run: `pytest tests/sword_duckdb/test_v17c_pipeline_graph.py -k PathFreq -v`
Expected: All PASS

Commit: `"Add compute_path_variables with BFS path_freq (#16)"`

---

### Task 3: Write tests for stream_order, path_segs, path_order

**Files:**
- Modify: `tests/sword_duckdb/test_v17c_pipeline_graph.py`

Add test classes:

- `TestComputeStreamOrder`: linear→1, confluence→2, side channel (main_side=1)→-9999
- `TestComputePathSegs`: linear→single segment, confluence→multiple segments, no -9999 for connected
- `TestComputePathOrder`: linear ranks by dist_out, different freq groups rank independently
- `TestComputePathVariablesIntegration`: 100-reach test DB → all path_freq=1, valid values

Run: `pytest tests/sword_duckdb/test_v17c_pipeline_graph.py -k "StreamOrder or PathSegs or PathOrder or Integration" -v`
Expected: All PASS

Commit: `"Add tests for stream_order, path_segs, path_order (#16)"`

---

### Task 4: Wire into process_region() and save_to_duckdb()

**Files:**
- Modify: `src/sword_v17c_pipeline/v17c_pipeline.py`

Changes:

1. **save_to_duckdb** (~line 856): add `path_vars: Optional[Dict]=None` param. When provided, add path_freq/stream_order/path_segs/path_order to UPDATE rows and SET clause.
2. **process_region** (~line 1092): add `skip_path_freq: bool=False`. Call `compute_path_variables(G, sections_df)` after `build_section_graph`. Pass `path_vars` to `save_to_duckdb`. Add stats.
3. **run_pipeline** (~line 1301): thread `skip_path_freq` through.
4. **main** (~line 1399): add `--skip-path-freq` CLI arg.

Run: `pytest tests/sword_duckdb/test_v17c_pipeline_graph.py -v` (no regressions)

Commit: `"Wire compute_path_variables into process_region and save_to_duckdb (#16)"`

---

### Task 5: Run against real v17c database

**Execution only — no code changes.**

1. Run NA only first:
```bash
python -m src.sword_v17c_pipeline.v17c_pipeline \
  --db data/duckdb/sword_v17c.duckdb --region NA --skip-swot --skip-facc
```

2. Verify: query DB for path_freq invalid count (should be 0 or near-0 for NA).

3. Run all regions:
```bash
python -m src.sword_v17c_pipeline.v17c_pipeline \
  --db data/duckdb/sword_v17c.duckdb --all --skip-swot --skip-facc
```

4. Verify all regions.

5. `ruff format` + `ruff check` on changed files.

6. Final commit + comment on issue #16.

7. Create PR to `v17c-updates`.
