# Reconstruction Engine Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 3 bugs (end_reach, path_freq, main_side) and implement 5 external-data stubs (trib_flag, grod_id/obstr_type, hfalls_id, river_name, iceflag) in `reconstruction.py`.

**Architecture:** All changes to `src/sword_duckdb/reconstruction.py` (4,375 lines). Each reconstructor follows existing `_reconstruct_{table}_{attr}(reach_ids, force, dry_run) -> Dict` pattern. External data via `self._source_data_dir`. Tests in `tests/sword_duckdb/test_reconstruction.py`.

**Tech Stack:** Python, DuckDB, scipy.spatial.cKDTree, geopandas, numpy, pytest

---

## Agent Assignment

| Agent | Plan File | Tasks |
|-------|-----------|-------|
| **bug-fixer** | `reconstruction-plan-bugfix.md` | Tasks 1-3: end_reach, path_freq, main_side |
| **topology-algo** | `reconstruction-plan-topology.md` | Tasks 4-5: path_order, path_segs |
| **external-data** | `reconstruction-plan-external.md` | Tasks 6-10: trib_flag, grod_id, hfalls_id, river_name, iceflag |

All agents work in isolated worktrees. Merge sequentially: bug-fixer first, topology-algo second, external-data third.

## Key References

- Current engine: `/Users/jakegearon/projects/SWORD/.dmux/worktrees/sword-reconstruct/src/sword_duckdb/reconstruction.py`
- Tests: `/Users/jakegearon/projects/SWORD/.dmux/worktrees/sword-reconstruct/tests/sword_duckdb/test_reconstruction.py`
- Conftest fixtures: `sword_writable` (fresh DB copy per test), `sword_readonly`
- Legacy trib_flag: `src/_legacy/development/reach_definition/post_formatting/Add_Trib_Flag.py`
- Legacy path_freq: `src/_legacy/updates/network_analysis/path_building_v17/path_variables_nc.py`
- Legacy stream_order: `src/_legacy/updates/network_analysis/path_building_v17/stream_order.py`
- Reactive end_reach (correct logic): `src/sword_duckdb/reactive.py`

## Common Patterns

All reconstructors use:
```python
def _reconstruct_{table}_{attr}(self, reach_ids=None, force=False, dry_run=False) -> Dict[str, Any]:
    # ... compute ...
    return self._update_reach_attribute("attr_name", result_df, dry_run)
```

`_update_reach_attribute` expects a DataFrame with columns `[reach_id, attr_name]`.
`_update_node_attribute` expects a DataFrame with columns `[node_id, attr_name]`.

External data accessed via `self._source_data_dir` (Path, set in `__init__`).

## Merge Order

1. bug-fixer → sword-reconstruct
2. topology-algo → sword-reconstruct (rebase on bug-fixer)
3. external-data → sword-reconstruct (rebase on topology-algo)
4. Run: `python -m pytest tests/sword_duckdb/test_reconstruction.py -v`
5. Run: `ruff check src/sword_duckdb/reconstruction.py && ruff format src/sword_duckdb/reconstruction.py`
