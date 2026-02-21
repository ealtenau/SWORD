# Legacy Code Port Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port data quality fixes and spatial utilities from `src/_legacy/` into the DuckDB pipeline.

**Architecture:** New `spatial_utils.py` module for reusable geo functions + BFS graph traversal. Four new `SWORDWorkflow` methods for data quality. Four corresponding lint checks. All use existing DuckDB tables and provenance logging.

**Tech Stack:** DuckDB SQL, geopy (existing dep), utm, pyproj, pytest

**Plan parts:**
- `2026-02-20-legacy-port-part1.md` — Tasks 1-2: spatial_utils.py
- `2026-02-20-legacy-port-part2.md` — Tasks 3-6: lint checks
- `2026-02-20-legacy-port-part3.md` — Tasks 7-11: workflow methods + integration

**Design doc:** `2026-02-20-legacy-code-port-design.md`

---

## Task Summary

| # | Component | Files | What |
|---|-----------|-------|------|
| 1 | spatial_utils — meters_to_degrees, reproject_utm | CREATE spatial_utils.py, test | Small geo helpers |
| 2 | spatial_utils — BFS traversal | MODIFY spatial_utils.py, test | get_all_upstream/downstream |
| 3 | Lint A030 — zero node width | MODIFY attributes.py, test_lint | WARNING check |
| 4 | Lint N011 — node ordering | MODIFY node.py, test_lint | WARNING check |
| 5 | Lint G013 — single node reaches | MODIFY geometry.py, test_lint | INFO check |
| 6 | Lint G014 — duplicate centerlines | MODIFY geometry.py, test_lint | INFO check |
| 7 | Workflow — fill_zero_width_nodes | MODIFY workflow.py, CREATE test_data_quality | SQL update |
| 8 | Workflow — remove_duplicate_centerline_points | MODIFY workflow.py, test | SQL delete |
| 9 | Workflow — find_and_merge_single_node_reaches | MODIFY workflow.py, test | Batch merge |
| 10 | Workflow — rederive_nodes | MODIFY workflow.py, test | Node rebuild |
| 11 | Integration — verify all checks pass | None | Test + lint + PR |
