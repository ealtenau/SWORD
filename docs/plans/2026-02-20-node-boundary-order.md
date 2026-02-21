# Add Node Boundary IDs and Node Order — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `dn_node_id`/`up_node_id` to reaches and `node_order` to nodes (issue #149).

**Architecture:** ALTER TABLE + UPDATE via dist_out-based SQL. RTREE drop/recreate for reaches. Standalone script + schema update + tests.

**Tech Stack:** DuckDB, Python, pytest

**Key insight:** Use dist_out ordering (not node_id) — flow direction changes can reorder node IDs but dist_out is always semantically correct.

---

### Task 1: Update schema.py

**Files:** Modify `src/sword_duckdb/schema.py`

- Add `dn_node_id BIGINT` and `up_node_id BIGINT` to REACHES_TABLE (after `n_nodes`)
- Add `node_order INTEGER` to NODES_TABLE (after `reach_id`)
- Bump SCHEMA_VERSION to "1.6.0"
- Commit: `schema: add dn_node_id, up_node_id, node_order columns (#149)`

### Task 2: Write maintenance script

**Files:** Create `scripts/maintenance/add_node_columns.py`

Script does:
1. ALTER TABLE to add columns (idempotent — skips if exist)
2. Drop RTREE indexes
3. UPDATE reaches boundary nodes: `FIRST(node_id ORDER BY dist_out ASC/DESC)`
4. UPDATE nodes order: `ROW_NUMBER() OVER (PARTITION BY reach_id ORDER BY dist_out ASC)`
5. Recreate RTREE indexes
6. Verify: no NULLs, max(node_order)==n_nodes, boundary nodes exist, dn dist_out <= up dist_out

CLI: `--db PATH` (required), `--verify-only` (optional)

Commit: `feat: add script to populate dn_node_id, up_node_id, node_order (#149)`

### Task 3: Write tests

**Files:** Create `tests/sword_duckdb/test_node_columns.py`

Tests (all use `ensure_test_db` fixture + tmp_path copy + subprocess):
- `test_script_adds_columns_and_populates` — columns exist, no NULLs
- `test_node_order_range_matches_n_nodes` — max(node_order) == n_nodes per reach
- `test_node_order_1_is_downstream` — node_order=1 has min dist_out
- `test_boundary_nodes_match_dist_out_extremes` — dn has min, up has max dist_out
- `test_idempotent` — running twice produces same result
- `test_verify_only` — --verify-only passes after script run

Run: `uv run pytest tests/sword_duckdb/test_node_columns.py -v`

Commit: `test: add tests for node boundary/order columns (#149)`

### Task 4: Run on production v17c

```bash
uv run python scripts/maintenance/add_node_columns.py --db data/duckdb/sword_v17c.duckdb
uv run python scripts/maintenance/add_node_columns.py --db data/duckdb/sword_v17c.duckdb --verify-only
```

Expected: ~248K reaches, ~11M nodes updated, all checks pass.

### Task 5: Update docs and PR

- Update design doc with final SQL
- Add new columns to CLAUDE.md schema section
- Create PR to v17c-updates
