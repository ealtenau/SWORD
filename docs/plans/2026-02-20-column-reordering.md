# Column Reordering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Single source of truth for column ordering, consumed by all code paths.

**Architecture:** New `column_order.py` with canonical tuples + `reorder_columns()` helper. All consumers import from it.

**Tech Stack:** Python, DuckDB, pandas, pytest

---

### Task 1: Create column_order.py + tests

**Files:**
- Create: `src/sword_duckdb/column_order.py`
- Create: `tests/sword_duckdb/test_column_order.py`

**Steps:**
1. Write tests: canonical list properties (starts with PK, ends with version, no dupes), `get_column_order()` dispatch, `reorder_columns()` behavior (reorder/skip missing/append extra), DB coverage (canonical covers all actual DB columns)
2. Run `pytest tests/sword_duckdb/test_column_order.py -m unit -v` — expect ImportError
3. Create `column_order.py` with `REACHES_COLUMN_ORDER`, `NODES_COLUMN_ORDER`, `CENTERLINES_COLUMN_ORDER` tuples (see design doc for exact ordering), `get_column_order(table_name)`, `reorder_columns(df, table_name)`
4. Run unit + db tests — all PASS
5. `ruff check && ruff format` both files
6. Commit: `feat: add column_order.py with canonical column ordering`

---

### Task 2: Wire sword_class.py

**Files:**
- Modify: `src/sword_duckdb/sword_class.py:1407-1446` (nodes cols)
- Modify: `src/sword_duckdb/sword_class.py:1549-1593` (reaches cols)

**Steps:**
1. Replace hardcoded nodes cols list (L1407-1446) with: `from src.sword_duckdb.column_order import NODES_COLUMN_ORDER` then `cols = [c for c in NODES_COLUMN_ORDER if c in nodes_df.columns]`
2. Replace hardcoded reaches cols list (L1549-1593) with same pattern using `REACHES_COLUMN_ORDER`
3. Run `pytest tests/sword_duckdb/test_sword_class.py -v` — all PASS
4. Commit: `refactor: wire sword_class.py to canonical column_order`

---

### Task 3: Wire export.py

**Files:**
- Modify: `src/sword_duckdb/export.py`

**Steps:**
1. Add `from src.sword_duckdb.column_order import reorder_columns`
2. In each `_export_*_to_pg` function, call `df = reorder_columns(df, table_name)` before writing
3. Same for any Parquet/GPKG/NetCDF export paths
4. Run `pytest tests/sword_duckdb/test_export.py -v` — all PASS
5. Commit: `refactor: wire export.py to canonical column_order`

---

### Task 4: Reorder schema.py DDL

**Files:**
- Modify: `src/sword_duckdb/schema.py:57-313`

**Steps:**
1. Write test: parse DDL column names, assert order matches canonical tuple
2. Reorder column definitions in `REACHES_TABLE`, `NODES_TABLE`, `CENTERLINES_TABLE` DDL strings to match canonical order. Keep types/comments/defaults.
3. Run `pytest tests/sword_duckdb/ -m "not slow" -v` — all PASS
4. Commit: `refactor: reorder schema DDL to match canonical column_order`

**Note:** Only affects new DB creation. Existing DBs use `reorder_columns()` at query/export time.

---

### Task 5: Wire migrations.py

**Files:**
- Modify: `src/sword_duckdb/migrations.py:248-392`

**Steps:**
1. After `df = pd.DataFrame(df_dict)` in `_migrate_nodes` and `_migrate_reaches`, add `df = reorder_columns(df, table_name)` then `columns = list(df.columns)`
2. Run `pytest tests/sword_duckdb/test_sword_class.py -v` — all PASS
3. Commit: `refactor: wire migrations.py to canonical column_order`

---

### Task 6: Export public API + final validation

**Files:**
- Modify: `src/sword_duckdb/__init__.py`

**Steps:**
1. Add column_order symbols to `__init__.py` public exports
2. Run full suite: `pytest tests/sword_duckdb/ -m "not slow" -v`
3. `ruff check` + `ruff format` all modified files
4. Commit: `feat: export column_order public API`

---

## Implementer Notes

- **RTREE:** If any step UPDATEs reaches/nodes, drop/recreate RTREE indexes
- **Regional views:** `SELECT *` inherits table order automatically
- **Test DB:** May lack v17c/SWOT columns — `reorder_columns()` handles missing gracefully
- **No physical reorder:** Existing DBs keep their storage order; reordering is logical (DataFrame/export)
