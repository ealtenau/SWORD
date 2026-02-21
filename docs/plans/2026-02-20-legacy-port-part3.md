# Part 3: Workflow Methods + Integration (Tasks 7-11)

> Parent plan: `2026-02-20-legacy-code-port.md`

All methods: add to `src/sword_duckdb/workflow.py`. Tests in `tests/sword_duckdb/test_data_quality.py`.
All use RTREE drop/recreate pattern. All have `dry_run` mode. All log provenance.

---

### Task 7: fill_zero_width_nodes

**Files:** Modify `workflow.py`, Create `tests/sword_duckdb/test_data_quality.py`

**Step 1: Test** — `TestFillZeroWidthNodes`:
- `test_dry_run_returns_dataframe` — returns DataFrame (possibly empty)
- `test_fills_zero_widths` — set a node width to 0, run fill, verify > 0

Mark: `pytestmark = [pytest.mark.db, pytest.mark.workflow]`

Test setup for `test_fills_zero_widths`:
1. Find multi-node reach with positive widths
2. RTREE drop → UPDATE node SET width=0 → RTREE recreate
3. Call `fill_zero_width_nodes(region="NA")`
4. Assert width now > 0 (should be median of reach)

**Step 2:** Run tests — FAIL (no method)

**Step 3:** Implement — SQL-based:

```python
def fill_zero_width_nodes(self, region=None, dry_run=False):
    # CTE: zero_nodes JOIN reach_medians(MEDIAN(width) WHERE width > 0)
    # UPDATE nodes SET width = median WHERE width <= 0
    # RTREE drop/recreate pattern
    # Provenance: FILL_ZERO_WIDTH on nodes table
```

Source: `src/_legacy/updates/formatting_scripts/fill_zero_node_wths.py`

**Step 4:** Run tests — PASS

**Step 5:** Commit: `feat: add fill_zero_width_nodes workflow method`

---

### Task 8: remove_duplicate_centerline_points

**Files:** Modify `workflow.py`, Modify `test_data_quality.py`

**Step 1: Test** — `TestRemoveDuplicateCenterlinePoints`:
- `test_dry_run_returns_dataframe`
- `test_removes_inserted_duplicate` — INSERT a dup, dry_run finds it, remove it, verify gone

**Step 2:** Run tests — FAIL

**Step 3:** Implement:

```python
def remove_duplicate_centerline_points(self, region=None, dry_run=False):
    # ROW_NUMBER() OVER (PARTITION BY reach_id, x, y ORDER BY cl_id)
    # DELETE WHERE rn > 1
    # Trigger reactive recalc for affected reaches
    # Provenance: REMOVE_DUPLICATE_CL
```

Source: `src/_legacy/updates/formatting_scripts/remove_duplicate_pts.py`

**Step 4:** Run tests — PASS

**Step 5:** Commit: `feat: add remove_duplicate_centerline_points workflow method`

---

### Task 9: find_and_merge_single_node_reaches

**Files:** Modify `workflow.py`, Modify `test_data_quality.py`

**Step 1: Test** — `TestFindAndMergeSingleNodeReaches`:
- `test_dry_run_returns_dataframe` — has columns: reach_id, merge_target, direction, status

**Step 2:** Run tests — FAIL

**Step 3:** Implement:

```python
def find_and_merge_single_node_reaches(self, region=None, dry_run=False):
    # Find: n_nodes=1 AND type NOT IN (4, 6)
    # For each candidate:
    #   if 1 downstream neighbor with n_rch_up=1 → merge downstream
    #   elif 1 upstream neighbor with n_rch_down=1 → merge upstream
    #   else → skip (log as unresolvable)
    # Call self.merge_reach(target, source) for each
    # Returns DataFrame(reach_id, merge_target, direction, status)
```

Source: `src/_legacy/updates/formatting_scripts/aggregate_1node_rchs.py`

**Step 4:** Run tests — PASS

**Step 5:** Commit: `feat: add find_and_merge_single_node_reaches workflow method`

---

### Task 10: rederive_nodes

**Files:** Modify `workflow.py`, Modify `test_data_quality.py`

**Step 1: Test** — `TestRederiveNodes`:
- `test_dry_run_returns_dataframe`
- `test_rederive_preserves_node_count` — find reach with ≥3 nodes, rederive, count unchanged

**Step 2:** Run tests — FAIL

**Step 3:** Implement:

```python
def rederive_nodes(self, reach_ids, region=None, dry_run=False):
    # Per reach:
    # 1. Get centerlines sorted by cl_id
    # 2. Compute geodesic distances (geopy.distance.geodesic)
    # 3. Divide into N groups (N = current n_nodes)
    # 4. Per group: x=median, y=median, length=geodesic sum, cl_id_min/max
    # 5. node dist_out = reach.dist_out - reach.reach_length + cumsum
    # 6. DELETE old nodes, INSERT new (preserving old node_ids)
    # 7. UPDATE centerlines.node_id references
    # RTREE drop/recreate
```

Source: `src/_legacy/updates/formatting_scripts/fix_problem_node_order_length.py`

**Step 4:** Run tests — PASS

**Step 5:** Commit: `feat: add rederive_nodes workflow method`

---

### Task 11: Integration verification + PR

**Step 1:** Run all new lint checks:
```bash
python -m pytest tests/sword_duckdb/test_lint.py -v -m lint -k "A030 or N011 or G013 or G014"
```

**Step 2:** Run all data quality tests:
```bash
python -m pytest tests/sword_duckdb/test_data_quality.py -v
```

**Step 3:** Run spatial_utils tests:
```bash
python -m pytest tests/sword_duckdb/test_spatial_utils.py -v
```

**Step 4:** Lint:
```bash
ruff check src/sword_duckdb/spatial_utils.py src/sword_duckdb/workflow.py src/sword_duckdb/lint/checks/
ruff format src/sword_duckdb/spatial_utils.py tests/sword_duckdb/test_spatial_utils.py tests/sword_duckdb/test_data_quality.py
```

**Step 5:** Push + PR to v17c-updates:
```bash
git push -u origin legacy-code
gh pr create --title "Port data quality fixes and spatial utils from legacy code" --base v17c-updates
```

PR body:
- 4 new spatial_utils functions
- 4 new workflow data quality methods
- 4 new lint checks (A030, N011, G013, G014)
- Ported from `src/_legacy/updates/formatting_scripts/` and `geo_utils.py`
