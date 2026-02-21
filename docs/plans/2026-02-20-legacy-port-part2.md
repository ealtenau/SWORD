# Part 2: Lint Checks (Tasks 3-6)

> Parent plan: `2026-02-20-legacy-code-port.md`

All checks follow same pattern: `@register_check` decorator, SQL query, return `CheckResult`.
Tests go in `tests/sword_duckdb/test_lint.py`.

---

### Task 3: A030 — zero_node_width (WARNING)

**Files:** Modify `src/sword_duckdb/lint/checks/attributes.py`, `tests/sword_duckdb/test_lint.py`

**Step 1:** Test — `TestA030ZeroNodeWidth::test_a030_registered` + `test_a030_runs`

**Step 2:** Run: `python -m pytest tests/sword_duckdb/test_lint.py::TestA030ZeroNodeWidth -v -m lint`

**Step 3:** Implement — append to `attributes.py`:

```python
@register_check("A030", Category.ATTRIBUTES, Severity.WARNING, "Nodes with width <= 0")
def check_zero_node_width(conn, region=None, threshold=None):
    # SELECT node_id, reach_id, region, width FROM nodes WHERE width <= 0
```

**Step 4:** Run tests — PASS

**Step 5:** Commit: `feat: add A030 zero_node_width lint check`

---

### Task 4: N011 — node_ordering_problems (WARNING)

**Files:** Modify `src/sword_duckdb/lint/checks/node.py`, `tests/sword_duckdb/test_lint.py`

**Step 1:** Test — `TestN011NodeOrdering::test_n011_registered` + `test_n011_runs`

**Step 2:** Run tests — FAIL

**Step 3:** Implement — append to `node.py`:

```python
@register_check("N011", Category.NETWORK, Severity.WARNING,
    "Nodes with ordering problems (zero length or length > 1000m)")
def check_node_ordering_problems(conn, region=None, threshold=None):
    # SELECT node_id, reach_id WHERE node_length <= 0 OR node_length > 1000
    # Returns issue_type: 'zero_length' or 'excessive_length'
```

**Step 4:** Run tests — PASS

**Step 5:** Commit: `feat: add N011 node_ordering_problems lint check`

---

### Task 5: G013 — single_node_reaches (INFO)

**Files:** Modify `src/sword_duckdb/lint/checks/geometry.py`, `tests/sword_duckdb/test_lint.py`

**Step 1:** Test — `TestG013SingleNodeReaches::test_g013_registered` + `test_g013_runs`

**Step 2:** Run tests — FAIL

**Step 3:** Implement — append to `geometry.py`:

```python
@register_check("G013", Category.GEOMETRY, Severity.INFO,
    "Non-ghost non-dam reaches with exactly 1 node (candidates for merge)")
def check_single_node_reaches(conn, region=None, threshold=None):
    # SELECT reach_id FROM reaches WHERE n_nodes = 1 AND type NOT IN (4, 6)
```

**Step 4:** Run tests — PASS

**Step 5:** Commit: `feat: add G013 single_node_reaches lint check`

---

### Task 6: G014 — duplicate_centerline_points (INFO)

**Files:** Modify `src/sword_duckdb/lint/checks/geometry.py`, `tests/sword_duckdb/test_lint.py`

**Step 1:** Test — `TestG014DuplicateCenterlines::test_g014_registered` + `test_g014_runs`

**Step 2:** Run tests — FAIL

**Step 3:** Implement — append to `geometry.py`:

```python
@register_check("G014", Category.GEOMETRY, Severity.INFO,
    "Duplicate centerline points (same x,y within a reach)")
def check_duplicate_centerline_points(conn, region=None, threshold=None):
    # ROW_NUMBER() OVER (PARTITION BY reach_id, x, y ORDER BY cl_id) as rn
    # WHERE rn > 1
```

**Step 4:** Run tests — PASS

**Step 5:** Commit: `feat: add G014 duplicate_centerline_points lint check`
