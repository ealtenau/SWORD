# Topology-Algo Agent Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix path_order and path_segs reconstructors to match legacy algorithm semantics.

**Files:**
- Modify: `src/sword_duckdb/reconstruction.py`
- Test: `tests/sword_duckdb/test_reconstruction.py`

---

## Task 4: Validate path_order

**Location:** `reconstruction.py:3809-3852` — `_reconstruct_reach_path_order`

**Context:** Current impl partitions by path_freq, orders by dist_out ASC. Legacy `path_variables_nc.py` derives path_order from shortest-path traces. The SQL approximation is reasonable. This task validates it and adds tests.

### Step 1: Write test

```python
class TestPathOrder:
    def test_starts_at_one(self, sword_writable):
        from src.sword_duckdb.reconstruction import ReconstructionEngine
        import numpy as np
        engine = ReconstructionEngine(sword_writable)
        result = engine._reconstruct_reach_path_order(dry_run=True)
        vals = np.array(result["values"])
        assert vals.min() >= 1

    def test_monotonic_with_dist_out(self, sword_writable):
        from src.sword_duckdb.reconstruction import ReconstructionEngine
        engine = ReconstructionEngine(sword_writable)
        result = engine._reconstruct_reach_path_order(dry_run=True)

        dist_outs = sword_writable._db.execute(
            "SELECT reach_id, dist_out, path_freq FROM reaches WHERE region = 'NA'"
        ).fetchdf()
        po_map = dict(zip(result["entity_ids"], result["values"]))
        dist_outs["path_order"] = dist_outs["reach_id"].map(po_map)

        violations = 0
        for pf, group in dist_outs.groupby("path_freq"):
            if len(group) < 2 or pf <= 0:
                continue
            sorted_g = group.sort_values("dist_out")
            if not sorted_g["path_order"].is_monotonic_increasing:
                violations += 1
        assert violations == 0, f"{violations} groups have non-monotonic path_order"
```

### Step 2: Run test

`python -m pytest tests/sword_duckdb/test_reconstruction.py::TestPathOrder -v`

### Step 3: Add documentation comment

If tests pass, add a reference comment to the implementation:

```python
        # path_order: position within path from outlet (1) upstream.
        # Legacy (path_variables_nc.py) derived from shortest-path traces.
        # This approximation partitions by path_freq and orders by dist_out,
        # which is equivalent for non-branching segments.
```

### Step 4: Run tests — expected PASS

### Step 5: Commit

```bash
git add src/sword_duckdb/reconstruction.py tests/sword_duckdb/test_reconstruction.py
git commit -m "test: validate path_order, add legacy algorithm reference"
```

---

## Task 5: Fix path_segs

**Location:** `reconstruction.py:3854-3895` — `_reconstruct_reach_path_segs`

**Bug:** Current impl returns COUNT of reaches per path_freq group. Per legacy `stream_order.py`, path_segs is a **unique segment ID** for each (path_order, path_freq) combination. It's an ID, not a count.

### Step 1: Write failing test

```python
class TestPathSegs:
    def test_positive_integers(self, sword_writable):
        from src.sword_duckdb.reconstruction import ReconstructionEngine
        import numpy as np
        engine = ReconstructionEngine(sword_writable)
        result = engine._reconstruct_reach_path_segs(dry_run=True)
        valid = [v for v in result["values"] if v != -9999]
        if valid:
            assert min(valid) >= 1

    def test_same_combo_same_id(self, sword_writable):
        from src.sword_duckdb.reconstruction import ReconstructionEngine
        engine = ReconstructionEngine(sword_writable)
        result = engine._reconstruct_reach_path_segs(dry_run=True)

        reaches = sword_writable._db.execute(
            "SELECT reach_id, path_order, path_freq FROM reaches WHERE region = 'NA'"
        ).fetchdf()
        ps_map = dict(zip(result["entity_ids"], result["values"]))
        reaches["path_segs"] = reaches["reach_id"].map(ps_map)

        for (po, pf), group in reaches.groupby(["path_order", "path_freq"]):
            if pf <= 0:
                continue
            assert group["path_segs"].nunique() == 1, (
                f"({po}, {pf}) has multiple path_segs values"
            )
```

### Step 2: Run test — may fail on semantics

`python -m pytest tests/sword_duckdb/test_reconstruction.py::TestPathSegs -v`

### Step 3: Replace with correct algorithm

Replace `_reconstruct_reach_path_segs` entirely:

```python
    def _reconstruct_reach_path_segs(
        self,
        reach_ids: Optional[List[int]] = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Reconstruct path_segs: unique segment ID per (path_order, path_freq) combo.

        Per legacy stream_order.py, path_segs is a sequential ID assigned to each
        unique (path_order, path_freq) combination, numbered from outlet upstream.
        Reaches with invalid path_freq get path_segs=-9999.
        """
        logger.info("Reconstructing reach.path_segs from (path_order, path_freq)")

        where_clause = ""
        params = [self._region, self._region]
        if reach_ids is not None:
            placeholders = ", ".join(["?"] * len(reach_ids))
            where_clause = f"AND r.reach_id IN ({placeholders})"
            params.extend(reach_ids)

        result_df = self._conn.execute(
            f"""
            WITH combos AS (
                SELECT DISTINCT path_order, path_freq
                FROM reaches
                WHERE region = ? AND path_freq > 0
                ORDER BY path_order, path_freq
            ),
            numbered AS (
                SELECT path_order, path_freq,
                       ROW_NUMBER() OVER (ORDER BY path_order, path_freq) as path_segs
                FROM combos
            )
            SELECT r.reach_id,
                   COALESCE(n.path_segs, -9999) as path_segs
            FROM reaches r
            LEFT JOIN numbered n
                ON r.path_order = n.path_order AND r.path_freq = n.path_freq
            WHERE r.region = ? {where_clause}
        """,
            params,
        ).fetchdf()

        return self._update_reach_attribute("path_segs", result_df, dry_run)
```

### Step 4: Run test — expected PASS

### Step 5: Commit

```bash
git add src/sword_duckdb/reconstruction.py tests/sword_duckdb/test_reconstruction.py
git commit -m "fix: path_segs assigns unique ID per (path_order, path_freq) combo"
```
