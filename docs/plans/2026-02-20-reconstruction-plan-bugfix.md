# Bug-Fixer Agent Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix end_reach bifurcation bug, path_freq=0 on connected reaches, and implement main_side from stub.

**Files:**
- Modify: `src/sword_duckdb/reconstruction.py`
- Test: `tests/sword_duckdb/test_reconstruction.py`

---

## Task 1: Fix end_reach Bug

**Location:** `reconstruction.py:2507-2519` — `classify_reach` inner function

**Bug:** Only checks `n_up > 1` for junction. Misses bifurcations where `n_down > 1`. The correct logic (from `reactive.py`) checks both.

### Step 1: Write failing test

```python
class TestEndReachBifurcation:
    def test_bifurcation_classified_as_junction(self, sword_writable):
        from src.sword_duckdb.reconstruction import ReconstructionEngine
        engine = ReconstructionEngine(sword_writable)

        # Find or create a bifurcation (reach with n_down > 1)
        result = sword_writable._db.execute("""
            SELECT rt.reach_id, COUNT(*) as n_down
            FROM reach_topology rt
            WHERE rt.region = 'NA' AND rt.direction = 'down'
            GROUP BY rt.reach_id HAVING COUNT(*) > 1 LIMIT 1
        """).fetchone()

        if result is None:
            reaches = sword_writable._db.execute(
                "SELECT reach_id FROM reaches WHERE region = 'NA' LIMIT 3"
            ).fetchall()
            parent, child1, child2 = reaches[0][0], reaches[1][0], reaches[2][0]
            sword_writable._db.execute("""
                INSERT INTO reach_topology (reach_id, region, direction, neighbor_rank, neighbor_reach_id)
                VALUES (?, 'NA', 'down', 0, ?), (?, 'NA', 'down', 1, ?)
                ON CONFLICT DO NOTHING
            """, [parent, child1, parent, child2])
            bifurc_id = parent
        else:
            bifurc_id = result[0]

        res = engine._reconstruct_reach_end_reach(reach_ids=[bifurc_id], dry_run=True)
        vals = dict(zip(res["entity_ids"], res["values"]))
        assert vals[bifurc_id] == 3, f"Bifurcation should be junction (3), got {vals[bifurc_id]}"
```

### Step 2: Run test — expected FAIL

`python -m pytest tests/sword_duckdb/test_reconstruction.py::TestEndReachBifurcation -v`

### Step 3: Fix classify_reach

In `reconstruction.py`, replace the `classify_reach` function at line 2508:

Old:
```python
        def classify_reach(row):
            n_up = row["n_up"]
            n_down = row["n_down"]
            if n_up == 0:
                return 1  # headwater
            elif n_down == 0:
                return 2  # outlet
            elif n_up > 1:
                return 3  # junction
            else:
                return 0  # main
```

New:
```python
        def classify_reach(row):
            n_up = row["n_up"]
            n_down = row["n_down"]
            if n_up == 0:
                return 1  # headwater
            elif n_down == 0:
                return 2  # outlet
            elif n_up > 1 or n_down > 1:
                return 3  # junction (confluence OR bifurcation)
            else:
                return 0  # main
```

Also update docstring comment at line 2507 to match.

### Step 4: Run test — expected PASS

`python -m pytest tests/sword_duckdb/test_reconstruction.py::TestEndReachBifurcation -v`

### Step 5: Commit

```bash
git add src/sword_duckdb/reconstruction.py tests/sword_duckdb/test_reconstruction.py
git commit -m "fix: end_reach detects bifurcations (n_down > 1) as junctions"
```

---

## Task 2: Fix path_freq for Broken Reaches

**Location:** `reconstruction.py:2375-2467` — `_reconstruct_reach_path_freq`

**Bug:** BFS from outlets gives path_freq=0 to side channels and topology-gap reaches. 4,952 connected non-ghost reaches globally. Fix: iterative repair after main BFS.

### Step 1: Write failing test

```python
class TestPathFreqRepair:
    def test_no_zeros_on_connected(self, sword_writable):
        from src.sword_duckdb.reconstruction import ReconstructionEngine
        engine = ReconstructionEngine(sword_writable)
        result = engine._reconstruct_reach_path_freq(dry_run=True)
        pf_map = dict(zip(result["entity_ids"], result["values"]))

        connected = sword_writable._db.execute("""
            SELECT DISTINCT rt.reach_id FROM reach_topology rt
            JOIN reaches r ON rt.reach_id = r.reach_id AND rt.region = r.region
            WHERE rt.region = 'NA' AND r.type != 6
        """).fetchdf()["reach_id"].tolist()

        zeros = [rid for rid in connected if pf_map.get(rid, 0) == 0]
        assert len(zeros) == 0, f"{len(zeros)} connected reaches have path_freq=0"
```

### Step 2: Run test

`python -m pytest tests/sword_duckdb/test_reconstruction.py::TestPathFreqRepair -v`

### Step 3: Add repair logic

After the BFS loop (after line ~2450, before `if reach_ids is not None`), insert:

```python
        # Repair connected reaches still at path_freq=0 (side channels, topology gaps)
        needs_repair = {rid for rid in all_reaches
                        if path_freq[rid] == 0
                        and (rid in upstream_map or rid in downstream_map)}
        for _ in range(10):
            if not needs_repair:
                break
            repaired = set()
            for rid in needs_repair:
                dn_pf = [path_freq[n] for n in downstream_map.get(rid, []) if path_freq[n] > 0]
                if dn_pf:
                    path_freq[rid] = max(dn_pf)
                    repaired.add(rid)
                    continue
                up_neighbors = upstream_map.get(rid, [])
                if up_neighbors and all(path_freq[n] > 0 for n in up_neighbors):
                    path_freq[rid] = sum(path_freq[n] for n in up_neighbors)
                    repaired.add(rid)
            needs_repair -= repaired
            if not repaired:
                break
        if needs_repair:
            logger.warning(f"path_freq: {len(needs_repair)} reaches still at 0 after repair")
```

### Step 4: Run test — expected PASS

### Step 5: Commit

```bash
git add src/sword_duckdb/reconstruction.py tests/sword_duckdb/test_reconstruction.py
git commit -m "fix: repair path_freq=0 on connected reaches via neighbor propagation"
```

---

## Task 3: Implement main_side

**Location:** `reconstruction.py:3730-3769` — `_reconstruct_reach_main_side` (currently stub)

**Algorithm:**
- 0 = main (highest path_freq branch at bifurcations) — default for all
- 1 = side (lower path_freq branch at bifurcation downstream)
- 2 = secondary outlet (outlet that isn't primary outlet of its network; primary = highest facc)

### Step 1: Write failing test

```python
class TestMainSide:
    def test_values_valid(self, sword_writable):
        from src.sword_duckdb.reconstruction import ReconstructionEngine
        engine = ReconstructionEngine(sword_writable)
        result = engine._reconstruct_reach_main_side(dry_run=True)
        if result.get("status") == "skipped":
            pytest.fail("main_side still a stub")
        assert set(result["values"]).issubset({0, 1, 2})

    def test_majority_zero(self, sword_writable):
        from src.sword_duckdb.reconstruction import ReconstructionEngine
        import numpy as np
        engine = ReconstructionEngine(sword_writable)
        result = engine._reconstruct_reach_main_side(dry_run=True)
        if result.get("status") == "skipped":
            pytest.fail("main_side still a stub")
        vals = np.array(result["values"])
        assert (vals == 0).sum() / len(vals) > 0.80
```

### Step 2: Run test — expected FAIL (stub returns skipped)

### Step 3: Replace stub

Replace `_reconstruct_reach_main_side` entirely:

```python
    def _reconstruct_reach_main_side(
        self,
        reach_ids: Optional[List[int]] = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Reconstruct main_side from topology and path_freq.

        0 = Main channel (highest path_freq at bifurcations)
        1 = Side channel (lower path_freq at bifurcations)
        2 = Secondary outlet (outlet that isn't primary of its network)
        """
        logger.info("Reconstructing reach.main_side from topology + path_freq")

        topology_df = self._conn.execute(
            "SELECT reach_id, direction, neighbor_reach_id FROM reach_topology WHERE region = ?",
            [self._region],
        ).fetchdf()

        reaches_df = self._conn.execute(
            "SELECT reach_id, path_freq, facc, network FROM reaches WHERE region = ?",
            [self._region],
        ).fetchdf()

        pf_map = dict(zip(reaches_df["reach_id"], reaches_df["path_freq"]))
        facc_map = dict(zip(reaches_df["reach_id"], reaches_df["facc"]))
        network_map = dict(zip(reaches_df["reach_id"], reaches_df["network"]))
        all_reach_ids = set(reaches_df["reach_id"])

        downstream_map = {}
        for _, row in topology_df.iterrows():
            if row["direction"] == "down":
                downstream_map.setdefault(row["reach_id"], []).append(row["neighbor_reach_id"])

        outlets = all_reach_ids - set(downstream_map.keys())

        # Primary outlet per network = highest facc
        primary_outlets = {}
        for oid in outlets:
            net = network_map.get(oid, -1)
            if net not in primary_outlets or facc_map.get(oid, 0) > facc_map.get(primary_outlets[net], 0):
                primary_outlets[net] = oid

        main_side = {rid: 0 for rid in all_reach_ids}

        # Secondary outlets
        for oid in outlets:
            net = network_map.get(oid, -1)
            if primary_outlets.get(net) != oid:
                main_side[oid] = 2

        # Side channels: at bifurcations, lower path_freq downstream = side
        for reach_id, dn_neighbors in downstream_map.items():
            if len(dn_neighbors) > 1:
                sorted_dn = sorted(dn_neighbors, key=lambda n: pf_map.get(n, 0), reverse=True)
                for side_reach in sorted_dn[1:]:
                    main_side[side_reach] = 1

        if reach_ids is not None:
            main_side = {k: v for k, v in main_side.items() if k in reach_ids}

        result_df = self._conn.execute(
            f"""SELECT reach_id FROM reaches WHERE region = ?
            {"AND reach_id IN (" + ",".join(["?"] * len(reach_ids)) + ")" if reach_ids else ""}""",
            [self._region] + (list(reach_ids) if reach_ids else []),
        ).fetchdf()
        result_df["main_side"] = result_df["reach_id"].map(main_side)

        return self._update_reach_attribute("main_side", result_df, dry_run)
```

### Step 4: Run test — expected PASS

### Step 5: Commit

```bash
git add src/sword_duckdb/reconstruction.py tests/sword_duckdb/test_reconstruction.py
git commit -m "feat: implement main_side from path_freq at bifurcations"
```
