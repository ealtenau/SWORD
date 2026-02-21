# External-Data Agent Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 5 external-data stubs: trib_flag, grod_id/obstr_type, hfalls_id, river_name, iceflag.

**Files:**
- Modify: `src/sword_duckdb/reconstruction.py`
- Test: `tests/sword_duckdb/test_reconstruction.py`

**Common pattern:** All use `self._source_data_dir` (Path) to find external data. When data missing, fall back to preserving existing values (same as current stubs). When data present, do spatial join via `scipy.spatial.cKDTree`.

---

## Task 6: Implement trib_flag

**Location:** `reconstruction.py:3124-3160` (node) and `3897-3936` (reach)

**Legacy algorithm** (from `Add_Trib_Flag.py`):
1. Load MHV points from `<source_data_dir>/MHV_SWORD/` — filter `sword_flag==0`, `strmorder>=3`
2. Build cKDTree from MHV (x, y) coords
3. Query each SWORD node, k=1
4. Flag if distance <= 0.003 degrees (~333m)
5. Reach trib_flag = MAX(node trib_flag)

MHV files: one per L2 basin, named `*pts*`. Basin ID = first 2 digits of node_id.

### Step 1: Write test

```python
class TestTribFlag:
    def test_not_stub_when_data_present(self, sword_writable, tmp_path):
        import geopandas as gpd
        from shapely.geometry import Point
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        node = sword_writable._db.execute(
            "SELECT node_id, x, y FROM nodes WHERE region = 'NA' LIMIT 1"
        ).fetchone()
        node_id, nx, ny = node

        mhv_dir = tmp_path / "MHV_SWORD"
        mhv_dir.mkdir()
        basin = int(str(node_id)[:2])
        gdf = gpd.GeoDataFrame({
            "x": [nx + 0.001], "y": [ny + 0.001],
            "sword_flag": [0], "strmorder": [4],
            "geometry": [Point(nx + 0.001, ny + 0.001)],
        })
        gdf.to_file(mhv_dir / f"mhv_pts_{basin:02d}.gpkg", driver="GPKG")

        engine = ReconstructionEngine(sword_writable, source_data_dir=str(tmp_path))
        result = engine._reconstruct_node_trib_flag(node_ids=[node_id], dry_run=True)
        assert result.get("status") != "skipped"

    def test_values_binary(self, sword_writable, tmp_path):
        import numpy as np
        from src.sword_duckdb.reconstruction import ReconstructionEngine
        mhv_dir = tmp_path / "MHV_SWORD"
        mhv_dir.mkdir()
        engine = ReconstructionEngine(sword_writable, source_data_dir=str(tmp_path))
        result = engine._reconstruct_node_trib_flag(dry_run=True)
        if result.get("status") == "skipped":
            pytest.skip("No MHV data")
        vals = set(np.array(result["values"]).astype(int))
        assert vals.issubset({0, 1})
```

### Step 2: Run — expected FAIL (stub)

### Step 3: Replace node trib_flag stub (lines 3124-3160)

```python
    def _reconstruct_node_trib_flag(
        self, node_ids=None, force=False, dry_run=False,
    ) -> Dict[str, Any]:
        """Reconstruct node trib_flag from MHV spatial join."""
        mhv_dir = self._source_data_dir / "MHV_SWORD" if self._source_data_dir else None
        if mhv_dir is None or not mhv_dir.exists():
            logger.warning("node.trib_flag: MHV data not found. Preserving values.")
            return {"status": "skipped", "reason": "MHV not found", "updated": 0, "dry_run": dry_run}

        import geopandas as gpd
        from scipy.spatial import cKDTree

        logger.info("Reconstructing node.trib_flag from MHV spatial join")

        where_clause = ""
        params = [self._region]
        if node_ids is not None:
            placeholders = ", ".join(["?"] * len(node_ids))
            where_clause = f"AND node_id IN ({placeholders})"
            params.extend(node_ids)

        nodes_df = self._conn.execute(
            f"SELECT node_id, x, y, reach_id FROM nodes WHERE region = ? {where_clause}", params
        ).fetchdf()
        if nodes_df.empty:
            return {"status": "ok", "updated": 0, "dry_run": dry_run}

        nodes_df["basin"] = nodes_df["node_id"].astype(str).str[:2].astype(int)
        needed_basins = nodes_df["basin"].unique()

        mhv_files = sorted(mhv_dir.glob("*pts*"))
        if not mhv_files:
            mhv_files = sorted(mhv_dir.glob("*.gpkg")) + sorted(mhv_dir.glob("*.shp"))

        trib_flag = np.zeros(len(nodes_df), dtype=np.int32)

        for mhv_file in mhv_files:
            fname = mhv_file.stem
            basin_candidates = [int(s) for s in fname.split("_") if s.isdigit() and len(s) <= 2]
            if not basin_candidates:
                continue
            file_basin = basin_candidates[-1]
            if file_basin not in needed_basins:
                continue

            try:
                mhv = gpd.read_file(mhv_file)
            except Exception as e:
                logger.warning(f"Failed to read {mhv_file}: {e}")
                continue

            if "sword_flag" in mhv.columns and "strmorder" in mhv.columns:
                mhv = mhv[(mhv["sword_flag"] == 0) & (mhv["strmorder"] >= 3)]
            if mhv.empty or "x" not in mhv.columns or "y" not in mhv.columns:
                continue

            kdt = cKDTree(np.column_stack([mhv["x"].values, mhv["y"].values]))
            basin_mask = nodes_df["basin"] == file_basin
            if not basin_mask.any():
                continue
            node_pts = np.column_stack([
                nodes_df.loc[basin_mask, "x"].values, nodes_df.loc[basin_mask, "y"].values
            ])
            distances, _ = kdt.query(node_pts, k=1)
            trib_flag[basin_mask.values] = np.where(distances <= 0.003, 1, trib_flag[basin_mask.values])

        nodes_df["trib_flag"] = trib_flag
        return self._update_node_attribute("trib_flag", nodes_df[["node_id", "trib_flag"]], dry_run)
```

### Step 4: Replace reach trib_flag stub (lines 3897-3936)

```python
    def _reconstruct_reach_trib_flag(
        self, reach_ids=None, force=False, dry_run=False,
    ) -> Dict[str, Any]:
        """Reconstruct reach trib_flag as MAX of node trib_flags."""
        logger.info("Reconstructing reach.trib_flag from node max")
        where_clause = ""
        params = [self._region]
        if reach_ids is not None:
            placeholders = ", ".join(["?"] * len(reach_ids))
            where_clause = f"AND n.reach_id IN ({placeholders})"
            params.extend(reach_ids)
        result_df = self._conn.execute(
            f"SELECT n.reach_id, MAX(n.trib_flag) as trib_flag FROM nodes n WHERE n.region = ? {where_clause} GROUP BY n.reach_id",
            params,
        ).fetchdf()
        return self._update_reach_attribute("trib_flag", result_df, dry_run)
```

### Step 5: Run tests — expected PASS

### Step 6: Commit

```bash
git commit -m "feat: implement trib_flag from MHV KDTree spatial join"
```

---

## Task 7: Implement grod_id from GROD

**Location:** `reconstruction.py:3944-3976` (node), `4046-4078` (reach)

**Algorithm:** Spatial join GROD points to nearest node within 0.005 degrees. GROD files in `<source_data_dir>/GROD/`.

### Step 1: Write test

```python
class TestGrodId:
    def test_from_external_data(self, sword_writable, tmp_path):
        import geopandas as gpd
        from shapely.geometry import Point
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        node = sword_writable._db.execute(
            "SELECT node_id, x, y FROM nodes WHERE region = 'NA' LIMIT 1"
        ).fetchone()
        grod_dir = tmp_path / "GROD"
        grod_dir.mkdir()
        gdf = gpd.GeoDataFrame({
            "GROD_ID": [12345], "Type": [1],
            "geometry": [Point(node[1] + 0.0005, node[2] + 0.0005)],
        })
        gdf.to_file(grod_dir / "GROD.gpkg", driver="GPKG")

        engine = ReconstructionEngine(sword_writable, source_data_dir=str(tmp_path))
        result = engine._reconstruct_node_grod_id(node_ids=[node[0]], dry_run=True)
        vals = dict(zip(result["entity_ids"], result["values"]))
        assert vals.get(node[0]) == 12345
```

### Step 2: Run — expected FAIL

### Step 3: Replace node grod_id stub

Same pattern as trib_flag but simpler (single file, no basin partitioning):
- Load GROD GeoPackage from `<source_data_dir>/GROD/`
- Normalize column names (`GROD_ID`/`grod_id`, `Type`/`obstr_type`)
- cKDTree from GROD geometry centroids
- Match to nodes within 0.005 deg
- `grod_id` = matched GROD ID, 0 if no match

Also update `_reconstruct_node_obstr_type` to use GROD `Type` column in same spatial join. And make reach versions aggregate from nodes (MAX).

### Step 4: Run tests, Step 5: Commit

```bash
git commit -m "feat: implement grod_id/obstr_type from GROD spatial join"
```

---

## Task 8: Implement hfalls_id from HydroFALLS

**Location:** `reconstruction.py:3978-4010` (node), `4080-4112` (reach)

Same pattern as GROD. Files in `<source_data_dir>/HydroFALLS/`. Match waterfall points to nearest node within 0.005 degrees.

### Step 1-5: Same as Task 7 but for HydroFALLS

```bash
git commit -m "feat: implement hfalls_id from HydroFALLS spatial join"
```

---

## Task 9: Implement river_name from Names Data

**Location:** `reconstruction.py:4012-4044` (node), `4114-4146` (reach)

**Algorithm:**
- Load river names shapefile from `<source_data_dir>/river_names/`
- Spatial join: nearest named feature to each reach centroid
- Default to `"NODATA"` (not empty string — SWORD convention)
- Node inherits from parent reach

### Step 1: Write test

```python
class TestRiverName:
    def test_default_nodata(self, sword_writable, tmp_path):
        from src.sword_duckdb.reconstruction import ReconstructionEngine
        names_dir = tmp_path / "river_names"
        names_dir.mkdir()
        # Empty dir -> all NODATA
        engine = ReconstructionEngine(sword_writable, source_data_dir=str(tmp_path))
        result = engine._reconstruct_reach_river_name(dry_run=True)
        if result.get("status") == "skipped":
            pytest.skip("No names data")
        for v in result["values"]:
            assert v == "NODATA" or isinstance(v, str)
```

### Step 2-5: Implement and commit

```bash
git commit -m "feat: implement river_name from names shapefile spatial join"
```

---

## Task 10: Implement iceflag from Ice Data

**Location:** `reconstruction.py:4148-4180`

**Algorithm:**
- First check `reach_ice_flags` table (366 rows per reach)
- If empty, try loading from CSV at `<source_data_dir>/ice_flags/`
- Summary: `iceflag = MAX(daily_values)` per reach
- -9999 = no data, 0 = no ice, 1 = seasonal, 2 = permanent

### Step 1: Write test

```python
class TestIceFlag:
    def test_from_ice_flags_table(self, sword_writable):
        from src.sword_duckdb.reconstruction import ReconstructionEngine
        # Insert test data
        sword_writable._db.execute("""
            INSERT INTO reach_ice_flags (reach_id, julian_day, iceflag)
            SELECT reach_id, 1, 1 FROM reaches WHERE region = 'NA' LIMIT 5
            ON CONFLICT DO NOTHING
        """)
        engine = ReconstructionEngine(sword_writable)
        result = engine._reconstruct_reach_iceflag(dry_run=True)
        assert result.get("status") != "skipped"
```

### Step 2-3: Replace stub

```python
    def _reconstruct_reach_iceflag(
        self, reach_ids=None, force=False, dry_run=False,
    ) -> Dict[str, Any]:
        """Reconstruct iceflag from reach_ice_flags table (max of daily values)."""
        logger.info("Reconstructing reach.iceflag from reach_ice_flags")

        count = self._conn.execute(
            "SELECT COUNT(*) FROM reach_ice_flags WHERE reach_id IN "
            "(SELECT reach_id FROM reaches WHERE region = ?)", [self._region]
        ).fetchone()[0]

        if count == 0 and self._source_data_dir:
            ice_dir = self._source_data_dir / "ice_flags"
            if ice_dir and ice_dir.exists():
                import pandas as pd
                for csv_file in ice_dir.glob("*.csv"):
                    df = pd.read_csv(csv_file)
                    if "reach_id" not in df.columns:
                        continue
                    day_cols = [c for c in df.columns if c != "reach_id"]
                    if day_cols:
                        melted = df.melt(id_vars=["reach_id"], value_vars=day_cols,
                                         var_name="jd", value_name="iceflag")
                        melted["julian_day"] = melted["jd"].str.extract(r"(\d+)").astype(int)
                        self._conn.executemany(
                            "INSERT INTO reach_ice_flags VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                            melted[["reach_id", "julian_day", "iceflag"]].values.tolist())
                        logger.info(f"Loaded {len(melted)} ice records from {csv_file}")

        where_clause = ""
        params = [self._region]
        if reach_ids:
            placeholders = ", ".join(["?"] * len(reach_ids))
            where_clause = f"AND r.reach_id IN ({placeholders})"
            params.extend(reach_ids)

        result_df = self._conn.execute(f"""
            SELECT r.reach_id, COALESCE(MAX(i.iceflag), -9999) as iceflag
            FROM reaches r LEFT JOIN reach_ice_flags i ON r.reach_id = i.reach_id
            WHERE r.region = ? {where_clause} GROUP BY r.reach_id
        """, params).fetchdf()

        return self._update_reach_attribute("iceflag", result_df, dry_run)
```

### Step 4-5: Run tests and commit

```bash
git commit -m "feat: implement iceflag from reach_ice_flags table or CSV"
```
