# Part 1: spatial_utils.py (Tasks 1-2)

> Parent plan: `2026-02-20-legacy-code-port.md`

---

### Task 1: meters_to_degrees + reproject_utm

**Files:**
- Create: `src/sword_duckdb/spatial_utils.py`
- Create: `tests/sword_duckdb/test_spatial_utils.py`

**Step 1: Write failing tests** in `tests/sword_duckdb/test_spatial_utils.py`

- `TestMetersToDegrees::test_equator` — 111320m at lat=0 ≈ 1.0 degree
- `TestMetersToDegrees::test_high_latitude` — 55660m at lat=60 ≈ 1.0 degree
- `TestMetersToDegrees::test_zero_meters` — 0m returns 0.0
- `TestReprojectUtm::test_single_point` — Washington DC → zone 18, easting >300k
- `TestReprojectUtm::test_multiple_points_same_zone` — 3 points return 3 values

Mark: `pytestmark = pytest.mark.unit`

**Step 2:** Run: `python -m pytest tests/sword_duckdb/test_spatial_utils.py -v -m unit`
Expected: FAIL (ModuleNotFoundError)

**Step 3:** Implement `src/sword_duckdb/spatial_utils.py`:

```python
def meters_to_degrees(meters: float, latitude: float) -> float:
    cos_lat = math.cos(math.radians(latitude))
    if cos_lat == 0:
        return float("inf")
    return meters / (111320.0 * cos_lat)

def reproject_utm(latitudes, longitudes):
    # utm.from_latlon per point → pick most common zone → Proj all
    # Returns (easting_arr, northing_arr, zone_num, zone_letter)
```

Source: `src/_legacy/updates/geo_utils.py:69` and `:293`

**Step 4:** Run: `python -m pytest tests/sword_duckdb/test_spatial_utils.py -v -m unit`
Expected: PASS (5 tests)

**Step 5:** Commit: `feat: add meters_to_degrees and reproject_utm spatial utils`

---

### Task 2: BFS get_all_upstream / get_all_downstream

**Files:**
- Modify: `src/sword_duckdb/spatial_utils.py`
- Modify: `tests/sword_duckdb/test_spatial_utils.py`

**Step 1: Write failing tests** — `TestBfsTraversal` class (mark `pytest.mark.db`):

- `test_get_all_upstream_returns_set` — outlet → set of upstream IDs
- `test_get_all_downstream_returns_set` — headwater → set of downstream IDs
- `test_upstream_of_headwater_is_empty` — headwater with n_rch_up=0 → empty set
- `test_downstream_of_outlet_is_empty` — outlet with n_rch_down=0 → empty set

Fixture: copy test DB to tmp_path, open duckdb.connect.

**Step 2:** Run: `python -m pytest tests/sword_duckdb/test_spatial_utils.py::TestBfsTraversal -v`
Expected: FAIL (ImportError)

**Step 3:** Implement in spatial_utils.py:

```python
def get_all_upstream(con, reach_id, region=None) -> set[int]:
    return _bfs_topology(con, reach_id, "up", region)

def get_all_downstream(con, reach_id, region=None) -> set[int]:
    return _bfs_topology(con, reach_id, "down", region)

def _bfs_topology(con, start_id, direction, region=None) -> set[int]:
    # BFS via reach_topology WHERE direction=? AND neighbor_reach_id != 0
    # Returns visited set (excludes start_id)
```

Pure DuckDB queries, no NetworkX. Uses `collections.deque`.

**Step 4:** Run: `python -m pytest tests/sword_duckdb/test_spatial_utils.py -v`
Expected: PASS (all)

**Step 5:** Commit: `feat: add BFS get_all_upstream/downstream graph traversal`
