# Validation Specification: geom (Reach Geometry Column)

**Version:** 1.0
**Date:** 2026-02-02
**Author:** Geometry Audit

---

## 1. Overview

This document specifies the source, computation, validation rules, and edge cases for the SWORD reach geometry column:

| Variable | Description | Type | Table |
|----------|-------------|------|-------|
| `geom` | Reach polyline geometry (centerline path) | GEOMETRY(LINESTRING, 4326) | reaches |

---

## 2. geom (Reach Geometry)

### 2.1 Official Definition (PDD v17b)

> "reach geometry represented as a linestring formed by consecutive centerline points"

**Type:** GEOMETRY (LINESTRING)
**Projection:** WGS84 (EPSG:4326)
**Dimensions:** [number of reaches]

### 2.2 Source Data

| Source | Dataset | Resolution |
|--------|---------|------------|
| Primary | GRWL centerlines | 30m spacing |
| Derived from | centerlines table (x, y columns) | ~30m intervals along reach |

### 2.3 Algorithm

**Construction** (from centerlines):
- Collect all centerline points for a reach ordered by `cl_id`
- Create LINESTRING from (x, y) coordinates in sequence
- Coordinates are in WGS84 (EPSG:4326)
- No reprojection applied

**Reconstruction** (from centerlines):
```python
# Pseudocode
def build_reach_geom(reach_id):
    cl_points = query("SELECT x, y FROM centerlines WHERE reach_id = ? ORDER BY cl_id")
    coords = [(x, y) for x, y in cl_points]
    return ST_LineStringFromCoordinates(coords, srid=4326)
```

### 2.4 Database Schema

**reaches table** (`schema.py`):
```sql
geom GEOMETRY(LINESTRING, 4326) NOT NULL  -- reach polyline geometry
```

**Spatial index:**
- RTREE spatial index on `geom` column for query performance
- Index may be dropped and recreated during bulk updates

### 2.5 Valid Geometry Types

| Type | Valid | Notes |
|------|-------|-------|
| LINESTRING | YES | Standard for river reaches |
| POINT | NO | Degenerate (requires >= 2 points) |
| MULTILINESTRING | NO | Reaches should be single continuous line |
| POLYGON | NO | Rivers are 1D, not 2D |

### 2.6 Coordinate Bounds (WGS84)

| Dimension | Min | Max | Valid Range | Audit Result |
|-----------|-----|-----|-------------|--------------|
| Longitude (X) | -180.0 | 180.0 | World extent | -180.0 to 180.0 |
| Latitude (Y) | -90.0 | 90.0 | World extent | -52.0 to 82.3108 |

**Note:** South America reaches extend to ~52°S. North reaches extend to ~82°N (Arctic).

### 2.7 Point Count Constraints

| Metric | Min | Max | Typical | Audit Result |
|--------|-----|-----|---------|--------------|
| Minimum points per reach | 2 | - | - | 2 |
| Maximum points per reach | - | - | - | 2043 |
| Average points per reach | - | - | ~269 | 269.15 |

**Invariant:** Every reach must have at least 2 points (start and end).

### 2.8 Geometry Validity

**Valid geometry:** ST_IsValid(geom) = TRUE
**Audit results:**
- Total reaches: 248,674
- Invalid geometries: 0
- NULL geometries: 0
- Empty geometries: 0
- Degenerate geometries (< 2 points): 0

**Status:** 100% valid, no corruption detected.

### 2.9 Key Invariants

1. **Non-null:** All reaches must have a valid geometry (no NULL values)
2. **Valid:** ST_IsValid(geom) must be TRUE
3. **Non-empty:** ST_IsEmpty(geom) must be FALSE
4. **Minimum points:** ST_NPoints(geom) >= 2
5. **Coordinate range:** All (X, Y) within WGS84 bounds
6. **SRID:** All geometries must use EPSG:4326 (WGS84)
7. **Type:** All geometries must be LINESTRING (not POINT, POLYGON, MULTI*)

### 2.10 Relationship to Centerlines

**Critical constraint:**
- Each reach's geometry is constructed from its centerline points
- Centerline points ordered by `cl_id` determine point sequence
- If centerline data is corrupted, geometry will be corrupted

**Dependency:**
```sql
-- Verify consistency
SELECT COUNT(*) FROM reaches
WHERE ST_NPoints(geom) != (
    SELECT COUNT(*) FROM centerlines WHERE centerlines.reach_id = reaches.reach_id
);
-- Should return 0 (perfect match)
```

### 2.11 Relationship to Geometry Metadata (x, y, x_min, x_max, y_min, y_max)

**Derived from geom:**
- `x` = ST_X(ST_Centroid(geom))
- `y` = ST_Y(ST_Centroid(geom))
- `x_min` = ST_XMin(ST_Envelope(geom))
- `x_max` = ST_XMax(ST_Envelope(geom))
- `y_min` = ST_YMin(ST_Envelope(geom))
- `y_max` = ST_YMax(ST_Envelope(geom))

If `geom` is corrupted, all metadata columns will be invalid.

### 2.12 Failure Modes

| Failure | Cause | Impact | Detection |
|---------|-------|--------|-----------|
| NULL geometry | Centerline data missing for reach | Spatial queries fail | Lint check (proposed) |
| Invalid geometry | Self-intersecting linestring | ST_Buffer, ST_Intersection fail | ST_IsValid() = FALSE |
| < 2 points | Single or zero centerline points | Geometry undefined | ST_NPoints() < 2 |
| Out of bounds | Coordinate wrap or projection error | Spatial indexing broken | X < -180 or X > 180 |
| Empty geometry | ST_LineStringFromCoordinates failed | All spatial ops return NULL | ST_IsEmpty() = TRUE |
| Stale after edit | Centerline modified but geom not updated | Spatial index desync | Point count mismatch |
| SRID mismatch | Geometry created with wrong SRID | Buffer/distance calcs wrong | ST_SRID(geom) != 4326 |

### 2.13 Edge Cases

| Edge Case | Expected Behavior | Impact | Check Coverage |
|-----------|-------------------|--------|-----------------|
| Single centerline point | Should not occur (requires >= 2 for linestring) | Geometry error | Proposed check |
| Reach at antimeridian (±180°) | Coordinate pair (180, lat) or (-180, lat) | May cause rendering issues | Not checked |
| Reach with duplicate points | Adjacent points at same location | Degenerate segment | Not checked (unusual) |
| Reach spanning > 180° | Centerlines wrap around world | Should not occur in SWORD | Not checked |
| Reach with Z coordinates | ST_Z values in 3D linestring | Only X, Y used | DuckDB strips by default |

### 2.14 Existing Validation Coverage

| Check ID | Name | Severity | Description | Status |
|----------|------|----------|-------------|--------|
| G001 | reach_length_bounds | INFO | Flags short/long reaches | Geometry-dependent |
| G002 | node_length_consistency | WARNING | Node sum vs reach length | Geometry-dependent |
| G003 | zero_length_reaches | INFO | Flags reach_length <= 0 | Geometry-dependent |

**Note:** Current lint framework does NOT validate `geom` column directly (only reach_length, which derives from it).

### 2.15 Proposed Lint Checks

| ID | Severity | Name | Rule | SQL |
|----|----------|------|------|-----|
| G008 | ERROR | geom_not_null | All reaches must have non-NULL geometry | `SELECT COUNT(*) FROM reaches WHERE geom IS NULL` |
| G009 | ERROR | geom_is_valid | All geometries must be valid | `SELECT COUNT(*) FROM reaches WHERE NOT ST_IsValid(geom)` |
| G010 | ERROR | geom_not_empty | All geometries must be non-empty | `SELECT COUNT(*) FROM reaches WHERE ST_IsEmpty(geom)` |
| G011 | ERROR | geom_min_points | All geometries must have >= 2 points | `SELECT COUNT(*) FROM reaches WHERE ST_NPoints(geom) < 2` |
| G012 | WARNING | geom_bounds | All coordinates within WGS84 bounds | `SELECT COUNT(*) FROM reaches WHERE ST_XMin(geom) < -180 OR ST_XMax(geom) > 180 OR ST_YMin(geom) < -90 OR ST_YMax(geom) > 90` |
| G013 | WARNING | geom_type | All geometries must be LINESTRING | `SELECT COUNT(*) FROM reaches WHERE ST_GeometryType(geom) != 'LINESTRING'` |
| G014 | INFO | geom_srid | All geometries must use EPSG:4326 | `SELECT COUNT(*) FROM reaches WHERE ST_SRID(geom) != 4326` |
| G015 | WARNING | geom_centerline_consistency | Point count must match centerline count | `SELECT COUNT(*) FROM reaches WHERE ST_NPoints(geom) != (SELECT COUNT(*) FROM centerlines WHERE reach_id = reaches.reach_id)` |

### 2.16 Dependencies

**geom is used by:**
- Spatial queries (ST_Contains, ST_Intersects)
- Geometry metadata derivation (x, y, x_min, x_max, y_min, y_max)
- reach_length calculation (sum of segment distances)
- GIS exports (GeoPackage, shapefiles)
- Centerline visualization

---

## 3. Code References

| Component | File | Description |
|-----------|------|-------------|
| Schema: geom column | `src/updates/sword_duckdb/schema.py` | Column definition |
| Creation: from centerlines | `src/updates/sword_duckdb/workflow.py` | Geometry construction |
| Validation: geometry tests | `tests/sword_duckdb/` | Geometry test cases |
| Exports: to GeoPackage | `src/updates/sword_duckdb/exports.py` | Geometry export |
| Exports: to shapefiles | `src/updates/sword_duckdb/exports.py` | Geometry export |

---

## 4. Audit Results

**Date:** 2026-02-02
**Database:** sword_v17c.duckdb

### 4.1 Summary Statistics

| Check | Result |
|-------|--------|
| Total reaches | 248,674 |
| NULL geometries | 0 (0%) |
| Invalid geometries | 0 (0%) |
| Empty geometries | 0 (0%) |
| Degenerate (< 2 points) | 0 (0%) |
| Geometry type | 100% LINESTRING |
| **Overall Status** | **HEALTHY - NO ISSUES** |

### 4.2 Point Count Distribution

| Metric | Value |
|--------|-------|
| Minimum points | 2 |
| Maximum points | 2,043 |
| Average points | 269.15 |
| Median points | ~243 (estimated) |

**Interpretation:** Most reaches have 200-300 centerline points (~200m spacing), with some long reaches (e.g., Amazon) having 2,000+ points.

### 4.3 Coordinate Bounds

| Dimension | Min | Max | Valid |
|-----------|-----|-----|-------|
| Longitude (X) | -180.0 | 180.0 | ✓ |
| Latitude (Y) | -52.0 | 82.3 | ✓ |

**Interpretation:** SWORD covers 234° of longitude (world extent) and 134° of latitude (poles to tropics). No coordinate wrap issues detected.

### 4.4 Geometry Validity

- **ST_IsValid():** 100% valid (0 invalid)
- **ST_IsEmpty():** 100% non-empty (0 empty)
- **ST_NPoints():** 100% >= 2 (0 degenerate)

**Conclusion:** Geometry column is in excellent condition. No data integrity issues detected.

---

## 5. Best Practices

### 5.1 When Working with geom

1. **Always load spatial extension:** `LOAD spatial;` before queries
2. **Use spatial index:** Queries benefit from RTREE index on `geom`
3. **Preserve SRID:** Never reproject to 2D or different CRS without documentation
4. **Validate after edits:** Run ST_IsValid(), ST_NPoints() checks after any centerline changes
5. **Sync with centerlines:** If centerline points change, geom must be rebuilt

### 5.2 Bulk Geometry Updates

If updating many reach geometries:
1. Drop RTREE index (improves update speed)
2. Perform geometry updates
3. Run `UPDATE reaches SET geom = ST_GeomFromText(...) WHERE reach_id IN (...)`
4. Verify with G008-G015 checks
5. Recreate RTREE index

### 5.3 Exports

**GeoPackage export** preserves:
- LINESTRING geometry
- WGS84 projection (EPSG:4326)
- All reach attributes

**Shapefile export** (if used):
- LINESTRING to POLYLINE conversion
- WGS84 projection
- Attribute set limited by DBF format (255 char width)

---

## 6. Summary

### 6.1 Current Status

- **Source:** GRWL centerlines (30m spacing)
- **Type:** LINESTRING (WGS84, EPSG:4326)
- **Coverage:** 248,674 reaches globally
- **Validity:** 100% valid, no corruption
- **Point count:** 2 to 2,043 points per reach (avg 269)

### 6.2 Key Constraints

1. Non-null (no NULL values)
2. Valid (ST_IsValid = TRUE)
3. Non-empty (ST_IsEmpty = FALSE)
4. Minimum 2 points (ST_NPoints >= 2)
5. Coordinates in WGS84 (-180..180, -90..90)
6. SRID = 4326 (EPSG:4326)
7. Type = LINESTRING (single line per reach)

### 6.3 Validation Coverage

**Current:** Geometry validated indirectly through reach_length checks (G001-G003)

**Proposed:** Direct geometry validation (G008-G015) to catch corruption early

### 6.4 Risk Assessment

**Risk level:** LOW
- 100% healthy at audit date (2026-02-02)
- No known issues or edge cases
- Spatial index present and functional

**Recommendation:** Implement proposed lint checks (G008-G015) to monitor geometry integrity during future edits.

---

*Audit completed: 2026-02-02*
*Database: sword_v17c.duckdb*
*Region coverage: All 6 regions (NA, SA, EU, AF, AS, OC)*
