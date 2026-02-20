# Validation Spec: wse (Water Surface Elevation)

## Summary

| Field | Value |
|-------|-------|
| **Source** | MERIT Hydro DEM (Yamazaki et al., 2019) |
| **Units** | meters |
| **Datum** | EGM96 geoid (via MERIT Hydro) |
| **Resolution** | 3 arc-seconds (~90m at equator) |
| **Tables** | `centerlines`, `nodes`, `reaches` |

**Official definition (v17b PDD, page 10):**
> "wse: node average water surface elevation" (units: meters)

> "wse_var: water surface elevation variance along the high-resolution centerline points used to calculate the average water surface elevation for each node"

**Reach-level definition (v17b PDD, page 15):**
> "wse: reach average water surface elevation" (units: meters)

> "wse_var: water surface elevation variance along the high-resolution centerline points used to calculate the average water surface elevation for each reach"

## Code Path

### Node WSE Reconstruction
- **Primary:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py:3366-3398`
- **Algorithm:**
  1. Uses existing node wse if available
  2. Falls back to reach wse (`COALESCE(n.wse, r.wse)`)
  3. Full reconstruction requires MERIT Hydro source data (not implemented)

```python
# node.wse reconstruction (simplified)
SELECT n.node_id, COALESCE(n.wse, r.wse) as wse
FROM nodes n
JOIN reaches r ON n.reach_id = r.reach_id
```

### Reach WSE Reconstruction
- **Primary:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py:1499-1563`
- **Algorithm:**
  1. Query all nodes for each reach
  2. Compute `MEDIAN(n.wse)` grouped by reach_id
  3. Update reaches table

```python
# reach.wse reconstruction
SELECT reach_id, MEDIAN(n.wse) as median_wse
FROM nodes n
WHERE n.region = ?
GROUP BY n.reach_id
```

### Reach Slope (derived from WSE)
- **Primary:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py:1565-1656`
- **Algorithm:**
  1. Get nodes ordered by dist_out (descending)
  2. Filter out NaN values
  3. Linear regression: `wse = slope * dist + intercept`
  4. Convert to m/km, store absolute value

```python
# slope = linear_regression(dist_out_km, wse)
dist_km = dist_valid / 1000.0
A = np.column_stack([dist_km, np.ones_like(dist_km)])
result, _, _, _ = np.linalg.lstsq(A, wse_valid, rcond=None)
slope = abs(result[0])  # m/km
```

### WSE Variance Reconstruction
- **Primary:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py:1923-1946`
- **Algorithm:** `VARIANCE(n.wse)` grouped by reach_id

### Node WSE Variance Reconstruction
- **Primary:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py:3586-3614`
- **Algorithm:** `VAR_SAMP(c.wse)` from centerlines grouped by node_id

## Schema Definition

**File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/schema.py`

### Nodes Table (lines 115-116)
```sql
wse DOUBLE,                  -- water surface elevation (m)
wse_var DOUBLE,              -- wse variance (m^2)
```

### Reaches Table (lines 213-214)
```sql
wse DOUBLE,                  -- water surface elevation (m)
wse_var DOUBLE,              -- wse variance (m^2)
```

### SWOT Observation Columns (v17c additions, lines 167-170, 275-278)
```sql
-- Nodes
wse_obs_mean DOUBLE,         -- mean observed WSE
wse_obs_median DOUBLE,       -- median observed WSE
wse_obs_std DOUBLE,          -- std dev of observed WSE
wse_obs_range DOUBLE,        -- range (max-min) of observed WSE

-- Reaches (same columns)
```

## Data Provenance

### MERIT Hydro (Original Source)
- **Citation:** Yamazaki et al. (2019), Water Resources Research
- **Resolution:** 3 arc-second (~90m)
- **Vertical datum:** EGM96 geoid
- **Derivation:** Multi-error-removed improved-terrain (MERIT) DEM adjusted for river surface

### Reconstruction Hierarchy
```
MERIT Hydro DEM
    |
    v
centerlines.wse  (30m resolution points, sampled from MERIT)
    |
    v
nodes.wse = MEDIAN(centerline.wse) for each ~200m node
    |
    v
reaches.wse = MEDIAN(node.wse) for each ~10km reach
    |
    v
reaches.slope = linreg(node.wse vs node.dist_out)
```

## Dependencies

### Upstream Dependencies (wse depends on)
- `centerline.x`, `centerline.y` - coordinates for MERIT sampling
- `centerline.node_id` - node assignment
- `node.reach_id` - reach assignment

### Downstream Dependencies (attributes derived from wse)
- `reach.slope` - linear regression of wse vs dist_out
- `reach.wse_var` - variance of node wse values
- `node.wse_var` - variance of centerline wse values
- Discharge models (MOMMA, BAM, etc.) - use wse for hydraulic calculations

## Failure Modes

| # | Mode | Description | Impact | Check |
|---|------|-------------|--------|-------|
| 1 | **WSE increases downstream** | DEM error or flow direction wrong | Breaks slope calculation, discharge estimation | A001 |
| 2 | **Negative or extreme WSE** | DEM artifacts, ocean/below sea level | Invalid elevation, -9999 fill values | A006 |
| 3 | **High wse_var in rivers** | Noisy DEM in narrow channels | Unreliable slope estimates | NEW |
| 4 | **Lake WSE not flat** | Improperly classified lakeflag | Violates expected flat surface | NEW |
| 5 | **SWOT vs MERIT mismatch** | Different temporal conditions, datums | Confusion in downstream analysis | NEW |
| 6 | **Missing wse values** | MERIT gaps (typically ocean/endorheic) | Breaks dependent calculations | A004 |
| 7 | **WSE jump at confluences** | Backwater effects or DEM seams | Topology appears incorrect | NEW |
| 8 | **Extreme slope from wse** | Short reaches with small elevation changes | Amplified noise | A002 |

## Proposed Checks

| ID | Severity | Rule | Rationale |
|----|----------|------|-----------|
| A001 | ERROR | WSE must decrease downstream (tolerance: 0.5m) | Physical constraint: water flows downhill. Existing check. |
| A006 | INFO | WSE must be <8000m | No rivers above this elevation. Existing check. |
| A007 | WARNING | WSE must be >-500m | Dead Sea lowest at -430m; allow margin. Catches -9999 fill. |
| A008 | INFO | wse_var should be <100 m^2 for rivers (lakeflag=0) | High variance indicates DEM noise or misclassification. |
| A009 | INFO | wse_var should be <10 m^2 for lakes (lakeflag=1) | Lakes should have flat water surface. |
| A010 | WARNING | WSE should not increase by >10m at confluence | Large jump suggests DEM seam or topology error. |
| A011 | INFO | SWOT wse_obs_mean should be within 20m of MERIT wse | Validates SWOT-MERIT consistency (v17c only). |
| A012 | WARNING | Slope derived from wse should match stored slope (10% tolerance) | Internal consistency check. |

## Edge Cases

### 1. Lakes (lakeflag=1)
- **Expected behavior:** Near-flat WSE (low variance)
- **Reality:** MERIT reflects one temporal snapshot; actual lake levels vary seasonally
- **Recommendation:** Apply looser WSE monotonicity checks for lakes; focus on wse_var instead

### 2. Tidal Rivers (lakeflag=3)
- **Expected behavior:** WSE varies with tide; may not be monotonically decreasing
- **Reality:** MERIT is a composite; SWORD wse is effectively mean tide
- **Recommendation:** Exclude lakeflag=3 from monotonicity checks (already done in A001)

### 3. DEM Errors
- **Types:** Striping, cloud artifacts, voids, datum shifts
- **Detection:** High wse_var, sudden jumps, values clustering at round numbers
- **Recommendation:** Flag reaches where wse_var > mean + 3*std for the region

### 4. Canals (lakeflag=2)
- **Expected behavior:** May have elevation control structures; not naturally monotonic
- **Reality:** Can flow uphill relative to natural terrain
- **Recommendation:** Exclude from monotonicity checks; validate separately

### 5. End Reaches (end_reach=1,2)
- **Headwaters (end_reach=1):** No upstream reference; wse error cannot be detected via monotonicity
- **Outlets (end_reach=2):** May be at sea level or lake level; special handling needed

### 6. SWOT vs MERIT Differences
- **Temporal:** SWOT observes current conditions; MERIT is ~2000-2015 composite
- **Datum:** SWOT uses EGM2008; MERIT uses EGM96 (differences up to ~1m)
- **Spatial:** SWOT 250m+ width requirement vs MERIT 90m resolution
- **Recommendation:** Systematic offset comparison, not direct equality check

### 7. Ghost Reaches (type=6)
- **Expected behavior:** Placeholder reaches with potentially invalid attributes
- **Reality:** May have wse=-9999 or copied from nearby reaches
- **Recommendation:** Exclude from all wse validation checks

### 8. Unreliable Topology (type=5)
- **Expected behavior:** Flow direction uncertain
- **Reality:** WSE monotonicity may be reversed
- **Recommendation:** Flag but don't fail; these are known issues

## Recommendations

### Short-term (v17c)
1. **Implement A007-A012 checks** - See proposed checks table above
2. **Add lakeflag-aware validation** - Different tolerances for rivers vs lakes
3. **Validate SWOT-MERIT consistency** - Once wse_obs_mean is populated

### Medium-term
1. **Improve reconstruction** - Integrate actual MERIT Hydro tiles for true re-sampling
2. **Add confidence scores** - Based on wse_var and source quality
3. **Regional calibration** - Different tolerances for mountainous vs flat regions

### Long-term (v18)
1. **SWOT as primary source** - Replace MERIT wse with SWOT observations where available
2. **Temporal variability** - Store min/max/mean from SWOT time series
3. **Uncertainty propagation** - Track wse uncertainty through to slope and discharge

## Existing Check (A001) Analysis

**Location:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/lint/checks/attributes.py:20-90`

**Current implementation:**
- Compares upstream wse to downstream neighbor wse via reach_topology
- Tolerance: 0.5m (configurable)
- Excludes: lakes (lakeflag=0 only), invalid values (-9999, <=0)
- Severity: ERROR

**Strengths:**
- Correctly joins through topology table
- Excludes non-river features
- Returns detailed issue list with coordinates

**Gaps:**
- Does not check within-reach node monotonicity
- No special handling for type=5 (unreliable topology)
- Hard-coded river_name which may be NULL

**Proposed enhancement:**
```python
# Add node-level monotonicity check within reaches
WITH node_pairs AS (
    SELECT n1.node_id, n1.wse as wse_up, n2.wse as wse_down
    FROM nodes n1
    JOIN nodes n2 ON n1.reach_id = n2.reach_id
        AND n2.dist_out < n1.dist_out  -- n2 is downstream
        AND n2.dist_out = (SELECT MAX(dist_out) FROM nodes
                          WHERE reach_id = n1.reach_id
                          AND dist_out < n1.dist_out)
    WHERE n1.lakeflag = 0
)
SELECT * FROM node_pairs WHERE wse_down > wse_up + 0.5
```

## References

1. Yamazaki, D., Ikeshima, D., Sosa, J., Bates, P. D., Allen, G., & Pavelsky, T. (2019). MERIT Hydro: A high-resolution global hydrography map based on latest topography datasets. Water Resources Research. https://doi.org/10.1029/2019WR024873

2. Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., & Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite data products. Water Resources Research, 57, e2021WR030054.

3. SWORD v17b Product Description Document (March 2025)

---

*Document version: 1.0*
*Created: 2026-02-02*
*Author: SWORD Validation Team*
