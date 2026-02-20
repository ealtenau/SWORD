# Validation Spec: width and slope

## Summary

| Field | width | slope |
|-------|-------|-------|
| **Source** | GRWL (Allen & Pavelsky, 2018) | Derived from WSE via linear regression |
| **Units** | meters | m/km |
| **Tables** | `centerlines`, `nodes`, `reaches` | `reaches` only |
| **Resolution** | 30m (centerlines) | Per-reach |

**Official width definition (v17b PDD, page 10):**
> "width: node average width" (units: meters)

> "width_var: width variance along the high-resolution centerline points used to calculate the average width for each node"

**Official reach width definition (v17b PDD, page 15):**
> "width: reach average width" (units: meters)

> "max_width: maximum width value across the channel for each reach that includes island and bar areas" (units: meters)

**Official slope definition (v17b PDD, page 16):**
> "slope: reach average slope calculated along the high-resolution centerline points" (units: m/km)

## Code Path

### Width

#### Centerline Width (Source)
- **Source:** GRWL dataset (Allen & Pavelsky, 2018)
- **Description:** River widths from Landsat at 30m resolution
- **RECONSTRUCTION_SPEC.md Section 5.2:**
```python
# Node level: median of centerline widths (from GRWL)
node_width = np.median(grwl_width[node_centerline_points])
node_width_var = np.var(grwl_width[node_centerline_points])

# Reach level: median of node widths
reach_width = np.median(node_width[reach_nodes])
reach_width_var = np.var(node_width[reach_nodes])
```

#### Node Width Reconstruction
- **Primary:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py:3527-3554`
- **Algorithm:**
  1. Query centerlines for each node
  2. Compute `MEDIAN(c.width)` where `width > 0`
  3. Update nodes table

```python
# node.width reconstruction (line 3545-3552)
SELECT
    c.node_id,
    MEDIAN(c.width) as width
FROM centerlines c
WHERE c.region = ? AND c.width > 0
GROUP BY c.node_id
```

#### Node Width Variance Reconstruction
- **Primary:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py:3556-3584`
- **Algorithm:** `VAR_SAMP(c.width)` from centerlines grouped by node_id

#### Reach Width Reconstruction
- **Primary:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py:1948-1971`
- **Algorithm:**
  1. Query all nodes for each reach
  2. Compute `MEDIAN(n.width)` grouped by reach_id
  3. Update reaches table

```python
# reach.width reconstruction (line 1964-1969)
SELECT n.reach_id, MEDIAN(n.width) as width
FROM nodes n
WHERE n.region = ?
GROUP BY n.reach_id
```

#### Reach Width Variance Reconstruction
- **Primary:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py:1973-1996`
- **Algorithm:** `VARIANCE(n.width)` grouped by reach_id

### Slope

#### Reach Slope Reconstruction
- **Primary:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py:1565-1662`
- **Algorithm:**
  1. Get nodes for each reach, ordered by `dist_out DESC` (upstream to downstream)
  2. Filter out NaN values in both `dist_out` and `wse`
  3. Convert distance to km: `dist_km = dist_valid / 1000.0`
  4. Linear least squares regression: `wse = slope * dist_km + intercept`
  5. Store absolute value of slope coefficient (m/km)

```python
# slope reconstruction (lines 1620-1631)
dist_km = dist_valid / 1000.0

# lstsq: solve Ax = b where A = [[dist, 1]], x = [[slope], [intercept]], b = wse
A = np.column_stack([dist_km, np.ones_like(dist_km)])
result, _, _, _ = np.linalg.lstsq(A, wse_valid, rcond=None)
slope = result[0]  # m/km (negative for downstream flow)

result_ids.append(reach_id)
result_values.append(abs(slope))  # Store absolute value
```

**Note:** The algorithm uses node-level data (not centerline-level), ordered by `dist_out DESC` to go from upstream to downstream within the reach.

## Schema Definition

**File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/schema.py`

### Nodes Table (lines 117-119)
```sql
width DOUBLE,                -- wth (m)
width_var DOUBLE,            -- wth_var (m^2)
max_width DOUBLE,            -- max_wth (m)
```

### Reaches Table (lines 215-218)
```sql
width DOUBLE,                -- wth (m)
width_var DOUBLE,            -- wth_var (m^2)
slope DOUBLE,                -- slope (m/km)
max_width DOUBLE,            -- max_wth (m)
```

### Additional Reach Columns (line 243)
```sql
low_slope_flag INTEGER,      -- low_slope: 1=too low for discharge estimation
```

### SWOT Observation Columns (v17c additions, lines 279-286)
```sql
-- Reaches
width_obs_mean DOUBLE,           -- mean observed width
width_obs_median DOUBLE,         -- median observed width
width_obs_std DOUBLE,            -- std dev of observed width
width_obs_range DOUBLE,          -- range (max-min) of observed width
slope_obs_mean DOUBLE,           -- mean observed slope
slope_obs_median DOUBLE,         -- median observed slope
slope_obs_std DOUBLE,            -- std dev of observed slope
slope_obs_range DOUBLE,          -- range (max-min) of observed slope
```

## Data Provenance

### Width: GRWL (Original Source)
- **Citation:** Allen, G. H., & Pavelsky, T. M. (2018). Global extent of rivers and streams. Science, 361(6402), 585-588.
- **Resolution:** 30m (Landsat-derived)
- **Methodology:** Water classification of Landsat imagery, centerline extraction, perpendicular width measurement
- **Coverage:** Rivers >= 30m wide globally
- **Temporal:** Composite of cloud-free scenes (primarily 2000-2015)

### Width Reconstruction Hierarchy
```
GRWL (30m Landsat widths)
    |
    v
centerlines.width  (30m resolution points from GRWL)
    |
    v
nodes.width = MEDIAN(centerline.width) for each ~200m node
    |
    v
reaches.width = MEDIAN(node.width) for each ~10km reach
```

### Slope: Derived from MERIT Hydro
- **Source:** WSE values from MERIT Hydro DEM
- **Methodology:** Linear regression of wse vs dist_out within each reach
- **Resolution:** Per-reach (typically ~10km)
- **Note:** Slope is NOT directly from MERIT Hydro slope products; it is computed from the wse profile

### Slope Derivation Hierarchy
```
MERIT Hydro DEM
    |
    v
centerlines.wse  (30m resolution)
    |
    v
nodes.wse = MEDIAN(centerline.wse)
    |
    v
reaches.slope = linreg(node.wse vs node.dist_out) converted to m/km
```

## Dependencies

### Width Upstream Dependencies
- `centerlines.width` - original GRWL widths
- `centerlines.node_id` - node assignment
- `nodes.reach_id` - reach assignment

### Width Downstream Dependencies
- `reach.max_width` - maximum width including islands
- `wth_coef` - RiverObs search window coefficient
- `ext_dist_coef` - extended distance coefficient (lake proximity)
- Discharge models (MOMMA, BAM, etc.) - use width for hydraulic calculations
- Sinuosity calculations - use width to determine meander wavelength

### Slope Upstream Dependencies
- `nodes.wse` - water surface elevation from MERIT Hydro
- `nodes.dist_out` - distance from outlet
- Requires >= 2 valid nodes per reach

### Slope Downstream Dependencies
- `low_slope_flag` - discharge estimation reliability flag
- Discharge models (MOMMA, BAM, etc.) - use slope directly in Manning's equation variants
- Flow routing calculations

## Failure Modes

### Width Failure Modes

| # | Mode | Description | Impact | Check |
|---|------|-------------|--------|-------|
| 1 | **width=1 placeholder** | Manually added nodes originally given width=1 | Unrealistically narrow | NEW |
| 2 | **Extreme width (>50km)** | Misclassified lake, multiple channels summed | Invalid, breaks models | A006 |
| 3 | **Width decreases dramatically downstream** | Tributary narrower than main channel | May indicate misclassification | A003 |
| 4 | **Wide headwaters (>500m)** | Missing upstream topology or lake | Suggests topology error | A008 |
| 5 | **Zero or negative width** | Data error or fill value | Invalid attribute | NEW |
| 6 | **Multi-channel width aggregation** | Single width doesn't capture braiding | Underestimates total channel area | INFO |
| 7 | **max_width >> width** | Islands/bars significantly widen channel | Consider for SWOT search window | INFO |
| 8 | **width_var extremely high** | Mixed water body types in node | Uncertain width estimate | NEW |

### Slope Failure Modes

| # | Mode | Description | Impact | Check |
|---|------|-------------|--------|-------|
| 1 | **Negative slope** | Physically impossible for steady flow | Indicates WSE or topology error | A002 |
| 2 | **Extreme slope (>100 m/km)** | Waterfall/error in WSE | Invalid for discharge | A002 |
| 3 | **Zero slope** | Lakes or extremely flat terrain | low_slope_flag should be set | INFO |
| 4 | **Insufficient nodes** | <2 valid nodes for regression | Cannot compute slope | NEW |
| 5 | **Noisy slope from short reach** | Few nodes + WSE noise = unstable regression | High uncertainty | NEW |
| 6 | **Slope inconsistent with facc** | High slope + high facc unusual | Possible topology error | NEW |
| 7 | **Slope derived from <3 nodes** | Low sample size for regression | High uncertainty in estimate | NEW |
| 8 | **Lakes with non-zero slope** | lakeflag=1 but slope > 0.01 m/km | Possible misclassification | NEW |

## Existing Lint Checks

### A002: slope_reasonableness
- **File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/lint/checks/attributes.py:20-80`
- **Severity:** WARNING
- **Rule:** Slope must be non-negative AND < 100 m/km
- **Scope:** Rivers only (lakeflag=0), excludes -9999 fill values
- **Assessment:** Well-designed, covers main failure modes 1 and 2

### A003: width_trend
- **File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/lint/checks/attributes.py:83-154`
- **Severity:** INFO
- **Rule:** Downstream width should be >= 30% of upstream width
- **Scope:** Rivers only (lakeflag=0), width > 100m
- **Assessment:** Good for detecting dramatic decreases, threshold may be too permissive

### A006: attribute_outliers
- **File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/lint/checks/attributes.py:297-354`
- **Severity:** INFO
- **Rule:** width < 50km, wse < 8000m, facc < 10M km²
- **Assessment:** Basic extreme value check; width limit reasonable

### A008: headwater_width
- **File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/lint/checks/attributes.py:414-469`
- **Severity:** WARNING
- **Rule:** Headwater rivers (n_rch_up=0, lakeflag=0) should be < 500m wide
- **Assessment:** Good topology validation check

## Proposed New Checks

| ID | Severity | Rule | Rationale |
|----|----------|------|-----------|
| A011 | WARNING | width > 0 AND width != 1 for non-ghost reaches | Catch placeholder values from manual additions |
| A012 | INFO | width_var / width < 2.0 for rivers | High relative variance suggests mixed features |
| A013 | INFO | Slope requires >= 2 nodes with valid wse and dist_out | Document coverage gaps |
| A014 | WARNING | Slope derived from >= 3 nodes has lower uncertainty | Flag low-confidence slopes |
| A015 | INFO | Lakes (lakeflag=1) should have slope < 0.01 m/km | Lakes should be flat |
| A016 | WARNING | Slope < 0.0001 m/km should trigger low_slope_flag=1 | Verify flag consistency |
| A017 | INFO | max_width / width ratio statistics | Document island/bar prevalence |
| A018 | INFO | SWOT width_obs_median within 50% of GRWL width | Cross-validate datasets (v17c) |
| A019 | INFO | SWOT slope_obs_median within 50% of derived slope | Cross-validate datasets (v17c) |

## Edge Cases

### 1. Lakes (lakeflag=1)
- **Width:** Lakes can be arbitrarily wide; max_width may be very large
- **Slope:** Should be near-zero; high slope suggests misclassification
- **Recommendation:** Exclude from width trend checks; add lake slope flatness check

### 2. Canals (lakeflag=2)
- **Width:** Typically uniform along length
- **Slope:** Controlled by engineering; may have locks/lifts
- **Recommendation:** Exclude from monotonicity checks; note that slope may be highly variable

### 3. Tidal Rivers (lakeflag=3)
- **Width:** May vary significantly with tide
- **Slope:** Effective slope varies with tidal cycle
- **Recommendation:** Apply wider tolerances; SWOT observations more relevant than MERIT

### 4. Manually Added Nodes (manual_add=1)
- **Width:** Originally set to 1, later updated to reach width
- **Issue:** May still have width=1 if not properly updated
- **Recommendation:** Check A011 specifically for manual_add=1 AND width=1

### 5. Multi-Channel Rivers (n_chan_max > 1)
- **Width:** Single-channel width underestimates total conveyance
- **Slope:** Should be consistent across channels (same water surface)
- **Recommendation:** Use max_width for SWOT search; report n_chan statistics

### 6. Dams/Waterfalls (obstr_type > 0, type=4)
- **Width:** May be anomalous at obstruction
- **Slope:** Artificially high at waterfall; flat upstream of dam
- **Recommendation:** Exclude from slope reasonableness if obstr_type=4

### 7. Ghost Reaches (type=6)
- **Width:** May be placeholder/invalid
- **Slope:** Undefined
- **Recommendation:** Exclude from all width/slope checks

### 8. Short Reaches (reach_length < 1km)
- **Width:** Few centerlines sampled
- **Slope:** Very few nodes for regression; high uncertainty
- **Recommendation:** Flag for higher uncertainty; may need special handling

### 9. End Reaches (end_reach=1,2)
- **Headwaters (end_reach=1):** Width check A008 already handles
- **Outlets (end_reach=2):** May be very wide (estuaries) or narrow (small streams)
- **Recommendation:** Report outlet width statistics separately

### 10. SWOT vs GRWL Width Differences
- **Temporal:** SWOT observes current conditions; GRWL is ~2000-2015 composite
- **Methodology:** SWOT measures water area and height; GRWL uses spectral classification
- **Spatial:** SWOT 50m-250m width requirement depending on product level
- **Recommendation:** Systematic comparison once SWOT data populated; expect 20-50% differences

### 11. Slope from Node-Level Data
- **Current implementation:** Uses node wse and dist_out, not centerline data
- **Implication:** ~40-50 points per reach (vs ~300+ centerline points)
- **Trade-off:** More robust to outliers but lower resolution
- **Recommendation:** Document this choice; consider centerline-level for v18

## Recommendations

### Short-term (v17c)
1. **Implement A011-A019 checks** - Especially A011 (width=1 placeholder) and A015 (lake slope)
2. **Validate SWOT consistency** - Once width_obs and slope_obs populated, compare to GRWL/derived
3. **Add slope confidence metric** - Based on number of nodes and R² of regression

### Medium-term
1. **Report n_nodes used for slope** - Store metadata about regression quality
2. **Add width confidence metric** - Based on width_var relative to width
3. **Regional width statistics** - Different expectations for arid vs humid climates

### Long-term (v18)
1. **Centerline-level slope** - Higher resolution regression using all centerline points
2. **Temporal width variability** - Integrate SWOT time series for seasonal width changes
3. **Multi-channel handling** - Better representation of braided/anastomosing rivers
4. **Uncertainty propagation** - Track width/slope uncertainty through to discharge

## References

1. Allen, G. H., & Pavelsky, T. M. (2018). Global extent of rivers and streams. Science, 361(6402), 585-588.

2. Yamazaki, D., Ikeshima, D., Sosa, J., Bates, P. D., Allen, G., & Pavelsky, T. (2019). MERIT Hydro: A high-resolution global hydrography map based on latest topography datasets. Water Resources Research. https://doi.org/10.1029/2019WR024873

3. Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., & Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite data products. Water Resources Research, 57, e2021WR030054.

4. SWORD v17b Product Description Document (March 2025)

5. SWORD RECONSTRUCTION_SPEC.md - Internal documentation of reconstruction algorithms

---

*Document version: 1.0*
*Created: 2026-02-02*
*Author: SWORD Validation Team*
