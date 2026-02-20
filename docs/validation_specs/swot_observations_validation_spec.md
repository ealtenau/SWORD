# Validation Spec: SWOT Observation Statistics

## Summary

| Field | Tables | Units | Source |
|-------|--------|-------|--------|
| **wse_obs_mean** | nodes, reaches | meters | SWOT L2 RiverSP |
| **wse_obs_median** | nodes, reaches | meters | SWOT L2 RiverSP |
| **wse_obs_std** | nodes, reaches | meters | SWOT L2 RiverSP |
| **wse_obs_range** | nodes, reaches | meters | SWOT L2 RiverSP |
| **width_obs_mean** | nodes, reaches | meters | SWOT L2 RiverSP |
| **width_obs_median** | nodes, reaches | meters | SWOT L2 RiverSP |
| **width_obs_std** | nodes, reaches | meters | SWOT L2 RiverSP |
| **width_obs_range** | nodes, reaches | meters | SWOT L2 RiverSP |
| **slope_obs_mean** | reaches only | m/m | SWOT L2 RiverSP (raw, may be negative) |
| **slope_obs_median** | reaches only | m/m | SWOT L2 RiverSP |
| **slope_obs_std** | reaches only | m/m | SWOT L2 RiverSP |
| **slope_obs_range** | reaches only | m/m | SWOT L2 RiverSP |
| **slope_obs_adj** | reaches only | m/m | GREATEST(slope_obs_mean, 0) |
| **slope_obs_reliable** | reaches only | boolean | TRUE if slope_obs_mean > noise floor |
| **n_obs** | nodes, reaches | count | SWOT L2 RiverSP |

**Note:** These are v17c additions - NOT documented in SWORD PDD v17b.

## Official Definition

These columns are **v17c additions** and do not appear in the SWORD v17b Product Description Document. They represent aggregated statistics from SWOT satellite observations.

**Intended definitions:**
- `*_obs_mean`: Arithmetic mean of SWOT observations
- `*_obs_median`: Median of SWOT observations
- `*_obs_std`: Sample standard deviation of SWOT observations
- `*_obs_range`: Max - Min of SWOT observations
- `n_obs`: Count of valid SWOT observations

## Source Data

### SWOT L2 RiverSP Products
- **Satellite:** Surface Water and Ocean Topography (SWOT)
- **Product Level:** Level 2 River Single Pass (RiverSP)
- **Format:** Parquet files
- **Location:** `/nas/cee-water/cjgleason/SWOT_grp/RiverSP_parquet/`
- **Temporal Coverage:** Mission start (2023) to present
- **Spatial Coverage:** Global rivers >= 100m wide

### SWOT Measurement Characteristics
| Parameter | SWOT Specification |
|-----------|-------------------|
| WSE accuracy | 10 cm (rivers > 100m wide) |
| WSE precision | ~1.7 cm/km along track |
| Width accuracy | ~20% for rivers 100-250m |
| Slope accuracy | ~1.7 cm/km |
| Minimum river width | 100m (nominal), 50m (threshold) |
| Repeat cycle | 21 days |

## Code Path

### Population Code
- **Primary:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/workflow.py:1622-1884`
- **Function:** `SWORDWorkflow.aggregate_swot_observations()`

#### Node-Level Aggregation (lines 1700-1750)
```python
# Query parquet files directly with DuckDB
node_agg_sql = f"""
SELECT
    CAST(node_id AS BIGINT) as node_id,
    AVG(wse) as wse_obs_mean,
    MEDIAN(wse) as wse_obs_median,
    CASE WHEN COUNT(*) > 1 THEN STDDEV_SAMP(wse) ELSE NULL END as wse_obs_std,
    MAX(wse) - MIN(wse) as wse_obs_range,
    AVG(width) as width_obs_mean,
    MEDIAN(width) as width_obs_median,
    CASE WHEN COUNT(*) > 1 THEN STDDEV_SAMP(width) ELSE NULL END as width_obs_std,
    MAX(width) - MIN(width) as width_obs_range,
    COUNT(*) as n_obs
FROM read_parquet('{glob_pattern}')
WHERE wse IS NOT NULL
  AND wse > -1000 AND wse < 10000
  AND width IS NOT NULL
  AND width > 0 AND width < 100000
  AND isfinite(wse) AND isfinite(width)
GROUP BY node_id
"""
```

#### Reach-Level Aggregation (lines 1800-1860)
```python
# Similar aggregation at reach level, includes slope
reach_agg_sql = f"""
SELECT
    CAST(reach_id AS BIGINT) as reach_id,
    AVG(wse) as wse_obs_mean,
    MEDIAN(wse) as wse_obs_median,
    CASE WHEN COUNT(*) > 1 THEN STDDEV_SAMP(wse) ELSE NULL END as wse_obs_std,
    MAX(wse) - MIN(wse) as wse_obs_range,
    AVG(width) as width_obs_mean,
    MEDIAN(width) as width_obs_median,
    CASE WHEN COUNT(*) > 1 THEN STDDEV_SAMP(width) ELSE NULL END as width_obs_std,
    MAX(width) - MIN(width) as width_obs_range,
    AVG(slope2) as slope_obs_mean,
    MEDIAN(slope2) as slope_obs_median,
    CASE WHEN COUNT(*) > 1 THEN STDDEV_SAMP(slope2) ELSE NULL END as slope_obs_std,
    MAX(slope2) - MIN(slope2) as slope_obs_range,
    COUNT(*) as n_obs
FROM read_parquet('{glob_pattern}')
WHERE wse IS NOT NULL
  AND wse > -1000 AND wse < 10000
  AND width IS NOT NULL
  AND width > 0 AND width < 100000
  AND slope2 IS NOT NULL
  AND isfinite(wse) AND isfinite(width) AND isfinite(slope2)
GROUP BY reach_id
"""
```

### SWOT Observation Aggregation (v17c pipeline)
- **File:** `/Users/jakegearon/projects/SWORD/src/sword_v17c_pipeline/reach_swot_obs.py`
- **Algorithm:** OLS regression (wse ~ p_dist_out) per cycle/pass for slope; direct aggregation for wse/width
- **Bounds checking:** width > 0 AND width < 100000, wse > -1000 AND wse < 10000 (prevents STDDEV_SAMP overflow)
- **Note:** Renamed from reach_slope_obs.py (2026-02-03) to reflect that it computes slope, wse, and width

## Schema Definition

**File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/schema.py`

### Nodes Table (lines 167-175)
```sql
wse_obs_mean DOUBLE,         -- mean observed WSE (m)
wse_obs_median DOUBLE,       -- median observed WSE (m)
wse_obs_std DOUBLE,          -- std dev of observed WSE (m)
wse_obs_range DOUBLE,        -- range (max-min) of observed WSE (m)
width_obs_mean DOUBLE,       -- mean observed width (m)
width_obs_median DOUBLE,     -- median observed width (m)
width_obs_std DOUBLE,        -- std dev of observed width (m)
width_obs_range DOUBLE,      -- range (max-min) of observed width (m)
n_obs INTEGER,               -- count of SWOT observations
```

### Reaches Table (lines 275-287)
```sql
-- Same as nodes, plus slope:
slope_obs_mean DOUBLE,       -- mean observed slope (m/km)
slope_obs_median DOUBLE,     -- median observed slope (m/km)
slope_obs_std DOUBLE,        -- std dev of observed slope (m/km)
slope_obs_range DOUBLE,      -- range (max-min) of observed slope (m/km)
```

## Population Statistics (v17c Database)

### Coverage by Region (Updated 2026-02-03)

| Region | Total | Slope | Slope % | WSE/Width | WSE % |
|--------|-------|-------|---------|-----------|-------|
| AF | 21,441 | 18,643 | 87.0% | 18,133 | 84.6% |
| AS | 100,185 | 79,749 | 79.6% | 83,208 | 83.1% |
| EU | 31,103 | 23,695 | 76.2% | 24,392 | 78.4% |
| NA | 38,696 | 31,873 | 82.4% | 26,241 | 67.8% |
| OC | 15,090 | 11,658 | 77.3% | 11,860 | 78.6% |
| SA | 42,159 | 36,325 | 86.2% | 34,912 | 82.8% |
| **Total** | **248,674** | **201,943** | **81.2%** | **198,746** | **79.9%** |

### n_obs Distribution

| Statistic | Value |
|-----------|-------|
| Min | 1 |
| 25th percentile | 39 |
| Median | 71 |
| Mean | 80 |
| 75th percentile | 110 |
| Max | 855 |

**Distribution by observation count:**
- 1-5 observations: 2.1%
- 6-10 observations: 1.8%
- 11-20 observations: 3.4%
- 21+ observations: 92.7%

## Valid Ranges

### Physical Constraints

| Variable | Min | Max | Rationale |
|----------|-----|-----|-----------|
| wse_obs_mean | -500 m | 8000 m | Dead Sea (-430m) to highest navigable rivers |
| wse_obs_std | 0 | 100 m | High std suggests mixed features or errors |
| wse_obs_range | 0 | 500 m | Extreme seasonal variation in large rivers |
| width_obs_mean | 50 m | 50000 m | SWOT minimum detectability to Amazon width |
| width_obs_std | 0 | 5000 m | High std suggests braided/varying channel |
| width_obs_range | 0 | 20000 m | Extreme seasonal/flood variation |
| slope_obs_mean | -0.1 | 100 m/km | **NOTE: Negative slopes physically impossible** |
| slope_obs_std | 0 | 10 m/km | High std suggests measurement noise |
| slope_obs_range | 0 | 50 m/km | Range of slope observations |
| n_obs | 1 | 1000 | Based on SWOT repeat cycle and mission duration |

### Observed Ranges (v17c Database)

| Variable | Observed Min | Observed Max | Issues |
|----------|--------------|--------------|--------|
| wse_obs_mean | -74.2 m | 5393.9 m | Within expected range |
| wse_obs_std | 0.0 m | 139.2 m | Some high values |
| wse_obs_range | 0.0 m | 6957.1 m | **EXTREME - investigate** |
| width_obs_mean | 50.0 m | 49854.6 m | Within expected range |
| width_obs_std | 0.0 m | 15782.4 m | Some very high values |
| width_obs_range | 0.0 m | 99844.0 m | **EXTREME - investigate** |
| slope_obs_mean | -0.23 m/km | 99.8 m/km | **24,129 negative values** |
| slope_obs_std | 0.0 m/km | 14.2 m/km | Within expected range |
| slope_obs_range | 0.0 m/km | 99.9 m/km | Within expected range |

## Relationship to Prior Values

### WSE: SWOT vs MERIT Hydro

| Metric | Value |
|--------|-------|
| Correlation | 0.9990 |
| Mean difference (SWOT - MERIT) | +1.25 m |
| Median difference | +0.72 m |
| Std dev of difference | 8.94 m |
| % within 5m | 85.3% |
| % within 10m | 93.6% |
| % differing > 10m | 6.4% |

**Assessment:** Excellent agreement. Mean offset of ~1.25m consistent with EGM96 vs EGM2008 datum difference.

### Width: SWOT vs GRWL

| Metric | Value |
|--------|-------|
| Correlation | 0.6423 |
| Mean difference (SWOT - GRWL) | +86.4 m |
| Median difference | +42.1 m |
| Std dev of difference | 342.7 m |
| % within 20% | 31.2% |
| % within 50% | 48.1% |
| % differing > 50% | 51.9% |

**Assessment:** Moderate correlation with systematic bias. SWOT measures wider than GRWL on average. Differences expected due to:
1. Temporal: SWOT (2023+) vs GRWL (~2000-2015)
2. Methodology: SWOT area-based vs GRWL spectral classification
3. Minimum width: SWOT 100m nominal vs GRWL 30m

### Slope: SWOT vs MERIT-Derived

| Metric | Value |
|--------|-------|
| Correlation | 0.0386 |
| Mean difference (SWOT - MERIT) | -0.012 m/km |
| % with negative SWOT slope | 13.4% (24,129 reaches) |
| % within 50% | 28.7% |

**Assessment:** **POOR correlation - major concern.** Possible causes:
1. SWOT slope from single-pass observations vs MERIT from DEM
2. Temporal variability in water surface slope
3. SWOT measurement noise in flat reaches
4. Different spatial scales (reach vs section)

## Failure Modes

### WSE Observation Failures

| # | Mode | Description | Impact | Detection |
|---|------|-------------|--------|-----------|
| 1 | **Extreme range** | wse_obs_range > 100m | Suggests data quality issues | A020 |
| 2 | **Large MERIT offset** | \|wse_obs - wse\| > 20m | Datum issue or temporal change | A021 |
| 3 | **Missing observations** | n_obs = NULL | No SWOT coverage | A022 |
| 4 | **Single observation** | n_obs = 1, std = NULL | No variability estimate | INFO |

### Width Observation Failures

| # | Mode | Description | Impact | Detection |
|---|------|-------------|--------|-----------|
| 1 | **Extreme range** | width_obs_range > 10000m | Data quality or mixed features | A023 |
| 2 | **Large GRWL divergence** | \|width_obs - width\| / width > 2.0 | Channel change or error | A024 |
| 3 | **Width < SWOT minimum** | width_obs_mean < 50m | Below detection threshold | A025 |
| 4 | **Extreme std** | width_obs_std > width_obs_mean | High relative variability | INFO |

### Slope Observation Failures

| # | Mode | Description | Impact | Detection |
|---|------|-------------|--------|-----------|
| 1 | **Negative slope** | slope_obs_mean < 0 | Physically impossible | **A026 (CRITICAL)** |
| 2 | **Extreme slope** | slope_obs_mean > 50 m/km | Waterfall or error | A027 |
| 3 | **High variability** | slope_obs_std > 5 m/km | Unreliable measurement | A028 |
| 4 | **Poor MERIT agreement** | Sign disagreement with derived slope | Temporal or method difference | A029 |

### n_obs Failures

| # | Mode | Description | Impact | Detection |
|---|------|-------------|--------|-----------|
| 1 | **Very few observations** | n_obs < 5 | Low statistical confidence | A030 |
| 2 | **Unexpectedly low** | n_obs < expected from orbit | Missing data | INFO |

## Existing Lint Checks

**None.** SWOT observation statistics currently have no dedicated lint checks.

Related existing checks:
- A002: slope_reasonableness (for MERIT-derived slope, not SWOT)
- A003: width_trend (for GRWL width, not SWOT)
- A006: attribute_outliers (does not include SWOT columns)

## Proposed Validation Checks

### High Priority (Implement for v17c)

| ID | Severity | Rule | Rationale |
|----|----------|------|-----------|
| A026 | **ERROR** | slope_obs_mean >= 0 | Negative slopes physically impossible; 24,129 violations |
| A027 | WARNING | slope_obs_mean < 50 m/km | Extreme slope suggests error |
| A021 | WARNING | \|wse_obs_median - wse\| < 20m | Validate SWOT-MERIT consistency |
| A024 | INFO | width_obs_median / width between 0.2 and 5.0 | Validate SWOT-GRWL consistency |

### Medium Priority

| ID | Severity | Rule | Rationale |
|----|----------|------|-----------|
| A020 | WARNING | wse_obs_range < 100m for rivers | Extreme range suggests data issues |
| A023 | WARNING | width_obs_range < 10000m | Extreme range suggests data issues |
| A028 | INFO | slope_obs_std < 5 m/km | High std suggests unreliable slope |
| A030 | INFO | n_obs >= 5 for statistical reliability | Flag low-confidence statistics |

### Low Priority / Informational

| ID | Severity | Rule | Rationale |
|----|----------|------|-----------|
| A031 | INFO | wse_obs_std < 50m for rivers | High std suggests variability |
| A032 | INFO | width_obs_std / width_obs_mean < 1.0 | Coefficient of variation |
| A033 | INFO | Report SWOT coverage by region | Monitoring metric |

## Edge Cases

### 1. Lakes (lakeflag=1)
- **WSE:** Should be relatively flat; low wse_obs_std expected
- **Width:** Can be very large; high width_obs_range acceptable
- **Slope:** Should be near-zero; non-zero slope is informational
- **Recommendation:** Apply lake-specific tolerances

### 2. Tidal Rivers (lakeflag=3)
- **WSE:** High variability expected due to tidal cycle
- **Width:** May vary significantly with tide
- **Slope:** Can be negative during flood tide
- **Recommendation:** Exclude from slope sign check; use wider tolerances

### 3. Canals (lakeflag=2)
- **WSE:** May have locks causing discontinuities
- **Width:** Typically uniform
- **Slope:** Controlled by engineering; may be flat or stepped
- **Recommendation:** Exclude from slope checks

### 4. Narrow Rivers (width < 100m GRWL, width_obs populated)
- **Issue:** SWOT may not reliably observe rivers below 100m
- **Impact:** Potential measurement bias
- **Recommendation:** Flag reaches where GRWL width < 100m but SWOT data exists

### 5. Seasonal Rivers
- **WSE:** High range expected
- **Width:** High range expected
- **Slope:** May vary with discharge
- **Recommendation:** Use wse_obs_std and width_obs_std as uncertainty indicators

### 6. Flat Reaches (slope < 0.01 m/km from MERIT)
- **Issue:** SWOT slope measurement noise dominates in flat reaches
- **Impact:** May produce negative or erratic slopes
- **Recommendation:** Apply wider tolerance for low-slope reaches

### 7. Short Reaches
- **Issue:** Fewer SWOT pixels per observation
- **Impact:** Higher noise in statistics
- **Recommendation:** Weight by n_obs in downstream analyses

## Known Issues (v17c)

### 1. Negative SWOT Slopes (RESOLVED)

**Scope:** 25,973 reaches (14.3% of reaches with SWOT data)

**Observed values:**
- Minimum slope_obs_mean: -0.23 m/m
- Distribution: Most between -0.05 and 0 m/m

**Root cause (investigated 2026-02-02):**
1. **Measurement noise** - SWOT slope accuracy is ~1.7 cm/km (0.000017 m/m). For very flat reaches (slope < this threshold), noise dominates, producing negative measurements.
2. **Temporal variability** - Backwater effects, tides, or local hydraulics can create temporarily negative water surface slopes in individual observations.
3. **Not a code bug** - The aggregation correctly averages SWOT observations; negative values are present in source data.

**Distribution by reach type:**
- Very flat reaches (<0.001 m/km MERIT slope): 31.6% negative
- Tidal reaches (lakeflag=3): 37.4% negative
- Lakes (lakeflag=1): 38.3% negative
- Rivers (lakeflag=0): 12.4% negative
- Steep reaches (>1 m/km): 7.0% negative

**Resolution (v17c):**
Two new columns added to handle this expected SWOT behavior:
- `slope_obs_adj = GREATEST(slope_obs_mean, 0.0)` - Clips negative slopes to 0
- `slope_obs_reliable = TRUE` if slope_obs_mean > 0.000017 (noise floor) AND std < |mean|

**Statistics:**
- 25,973 reaches clipped to 0 in slope_obs_adj
- 91,463 reaches (50.2%) marked as slope_obs_reliable = TRUE

**Recommendation for users:** Use `slope_obs_adj` for hydraulic calculations; check `slope_obs_reliable` flag; interpret low/negative slopes as "below SWOT detection limit" rather than actual reverse flow.

### 2. Low Slope Correlation

**Scope:** Correlation of 0.0386 between SWOT and MERIT-derived slopes

**Impact:** SWOT slopes may not be directly comparable to MERIT-derived slopes

**Possible causes:**
1. Different measurement methods (satellite vs DEM)
2. Different temporal conditions
3. Different spatial scales
4. SWOT captures hydraulic slope; MERIT captures geometric slope

**Recommendation:** Document as known limitation; consider separate validation approach for SWOT slopes.

### 3. SWOT-GRWL Width Divergence

**Scope:** 52% of reaches have width difference > 50%

**Impact:** Inconsistent width values between prior and observed

**Likely causes:**
1. Temporal change (decades between datasets)
2. Methodological differences
3. SWOT minimum width constraint
4. Seasonal differences in observations

**Recommendation:** Document as expected; use ratio as informational metric.

### 4. Extreme Range Values

**Observed extremes:**
- wse_obs_range: 6957.1 m (physically implausible)
- width_obs_range: 99844.0 m (possible for wide floodplains but extreme)

**Recommendation:** Investigate specific reaches; may indicate data quality issues.

## Recommendations

### Short-term (v17c Release)

1. **Implement A026 check** - Flag negative SWOT slopes as ERROR
2. **Implement A021, A024 checks** - Validate SWOT-MERIT and SWOT-GRWL consistency
3. **Create GitHub issue** for negative slope investigation
4. **Document SWOT limitations** in release notes:
   - Low correlation with MERIT slopes
   - Known negative slope issue
   - Width divergence from GRWL expected

### Medium-term

1. **Improve slope aggregation** - Consider filtering outliers before mean/median
2. **Add confidence metrics** - n_obs-weighted uncertainty estimates
3. **Regional analysis** - Different tolerances for arid vs humid climates
4. **Tidal river handling** - Separate validation rules for lakeflag=3

### Long-term (v18)

1. **SWOT as primary source** - Migrate from MERIT to SWOT for WSE/slope
2. **Time series statistics** - Min/max/seasonal amplitude
3. **Uncertainty propagation** - Track SWOT uncertainty through to discharge
4. **Multi-mission fusion** - Combine SWOT with ICESat-2, Sentinel-3

## References

1. Durand, M., et al. (2023). The Surface Water and Ocean Topography (SWOT) Mission. IEEE Transactions on Geoscience and Remote Sensing.

2. Frasson, R. P. d. M., et al. (2019). Global relationships between river width, slope, catchment area, meander wavelength, sinuosity, and discharge. Geophysical Research Letters.

3. Allen, G. H., & Pavelsky, T. M. (2018). Global extent of rivers and streams. Science, 361(6402), 585-588.

4. Yamazaki, D., et al. (2019). MERIT Hydro: A high-resolution global hydrography map. Water Resources Research.

5. SWOT Product Description Document (2023). JPL D-56411.

---

*Document version: 1.0*
*Created: 2026-02-02*
*Author: SWORD Validation Team*
