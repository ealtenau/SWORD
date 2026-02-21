# Validation Spec: SWOT Observation Statistics

## Summary

| Field | Tables | Units | Source |
|-------|--------|-------|--------|
| **{var}_obs_p10–p90** | nodes (wse, width), reaches (wse, width, slope) | m or m/m | SWOT L2 RiverSP percentiles |
| **{var}_obs_range** | nodes, reaches | m or m/m | max - min |
| **{var}_obs_mad** | nodes, reaches | m or m/m | median absolute deviation |
| **slope_obs_adj** | reaches only | m/m | GREATEST(slope_obs_p50, 0) |
| **slope_obs_slopeF** | reaches only | unitless | weighted sign fraction (-1 to +1) |
| **slope_obs_reliable** | reaches only | boolean | \|slopeF\| > 0.5 AND \|p50\| > 0.000017 |
| **slope_obs_quality** | reaches only | category | reliable / below_ref_uncertainty / high_uncertainty / negative |
| **n_obs** | nodes, reaches | count | SWOT L2 RiverSP |

**Note:** These are v17c additions — NOT documented in SWORD PDD v17b.

### Column Counts

- **Nodes**: wse(11) + width(11) + n_obs(1) = **23 columns**
- **Reaches**: wse(11) + width(11) + slope(11) + derived(4) + n_obs(1) = **38 columns**

### Per-Variable Columns (wse, width, slope)

Each variable gets 11 columns: p10, p20, p30, p40, p50, p60, p70, p80, p90, range, mad.

### Removed Columns (replaced)

| Old Column | Replacement | Rationale |
|------------|-------------|-----------|
| `*_obs_mean` | `*_obs_p50` | Median more robust to outliers |
| `*_obs_median` | `*_obs_p50` | Same value, consistent naming |
| `*_obs_std` | `*_obs_mad` | MAD more robust to outliers |

## Official Definition

These columns are **v17c additions** and do not appear in the SWORD v17b Product Description Document. They represent aggregated statistics from SWOT satellite observations.

**Definitions:**
- `*_obs_pNN`: NN-th percentile of filtered SWOT observations (QUANTILE_CONT)
- `*_obs_range`: max - min of filtered observations
- `*_obs_mad`: median absolute deviation = MEDIAN(|x - MEDIAN(x)|)
- `slope_obs_adj`: GREATEST(slope_obs_p50, 0) — clips negative medians to 0
- `slope_obs_slopeF`: SUM(weight * SIGN(slope)) / SUM(weight), weighted by n_good_nod
- `slope_obs_reliable`: |slopeF| > 0.5 AND |p50| > SLOPE_REF_UNCERTAINTY
- `slope_obs_quality`: categorical label based on p50 and slopeF
- `n_obs`: count of valid (post-filter) SWOT observations

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

## Filter Constants

All filter thresholds are defined in `src/sword_duckdb/swot_filters.py` (single source of truth).

### Value Range Filters

| Constant | Value | Description |
|----------|-------|-------------|
| `WSE_MIN` | -1000 m | Dead Sea ~-430 m |
| `WSE_MAX` | 10000 m | Highest navigable ~5000 m |
| `WIDTH_MIN` | 0 m | Exclusive (width > 0) |
| `WIDTH_MAX` | 100000 m | 100 km, generous |
| `SLOPE_MIN` | -1 m/m | = -1000 m/km, garbage filter |
| `SLOPE_MAX` | 1 m/m | = 1000 m/km, garbage filter |

### Quality Thresholds

| Constant | Value | Description |
|----------|-------|-------------|
| `SENTINEL` | -999999999999 | SWOT fill value |
| `DARK_FRAC_MAX` | 0.5 | Maximum dark water fraction |
| `XTRACK_MIN` | 10000 m | Minimum cross-track distance |
| `XTRACK_MAX` | 60000 m | Maximum cross-track distance |
| `QUALITY_MAX` | 1 | 0=good, 1=suspect |
| `SLOPE_REF_UNCERTAINTY` | 0.000017 m/m | ~17 mm/km, pragmatic SNR~1 threshold |

### Node Filters (build_node_filter_sql)

Applied when aggregating node observations:
- WSE not NULL, not sentinel, in (WSE_MIN, WSE_MAX), finite
- Width not NULL, not sentinel, in (WIDTH_MIN, WIDTH_MAX), finite
- wse_q or wse_sm_q <= QUALITY_MAX (if present)
- dark_frac <= DARK_FRAC_MAX (if present)
- |xtrk_dist| between XTRACK_MIN and XTRACK_MAX (if present)
- xovr_cal_q <= QUALITY_MAX (if present)
- ice_clim_f = 0 (if present)
- time_str not NULL/empty (if present)

### Reach Filters (build_reach_filter_sql)

Applied when aggregating reach observations:
- WSE, width, slope: not NULL, not sentinel, in range, finite
- reach_q <= QUALITY_MAX (if present)
- dark_frac <= DARK_FRAC_MAX (if present)
- xovr_cal_q <= QUALITY_MAX (if present)
- ice_clim_f = 0 (if present)

## Code Path

### Population Code
- **Filter constants:** `src/sword_duckdb/swot_filters.py`
- **Schema migration:** `src/sword_duckdb/schema.py` → `add_swot_obs_columns()`
- **Aggregation:** `src/sword_duckdb/workflow.py`
  - `aggregate_swot_observations()` — entry point
  - `_aggregate_node_observations()` — node-level percentiles
  - `_aggregate_reach_observations()` — reach-level percentiles + slope derived

### Node-Level Aggregation

Two-pass SQL strategy:
1. **filtered CTE**: read parquet with WHERE clause from `build_node_filter_sql()`
2. **medians CTE**: compute `QUANTILE_CONT(wse, 0.5)` and `QUANTILE_CONT(width, 0.5)` per node
3. **Main query**: JOIN filtered × medians → compute all percentiles, range, MAD

```sql
WITH filtered AS (
    SELECT node_id, wse, width
    FROM read_parquet(glob)
    WHERE {node_filter_where}
),
medians AS (
    SELECT node_id,
        QUANTILE_CONT(wse, 0.5) as med_wse,
        QUANTILE_CONT(width, 0.5) as med_width
    FROM filtered GROUP BY node_id
)
SELECT f.node_id,
    QUANTILE_CONT(f.wse, 0.1) as wse_obs_p10,
    ...
    QUANTILE_CONT(f.wse, 0.9) as wse_obs_p90,
    MAX(f.wse) - MIN(f.wse) as wse_obs_range,
    MEDIAN(ABS(f.wse - m.med_wse)) as wse_obs_mad,
    -- same for width --
    COUNT(*) as n_obs
FROM filtered f JOIN medians m ON f.node_id = m.node_id
GROUP BY f.node_id
```

### Reach-Level Aggregation

Three-pass SQL strategy (extends node pattern):
1. **filtered CTE**: read parquet with WHERE clause from `build_reach_filter_sql()`
2. **medians CTE**: compute p50 for wse, width, slope
3. **Main query**: all percentiles + range + MAD + weighted slopeF + derived columns

```sql
-- Slope slopeF: weighted sign fraction
SUM(COALESCE(n_good_nod, 1) * SIGN(slope))
    / SUM(COALESCE(n_good_nod, 1)) as slope_obs_slopeF

-- Derived from p50
GREATEST(QUANTILE_CONT(slope, 0.5), 0.0) as slope_obs_adj

-- slope_obs_quality: CASE expression
CASE
    WHEN p50 < -ref_uncertainty AND |slopeF| > 0.5 THEN 'negative'
    WHEN |p50| <= ref_uncertainty THEN 'below_ref_uncertainty'
    WHEN |slopeF| <= 0.5 THEN 'high_uncertainty'
    ELSE 'reliable'
END
```

### Slope Quality Categories

| Category | Condition | Interpretation |
|----------|-----------|----------------|
| `reliable` | \|slopeF\| > 0.5 AND \|p50\| > ref_uncertainty | High confidence |
| `below_ref_uncertainty` | \|p50\| <= ref_uncertainty | Signal below noise floor |
| `high_uncertainty` | \|slopeF\| <= 0.5 AND \|p50\| > ref_uncertainty | Inconsistent sign |
| `negative` | p50 < -ref_uncertainty AND \|slopeF\| > 0.5 | Consistently negative |

Where `ref_uncertainty` = SLOPE_REF_UNCERTAINTY (0.000017 m/m ≈ 17 mm/km). This is a
pragmatic SNR~1 threshold, not a universal detectability limit. Actual uncertainty
scales with reach length, width, node count, and cross-track position.

### SWOT Observation Aggregation (v17c pipeline)
- **File:** `src/sword_v17c_pipeline/reach_swot_obs.py`
- **Algorithm:** OLS regression (wse ~ p_dist_out) per cycle/pass for slope; direct aggregation for wse/width
- **Filter constants:** Imported from `src/sword_duckdb/swot_filters.py`

## Schema Definition

**File:** `src/sword_duckdb/schema.py`

### Nodes Table (23 SWOT columns)
```sql
-- Per percentile: wse_obs_p10, wse_obs_p20, ..., wse_obs_p90  (9 cols)
-- wse_obs_range DOUBLE, wse_obs_mad DOUBLE                     (2 cols)
-- width_obs_p10, width_obs_p20, ..., width_obs_p90              (9 cols)
-- width_obs_range DOUBLE, width_obs_mad DOUBLE                  (2 cols)
-- n_obs INTEGER                                                  (1 col)
```

### Reaches Table (38 SWOT columns)
```sql
-- Same as nodes (without n_obs): 22 cols
-- slope_obs_p10, slope_obs_p20, ..., slope_obs_p90              (9 cols)
-- slope_obs_range DOUBLE, slope_obs_mad DOUBLE                  (2 cols)
-- slope_obs_adj DOUBLE         -- GREATEST(p50, 0)
-- slope_obs_slopeF DOUBLE      -- weighted sign fraction
-- slope_obs_reliable BOOLEAN   -- quality flag
-- slope_obs_quality VARCHAR    -- categorical label
-- n_obs INTEGER
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
| wse_obs_p50 | -500 m | 8000 m | Dead Sea (-430m) to highest navigable rivers |
| wse_obs_mad | 0 | 100 m | High MAD suggests mixed features or errors |
| wse_obs_range | 0 | 500 m | Extreme seasonal variation in large rivers |
| width_obs_p50 | 50 m | 50000 m | SWOT minimum detectability to Amazon width |
| width_obs_mad | 0 | 5000 m | High MAD suggests braided/varying channel |
| width_obs_range | 0 | 20000 m | Extreme seasonal/flood variation |
| slope_obs_p50 | -0.1 m/m | 0.1 m/m | NOTE: Negative p50 possible from noise |
| slope_obs_mad | 0 | 0.01 m/m | High MAD suggests measurement noise |
| slope_obs_range | 0 | 0.05 m/m | Range of slope observations |
| slope_obs_slopeF | -1 | +1 | Weighted sign fraction |
| n_obs | 1 | 1000 | Based on SWOT repeat cycle and mission duration |

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
| % with negative SWOT slope p50 | ~13% |

**Assessment:** **POOR correlation - expected.** Causes:
1. SWOT slope from single-pass observations vs MERIT from DEM
2. Temporal variability in water surface slope
3. SWOT measurement noise in flat reaches
4. Different spatial scales (reach vs section)

## Failure Modes

### WSE Observation Failures

| # | Mode | Description | Detection |
|---|------|-------------|-----------|
| 1 | Extreme range | wse_obs_range > 100m | A020 |
| 2 | Large MERIT offset | \|wse_obs_p50 - wse\| > 20m | A021 |
| 3 | Missing observations | n_obs = NULL | A022 |
| 4 | Single observation | n_obs = 1, mad = 0 | INFO |

### Width Observation Failures

| # | Mode | Description | Detection |
|---|------|-------------|-----------|
| 1 | Extreme range | width_obs_range > 10000m | A023 |
| 2 | Large GRWL divergence | width_obs_p50 / width > 5.0 | A024 |
| 3 | Width < SWOT minimum | width_obs_p50 < 50m | A025 |
| 4 | Extreme MAD | width_obs_mad > width_obs_p50 | INFO |

### Slope Observation Failures

| # | Mode | Description | Detection |
|---|------|-------------|-----------|
| 1 | Negative p50 | slope_obs_p50 < 0 | A026 |
| 2 | Extreme slope | slope_obs_p50 > 0.05 m/m | A027 |
| 3 | High uncertainty | slope_obs_quality = 'high_uncertainty' | INFO |
| 4 | Below noise | slope_obs_quality = 'below_ref_uncertainty' | INFO |

### n_obs Failures

| # | Mode | Description | Detection |
|---|------|-------------|-----------|
| 1 | Very few observations | n_obs < 5 | A030 |
| 2 | Unexpectedly low | n_obs < expected from orbit | INFO |

## Existing Lint Checks

| ID | Severity | Rule | Column |
|----|----------|------|--------|
| A021 | WARNING | \|wse_obs_p50 - wse\| < 20m | wse_obs_p50 |
| A024 | INFO | width_obs_p50 / width between 0.2 and 5.0 | width_obs_p50 |
| A026 | ERROR | slope_obs_p50 >= 0 | slope_obs_p50 |
| A027 | WARNING | slope_obs_p50 < 50 m/km | slope_obs_p50 |
| FL001 | INFO | SWOT observation coverage statistics | n_obs |

## Proposed Additional Checks

### Medium Priority

| ID | Severity | Rule | Rationale |
|----|----------|------|-----------|
| A020 | WARNING | wse_obs_range < 100m for rivers | Extreme range suggests data issues |
| A023 | WARNING | width_obs_range < 10000m | Extreme range suggests data issues |
| A028 | INFO | slope_obs_mad < 0.005 m/m | High MAD suggests unreliable slope |
| A030 | INFO | n_obs >= 5 for statistical reliability | Flag low-confidence statistics |

### Low Priority / Informational

| ID | Severity | Rule | Rationale |
|----|----------|------|-----------|
| A031 | INFO | wse_obs_mad < 50m for rivers | High MAD suggests variability |
| A032 | INFO | width_obs_mad / width_obs_p50 < 1.0 | Coefficient of variation |
| A033 | INFO | Report SWOT coverage by region | Monitoring metric |

## Edge Cases

### 1. Lakes (lakeflag=1)
- WSE: low mad expected (flat surface)
- Width: large range acceptable
- Slope: near-zero p50 expected; quality likely `below_ref_uncertainty`

### 2. Tidal Rivers (lakeflag=3)
- WSE: high variability expected
- Width: may vary significantly with tide
- Slope: negative p50 possible during flood tide; quality may be `negative`
- Exclude from slope sign check; use wider tolerances

### 3. Canals (lakeflag=2)
- WSE: may have locks causing discontinuities
- Width: typically uniform
- Slope: controlled; may be flat or stepped

### 4. Narrow Rivers (width < 100m GRWL)
- SWOT may not reliably observe rivers below 100m
- Potential measurement bias

### 5. Single Observation (n_obs = 1)
- range = 0, mad = 0
- All percentiles equal
- slopeF = sign(slope) (no averaging)
- Reliable flag may still be TRUE if |slope| > ref_uncertainty

### 6. Flat Reaches (slope < 0.01 m/km from MERIT)
- SWOT slope noise dominates
- p50 may be negative
- quality likely `below_ref_uncertainty` or `high_uncertainty`

## Known Issues (v17c)

### 1. Negative SWOT Slopes (ADDRESSED)

**Scope:** ~25,000 reaches with negative slope_obs_p50

**Root cause:**
1. **Measurement noise** — SWOT slope accuracy ~17 mm/km. For flat reaches, noise produces negative measurements.
2. **Temporal variability** — Backwater, tides, local hydraulics can create temporarily negative slopes.
3. **Not a code bug** — Aggregation correctly computes percentiles; negative values are in source data.

**Resolution (v17c):** Four derived columns address this:
- `slope_obs_adj = GREATEST(slope_obs_p50, 0.0)` — clips negatives
- `slope_obs_slopeF` — weighted sign consistency (-1 to +1)
- `slope_obs_reliable` — TRUE when signal exceeds noise floor with consistent sign
- `slope_obs_quality` — categorical label for downstream use

**Recommendation:** Use `slope_obs_adj` for hydraulic calculations; check `slope_obs_quality` for confidence; `below_ref_uncertainty` means "below SWOT detection limit."

### 2. Low Slope Correlation

Correlation of 0.0386 between SWOT and MERIT-derived slopes. Expected — different measurement methods, temporal conditions, and spatial scales.

### 3. SWOT-GRWL Width Divergence

52% of reaches have width difference > 50%. Expected due to temporal change (decades), methodological differences, and SWOT minimum width constraint.

## Recommendations

### Short-term (v17c Release)

1. Run aggregation on all regions with production SWOT data
2. Validate percentile distributions against expected physical ranges
3. Document SWOT limitations in release notes

### Medium-term

1. Add n_obs-weighted uncertainty estimates
2. Regional analysis — different tolerances for arid vs humid climates
3. Tidal river handling — separate validation rules for lakeflag=3

### Long-term (v18)

1. SWOT as primary WSE/slope source (migrate from MERIT)
2. Time series statistics (seasonal amplitude)
3. Multi-mission fusion (SWOT + ICESat-2 + Sentinel-3)

## Test Coverage

**File:** `tests/sword_duckdb/test_swot_obs.py`

| Test Class | Count | Tests |
|------------|-------|-------|
| TestSwotFilters | 5 | constants, node filter WSE col, sentinel in filter, slope bounds, quality cols |
| TestAddSwotObsColumns | 4 | adds columns, idempotent, drops legacy, column types |
| TestNodeAggregation | 2 | basic (p50/mad/n_obs, percentile ordering), sentinel filtered |
| TestReachAggregation | 5 | basic (derived cols), reliable slope, single obs, all filtered, percentile ordering |

## References

1. Durand, M., et al. (2023). The Surface Water and Ocean Topography (SWOT) Mission. IEEE Transactions on Geoscience and Remote Sensing.
2. Frasson, R. P. d. M., et al. (2019). Global relationships between river width, slope, catchment area, meander wavelength, sinuosity, and discharge. Geophysical Research Letters.
3. Allen, G. H., & Pavelsky, T. M. (2018). Global extent of rivers and streams. Science, 361(6402), 585-588.
4. Yamazaki, D., et al. (2019). MERIT Hydro: A high-resolution global hydrography map. Water Resources Research.
5. SWOT Product Description Document (2023). JPL D-56411.

---

*Document version: 2.0*
*Updated: 2026-02-20*
*Author: SWORD Validation Team*
