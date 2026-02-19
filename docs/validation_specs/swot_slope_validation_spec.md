# Validation Spec: SWOT Slope Variables (swot_slope, swot_slope_se, swot_slope_confidence)

## ⚠️ STATUS: COLUMNS REMOVED (2026-02-02)

These columns were **removed from the v17c database** because:
1. They were 0% populated
2. The pipeline integration is incomplete
3. Section-to-reach mapping is undefined

**GitHub Issue:** #117 - swot_slope columns are empty - pipeline integration incomplete

If/when the pipeline is completed, these columns can be re-added.

---

## Original Summary (for reference)

| Field | Table | Units | Source | Status |
|-------|-------|-------|--------|--------|
| **swot_slope** | reaches only | m/km | SWOT L2 RiverSP + Linear Mixed Effects | **REMOVED** |
| **swot_slope_se** | reaches only | m/km | SWOT L2 RiverSP (LME model) | **REMOVED** |
| **swot_slope_confidence** | reaches only | categorical | Quality flag from LME analysis | **REMOVED** |

**Note:** These were v17c additions - NOT documented in SWORD PDD v17b.

---

## Current Status: EMPTY COLUMNS

### Population Statistics (v17c Database)
```
Total reaches: 248,674
Non-null swot_slope: 0 (0.00%)
Non-null swot_slope_se: 0 (0.00%)
Non-null swot_slope_confidence: 0 (0.00%)
```

### Why Empty?

The v17c pipeline requires a multi-step process:

1. **SWOT Slope Computation Pipeline**
   - File: `/Users/jakegearon/projects/SWORD/src/updates/sword_v17c_pipeline/SWOT_slopes.py`
   - Status: Implemented but NOT integrated into main v17c pipeline
   - Requires: External SWOT data at `/Volumes/SWORD_DATA/data/swot/RiverSP_D_parq/node`
   - Output: Section-level slopes (junction-to-junction), not reach-level

2. **v17c Pipeline Integration**
   - File: `/Users/jakegearon/projects/SWORD/src/updates/sword_v17c_pipeline/v17c_pipeline.py:707-787`
   - Function: `apply_swot_slopes()`
   - Status: Partially implemented - loads slope data but DOES NOT map to reaches
   - Issue: Requires section-to-reach mapping which is incomplete

3. **Data Flow Blocker**
   - SWOT_slopes.py produces **section-level** slopes (between junctions)
   - Reaches database needs **reach-level** slopes
   - Section-reach mapping table needed but not yet created
   - No aggregation rule defined (average? first reach? weighted by distance?)

---

## Official Definition (v17c Intended)

**swot_slope:**
- Water surface slope at each reach, derived from SWOT L2 RiverSP satellite observations
- Computed using Linear Mixed Effects (LME) regression on node-level SWOT water surface elevation data
- Units: meters per kilometer (m/km)
- Sign convention: Positive = downstream decrease in elevation

**swot_slope_se:**
- Standard error (uncertainty) of swot_slope estimate
- From the LME model's fixed effects variance
- Units: same as swot_slope (m/km)
- Interpretation: 95% CI ≈ swot_slope ± 1.96*swot_slope_se

**swot_slope_confidence:**
- Categorical quality flag indicating slope reliability
- Expected values: (documentation needed - likely 'U'/'R' for unreliable/reliable, or confidence level)
- Based on: Fraction of positive slopes (slopeF), convergence of LME model, sample size

---

## Source Data

### SWOT L2 RiverSP Products
- **Satellite:** Surface Water and Ocean Topography (SWOT)
- **Product Level:** Level 2 River Single Pass (RiverSP)
- **Format:** Parquet files
- **Location:** `/Volumes/SWORD_DATA/data/swot/RiverSP_D_parq/node/` (external volume)
- **Temporal Coverage:** 2023-present (SWOT operational phase)
- **Spatial Coverage:** Global rivers >= 100m wide (nominal)

### SWOT Measurement Characteristics
| Parameter | SWOT Specification |
|-----------|-------------------|
| WSE accuracy | 10 cm (rivers > 100m wide) |
| WSE precision | ~1.7 cm/km along-track |
| Slope accuracy | ~1.7 cm/km (from WSE precision) |
| Minimum river width | 100m (nominal), 50m (threshold) |
| Repeat cycle | 21 days |
| Mission duration | 3+ years (2023-2026+) |

---

## Code Path

### Main Computation: SWOT_slopes.py
**File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_v17c_pipeline/SWOT_slopes.py`

#### Overview
Computes water surface slope for each section (junction-to-junction) using Linear Mixed Effects (LME) regression on SWOT node observations.

#### Key Functions

1. **`open_SWOT_files()` (lines 32-284)**
   - Opens SWOT RiverSP parquet files
   - Filters by quality flags (WSE quality, dark water fraction, cross-track distance, calibration)
   - Deduplicates observations by SWOT cycle/pass
   - Returns DataFrame with columns: `node_id, time_str, wse, wse_u, cycle, pass, width, width_u`

2. **`lme_tuning()` (lines 601-636)**
   - Fits Linear Mixed Effects model with formula: `z ~ x` with random slopes per time group
   - Centers and scales predictor (distance) for numerical stability
   - Returns: degree, random_setting, convergence flag, fitted model

3. **`lme_results()` (lines 638-685)**
   - Extracts results from fitted LME model
   - Rescales from standardized to original units
   - Computes standard error and 95% confidence intervals
   - Returns: `beta_orig, se_beta_orig, p_value, fraction_positive, predictions`

4. **`compute_clean_slopes()` (lines 853-1087)**
   - Advanced outlier filtering using Median Absolute Deviation (MAD)
   - Identifies and removes outlier SWOT passes that deviate from section median
   - Recomputes section-level slopes with outliers removed
   - Returns: DataFrame with columns `junction_id, section_id, slope, SE, slopeF, intercept, outlier_removed_flag`

5. **`junction_slope_calc()` (lines 687-750)**
   - Alternative to LME: junction-level slope aggregation
   - Slower, less sophisticated
   - Not used in current workflow (superseded by `compute_clean_slopes`)

#### Output Format

CSV file: `{region}_swot_slopes.csv` with columns:
```
junction_id         - Start junction of section
section_id          - Unique section identifier
slope               - Water surface slope (m/km)
SE                  - Standard error of slope
slopeF              - Fraction of positive slopes (0-1)
convergence         - LME model convergence flag (True/False)
random_effects      - Random effects formula used
distance            - Length of section (m)
outlier_removed_flag - Whether outliers were filtered
```

**Example values:**
```
junction_id=3456, section_id=891, slope=0.0045, SE=0.0012, slopeF=0.87
junction_id=5678, section_id=892, slope=-0.0001, SE=0.0025, slopeF=0.31
junction_id=7890, section_id=893, slope=0.0123, SE=0.0089, slopeF=0.95
```

### v17c Pipeline Integration (Incomplete)
**File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_v17c_pipeline/v17c_pipeline.py:707-787`

**Function:** `apply_swot_slopes(conn, region, swot_path)`

**Current behavior:**
1. Checks for SWOT parquet files
2. Looks for pre-computed slopes CSV file
3. **DOES NOT actually apply to database** - just logs that slopes are available
4. Returns 0 (no reaches updated)

**Missing components:**
- Section-to-reach mapping
- Aggregation rule (how to map section slopes to individual reaches)
- Database update statement
- Confidence score computation

---

## Schema Definition

**File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/schema.py:805-850`

### Reaches Table Columns

```sql
-- v17c columns (added by add_v17c_columns())
swot_slope DOUBLE,              -- water surface slope from SWOT (m/km)
swot_slope_se DOUBLE,           -- standard error of swot_slope (m/km)
swot_slope_confidence VARCHAR   -- quality flag (U/R or categorical)
```

**Column creation code (lines 846-849):**
```python
reaches_v17c_columns = [
    # ... other columns ...
    ("swot_slope", "DOUBLE"),
    ("swot_slope_se", "DOUBLE"),
    ("swot_slope_confidence", "VARCHAR"),
]
```

---

## Valid Ranges

### Physical Constraints

| Variable | Min | Max | Rationale |
|----------|-----|-----|-----------|
| swot_slope | 0 | 100 m/km | Positive = downstream decrease in elevation |
| swot_slope_se | 0 | 50 m/km | Standard error must be positive, but should be much smaller than slope |
| swot_slope | -0.1 | 0.1 m/km (small magnitude) | Small slopes expected in flat terrain |
| swot_slope_confidence | NULL or categorical | N/A | Expected: 'R'/'U', 'high'/'medium'/'low', or numeric 0-1 |

### Observed Ranges (Once Populated)

Not yet populated - these will be determined after computation.

### Comparison to Derived Slope

**Reference:** `slope` column computed from MERIT Hydro WSE via linear regression (see `width_slope_validation_spec.md`)

- **MERIT slope:** From DEM-derived WSE profile, more stable, lower temporal variability
- **SWOT slope:** From satellite observations, more temporally variable, affected by discharge/season
- **Expected correlation:** Moderate (likely 0.4-0.6), not perfect due to temporal differences
- **Expected bias:** SWOT may show higher variability and occasional sign flips in backwater reaches

---

## Failure Modes

### Algorithm-Level Failures

| # | Mode | Description | Impact | Prevention |
|---|------|-------------|--------|------------|
| 1 | **LME non-convergence** | Model fails to converge for a section | Slope = NULL | Check convergence flag |
| 2 | **Insufficient observations** | < 3 SWOT observations per section | Cannot compute LME | Set minimum sample size |
| 3 | **All same WSE** | SWOT observations all have identical elevation | Slope = 0 with zero SE | Flag as flat/lake |
| 4 | **Negative slope** | LME produces negative slope (backwater) | Physically possible but rare | Document occurrence rate |
| 5 | **Extreme SE** | Standard error > slope magnitude | Very uncertain result | Flag low confidence |
| 6 | **Outlier removal fails** | MAD filtering removes all data | Cannot compute slope | Reduce outlier threshold |

### Data Quality Failures

| # | Mode | Description | Impact | Detection |
|---|------|-------------|--------|-----------|
| 1 | **Dark water contamination** | SWOT WSE from dark water features | Biased slope (usually too low) | Check dark_water_frac filter |
| 2 | **Cross-track distance error** | Observations far from nadir (> 60km) | Higher WSE uncertainty | Applied filter: 10-60km |
| 3 | **Summary quality flag low** | WSE quality = 2 or 3 (poor) | Unreliable elevation | Applied filter: quality <= 1 |
| 4 | **Temporal aliasing** | Mixed observations from different seasons | High slopeF variability | Document seasonal effects |
| 5 | **Ice-covered sections** | SWOT over ice/frozen rivers | Unreliable WSE | Apply ice flag filter |

### Output Mapping Failures (Once Integration Completes)

| # | Mode | Description | Impact | Prevention |
|---|------|-------------|--------|-----------|
| 1 | **Section-reach mismatch** | Multiple reaches per section | Ambiguous slope assignment | Define aggregation rule |
| 2 | **Reach spans multiple sections** | Reaches crossing junctions | Doesn't fit section paradigm | Rebuild reaches or junction topology |
| 3 | **Section has no reaches** | Junction-to-junction with no reaches | Cannot populate slope | Review reach/junction definitions |
| 4 | **Orphan reaches** | Reaches not in any section | Slope remains NULL | Quality check section-reach mapping |

---

## Existing Lint Checks

**None.** SWOT slope variables currently have no dedicated lint checks because columns are not yet populated.

**Related existing checks (for derived slope only):**
- **A002: slope_reasonableness** - Checks that slope is non-negative and < 100 m/km (for MERIT-derived slope, not SWOT)
- **A006: attribute_outliers** - General outlier detection (does not include SWOT columns)

---

## Proposed Validation Checks

### High Priority (When Populated)

| ID | Severity | Rule | Rationale |
|----|----------|------|-----------|
| **V001** | **ERROR** | swot_slope >= 0 | Negative slopes physically impossible except in backwater |
| **V002** | **ERROR** | swot_slope < 100 m/km | Extreme slopes suggest error |
| **V003** | **ERROR** | swot_slope_se > 0 | Standard error must be positive |
| **V004** | **WARNING** | swot_slope_se < swot_slope OR swot_slope < 0.01 | SE should not exceed slope; tiny slopes less certain |
| **V005** | **WARNING** | swot_slope_confidence NOT NULL | Confidence flag required |

### Medium Priority

| ID | Severity | Rule | Rationale |
|----|----------|------|-----------|
| **V006** | **INFO** | swot_slope > 0 for non-lake rivers | Most rivers have positive slope |
| **V007** | **INFO** | \|swot_slope - slope\| < 0.01 m/km OR ratio < 2.0 | Cross-validate vs MERIT-derived slope |
| **V008** | **WARNING** | Reaches with swot_slope=0 AND lakeflag=0 | Zero slope should indicate lakes |
| **V009** | **INFO** | swot_slope_se / swot_slope < 0.5 for rivers | Relative SE indicates confidence |

### Low Priority / Informational

| ID | Severity | Rule | Rationale |
|----|----------|------|-----------|
| **V010** | **INFO** | Report % reaches with swot_slope populated | Coverage metric |
| **V011** | **INFO** | Report swot_slope distribution by region | Regional statistics |
| **V012** | **INFO** | Report correlation between swot_slope and slope | Validation metric |
| **V013** | **INFO** | Flag confidence='U' (unreliable) reaches | Quality monitoring |

---

## Edge Cases

### 1. Lakes (lakeflag=1)
- **Expected:** swot_slope ≈ 0 (lakes are flat)
- **Issue:** SWOT may detect slight tilt due to wind setup or bathymetry
- **Recommendation:** Apply lake-specific tolerance; allow small negative slopes in large lakes

### 2. Tidal Rivers (lakeflag=3)
- **Expected:** swot_slope varies with tide phase
- **Issue:** Single-pass SWOT slope may not be representative of average
- **Recommendation:** Flag with lower confidence; use multiple passes for averaging

### 3. Canals (lakeflag=2)
- **Expected:** swot_slope controlled by locks/steps
- **Issue:** Can have zero slope or large discontinuities
- **Recommendation:** Exclude from monotonicity checks

### 4. Backwater Reaches
- **Expected:** swot_slope can be slightly negative
- **Issue:** Physically possible but uncommon
- **Recommendation:** Flag but don't reject; investigate cause

### 5. Short Reaches (reach_length < 1 km)
- **Expected:** Few SWOT pixels, noisy slope
- **Issue:** High relative uncertainty (SE/slope ratio large)
- **Recommendation:** Weight analyses by swot_slope_se or reach_length

### 6. Narrow Rivers (width < 100m from GRWL)
- **Expected:** Below SWOT nominal detectability
- **Issue:** SWOT may not observe these rivers
- **Recommendation:** swot_slope will likely remain NULL

### 7. Braided/Anastomosing Rivers
- **Expected:** Multiple channels with different slopes
- **Issue:** SWOT slope averages all water bodies
- **Recommendation:** May not represent single main channel

### 8. Dams/Waterfalls (obstr_type > 0)
- **Expected:** Extremely high slope at obstruction
- **Issue:** Section may span across dam, creating bimodal WSE distribution
- **Recommendation:** Exclude sections containing major obstructions from LME analysis

### 9. Flat Reaches (slope < 0.001 m/km from MERIT)
- **Expected:** Very small slope, high noise relative to signal
- **Issue:** SWOT measurement noise > true signal
- **Recommendation:** Lower confidence for these reaches; flag with V004

### 10. Seasonal/Flood Rivers
- **Expected:** Slope varies with discharge and season
- **Issue:** SWOT slope represents one phase only
- **Recommendation:** Document temporal context; may need seasonal variants

---

## Known Issues / Incomplete Status

### 1. **Not Yet Integrated into Pipeline**

The SWOT_slopes.py script produces section-level slopes, but:
- v17c_pipeline.py does not call SWOT_slopes.py
- No automatic slope computation during standard `python -m src.updates.sword_v17c_pipeline.v17c_pipeline`
- External execution required: `python SWOT_slopes.py --dir {dir} --continent {continent}`

**Workaround:** Manual execution after running main v17c pipeline.

### 2. **Section-to-Reach Mapping Undefined**

The output is at junction-to-junction section level, but database is at reach level:
- No automatic aggregation rule defined
- Multiple options possible:
  - Average slope of reaches in section
  - Slope of first reach in section
  - Length-weighted average
  - Slope at reach centroid (requires interpolation)

**Decision needed:** Which aggregation rule to use?

### 3. **Confidence Flag Not Specified**

The `swot_slope_confidence` column exists but its expected values are undocumented:
- Possible values: 'U'/'R', 'high'/'medium'/'low', numeric 0-1, or fraction_positive?
- Current code output: `slopeF` (fraction of positive slopes)
- Need to define mapping rule

**Decision needed:** What value/encoding to use for confidence?

### 4. **External Data Dependency**

SWOT parquet files must be mounted at `/Volumes/SWORD_DATA/` (external volume):
- Not included in git repository
- May not be available on all systems
- v17c_pipeline skips SWOT step if directory not found

**Workaround:** `--skip-swot` flag in v17c_pipeline to run main attributes without SWOT slopes.

### 5. **Negative Slope Edge Case**

LME model can produce negative slopes in backwater reaches:
- Physically possible but should be rare
- No handling for sign flipping in current code
- May indicate hydraulic backwater, tides, or measurement error

**Recommendation:** Allow negative slopes but flag for investigation.

---

## Testing & Validation Plan

### Before Population

1. **Verify SWOT data access**
   - Confirm `/Volumes/SWORD_DATA/` path is accessible
   - Check parquet file format and content

2. **Unit test SWOT_slopes.py functions**
   - Create test fixtures with synthetic SWOT-like data
   - Validate LME model fitting
   - Check outlier filtering behavior

3. **Test section-reach mapping**
   - Verify all sections have corresponding reaches
   - Check for orphan sections or reaches
   - Define and test aggregation logic

### After Population

1. **Distribution validation**
   - Plot histogram of swot_slope values
   - Compare to MERIT-derived slope distribution
   - Identify outliers (V001-V004 checks)

2. **Cross-validation**
   - Compute correlation with MERIT-derived slope
   - Compare with width and facc distributions
   - Check for systematic biases by region

3. **Quality metrics**
   - Report coverage % by region
   - Distribution of swot_slope_se
   - Distribution of swot_slope_confidence
   - Count of convergence failures

4. **Spot checks**
   - Manually verify 10-20 reaches with extreme slopes
   - Check reaches with negative slopes
   - Verify section-reach mapping accuracy

---

## Implementation Roadmap

### Phase 1: Setup (Ready)
- [x] Schema columns created (add_v17c_columns)
- [x] SWOT_slopes.py implemented
- [ ] Test fixtures created
- [ ] Documentation complete

### Phase 2: Computation (Blocked)
- [ ] Integrate SWOT_slopes.py into v17c_pipeline.py
- [ ] Define section-to-reach mapping strategy
- [ ] Define confidence flag encoding
- [ ] Run computation for all regions
- [ ] Validate output quality

### Phase 3: Integration (Future)
- [ ] Implement proposed lint checks (V001-V013)
- [ ] Database update queries
- [ ] Population statistics reporting
- [ ] Documentation update

### Phase 4: Validation (Future)
- [ ] Cross-validation with MERIT slopes
- [ ] Regional performance analysis
- [ ] Known issues investigation
- [ ] Release notes documentation

---

## Recommendations

### Immediate (Blocking Issues)

1. **Decide on section-to-reach mapping**
   - Define aggregation rule (average vs first vs weighted)
   - Update v17c_pipeline.apply_swot_slopes() accordingly
   - Create section-reach mapping table if needed

2. **Specify confidence flag encoding**
   - Decide on format for swot_slope_confidence
   - Map LME output (convergence, slopeF, etc.) to flag value
   - Document in schema and code

3. **Handle external data dependency**
   - Verify SWOT data access paths
   - Document data mounting requirements
   - Consider fallback behavior if data unavailable

### Short-term (v17c Release)

1. **Run full computation**
   - Execute SWOT_slopes.py for all regions
   - Validate output quality
   - Document any computation failures

2. **Implement lint checks**
   - Focus on high-priority checks (V001-V005)
   - Add to lint framework
   - Run against populated data

3. **Create validation report**
   - Coverage by region
   - Distribution statistics
   - Known issues and limitations
   - Cross-validation results

### Long-term (v18+)

1. **Improve slope computation**
   - Consider centerline-level regression (vs node-level)
   - Add temporal variability metrics
   - Integrate with other satellite data (ICESat-2, Sentinel-3)

2. **Refine SWOT integration**
   - Move to primary slope source (replace MERIT-derived)
   - Add seasonal variants
   - Propagate uncertainty to discharge estimates

---

## References

1. **SWOT_slopes.py:** `/Users/jakegearon/projects/SWORD/src/updates/sword_v17c_pipeline/SWOT_slopes.py` - Main computation script

2. **v17c_pipeline.py:** `/Users/jakegearon/projects/SWORD/src/updates/sword_v17c_pipeline/v17c_pipeline.py:707-787` - Pipeline integration (incomplete)

3. **schema.py:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/schema.py:805-850` - Column definitions

4. **SWOT PDD:** JPL D-56411 - Surface Water and Ocean Topography Mission Product Description Document

5. **Linear Mixed Effects (LME) Model:** Statsmodels documentation - `smf.mixedlm()` for hierarchical regression

6. **Median Absolute Deviation (MAD):** Statistical outlier detection method used in `compute_clean_slopes()`

7. **Related Validation Specs:**
   - `width_slope_validation_spec.md` - MERIT-derived slope documentation
   - `swot_observations_validation_spec.md` - SWOT observation statistics (WSE, width)

---

*Document version: 1.0*
*Created: 2026-02-02*
*Author: SWORD Validation Team*
*Status: DRAFT - Waiting for section-to-reach mapping design*
