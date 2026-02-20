# Validation Specification: Flag Variables

## Overview

This document provides a comprehensive audit of the flag variables in SWORD v17b: `swot_obs`, `iceflag`, `low_slope_flag`, `edit_flag`, and `add_flag`. It documents their provenance, derivation logic, valid values, failure modes, and validation checks.

---

## 1. swot_obs (SWOT Observations Count)

### 1.1 Official Definition

**Source:** SWORD Product Description Document v17b (pages 16, 17, 26)

> **swot_obs**: The maximum number of SWOT passes to intersect each reach during the 21 day orbit cycle.

### 1.2 Valid Values

| Value | Meaning |
|-------|---------|
| 0 | Reach not observed by SWOT during 21-day cycle |
| 1-31 | Number of SWOT passes intersecting the reach |

**Observed Distribution (v17b):**
- swot_obs=0: 8,457 reaches (3.4%)
- swot_obs=1: 47,121 reaches (18.9%)
- swot_obs=2: 110,436 reaches (44.4%)
- swot_obs=3: 24,780 reaches (10.0%)
- swot_obs=4+: remaining reaches
- Max observed: 31

### 1.3 Source and Derivation

**Source Dataset:** SWOT Orbit Tracks
- Source: https://www.aviso.altimetry.fr/en/missions/future-missions/swot/orbit.html
- Provides polygons containing SWOT track coverage for each pass in the 21-day cycle

**Derivation Method:** Spatial intersection count (MAX)

**Code Path:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py`

**Function:** `_reconstruct_reach_swot_obs` (lines 3296-3326)

```python
def _reconstruct_reach_swot_obs(self, ...):
    """
    Reconstruct reach swot_obs (SWOT observation count).

    Note: Full implementation requires external SWOT orbit data.
    This is a stub that returns 0 (unknown).
    """
    logger.info("Reconstructing reach.swot_obs (stub - requires external SWOT data)")

    result_df = self._conn.execute(f"""
        SELECT
            reach_id,
            COALESCE(swot_obs, 0) as swot_obs
        FROM reaches
        WHERE region = ? {where_clause}
    """, params).fetchdf()
```

**Status:** STUB - preserves existing values. Full reconstruction requires external SWOT orbit shapefiles.

### 1.4 Attribute Specification

**File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py` (lines 367-374)

```python
"reach.swot_obs": AttributeSpec(
    name="reach.swot_obs",
    source=SourceDataset.SWOT_TRACKS,
    method=DerivationMethod.MAX,
    source_columns=["num_observations"],
    dependencies=["reach.geom"],
    description="Max SWOT observations: np.max(num_obs[reach]) in 21-day cycle"
),
```

### 1.5 Related Table: reach_swot_orbits

**Schema:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/schema.py` (lines 318-330)

```python
REACH_SWOT_ORBITS_TABLE = """
CREATE TABLE IF NOT EXISTS reach_swot_orbits (
    reach_id BIGINT NOT NULL,
    region VARCHAR(2) NOT NULL,
    orbit_rank TINYINT NOT NULL,  -- 0-74
    orbit_id BIGINT NOT NULL,     -- SWOT orbit pass_tile ID
    PRIMARY KEY (reach_id, region, orbit_rank)
);
"""
```

**Data:** 623,697 records in v17b (normalized from [75, number_of_reaches] array)

### 1.6 Consistency Rules

| Rule | Description | Check |
|------|-------------|-------|
| R1 | swot_obs >= 0 | Non-negative values only |
| R2 | swot_obs <= 75 | Max orbits per cycle (theoretical) |
| R3 | swot_obs should match orbit count | COUNT(reach_swot_orbits) for reach = swot_obs |

### 1.7 Regional Distribution (swot_obs=0)

| Region | Count | Notes |
|--------|-------|-------|
| AF | 908 | Outside SWOT coverage (78N-78S swath) |
| AS | 2,558 | High latitude reaches |
| EU | 873 | Edge of coverage |
| NA | 1,064 | High Arctic reaches |
| OC | 1,694 | Small islands, Pacific reaches |
| SA | 1,360 | Edge cases |

### 1.8 Failure Modes

| Failure Mode | Description | Likely Cause | Severity |
|--------------|-------------|--------------|----------|
| FM1 | swot_obs < 0 | Data corruption | ERROR |
| FM2 | swot_obs > actual orbit count | Calculation error | WARNING |
| FM3 | swot_obs mismatch with reach_swot_orbits | Stale data | WARNING |

### 1.9 Existing Lint Checks

**None currently implemented for swot_obs.**

### 1.10 Proposed New Checks

#### F001: swot_obs Non-Negative

```python
@register_check("F001", Category.FLAGS, Severity.ERROR, "swot_obs should be non-negative")
def check_swot_obs_non_negative(conn, region=None, threshold=None):
    """Check that swot_obs >= 0 for all reaches."""
    query = """
    SELECT reach_id, region, swot_obs
    FROM reaches
    WHERE swot_obs < 0
    """
```

#### F002: swot_obs vs reach_swot_orbits Consistency

```python
@register_check("F002", Category.FLAGS, Severity.INFO, "swot_obs matches orbit count")
def check_swot_obs_orbit_consistency(conn, region=None, threshold=None):
    """Check that swot_obs equals count of reach_swot_orbits entries."""
    query = """
    SELECT r.reach_id, r.swot_obs, COUNT(o.orbit_id) as orbit_count
    FROM reaches r
    LEFT JOIN reach_swot_orbits o ON r.reach_id = o.reach_id AND r.region = o.region
    GROUP BY r.reach_id, r.swot_obs
    HAVING r.swot_obs != COUNT(o.orbit_id)
    """
```

---

## 2. iceflag (Ice Flag)

### 2.1 Official Definition

**Source:** SWORD Product Description Document v17b (pages 15-17)

> **ice_flag**: Ice flag values for each SWOT reach are modeled river ice conditions based on an empirical river ice model [Yang et al., 2019], that takes surface air temperature (SAT) data from ERA5 Land (9 km resolution) as model input. Values include 0 - ice free, 1 - mixed, 2 - ice cover.

**Historical Note:** Added in Beta v0.9 (January 2021):
> Added climatological ice flag values to netCDF.

### 2.2 Valid Values

| Value | Meaning |
|-------|---------|
| -9999 | No data / fill value |
| 0 | Ice free |
| 1 | Mixed (partial ice) |
| 2 | Ice cover |

**Observed Distribution (v17b):**
- iceflag=-9999: 7,010 reaches (2.8%)
- iceflag=0: 132,675 reaches (53.3%)
- iceflag=1: 23,813 reaches (9.6%)
- iceflag=2: 85,175 reaches (34.3%)

### 2.3 Storage Structure

**IMPORTANT:** The PDD describes ice_flag as a 366-day array per reach (dimensions: [366, number_of_reaches]). However, in the DuckDB implementation:

1. **Normalized Table:** `reach_ice_flags` with columns (reach_id, julian_day, iceflag)
   - Currently EMPTY in v17b (0 records)

2. **Scalar Column:** `reaches.iceflag` stores a single INTEGER
   - Appears to be a summary/dominant value, NOT the 366-day array

**Schema:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/schema.py` (lines 332-343)

```python
REACH_ICE_FLAGS_TABLE = """
CREATE TABLE IF NOT EXISTS reach_ice_flags (
    reach_id BIGINT NOT NULL,
    julian_day SMALLINT NOT NULL,  -- 1-366
    iceflag INTEGER NOT NULL,      -- 0=ice free, 1=mixed, 2=ice cover
    PRIMARY KEY (reach_id, julian_day)
);
"""
```

### 2.4 Source and Derivation

**Source Dataset:** ICE_FLAGS (external CSV)
- Model: Empirical river ice model (Yang et al., 2019)
- Input: ERA5 Land surface air temperature (9 km resolution)

**Derivation Method:** SPATIAL_JOIN

**Code Path:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py`

**Function:** `_reconstruct_reach_iceflag` (lines 3999-4027)

```python
def _reconstruct_reach_iceflag(self, ...):
    """
    Reconstruct reach iceflag from external ice flag CSV.

    REQUIRES EXTERNAL DATA: Ice flag CSV (366-day array per reach)
    This is a stub - preserves existing values.
    """
    logger.warning("reach.iceflag requires external ice flag data - preserving existing values")

    result_df = self._conn.execute(f"""
        SELECT reach_id, iceflag
        FROM reaches
        WHERE region = ? {where_clause}
    """, params).fetchdf()
```

**Status:** STUB - preserves existing values. Full reconstruction requires external ice flag data.

### 2.5 Attribute Specification

**File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py` (lines 376-383)

```python
"reach.iceflag": AttributeSpec(
    name="reach.iceflag",
    source=SourceDataset.ICE_FLAGS,
    method=DerivationMethod.SPATIAL_JOIN,
    source_columns=["ice_flag"],
    dependencies=["reach.reach_id"],
    description="Ice flag: 366-day array per reach from external CSV spatial join"
),
```

### 2.6 Regional Distribution

| Region | -9999 | 0 (Ice Free) | 1 (Mixed) | 2 (Ice Cover) |
|--------|-------|--------------|-----------|---------------|
| AF | 948 | 20,493 | 0 | 0 |
| AS | 2,917 | 40,874 | 5,848 | 50,546 |
| EU | 760 | 8,714 | 11,451 | 10,178 |
| NA | 556 | 8,155 | 5,576 | 24,409 |
| OC | 529 | 14,484 | 76 | 0 |
| SA | 1,300 | 39,955 | 862 | 42 |

**Observations:**
- Africa has no ice (as expected)
- Asia has most ice-covered reaches (high-latitude rivers like Siberian rivers)
- Oceania has minimal ice
- South America has very little ice

### 2.7 Consistency Rules

| Rule | Description | Check |
|------|-------------|-------|
| R1 | Valid values only | iceflag IN (-9999, 0, 1, 2) |
| R2 | Latitude correlation | Ice (1,2) more common at high latitudes |
| R3 | Africa no ice | AF region should have iceflag=0 or -9999 only |

### 2.8 Failure Modes

| Failure Mode | Description | Likely Cause | Severity |
|--------------|-------------|--------------|----------|
| FM1 | Invalid iceflag value | Data corruption | ERROR |
| FM2 | reach_ice_flags table empty | Migration incomplete | WARNING |
| FM3 | Tropical reach with iceflag=2 | Data error | WARNING |
| FM4 | iceflag=-9999 in well-covered region | Missing source data | INFO |

### 2.9 Existing Lint Checks

**None currently implemented for iceflag.**

### 2.10 Proposed New Checks

#### F003: iceflag Valid Values

```python
@register_check("F003", Category.FLAGS, Severity.ERROR, "iceflag should have valid values")
def check_iceflag_valid_values(conn, region=None, threshold=None):
    """Check that iceflag is one of valid values."""
    query = """
    SELECT reach_id, region, iceflag
    FROM reaches
    WHERE iceflag NOT IN (-9999, 0, 1, 2)
    """
```

#### F004: iceflag Latitude Sanity

```python
@register_check("F004", Category.FLAGS, Severity.INFO, "iceflag correlates with latitude")
def check_iceflag_latitude_sanity(conn, region=None, threshold=None):
    """Check that ice flags correlate with latitude."""
    query = """
    SELECT reach_id, region, y as latitude, iceflag
    FROM reaches
    WHERE (ABS(y) < 23.5 AND iceflag = 2)  -- Tropical with ice cover
       OR (region = 'AF' AND iceflag IN (1, 2))  -- Africa with ice
    """
```

---

## 3. low_slope_flag

### 3.1 Official Definition

**Source:** SWORD Product Description Document v17b (pages 17)

> **low_slope_flag**: binary flag where a value of 1 indicates the reach slope is too low for effective discharge estimation.

**Historical Note:** Added in Release v13 (July 2022):
> Added "low_slope_flag" to reaches group.

### 3.2 Valid Values

| Value | Meaning |
|-------|---------|
| 0 | Slope adequate for discharge estimation |
| 1 | Slope too low for discharge estimation |

**Observed Distribution (v17b):**
- low_slope_flag=0: 248,673 reaches (100%)
- low_slope_flag=1: 0 reaches (0%)

**NOTE:** All reaches have low_slope_flag=0 in v17b. This suggests either:
1. The flag was never populated
2. No reaches meet the threshold
3. Different calculation method than expected

### 3.3 Source and Derivation

**Source Dataset:** COMPUTED (derived from slope)

**Derivation Method:** THRESHOLD

**Threshold:** slope < 0.01 m/km (0.00001 m/m)

**Code Path:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py`

**Function:** `_reconstruct_reach_low_slope_flag` (lines 3259-3294)

```python
def _reconstruct_reach_low_slope_flag(self, ...):
    """
    Reconstruct reach low_slope_flag from slope values.

    Based on SWORD processing:
    low_slope_flag = 1 if slope is too low for reliable discharge estimation.

    Typical threshold: slope < 0.00001 m/m (0.01 m/km)
    """
    result_df = self._conn.execute(f"""
        SELECT
            reach_id,
            CASE
                WHEN slope IS NULL OR slope < 0.01 THEN 1
                ELSE 0
            END as low_slope_flag
        FROM reaches
        WHERE region = ? {where_clause}
    """, params).fetchdf()
```

### 3.4 Attribute Specification

**File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py` (lines 385-392)

```python
"reach.low_slope_flag": AttributeSpec(
    name="reach.low_slope_flag",
    source=SourceDataset.COMPUTED,
    method=DerivationMethod.COMPUTED,
    source_columns=[],
    dependencies=["reach.slope"],
    description="Low slope flag: 1 if slope < threshold for discharge calc"
),
```

### 3.5 Threshold Analysis

**Question:** Why are all low_slope_flag=0 when reconstruction uses slope < 0.01?

```sql
-- Check slope distribution
SELECT
    COUNT(*) FILTER (WHERE slope IS NULL) as null_slopes,
    COUNT(*) FILTER (WHERE slope < 0.01) as very_low_slopes,
    COUNT(*) FILTER (WHERE slope >= 0.01) as adequate_slopes
FROM reaches;
```

**Possible Explanations:**
1. Original SWORD uses different threshold (possibly stricter)
2. Original calculation excludes certain reach types
3. Flag was never run on v17b data

### 3.6 Consistency Rules

| Rule | Description | Check |
|------|-------------|-------|
| R1 | Valid values only | low_slope_flag IN (0, 1) |
| R2 | Threshold alignment | low_slope_flag=1 implies slope < threshold |
| R3 | NULL slope handling | NULL slope should trigger flag |

### 3.7 Failure Modes

| Failure Mode | Description | Likely Cause | Severity |
|--------------|-------------|--------------|----------|
| FM1 | Invalid value | Data corruption | ERROR |
| FM2 | All zeros despite low slopes | Threshold mismatch | WARNING |
| FM3 | Flag=1 but slope > threshold | Stale flag | WARNING |

### 3.8 Existing Lint Checks

**A002: slope_reasonableness** checks slope values but not low_slope_flag.

### 3.9 Proposed New Checks

#### F005: low_slope_flag Valid Values

```python
@register_check("F005", Category.FLAGS, Severity.ERROR, "low_slope_flag should be 0 or 1")
def check_low_slope_flag_valid(conn, region=None, threshold=None):
    """Check that low_slope_flag is binary."""
    query = """
    SELECT reach_id, region, low_slope_flag
    FROM reaches
    WHERE low_slope_flag NOT IN (0, 1)
    """
```

#### F006: low_slope_flag Threshold Consistency

```python
@register_check("F006", Category.FLAGS, Severity.INFO, "low_slope_flag matches slope threshold")
def check_low_slope_flag_consistency(conn, region=None, threshold=0.01):
    """Check that low_slope_flag aligns with slope values."""
    query = f"""
    SELECT reach_id, region, slope, low_slope_flag
    FROM reaches
    WHERE (low_slope_flag = 1 AND slope >= {threshold})
       OR (low_slope_flag = 0 AND slope < {threshold})
    """
```

---

## 4. edit_flag

### 4.1 Official Definition

**Source:** SWORD Product Description Document v17b (pages 11-12, 17-18)

> **edit_flag**: numerical flag indicating the type of update applied to SWORD nodes/reaches from the previous version.

### 4.2 Valid Values

| Code | Meaning | v17b | v17c |
|------|---------|------|------|
| 1 | Reach type change | ✓ | ✓ |
| 2 | Node order change | ✓ | ✓ |
| 3 | Reach neighbor change | ✓ | ✓ |
| 41 | Flow accumulation update | ✓ | ✓ |
| 42 | Elevation update | ✓ | ✓ |
| 43 | Width update | ✓ | ✓ |
| 44 | Slope update | ✓ | ✓ |
| 45 | River name update | ✓ | ✓ |
| 5 | Reach ID change | ✓ | ✓ |
| 6 | Reach boundary change | ✓ | ✓ |
| 7 | Reach/node addition | ✓ | ✓ |
| facc_suspect | Flow accumulation quality flag | - | ✓ NEW |
| facc_traced | Flow accumulation traced/verified | - | ✓ NEW |
| NaN | No edits | ✓ | ✓ |

**Note:**
- Multiple updates are comma-separated (e.g., "41,2" or "7,1")
- v17c introduces new codes for facc QA workflow (facc_suspect, facc_traced)

### 4.3 Data Type

**Schema:** VARCHAR (not INTEGER) to support comma-separated values.

**File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/schema.py` (lines 148, 249)

```python
edit_flag VARCHAR,           -- comma-separated update codes
```

### 4.4 Observed Distribution (v17b Reaches)

| edit_flag | Count | Percentage |
|-----------|-------|------------|
| "NaN" | 239,791 | 96.4% |
| "7" | 6,905 | 2.8% |
| "1" | 1,866 | 0.8% |
| "7,1" | 65 | 0.03% |
| "['7'],1" | 40 | 0.02% |
| "['1'],1" | 6 | <0.01% |

**Observations:**
- Most reaches have no edits ("NaN")
- "7" (reach/node addition) is most common edit
- Some values have formatting issues (e.g., "['7'],1" should be "7,1")

### 4.5 Source and Derivation

**Source Dataset:** MANUAL (edit tracking)

**Derivation Method:** DIRECT (not reconstructable)

**Code Path:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py`

**Function:** `_reconstruct_reach_edit_flag` (lines 4093-4120)

```python
def _reconstruct_reach_edit_flag(self, ...):
    """
    Reach edit_flag tracks manual edits - NOT RECONSTRUCTABLE.

    This preserves existing values. Edit flags should only be set
    through manual edit operations.
    """
    logger.warning("reach.edit_flag tracks manual edits - cannot reconstruct, preserving values")

    result_df = self._conn.execute(f"""
        SELECT reach_id, COALESCE(edit_flag, '') as edit_flag
        FROM reaches
        WHERE region = ? {where_clause}
    """, params).fetchdf()
```

### 4.6 Attribute Specification

**File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py` (lines 422-428)

```python
"reach.edit_flag": AttributeSpec(
    name="reach.edit_flag",
    source=SourceDataset.MANUAL,
    method=DerivationMethod.DIRECT,
    source_columns=[],
    dependencies=[],
    description="Edit flag: comma-separated update codes from manual edits"
),
```

### 4.7 Format Issues

**Bug Found:** Some edit_flag values have inconsistent formatting:
- "['7'],1" should be "7,1"
- "['1'],1" should be "1,1"

These appear to be Python list stringification artifacts.

### 4.8 Consistency Rules

| Rule | Description | Check |
|------|-------------|-------|
| R1 | Valid codes only | All codes in comma-separated list should be valid |
| R2 | No list artifacts | Should not contain "[", "]", "'" characters |
| R3 | Consistent delimiter | Use comma, not semicolon or space |

### 4.9 Failure Modes

| Failure Mode | Description | Likely Cause | Severity |
|--------------|-------------|--------------|----------|
| FM1 | Invalid code | Typo or bug | WARNING |
| FM2 | List formatting | Python serialization bug | WARNING |
| FM3 | Inconsistent delimiter | Processing inconsistency | INFO |

### 4.10 Existing Lint Checks

**None currently implemented for edit_flag.**

### 4.11 Proposed New Checks

#### F007: edit_flag Valid Format

```python
@register_check("F007", Category.FLAGS, Severity.WARNING, "edit_flag should have valid format")
def check_edit_flag_format(conn, region=None, threshold=None):
    """Check that edit_flag values are properly formatted."""
    query = """
    SELECT reach_id, region, edit_flag
    FROM reaches
    WHERE edit_flag IS NOT NULL
      AND edit_flag != 'NaN'
      AND (edit_flag LIKE '%[%' OR edit_flag LIKE '%]%' OR edit_flag LIKE '%''%')
    """
```

#### F008: edit_flag Valid Codes

```python
@register_check("F008", Category.FLAGS, Severity.INFO, "edit_flag contains valid codes")
def check_edit_flag_valid_codes(conn, region=None, threshold=None):
    """Check that edit_flag codes are from valid set."""
    # Valid codes: 1,2,3,41,42,43,44,45,5,6,7,NaN
```

---

## 5. add_flag

### 5.1 Official Definition

**Source:** SWORD Product Description Document v17b

**Note:** `add_flag` is NOT documented in the official PDD. It appears to be an internal flag for tracking reaches/nodes added from MERIT Hydro during v17 topology updates.

### 5.2 Valid Values

| Value | Meaning |
|-------|---------|
| NULL | Unknown/not set |
| 0 | Not added (original SWORD) |
| 1 | Added from MERIT Hydro |

**Observed Distribution (v17b):**
- add_flag=NULL: 248,673 reaches (100%)
- add_flag=NULL: 11,112,454 nodes (100%)

**Observed Distribution (v17c):**
- add_flag=NULL: 248,673 reaches (99.9999%)
- add_flag=0: 1 reach (0.0004%)
- add_flag=1: 0 reaches (0%)

**Status:** Flag was added to the schema but rarely populated in v17b. Single add_flag=0 reach appears in v17c.

### 5.3 Schema Definition

**File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/schema.py` (lines 178, 290)

```python
# In NODES_TABLE (line 178):
add_flag INTEGER,            -- 0=not added, 1=added from MERIT Hydro

# In REACHES_TABLE (line 290):
add_flag INTEGER,            -- 0=not added, 1=added from MERIT Hydro
```

### 5.4 Source and Derivation

**Source Dataset:** COMPUTED (during MERIT Hydro integration)

**Derivation Method:** Set during reach/node addition operations

**Code Context:** Used when adding reaches from MERIT Hydro Vector to fill gaps in SWORD network.

### 5.5 Code Usage

**Primary Implementation Files:**

1. **delta_utils.py** - Delta region MERIT Hydro integration
   - Nodes added from MERIT Hydro are marked with add_flag=1 (line 1586)
   - Reaches added from MERIT Hydro are marked with add_flag=1 (line 1840)
   - Used when filling gaps in SWORD delta network with MERIT Hydro Vector data

2. **sword_class.py** - Core SWORD DuckDB implementation
   - Exported with default value 0 if not present (lines 572, 688)
   - Set to 0 for new ghost reaches (line 3463)
   - Preserved from source data when loading (lines 1258, 1359)
   - Inserted into database when creating new ghost reaches (line 1873)

**Semantic Meaning:**
- `add_flag=0`: Original SWORD reach/node (from GRWL centerlines)
- `add_flag=1`: Added from MERIT Hydro Vector (fill-in reaches for network connectivity)
- `add_flag=NULL`: Unknown/unset (treated as 0 in most contexts)

### 5.6 Consistency Rules

| Rule | Description | Check |
|------|-------------|-------|
| R1 | Valid values only | add_flag IN (NULL, 0, 1) |
| R2 | NULL interpretation | NULL should be treated as 0 |
| R3 | Correlation with edit_flag | add_flag=1 should have edit_flag containing "7" |

### 5.7 Failure Modes

| Failure Mode | Description | Likely Cause | Severity |
|--------------|-------------|--------------|----------|
| FM1 | Invalid value | Data corruption | ERROR |
| FM2 | All NULL | Feature not implemented | INFO |
| FM3 | Mismatch with edit_flag | Incomplete tracking | INFO |

### 5.8 Existing Lint Checks

**None currently implemented for add_flag.**

### 5.9 Proposed New Checks

#### F009: add_flag Valid Values

```python
@register_check("F009", Category.FLAGS, Severity.ERROR, "add_flag should be NULL, 0, or 1")
def check_add_flag_valid(conn, region=None, threshold=None):
    """Check that add_flag has valid values."""
    query = """
    SELECT reach_id, region, add_flag
    FROM reaches
    WHERE add_flag IS NOT NULL AND add_flag NOT IN (0, 1)
    """
```

---

## 6. Summary Table

| Variable | Source | Values | v17b Status | v17c Status | Lint Check |
|----------|--------|--------|-------------|-------------|------------|
| swot_obs | SWOT Tracks | 0-31 | STUB | STUB | None |
| iceflag | ERA5/Ice Model | -9999,0,1,2 | STUB | STUB | None |
| low_slope_flag | Computed | 0,1 | All 0 | All 0 | None |
| edit_flag | Manual | Codes CSV | 96.4% NaN | Mixed format | Proposed: F007, F008 |
| add_flag | Computed | NULL,0,1 | All NULL | 99.9999% NULL, 0.0004% 0 | Proposed: F009 |

---

## 7. Bugs and Issues Discovered

### 7.1 edit_flag Formatting Bug (WARNING)

**Issue:** Some edit_flag values have Python list formatting artifacts.

**Examples:**
- "['7'],1" should be "7,1"
- "['1'],1" should be "1,1"

**Affected Reaches:** 46 reaches (0.02%)

**Fix:**
```sql
UPDATE reaches
SET edit_flag = REPLACE(REPLACE(REPLACE(edit_flag, '[', ''), ']', ''), '''', '')
WHERE edit_flag LIKE '%[%' OR edit_flag LIKE '%]%';
```

### 7.2 reach_ice_flags Table Empty (INFO)

**Issue:** The `reach_ice_flags` table has 0 records, though the schema exists for 366-day ice flag arrays.

**Impact:** Daily ice flag data is not available; only the summary `iceflag` column in `reaches` table.

**Resolution:** Either populate the table from NetCDF source or document that scalar `iceflag` is the summary value.

### 7.3 low_slope_flag All Zeros (INFO)

**Issue:** All reaches have low_slope_flag=0, even though some may have very low slopes.

**Investigation Needed:** Verify the threshold used in original SWORD processing matches reconstruction threshold (0.01 m/km).

### 7.4 add_flag All NULL (INFO)

**Issue:** add_flag column exists but all values are NULL in v17b.

**Resolution:** Either populate during v17c MERIT Hydro integration or document as reserved for future use.

---

## 8. Proposed GitHub Issue

### Title: Add lint checks for flag variables (swot_obs, iceflag, low_slope_flag, edit_flag, add_flag)

### Labels: `type:feature`, `comp:lint`, `P2-medium`

### Milestone: `v17c-verify`

### Body:

**Summary:**
Flag variables currently have no lint checks. This issue proposes adding 9 new checks (F001-F009) for validation.

**Flag Variables:**
- `swot_obs` - SWOT observation count per 21-day cycle
- `iceflag` - River ice condition (0=free, 1=mixed, 2=cover, -9999=no data)
- `low_slope_flag` - Low slope warning for discharge estimation
- `edit_flag` - Manual edit tracking codes
- `add_flag` - MERIT Hydro addition tracking

**Proposed Checks:**
1. **F001**: swot_obs non-negative (ERROR)
2. **F002**: swot_obs vs reach_swot_orbits consistency (INFO)
3. **F003**: iceflag valid values (ERROR)
4. **F004**: iceflag latitude sanity (INFO)
5. **F005**: low_slope_flag valid values (ERROR)
6. **F006**: low_slope_flag threshold consistency (INFO)
7. **F007**: edit_flag valid format (WARNING)
8. **F008**: edit_flag valid codes (INFO)
9. **F009**: add_flag valid values (ERROR)

**Bugs Found:**
- [ ] Fix edit_flag formatting (46 reaches have list artifacts)
- [ ] Investigate low_slope_flag all zeros
- [ ] Document reach_ice_flags table status

**Validation Spec:** `docs/validation_specs/flags_validation_spec.md`

---

## 9. Audit Findings (v17c - February 2026)

### 9.1 add_flag Audit

**Query Date:** February 2, 2026

**v17b Baseline:**
```sql
SELECT add_flag, COUNT(*) FROM reaches GROUP BY 1 ORDER BY 2 DESC;
```
Result: All 248,673 reaches have add_flag=NULL (100%)

**v17c Current State:**
```sql
SELECT add_flag, COUNT(*) FROM reaches GROUP BY 1 ORDER BY 2 DESC;
```
Result:
- add_flag=NULL: 248,673 reaches (99.9960%)
- add_flag=0: 1 reach (0.0040%)
- add_flag=1: 0 reaches (0%)

**Interpretation:** The single add_flag=0 reach suggests minimal MERIT Hydro integration in v17c to date. Expected population when delta regions are processed.

### 9.2 edit_flag Audit

**v17b Baseline:**
```sql
SELECT edit_flag, COUNT(*) FROM reaches GROUP BY 1 ORDER BY 2 DESC;
```
Result:
| edit_flag | Count | % |
|-----------|-------|---|
| NaN | 239,791 | 96.4% |
| 7 | 6,905 | 2.8% |
| 1 | 1,866 | 0.8% |
| 7,1 | 65 | 0.03% |
| ['7'],1 | 40 | 0.02% |
| ['1'],1 | 6 | <0.01% |

**v17c Current State:**
```sql
SELECT edit_flag, COUNT(*) FROM reaches GROUP BY 1 ORDER BY 2 DESC;
```
Result:
| edit_flag | Count | % |
|-----------|-------|---|
| NaN | 239,601 | 96.3% |
| 7 | 6,784 | 2.7% |
| 1 | 1,866 | 0.8% |
| 7,facc_suspect | 121 | 0.05% |
| facc_traced | 116 | 0.05% |
| facc_suspect | 73 | 0.03% |
| 7,1 | 65 | 0.03% |
| ['7'],1 | 40 | 0.02% |
| ['1'],1 | 6 | <0.01% |
| 6 | 2 | <0.01% |

**Key Findings:**
1. New codes introduced: `facc_suspect`, `facc_traced` (likely from facc QA work)
2. Formatting bugs persisting: `['7'],1` and `['1'],1` still present
3. Total reach count decreased by 72 (from 248,673 to 248,674 total with new entry)
4. edit_flag shows active QA/editing workflow

---

## 9. References

1. SWORD Product Description Document v17b (March 2025)
2. `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py`
3. `/Users/jakegearon/projects/SWORD/src/sword_duckdb/schema.py`
4. `/Users/jakegearon/projects/SWORD/src/sword_duckdb/sword_class.py`
5. `/Users/jakegearon/projects/SWORD/src/_legacy/updates/delta_updates/delta_utils.py`
6. `/Users/jakegearon/projects/SWORD/src/sword_duckdb/lint/checks/attributes.py`
7. Yang, X., Pavelsky, T. M., Allen, G. H. (2019). The past and future of global river ice. Nature.
