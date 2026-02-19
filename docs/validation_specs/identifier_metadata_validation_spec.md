# Identifier & Metadata Variables Validation Specification

**Version:** v17b/v17c
**Date:** February 2025
**Author:** Data Audit
**Status:** Research

## Overview

This specification documents the three core identifier/metadata variables in the SWORD reaches table:
- `reach_id` - Unique reach identifier with embedded geographic and type information
- `region` - Continental region code
- `version` - Database version identifier

These variables establish the primary key and location context for all reaches.

## Summary Statistics

| Variable | Type | Count | Distribution | Notes |
|----------|------|-------|---------------|-------|
| `reach_id` | BIGINT (11 digits) | 248,674 | 100% unique, globally | CBBBBBRRRRT format (Pfafstetter-based) |
| `region` | VARCHAR(2) | 248,674 | AF:8.62%, AS:40.29%, EU:12.51%, NA:15.56%, OC:6.07%, SA:16.95% | Geographic continental identifier |
| `version` | VARCHAR(5) | 248,674 | v17b:100% | Database release version |

## Detailed Variable Specifications

### 1. reach_id

**Purpose:** Unique identifier for each reach, encoding Pfafstetter basin hierarchy, reach sequence, and feature type.

**Format:** CBBBBBRRRRT (11 digits total)
- **C** (1 digit): Continent code from Pfafstetter basin level 1
  - `1` = Africa (actually 10-19 in Pfafstetter, first digit = 1)
  - `2` = Europe/Middle East (Pfafstetter level 1 = 2)
  - `3` = Asia (Pfafstetter level 1 = 3, includes Asia proper)
  - `4` = Asia (Pfafstetter level 1 = 4, continuation)
  - `5` = Oceania (Pfafstetter level 1 = 5)
  - `6` = South America (Pfafstetter level 1 = 6)
  - `7` = North America (Pfafstetter level 1 = 7, includes NA, Arctic, Greenland)
  - `8` = North America (Pfafstetter level 1 = 8, Arctic/Greenland)
  - `9` = North America (Pfafstetter level 1 = 9, Greenland)
- **BBBBB** (5 digits): Pfafstetter basin codes for levels 2-6 (nested hierarchy, right-aligned)
- **RRRR** (4 digits): Sequential reach ID assigned within level-6 basin, downstream to upstream
- **T** (1 digit): Type/feature class (see Type Distribution section)

**Data Type:** BIGINT (stored as integer), typically displayed as VARCHAR for leading zero preservation

**Length:** Always exactly 11 digits (leading zeros preserved in NetCDF/vector formats)

**Valid Range:**
- Min across all regions: 11410000016 (Africa)
- Max across all regions: 91490000016 (North America, Pfafstetter 9=Greenland)
- All values fall within expected range for Pfafstetter coding

**Regional Distribution:**

| Region | Count | Reach ID Range | Pfafstetter Level 1 | Notes |
|--------|-------|-----------------|---------------------|-------|
| AF | 21,441 | 11410000016–18199600226 | 1 | Africa |
| AS | 100,185 | 31101000016–49207000326 | 3, 4 | Asia & Siberia |
| EU | 31,103 | 21101200016–29510300165 | 2 | Europe/Middle East |
| NA | 38,696 | 71120000013–91490000016 | 7, 8, 9 | North America, Arctic, Greenland |
| OC | 15,090 | 51111100016–57208000215 | 5 | Oceania/Australia |
| SA | 42,159 | 61100200016–67209900286 | 6 | South America |
| **Global** | **248,674** | — | All | — |

**Pfafstetter Basin Encoding:**
- Pfafstetter is a nested basin coding system (levels 1-9)
- SWORD uses up to level 6 (country/region scale)
- Level 1 digit = continent, embedded in first digit of reach_id
- Levels 2-6 = nested subdivision codes (5 digits)
- Each level represents finer geographic/hydrographic resolution

**Type Distribution (Last Digit):**

| Type | Code | Meaning | Count | % |
|------|------|---------|-------|-----|
| River | 1 | River/main channel | 163,580 | 65.78% |
| Lake on river | 3 | Reach containing lake | 25,737 | 10.35% |
| Dam/waterfall | 4 | Obstruction feature | 21,372 | 8.59% |
| Unreliable topology | 5 | Known topology issues | 14,883 | 5.98% |
| Ghost reach | 6 | Non-physical, connective | 23,102 | 9.29% |

**Invariants:**
1. reach_id is globally unique (0 duplicates across 248,674 reaches)
2. reach_id first digit matches region's Pfafstetter continent code:
   - C=1 → region=AF
   - C=2 → region=EU
   - C=3,4 → region=AS
   - C=5 → region=OC
   - C=6 → region=SA
   - C=7,8,9 → region=NA
3. reach_id always 11 digits (leading zeros preserved)
4. Type (last digit) must be 1, 3, 4, 5, or 6
5. Reach ID (RRRR) assigned sequentially downstream→upstream within level-6 basin

**Data Quality:**
- 100% non-null
- 100% unique
- All values fit Pfafstetter encoding scheme
- Consistent across regions

---

### 2. region

**Purpose:** Continental region identifier for data organization and access.

**Valid Values:** Two-letter uppercase codes

| Code | Continent | Count | % of Total | Notes |
|------|-----------|-------|-----------|-------|
| NA | North America | 38,696 | 15.56% | Includes Canada, USA, Mexico, Arctic, Greenland (Pfafstetter 7,8,9) |
| SA | South America | 42,159 | 16.95% | Includes Amazon, Paraná, Orinoco basins (Pfafstetter 6) |
| EU | Europe/Middle East | 31,103 | 12.51% | Includes Europe, Caucasus, Middle East through Persia (Pfafstetter 2) |
| AF | Africa | 21,441 | 8.62% | Includes Nile, Congo, Niger, Zambezi (Pfafstetter 1) |
| AS | Asia | 100,185 | 40.29% | Includes Asia proper, Siberia, East Asia (Pfafstetter 3,4) |
| OC | Oceania | 15,090 | 6.07% | Includes Australia, New Zealand, Pacific islands (Pfafstetter 5) |
| **Global** | **All regions** | **248,674** | **100%** | — |

**Data Type:** VARCHAR(2)

**Case Sensitivity:** Uppercase (always 2 uppercase letters, no lowercase)

**Relationship to reach_id:**
- Region is derived from reach_id's first digit (Pfafstetter continent level 1)
- Region code is human-readable alias for Pfafstetter continent mapping
- Mismatch between region and reach_id prefix indicates data corruption

**Invariant:**
- region must match reach_id's Pfafstetter continent encoding (first digit)

**Data Quality:**
- 100% non-null
- 6 distinct values only
- Region values match reach_id prefixes globally (0 mismatches detected)

---

### 3. version

**Purpose:** Database version identifier, tracking which release the reach belongs to.

**Valid Values:** Version strings in format `vXXb` or `vXX`

| Value | Release Date | Count | % | Notes |
|-------|--------------|-------|---|-------|
| v17b | March 2025 | 248,674 | 100% | Latest (bug fixes to v17, type/length corrections) |
| (v17) | October 2024 | 0 | 0% | v17 not present in v17b/v17c databases |

**Data Type:** VARCHAR(5)

**Current Status:**
- All 248,674 reaches in sword_v17c.duckdb are labeled as `v17b`
- This is the March 2025 release with corrections applied

**Historical Versions (Not in Current DB):**
- v10–v16 were previous releases; all have been superseded
- v17 (October 2024) was the topology rewrite release
- v17b (March 2025) corrected bugs in v17

**Data Quality:**
- 100% non-null
- Single value (v17b) across entire database
- Indicates database is mature stable release

---

## Encoding Scheme Details

### Pfafstetter Basin Coding System

**Source:** HydroBASINS (Lehner & Grill, 2013)

**Hierarchy:**
- Level 1: Continent (9 codes, but SWORD maps to 6 geographic regions)
- Levels 2–6: Nested basin subdivisions at increasing detail
- Each level refines geographic/hydrographic boundaries

**SWORD Implementation:**
- Uses Pfafstetter levels 1–6 for nested basin context
- Level 1 embedded in reach_id first digit (C)
- Levels 2–6 embedded as 5-digit code (B)
- Example: reach_id `23190200061` breaks down as:
  - C=2 (Europe/Middle East, Pfafstetter level 1)
  - B=31902 (nested basin within Europe)
  - R=0006 (6th reach within level-6 basin, counting downstream→upstream)
  - T=1 (river type)

**Regional Grouping:**
SWORD groups HydroBASINS Pfafstetter regions into 6 continents for distribution:
- NA bundles Pfafstetter 7 (North America), 8 (Arctic), 9 (Greenland)
- AS bundles Pfafstetter 3 (Asia), 4 (Siberia)
- Other regions (SA, AF, EU, OC) map 1:1

---

## Validation Rules

### Primary Key Constraint
```sql
-- Verify global uniqueness of reach_id
SELECT COUNT(DISTINCT reach_id) FROM reaches;  -- Should equal row count
SELECT COUNT(*) FROM reaches;                  -- Should equal 248,674
```

**Expected:** COUNT(DISTINCT reach_id) = COUNT(*) = 248,674

### Region-to-reach_id Consistency
```sql
-- Verify reach_id prefix matches region
WITH id_regions AS (
  SELECT
    reach_id,
    region,
    SUBSTRING(CAST(reach_id AS VARCHAR), 1, 1) as id_first_digit,
    CASE
      WHEN SUBSTRING(CAST(reach_id AS VARCHAR), 1, 1) IN ('7','8','9') THEN 'NA'
      WHEN SUBSTRING(CAST(reach_id AS VARCHAR), 1, 1) = '2' THEN 'EU'
      WHEN SUBSTRING(CAST(reach_id AS VARCHAR), 1, 1) IN ('3','4') THEN 'AS'
      WHEN SUBSTRING(CAST(reach_id AS VARCHAR), 1, 1) = '5' THEN 'OC'
      WHEN SUBSTRING(CAST(reach_id AS VARCHAR), 1, 1) = '6' THEN 'SA'
      WHEN SUBSTRING(CAST(reach_id AS VARCHAR), 1, 1) = '1' THEN 'AF'
      ELSE 'UNKNOWN'
    END as computed_region
  FROM reaches
)
SELECT
  COUNT(*) as total,
  COUNT(CASE WHEN region = computed_region THEN 1 END) as matches,
  COUNT(CASE WHEN region != computed_region THEN 1 END) as mismatches
FROM id_regions;
```

**Expected:** mismatches = 0

### reach_id Format Validation
```sql
-- Verify reach_id is always 11 digits
SELECT
  LENGTH(CAST(reach_id AS VARCHAR)) as id_length,
  COUNT(*) as count
FROM reaches
GROUP BY id_length
ORDER BY id_length;
```

**Expected:** Single row with id_length=11, count=248,674

### Type Code Validation
```sql
-- Verify type digit (last digit) is valid (1,3,4,5,6)
SELECT
  CAST(reach_id AS BIGINT) % 10 as type_code,
  COUNT(*) as count
FROM reaches
GROUP BY type_code
ORDER BY type_code;
```

**Expected:** Only rows with type_code in (1,3,4,5,6), no other values

### Region Value Validation
```sql
-- Verify region contains only valid 2-letter codes
SELECT DISTINCT region FROM reaches ORDER BY region;
```

**Expected:** NA, SA, EU, AF, AS, OC (6 values only)

### Version Value Validation
```sql
-- Verify version is consistent and valid
SELECT DISTINCT version FROM reaches;
```

**Expected:** v17b (single value for current database)

---

## Failure Modes & Detection

### Failure Mode 1: reach_id Duplicates
**Symptom:** Duplicate reach_id values violate primary key uniqueness
**Cause:** Data import error, record duplication, ID collision
**Detection:** `COUNT(DISTINCT reach_id) < COUNT(*)`
**Impact:** Cannot uniquely identify reaches; breaks foreign keys
**Mitigation:** Check import logs, deduplicate on reach_id + geometry, reassign IDs if needed

### Failure Mode 2: reach_id-region Mismatch
**Symptom:** reach_id first digit doesn't match region code
**Example:** reach_id='21410...' (Pfafstetter 2=EU) but region='NA'
**Cause:** Region column overwritten without reach_id update, data import error
**Detection:** See region-to-reach_id consistency query above
**Impact:** Query filters by region miss reaches; geographic metadata becomes unreliable
**Mitigation:** Audit recent updates to region; rebuild region from reach_id prefix

### Failure Mode 3: Invalid Type Code
**Symptom:** reach_id ends in digit not in (1,3,4,5,6)
**Example:** reach_id='71234500007' (type=7 invalid)
**Cause:** ID reassignment, data corruption, development artifact
**Detection:** See type code validation query above
**Impact:** Type classification breaks; code assumes specific type values
**Mitigation:** Determine correct type; reissue ID with valid type digit

### Failure Mode 4: Non-11-digit reach_id
**Symptom:** reach_id shorter or longer than 11 digits (e.g., '7120000013' = 10 digits, '712340000123' = 12)
**Cause:** Leading zero stripped by BIGINT conversion, extra/missing digit in ID generation
**Detection:** `LENGTH(CAST(reach_id AS VARCHAR)) != 11`
**Impact:** Pfafstetter decoding fails; basin codes misaligned; lookup errors
**Mitigation:** Reformat to 11 digits; verify Pfafstetter encoding is correct

### Failure Mode 5: Wrong Region Distribution
**Symptom:** Unexpected change in region counts (e.g., AS drops from 40% to 30%)
**Cause:** Incorrect regional subset loaded; deletion of reaches by region; erroneous filter applied
**Detection:** `SELECT region, COUNT(*) FROM reaches GROUP BY region`
**Expected:** AF:21441, AS:100185, EU:31103, NA:38696, OC:15090, SA:42159
**Impact:** Incomplete data; analysis biased; validation checks invalid
**Mitigation:** Reload full database; verify import parameters

### Failure Mode 6: Non-v17b Version
**Symptom:** `version` contains values other than 'v17b'
**Cause:** Mixed database versions, incomplete upgrade, rollback without cleanup
**Detection:** `SELECT DISTINCT version FROM reaches WHERE version != 'v17b'`
**Impact:** Attributes may not match v17b spec; topology different; analysis unreliable
**Mitigation:** Reload v17b database; upgrade to consistent version

### Failure Mode 7: Null/Missing Identifiers
**Symptom:** NULL values in reach_id, region, or version
**Cause:** Incomplete import, schema change, filtering error
**Detection:** `SELECT COUNT(*) FROM reaches WHERE reach_id IS NULL OR region IS NULL OR version IS NULL`
**Expected:** 0 (all non-null)
**Impact:** Cannot identify or locate reach; breaks joins and lookups
**Mitigation:** Reload data; implement NOT NULL constraints

---

## Proposed Lint Checks

### ID001: reach_id Uniqueness (ERROR)
**Category:** Topology (T)
**Severity:** ERROR
**Check:** `COUNT(DISTINCT reach_id) = COUNT(*)`
**Message:** "reach_id must be globally unique; {count_dup} duplicates detected"
**Action:** Fail on error; review import logs; deduplicate

### ID002: reach_id Format (ERROR)
**Category:** Topology (T)
**Severity:** ERROR
**Check:** `LENGTH(CAST(reach_id AS VARCHAR)) = 11`
**Message:** "reach_id must be exactly 11 digits; {count_bad} malformed IDs"
**Action:** Fail on error; fix Pfafstetter encoding

### ID003: Region-reach_id Consistency (ERROR)
**Category:** Topology (T)
**Severity:** ERROR
**Check:** Pfafstetter continent digit matches region code
**Message:** "{count_mismatch} reaches have reach_id prefix ≠ region code"
**Action:** Fail on error; rebuild region from reach_id or vice versa

### ID004: Type Code Validity (WARNING)
**Category:** Attributes (A)
**Severity:** WARNING
**Check:** Last digit of reach_id in (1,3,4,5,6)
**Message:** "{count_bad} reaches have invalid type code {invalid_types}"
**Action:** Warn; investigate; fix type assignments

### ID005: Region Distribution (INFO)
**Category:** Attributes (A)
**Severity:** INFO
**Check:** Region count distribution
**Message:** "Region distribution: {region_counts}"
**Action:** Info only; flag anomalies in expected distribution

### ID006: Version Homogeneity (WARNING)
**Category:** Attributes (A)
**Severity:** WARNING
**Check:** Single version value (v17b)
**Message:** "Found {count_versions} distinct versions: {versions}; expected 1 (v17b)"
**Action:** Warn if mixed versions; fail if non-v17b

### ID007: null Identifiers (ERROR)
**Category:** Attributes (A)
**Severity:** ERROR
**Check:** No NULL reach_id, region, or version
**Message:** "{count_null} reaches with NULL identifiers"
**Action:** Fail on error; reload data

---

## Reconstruction Rules

**NEVER reconstruct these variables from other attributes.** They are foundational identifiers.

1. **reach_id**: Assigned during network definition by HydroBASINS integration. Do not reassign unless topology is rebuilt from scratch.
   - If corrupted, rebuild from v17b NetCDF source or prior backup
   - If single digit corrupted, fix character by character (preserves Pfafstetter encoding)
   - If completely missing, reload full database

2. **region**: Always derive from reach_id first digit if mismatch detected
   ```python
   region = {
       '1': 'AF', '2': 'EU', '3': 'AS', '4': 'AS',
       '5': 'OC', '6': 'SA', '7': 'NA', '8': 'NA', '9': 'NA'
   }.get(str(reach_id)[0])
   ```

3. **version**: Set based on database release. Do not change without database upgrade.
   - v17c inherits v17b version marker (all reaches labeled v17b)
   - Do not manually update version on individual reaches

---

## Dependencies & Impact

**Primary Key:** `(reach_id, region)` - Though reach_id alone is globally unique
**Referenced By:**
- nodes.reach_id (foreign key)
- reach_topology table (reach_id pairs)
- all topology/descendant tables

**Impact of Changes:**
- Changing reach_id invalidates all downstream references
- Changing region requires Pfafstetter audit
- Changing version affects attribute interpretation

---

## Sources & References

1. **SWORD Product Description Document v17b** (March 2025)
   - Table 3, Table 5: reach_id format specification (CBBBBBRRRRT)
   - Section 4.1 NetCDF: Pfafstetter encoding details
   - Section 4.2 Geopackage: region definitions

2. **HydroBASINS Pfafstetter Coding** (Lehner & Grill, 2013)
   - Source for nested basin hierarchy
   - Reference: https://www.hydrosheds.org

3. **SWORD Database v17b/v17c**
   - Data validation queries confirm all invariants (248,674 reaches)
   - Region distribution: AF:21441, AS:100185, EU:31103, NA:38696, OC:15090, SA:42159

---

## Appendix: Example Queries

### Decode reach_id for a specific reach
```sql
SELECT
  reach_id,
  region,
  SUBSTRING(CAST(reach_id AS VARCHAR), 1, 1) as continent,
  SUBSTRING(CAST(reach_id AS VARCHAR), 2, 5) as pfafstetter_2_6,
  SUBSTRING(CAST(reach_id AS VARCHAR), 7, 4) as reach_seq,
  SUBSTRING(CAST(reach_id AS VARCHAR), 11, 1) as type_code,
  CASE
    WHEN SUBSTRING(CAST(reach_id AS VARCHAR), 11, 1) = '1' THEN 'River'
    WHEN SUBSTRING(CAST(reach_id AS VARCHAR), 11, 1) = '3' THEN 'Lake on river'
    WHEN SUBSTRING(CAST(reach_id AS VARCHAR), 11, 1) = '4' THEN 'Dam/waterfall'
    WHEN SUBSTRING(CAST(reach_id AS VARCHAR), 11, 1) = '5' THEN 'Unreliable topology'
    WHEN SUBSTRING(CAST(reach_id AS VARCHAR), 11, 1) = '6' THEN 'Ghost reach'
  END as type_name
FROM reaches
WHERE reach_id = 23190200061;
```

### Find all reaches in a specific Pfafstetter basin
```sql
-- Find all reaches in Pfafstetter basin 21 (Europe)
SELECT reach_id, region, width, dist_out
FROM reaches
WHERE SUBSTRING(CAST(reach_id AS VARCHAR), 1, 2) = '21'
ORDER BY dist_out DESC;
```

### Identify reaches by type across regions
```sql
SELECT
  region,
  CAST(reach_id AS BIGINT) % 10 as type_code,
  CASE
    WHEN CAST(reach_id AS BIGINT) % 10 = 1 THEN 'River'
    WHEN CAST(reach_id AS BIGINT) % 10 = 3 THEN 'Lake'
    WHEN CAST(reach_id AS BIGINT) % 10 = 4 THEN 'Dam/waterfall'
    WHEN CAST(reach_id AS BIGINT) % 10 = 5 THEN 'Unreliable'
    WHEN CAST(reach_id AS BIGINT) % 10 = 6 THEN 'Ghost'
  END as type_name,
  COUNT(*) as count
FROM reaches
GROUP BY region, type_code, type_name
ORDER BY region, type_code;
```

---

**Document Version:** 1.0
**Last Updated:** February 2, 2026
**Classification:** Technical Documentation
