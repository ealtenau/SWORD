# Validation Spec: Obstruction Variables (obstr_type, grod_id, hfalls_id)

## Summary

| Property | obstr_type | grod_id | hfalls_id |
|----------|------------|---------|-----------|
| **Source** | GROD + HydroFALLS | GROD | HydroFALLS |
| **Units** | Categorical (0-4) | Database ID | Database ID |
| **Applied to** | Nodes, Reaches | Nodes, Reaches | Nodes, Reaches |

**Official definitions (v17b PDD, Tables 3-5):**

- **obstr_type**: "Type of obstruction for each [node/reach] based on GROD and HydroFALLS databases. Obstr_type values: 0 - No Dam, 1 - Dam, 2 - Lock, 3 - Low Permeable Dam, 4 - Waterfall."
- **grod_id**: "The unique GROD ID for each [node/reach] with obstr_type values 1-3."
- **hfalls_id**: "The unique HydroFALLS ID for each [node/reach] with obstr_type value 4."

---

## Source Datasets

### GROD (Global River Obstruction Database)

**Reference:** Whittemore et al. (2020). "A Participatory Science Approach to Expanding Instream Infrastructure Inventories." Earth's Future, 8(11), e2020EF001558.

**Description:** GROD provides global locations of anthropogenic river obstructions (dams, locks, low-permeable barriers) along the GRWL river network.

**File:** `data/inputs/GROD/GROD_ALL.csv`

**Key fields:**
- `lat`, `lon` - Location coordinates
- `grod_fid` - Unique GROD feature ID (mapped to `grod_id` in SWORD)
- Obstruction type classification (mapped to `obstr_type` 1-3)

### HydroFALLS

**Reference:** http://wp.geog.mcgill.ca/hydrolab/hydrofalls/

**Description:** HydroFALLS provides global locations of waterfalls and natural river obstructions.

**File:** `data/inputs/HydroFalls/hydrofalls.csv` (filtered version: `hydrofalls_filt.csv`)

**Key fields:**
- `FALLS_ID` - Unique waterfall ID (mapped to `hfalls_id` in SWORD)
- `LAT_HYSD`, `LONG_HYSD` - Location coordinates
- `CONFIDENCE` - Confidence rating
- `CONTINENT` - Continental region

**Filtering:** HydroFALLS points within 500m of GROD locations are removed to avoid double-counting (see `hydrofalls_filtering.py`).

---

## Code Path

### Primary Implementation

**Original Production:**
- **File:** `src/development/reach_definition/Reach_Definition_Tools_v11.py`
- **Node computation:** `basin_node_attributes()` lines 4534-4556
- **Reach computation:** `reach_attributes()` lines 4927-4937

### Algorithm (Original Production)

```python
# From Reach_Definition_Tools_v11.py, basin_node_attributes()

# 1. Get GROD values for centerlines in this node
GROD = np.copy(grod_id[nodes])

# 2. Reset values > 4 to 0 (invalid obstruction codes)
GROD[np.where(GROD > 4)] = 0

# 3. Node obstr_type = MAX of centerline GROD values
node_grod_id[ind] = np.max(GROD)

# 4. Assign corresponding grod_id or hfalls_id based on obstr_type
ID = np.where(GROD == np.max(GROD))[0][0]  # First matching index
if np.max(GROD) == 0:
    node_grod_fid[ind] = 0           # No obstruction
elif np.max(GROD) == 4:
    node_hfalls_fid[ind] = hfalls_fid[nodes[ID]]  # Waterfall -> HydroFALLS ID
else:
    node_grod_fid[ind] = grod_fid[nodes[ID]]      # Dam/Lock/Low-perm -> GROD ID
```

### Current Reconstruction (DuckDB)

**File:** `src/sword_duckdb/reconstruction.py`

**AttributeSpec definitions (lines 301-326):**
```python
"reach.obstr_type": AttributeSpec(
    name="reach.obstr_type",
    source=SourceDataset.GROD,
    method=DerivationMethod.MAX,
    source_columns=["obstruction_type"],
    dependencies=["centerline.grod"],
    description="Obstruction type: np.max(GROD[reach]), values >4 reset to 0. 0=none, 1=dam, 2=lock, 3=low-perm, 4=waterfall"
),

"reach.grod_id": AttributeSpec(
    name="reach.grod_id",
    source=SourceDataset.GROD,
    method=DerivationMethod.SPATIAL_JOIN,
    source_columns=["grod_fid"],
    dependencies=["reach.obstr_type"],
    description="GROD database ID at max obstruction point"
),

"reach.hfalls_id": AttributeSpec(
    name="reach.hfalls_id",
    source=SourceDataset.HYDROFALLS,
    method=DerivationMethod.SPATIAL_JOIN,
    source_columns=["hfalls_fid"],
    dependencies=["reach.obstr_type"],
    description="HydroFALLS ID (only if obstr_type == 4)"
),
```

**Reconstruction methods:**
- `_reconstruct_reach_obstr_type()` (lines 3661-3694): MAX of node obstr_type
- `_reconstruct_node_obstr_type()` (lines 3067-3105): Inherits from reach (fallback)
- `_reconstruct_reach_grod_id()` (lines 3912-3939): **STUB** - preserves existing values
- `_reconstruct_reach_hfalls_id()` (lines 3941-3968): **STUB** - preserves existing values

**Note:** `grod_id` and `hfalls_id` reconstruction are stubs requiring external GROD/HydroFALLS spatial data not currently integrated into DuckDB.

---

## Valid Values

### obstr_type

| Value | Meaning | Source | ID Field |
|-------|---------|--------|----------|
| 0 | No obstruction | N/A | Neither (both = 0) |
| 1 | Dam | GROD | grod_id |
| 2 | Lock | GROD | grod_id |
| 3 | Low Permeable Dam | GROD | grod_id |
| 4 | Waterfall | HydroFALLS | hfalls_id |
| **5** | **Undocumented** | Unknown | Should not exist |

### grod_id and hfalls_id

| Field | Valid Range | Null/Zero Meaning |
|-------|-------------|-------------------|
| grod_id | 1 - ~30,350 | No GROD obstruction |
| hfalls_id | 1 - ~3,934 | No waterfall |

---

## Database Statistics (v17b)

### Reach obstr_type Distribution

| obstr_type | Count | Percentage | Description |
|------------|-------|------------|-------------|
| 0 | 227,194 | 91.36% | No obstruction |
| 1 | 8,104 | 3.26% | Dam |
| 2 | 1,622 | 0.65% | Lock |
| 3 | 10,500 | 4.22% | Low permeable dam |
| 4 | 1,248 | 0.50% | Waterfall |
| 5 | 5 | 0.00% | **UNDOCUMENTED** |

### Node obstr_type Distribution

| obstr_type | Count | Percentage |
|------------|-------|------------|
| 0 | 11,090,464 | 99.80% |
| 1 | 8,260 | 0.07% |
| 2 | 1,670 | 0.02% |
| 3 | 10,804 | 0.10% |
| 4 | 1,251 | 0.01% |
| 5 | 5 | 0.00% |

### Regional Breakdown (Reaches)

| Region | Dams | Locks | Low-Perm | Waterfalls | Total |
|--------|------|-------|----------|------------|-------|
| AF | 199 | 23 | 267 | 155 | 644 |
| AS | 5,045 | 264 | 6,169 | 109 | 11,587 |
| EU | 1,473 | 1,036 | 2,664 | 150 | 5,323 |
| NA | 814 | 276 | 776 | 264 | 2,130 |
| OC | 173 | 11 | 253 | 33 | 470 |
| SA | 400 | 12 | 371 | 537 | 1,320 |

### ID Statistics

| Metric | grod_id | hfalls_id |
|--------|---------|-----------|
| Unique IDs (reaches) | 20,129 | 1,230 |
| Non-zero count (reaches) | 20,247 | 1,232 |
| Min value | 1 | 1 |
| Max value | 30,350 | 3,934 |

---

## Failure Modes

### 1. Undocumented obstr_type=5 (BUG IDENTIFIED)

**Description:** 5 reaches have `obstr_type=5`, which is not documented in the PDD. All 5 have `hfalls_id != 0`, suggesting they should be waterfalls (type=4).

**Affected reaches:**
| reach_id | region | hfalls_id | lakeflag |
|----------|--------|-----------|----------|
| 82291000364 | NA | 18 | 0 (river) |
| 72510000035 | NA | 514 | 3 (tidal) |
| 78220000044 | NA | 628 | 0 (river) |
| 73120000244 | NA | 862 | 1 (lake) |
| 73120000424 | NA | 783 | 0 (river) |

**Root cause:** Values >4 should be reset to 0, but these appear to have been assigned `obstr_type=5` incorrectly. The algorithm clamps values >4 to 0, but `obstr_type=5` may have been assigned through a different code path.

**Impact:** 5 reaches with incorrectly classified obstruction type; these should be `obstr_type=4` (waterfall).

**Severity:** LOW (only 5 reaches)

### 2. obstr_type/grod_id Mismatch

**Description:** 21 reaches have `grod_id != 0` but `obstr_type = 4` (waterfall). These should have `grod_id = 0` since waterfalls use HydroFALLS.

**Investigation:** All 21 are in region NA and have reach IDs ending in 4 (dam/waterfall type). Sample:
- 72510000854, 72510000864, 72510000874, etc.

**Root cause:** These appear to be reaches where GROD and HydroFALLS locations overlap. The GROD 500m filtering may have been incomplete.

**Impact:** grod_id contains valid GROD IDs, but obstr_type indicates waterfall. The obstruction is likely a dam co-located with a waterfall.

**Severity:** LOW (21 reaches)

### 3. obstr_type/hfalls_id Mismatch

**Description:** 5 reaches have `hfalls_id != 0` but `obstr_type != 4`. All 5 have `obstr_type = 5` (undocumented).

**This is the same as Failure Mode 1.** The `obstr_type=5` bug causes these reaches to show hfalls_id without proper obstr_type classification.

### 4. Missing IDs for Obstructed Reaches

**Check:** Are there reaches with `obstr_type in (1,2,3)` but `grod_id = 0`?

**Result:** 0 reaches - **No issues found.** All GROD-sourced obstruction types have valid grod_id.

**Check:** Are there reaches with `obstr_type = 4` but `hfalls_id = 0`?

**Result:** 21 reaches have this condition. This is because these 21 reaches have both GROD and HydroFALLS locations (see Failure Mode 2), and the algorithm assigned `hfalls_id` from the first matching waterfall point which may be 0.

### 5. Spatial Join Ambiguity

**Description:** When multiple GROD or HydroFALLS points fall within a reach, the algorithm takes the first matching point at the MAX obstruction type location.

**Impact:** If multiple obstructions exist, only one grod_id/hfalls_id is recorded. The choice of which ID is recorded depends on centerline point ordering, not necessarily the most significant obstruction.

**Mitigation:** Consider storing all obstruction IDs in an array column for v18.

### 6. Centerline-Node-Reach Aggregation

**Description:** obstr_type is computed as MAX from centerlines to nodes, then MAX from nodes to reaches. This means a single obstructed centerline point propagates to the entire reach.

**Impact:** A short dam section influences the entire reach's obstr_type.

**Acceptable behavior:** This is intentional - reaches containing ANY obstruction should be flagged.

---

## Existing Lint Checks

**None found.** The current lint framework (`src/sword_duckdb/lint/checks/`) does not include any checks for obstruction variables.

---

## Proposed New Checks

### O001: obstr_type_validity (ERROR)

**Description:** Check that obstr_type is in valid range [0-4].

**Rule:** `obstr_type NOT IN (0, 1, 2, 3, 4)` should be 0 count.

**SQL:**
```sql
SELECT reach_id, region, obstr_type, grod_id, hfalls_id
FROM reaches
WHERE obstr_type NOT IN (0, 1, 2, 3, 4)
```

**Expected failures:** 5 reaches (obstr_type=5)

### O002: grod_id_consistency (WARNING)

**Description:** Check that grod_id is non-zero only when obstr_type in (1,2,3).

**Rules:**
1. If `obstr_type IN (1,2,3)` then `grod_id > 0`
2. If `grod_id > 0` then `obstr_type IN (1,2,3)`

**SQL:**
```sql
-- Check 1: GROD obstructions should have grod_id
SELECT reach_id FROM reaches
WHERE obstr_type IN (1, 2, 3) AND (grod_id IS NULL OR grod_id = 0)

-- Check 2: grod_id should imply GROD obstruction type
SELECT reach_id FROM reaches
WHERE grod_id > 0 AND obstr_type NOT IN (1, 2, 3)
```

**Expected failures:** 0 for Check 1; 21 for Check 2 (obstr_type=4 with grod_id, co-located obstructions)

### O003: hfalls_id_consistency (WARNING)

**Description:** Check that hfalls_id is non-zero only when obstr_type = 4.

**Rules:**
1. If `obstr_type = 4` then `hfalls_id > 0`
2. If `hfalls_id > 0` then `obstr_type = 4`

**SQL:**
```sql
-- Check 1: Waterfalls should have hfalls_id
SELECT reach_id FROM reaches
WHERE obstr_type = 4 AND (hfalls_id IS NULL OR hfalls_id = 0)

-- Check 2: hfalls_id should imply waterfall type
SELECT reach_id FROM reaches
WHERE hfalls_id > 0 AND obstr_type != 4
```

**Expected failures:** 21 for Check 1 (co-located with GROD); 5 for Check 2 (obstr_type=5 bug)

### O004: obstruction_mutual_exclusivity (INFO)

**Description:** Check that reaches don't have both grod_id and hfalls_id non-zero (should be mutually exclusive).

**SQL:**
```sql
SELECT reach_id, grod_id, hfalls_id, obstr_type
FROM reaches
WHERE grod_id > 0 AND hfalls_id > 0
```

**Expected failures:** 0 (confirmed)

### O005: node_reach_obstr_consistency (WARNING)

**Description:** Check that reach obstr_type equals MAX of its node obstr_types.

**SQL:**
```sql
SELECT r.reach_id, r.obstr_type as reach_obstr, MAX(n.obstr_type) as max_node_obstr
FROM reaches r
JOIN nodes n ON r.reach_id = n.reach_id AND r.region = n.region
GROUP BY r.reach_id, r.obstr_type
HAVING r.obstr_type != MAX(n.obstr_type)
```

### O006: reach_type_obstr_consistency (INFO)

**Description:** Check that reaches with obstr_type=4 (waterfall) or obstr_type in (1,2,3) (dams) have reach type=4 (dam/waterfall type).

**SQL:**
```sql
SELECT reach_id, obstr_type, reach_id % 10 as reach_type
FROM reaches
WHERE obstr_type IN (1, 2, 3, 4) AND reach_id % 10 != 4
```

**Note:** Most obstructed reaches have type=4 (dam_type), but some have type=1 (river) or type=3 (lake_on_river). This may be acceptable but worth reporting.

---

## Edge Cases

### Co-located GROD and HydroFALLS

Some locations have both anthropogenic obstructions (dams) and natural obstructions (waterfalls) nearby. The HydroFALLS filtering (500m buffer) removes most but not all overlaps.

**Recommendation:** For v18, consider allowing both grod_id and hfalls_id to be non-zero, or add a flag for "co-located obstructions."

### Multiple Obstructions per Reach

A reach may contain multiple dams or waterfalls. Only one ID is stored (the first at the MAX obstruction type location).

**Recommendation:** For v18, consider storing obstruction IDs as arrays: `grod_ids BIGINT[]`, `hfalls_ids BIGINT[]`.

### Reach Type Encoding

Reach IDs encode type in the last digit. Type=4 indicates "dam or waterfall." However, some obstructed reaches have type=1 (river) or type=3 (lake_on_river) based on original GRWL classification.

**Current behavior:** obstr_type is computed independently from reach ID type.

**Consideration:** The reach ID type may not always match obstr_type. For example:
- Reach type=4 with obstr_type=0: Reach was originally classified as dam but later found to have no obstruction
- Reach type=1 with obstr_type=1: River reach with a dam not in original classification

---

## Recommendations

### Short-term (v17c)

1. **Fix obstr_type=5 bug:** Change 5 reaches from obstr_type=5 to obstr_type=4 (they all have hfalls_id)
2. **Add lint checks O001-O004:** Implement basic consistency validation
3. **Document co-located obstructions:** Note the 21 reaches with both GROD and HydroFALLS IDs

### Medium-term (v18)

1. **Re-run HydroFALLS filtering:** Increase buffer or use point-in-polygon instead of distance
2. **Support multiple obstruction IDs per reach:** Use array columns
3. **Add obstruction metadata:** Include obstruction name, height, year built from source datasets
4. **Full reconstruction:** Integrate GROD/HydroFALLS spatial data into DuckDB pipeline

---

## References

1. **SWORD v17b PDD** - Tables 3-5 (variable definitions)
2. **Whittemore et al., 2020** - GROD dataset documentation
3. **HydroFALLS** - http://wp.geog.mcgill.ca/hydrolab/hydrofalls/
4. **reconstruction.py** - AttributeSpecs and reconstruction methods
5. **Reach_Definition_Tools_v11.py** - Original production algorithm
6. **hydrofalls_filtering.py** - HydroFALLS preprocessing

---

## Appendix: SQL Validation Queries

```sql
-- Complete obstruction audit query
SELECT
    obstr_type,
    COUNT(*) as count,
    SUM(CASE WHEN grod_id > 0 THEN 1 ELSE 0 END) as has_grod_id,
    SUM(CASE WHEN hfalls_id > 0 THEN 1 ELSE 0 END) as has_hfalls_id,
    SUM(CASE WHEN grod_id > 0 AND hfalls_id > 0 THEN 1 ELSE 0 END) as has_both
FROM reaches
GROUP BY obstr_type
ORDER BY obstr_type;

-- Find obstr_type=5 reaches (bug)
SELECT reach_id, region, grod_id, hfalls_id, lakeflag, river_name
FROM reaches
WHERE obstr_type = 5;

-- Find inconsistent grod_id/obstr_type
SELECT reach_id, obstr_type, grod_id, hfalls_id
FROM reaches
WHERE (grod_id > 0 AND obstr_type NOT IN (1, 2, 3))
   OR (obstr_type IN (1, 2, 3) AND (grod_id IS NULL OR grod_id = 0));

-- Find inconsistent hfalls_id/obstr_type
SELECT reach_id, obstr_type, grod_id, hfalls_id
FROM reaches
WHERE (hfalls_id > 0 AND obstr_type != 4)
   OR (obstr_type = 4 AND (hfalls_id IS NULL OR hfalls_id = 0));
```
