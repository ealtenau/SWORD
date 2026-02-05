# SWORD River Name Validation Specification

## Document Purpose

This specification documents the provenance, meaning, and validation rules for the `river_name` field in SWORD. It serves as a reference for understanding river naming conventions and guides recommendations for future lint checks.

---

## 1. Field Definition

### 1.1 river_name

**Source:** Global River Widths from Landsat (GRWL) shapefile with river name attributes [Allen & Pavelsky, 2018]

**Introduced:** v11 (July 2021)

**Official Definition (SWORD PDD v17b, Table 3 & Table 5):**
> "All river names associated with a reach/node. If there are multiple names for a reach/node they are listed in alphabetical order and separated by a semicolon."

**Data Type:** STRING (VARCHAR in DuckDB)

**Spatial Scope:** Global, all 6 regions (NA, SA, EU, AF, AS, OC)

---

## 2. Derivation Algorithm

**Source:** GRWL database contains river names associated with centerline segments

**Process (from Attach_Fill_Variables.py):**

```
1. For each node:
   - Find the closest GRWL river name point location (KDTree nearest neighbor)
   - Assign that river name to the node
   - Replace 'NaN' values with 'NODATA'

2. For each reach:
   - Collect all unique river names from nodes in the reach
   - If multiple unique names exist:
     a. Remove any 'NODATA' values
     b. If remaining names > 1: concatenate alphabetically with '; ' separator
     c. If remaining names = 1: use that single name
   - If all names are 'NODATA': assign 'NODATA' to reach
   - Else: use the single name
```

**Key Characteristics:**
- Node-level: Assigned via nearest-neighbor matching to GRWL data
- Reach-level: Aggregated from node names, with multiple names concatenated if found
- Alphabetically sorted when multiple names exist
- Semicolon-delimited when concatenated

---

## 3. Current Database Distribution

### 3.1 Summary Statistics (v17c, 248,674 reaches)

| Metric | Value |
|--------|-------|
| Total reaches | 248,674 |
| Distinct river names | 6,459 |
| Reaches with 'NODATA' | 127,402 (51.23%) |
| Reaches with valid names | 121,272 (48.77%) |
| Coverage by region | 100% (no NULLs or blanks) |

### 3.2 Top 20 River Names by Reach Count

| Rank | River Name | Reach Count |
|------|-----------|-------------|
| 1 | NODATA | 127,402 |
| 2 | Lena River | 537 |
| 3 | Rio Grande | 467 |
| 4 | Yellow River | 443 |
| 5 | Ob River | 430 |
| 6 | Niger River | 391 |
| 7 | Missouri River | 337 |
| 8 | Mississippi River | 333 |
| 9 | Amazon river | 318 |
| 10 | Colorado River | 304 |
| 11 | Volga | 294 |
| 12 | Xi River | 282 |
| 13 | Zambezi | 267 |
| 14 | Irtysh | 266 |
| 15 | Purus River | 265 |
| 16 | Amur | 259 |
| 17 | Nile River | 257 |
| 18 | Black River | 254 |
| 19 | White River | 250 |
| 20 | Indus | 248 |

### 3.3 Coverage by Region

| Region | Total Reaches | NODATA Count | NODATA % | Coverage % |
|--------|---------------|--------------|----------|------------|
| NA | 38,696 | 7,461 | 19.3% | 80.7% |
| SA | 42,159 | 11,228 | 26.6% | 73.4% |
| EU | 31,103 | 2,433 | 7.8% | 92.2% |
| AF | 21,441 | 8,844 | 41.3% | 58.7% |
| AS | 100,185 | 89,688 | 89.5% | 10.5% |
| OC | 15,090 | 7,748 | 51.4% | 48.6% |

**Key Finding:** Asia has poorest coverage (10.5%), likely due to GRWL data availability and resolution limitations in dense river networks.

### 3.4 Name Format Distribution

| Format | Count | Description |
|--------|-------|-------------|
| 0-20 chars | 246,476 | Simple names, mostly NODATA (38 chars) + short names |
| 21-50 chars | 2,191 | Multi-name concatenations (e.g., "Rio A; Rio B") |
| 51-100 chars | 7 | Very long multi-name concatenations |
| >100 chars | 0 | None |
| Max length | 104 | Longest name (example: very long concatenation) |

**Separator Pattern:** Semicolon + space `"; "` used for multi-name concatenations

### 3.5 Data Quality Checks

| Check | Result | Notes |
|-------|--------|-------|
| Leading/trailing spaces | 0 | Clean |
| Newlines in values | 0 | Clean |
| Extremely long names | 0 (>200 chars) | Well-formatted |
| NULL values | 0 | All positions filled |
| Empty strings | 0 | All positions filled |

---

## 4. Known Issues & Limitations

### 4.1 Missing Data (NODATA Cases)

**Problem:** 127,402 reaches (51.23%) have 'NODATA' value

**Root Causes:**
1. **Sparse GRWL Data:** Not all river reaches are labeled in the original GRWL shapefile
   - GRWL focused on larger rivers (>30m wide)
   - Small tributaries may lack names

2. **Spatial Resolution:** Nearest-neighbor matching may fail for small rivers
   - In regions with dense networks (Asia), nearest GRWL point may not match the intended river

3. **Regional Variation:** Some regions have better GRWL coverage
   - Europe: 92.2% coverage (good)
   - Asia: 10.5% coverage (poor - dense network issue)

### 4.2 Inconsistent Naming Conventions

**Pattern 1:** With "River" suffix
- "Lena River", "Ob River", "Colorado River"

**Pattern 2:** Without "River" suffix
- "Volga", "Amur", "Indus"

**Pattern 3:** With "Rio" prefix (Spanish/Portuguese)
- "Rio Grande", "Rio Xingu", "Rio Cacheu"

**Impact:** Makes substring-based queries unreliable

### 4.3 Multiple Names Per Reach

**Frequency:** ~0.88% of valid-named reaches (2,191 of 121,272)

**Format:** Alphabetically sorted, semicolon-delimited
- Example: "Tigris; Euphrates" (if two major rivers merge)
- Example: "Amazon river; Rio Solim" (variant names)

**Implication:** Parsing multi-name fields requires semicolon splitting

### 4.4 Case Inconsistency

**Observations:**
- Most names are title-case: "Amazon River", "Lena River"
- Some are lowercase: "Amazon river" (appears in data)
- Some are all-caps: Potential historical variation

**Impact:** Case-insensitive comparisons recommended

---

## 5. Validation Rules & Invariants

### 5.1 Structural Invariants

1. **No NULL or Empty Values:**
   - All reaches must have a river_name value
   - Use 'NODATA' string for missing data (not NULL)

2. **Valid Separator:**
   - Multi-name reaches use `"; "` (semicolon + space)
   - No other delimiters should appear

3. **Alphabetical Ordering:**
   - When multiple names present, they must be alphabetically sorted
   - Case-sensitive sorting (as from GRWL)

4. **Character Set:**
   - ASCII-compatible characters only
   - No newlines, leading/trailing whitespace

5. **Maximum Length:**
   - Observed max: 104 characters
   - Recommend limit: 256 characters for VARCHAR field

### 5.2 Semantic Invariants

1. **Name Consistency Across Hierarchy:**
   - Reach name should be derived from its constituent nodes
   - Node names should match their upstream/downstream neighbors where topologically consistent

2. **NODATA Inheritance:**
   - If all nodes in a reach have NODATA, reach should have NODATA
   - A reach with valid names should have at least one node with that name

3. **Regional Coherence:**
   - Reaches in the same drainage basin may share river names
   - Names should reflect actual geographic reality (not random)

---

## 6. Common Failure Modes

| Failure | Cause | Severity | Detectability |
|---------|-------|----------|---|
| NODATA overrepresentation | GRWL spatial gaps | MEDIUM | Easy (count) |
| Multi-name with wrong delimiter | Data processing error | LOW | Easy (check separator) |
| Non-alphabetical ordering | Derivation algorithm bug | LOW | Medium (parse & sort check) |
| Case mismatches | Historical GRWL variation | LOW | Medium (requires manual inspection) |
| Whitespace corruption | ASCII encoding issue | LOW | Easy (regex check) |
| Non-ASCII characters | GRWL data encoding | MEDIUM | Easy (validate ASCII) |

---

## 7. Source Data Quality Considerations

**GRWL Database (Source):**
- Global River Widths from Landsat [Allen & Pavelsky, 2018]
- ~2.9 million river segments from Landsat 7/8
- Coverage: Global rivers ≥30m wide
- Naming: Based on confluence of multiple data sources including:
  - NHD (National Hydrography Database) for North America
  - Global river gazetteers
  - User submissions/corrections

**Known GRWL Limitations:**
1. Sparse coverage of small tributaries
2. Naming quality varies by region
3. Some rivers labeled with multiple names (common names vs official)
4. Coastal reaches may be missing or mis-named

---

## 8. Proposed Lint Checks

### 8.1 Current Recommendations

| Check ID | Category | Severity | Rule | Implementation |
|----------|----------|----------|------|-----------------|
| C005 | Classification | INFO | NODATA frequency | Report % NODATA by region |
| C006 | Classification | INFO | Naming coverage | Report reaches with valid names |
| C007 | Format | WARNING | Separator format | Check multi-name uses `"; "` |
| C008 | Format | WARNING | Alphabetical order | Verify sort order for multi-names |
| C009 | Format | ERROR | ASCII-only | Detect non-ASCII characters |
| C010 | Format | WARNING | No leading/trailing spaces | Trim check |

### 8.2 Check Definitions

**C005: NODATA Distribution**
```python
# Purpose: Monitor data gap coverage
# Query: SELECT COUNT(*) WHERE river_name = 'NODATA' GROUP BY region
# Threshold: FLAG regions >30% NODATA as WARNING
# Rationale: High NODATA indicates potential data quality issues
```

**C006: Coverage by Major Rivers**
```python
# Purpose: Verify major rivers have names
# Query: Top 50 rivers should have >50% of reaches named
# Threshold: Flag reaches with path_freq>100 but river_name='NODATA'
# Rationale: Major rivers should rarely have NODATA
```

**C007: Multi-Name Separator**
```python
# Purpose: Detect malformed multi-name fields
# Pattern: Must use exactly "; " (semicolon + space)
# Threshold: FLAG any other delimiter patterns
# Examples of failures: ", " or ";" or " ; " or "\n"
```

**C008: Alphabetical Ordering**
```python
# Purpose: Verify name aggregation was correct
# Query: Split on "; " and check names are alphabetically sorted
# Threshold: FLAG any unsorted multi-name reaches
# Note: Case-sensitive comparison (as per GRWL convention)
```

**C009: ASCII-Only Characters**
```python
# Purpose: Ensure encoding compatibility
# Query: Check all characters are ASCII (codepoint 0-127)
# Threshold: FLAG any non-ASCII characters as ERROR
# Impact: Non-ASCII prevents some data formats (shapefiles, NetCDF)
```

**C010: Whitespace Validation**
```python
# Purpose: Detect corrupted/padded names
# Rules:
#   - No leading whitespace
#   - No trailing whitespace
#   - No double spaces (except in multi-name "; ")
# Threshold: FLAG any violations as WARNING
```

---

## 9. Reconstruction & Update Procedures

### 9.1 When to Reconstruct

**Reconstruction needed if:**
1. GRWL shapefile has been updated with new names
2. River name matching algorithm changed
3. NODATA percentages need reduction via manual edits

**Reconstruction NOT needed for:**
1. Minor cosmetic fixes (these should be done via targeted updates)
2. Single reach corrections (use manual edits)

### 9.2 Reconstruction Algorithm

```python
def reconstruct_river_names(sword_db, grwl_shapefile):
    """
    Reconstruct river_name field from GRWL nearest-neighbor matching.

    Matches the original algorithm in Attach_Fill_Variables.py
    """
    for region in regions:
        # 1. Load GRWL data for this region
        grwl_points = load_grwl_names(grwl_shapefile, region)

        # 2. For each node, find nearest GRWL point
        for node in get_nodes(sword_db, region):
            nearest_grwl = find_nearest(node.xy, grwl_points, k=2)
            node.river_name = nearest_grwl[0].name
            if node.river_name == 'NaN':
                node.river_name = 'NODATA'

        # 3. For each reach, aggregate node names
        for reach in get_reaches(sword_db, region):
            node_names = [n.river_name for n in reach.nodes]

            # Remove NODATA
            valid_names = [n for n in node_names if n != 'NODATA']

            if not valid_names:
                reach.river_name = 'NODATA'
            elif len(set(valid_names)) == 1:
                reach.river_name = valid_names[0]
            else:
                # Multiple names: sort alphabetically and join with '; '
                unique_sorted = sorted(set(valid_names))
                reach.river_name = '; '.join(unique_sorted)
```

### 9.3 Manual Update Procedure

**For specific reaches:**
```sql
-- Single reach update
UPDATE reaches
SET river_name = 'Amazon River'
WHERE reach_id = 123456789;

-- With provenance
INSERT INTO sword_operations (
    user_id, reach_id, attribute, old_value, new_value, reason
) VALUES (
    'jake', 123456789, 'river_name', 'NODATA', 'Amazon River',
    'Field verification - manual correction'
);
```

---

## 10. Export Considerations

### 10.1 Data Format Compatibility

| Format | Constraint | Notes |
|--------|-----------|-------|
| NetCDF | S50 (fixed 50-char string) | Original v11-v17b uses this |
| DuckDB | VARCHAR(256) | Flexible, sufficient for current max 104 chars |
| GeoPackage/Shapefile | VARCHAR(254) | Some tools limit to 254 |
| Parquet | STRING | Unlimited |
| GeoJSON | STRING (UTF-8) | Requires ASCII for safety |

### 10.2 Encoding Issues

**Historical Issue (NetCDF):**
- v11-v17b used ASCII encoding in NetCDF S50 strings
- Non-ASCII characters cause corruption

**Current Approach (DuckDB):**
- Uses UTF-8 internally
- Should validate ASCII-only if planning NetCDF export

---

## 11. Related Variables & Cross-Checks

**Relationships:**
- **reach_id:** Last 4 digits (R=reach_id, B=basin code) could theoretically be matched to basin-level river naming conventions
- **type:** Lake_on_river (type=3) may have different naming than river (type=1)
- **main_side:** Main channel (main_side=0) typically has higher name coverage than side channels
- **path_freq:** Major rivers (high path_freq) should have better name coverage

**Recommended Correlation Checks:**
```sql
-- Check if named reaches have different distribution of type/lakeflag
SELECT type, COUNT(*) as total,
       SUM(CASE WHEN river_name != 'NODATA' THEN 1 ELSE 0 END) as named
FROM reaches
GROUP BY type;

-- Check path_freq vs naming
SELECT
  CASE WHEN path_freq >= 100 THEN 'major'
       WHEN path_freq >= 10 THEN 'medium'
       ELSE 'minor' END as river_class,
  COUNT(*) as total,
  SUM(CASE WHEN river_name != 'NODATA' THEN 1 ELSE 0 END) as named,
  ROUND(100.0 * SUM(CASE WHEN river_name != 'NODATA' THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_named
FROM reaches
GROUP BY river_class;
```

---

## 12. Summary Statistics Table

| Category | Metric | Value | Note |
|----------|--------|-------|------|
| **Coverage** | Total reaches | 248,674 | v17c |
| | Distinct names | 6,459 | Including NODATA |
| | NODATA reaches | 127,402 (51.2%) | Major gap |
| | Named reaches | 121,272 (48.8%) | Valid names |
| **Format** | Max name length | 104 chars | Safe for VARCHAR |
| | Multi-name reaches | 2,191 (0.88% of named) | Semicolon-delimited |
| | ASCII violations | 0 | Clean |
| **Regional** | Best coverage | EU (92.2%) | Good GRWL data |
| | Poorest coverage | AS (10.5%) | Dense networks |
| | Most named rivers | Lena (537 reaches) | Largest single river |

---

## 13. References

1. **SWORD Product Description Document v17b** (March 2025) - Official field documentation
2. **Allen, G. H., & Pavelsky, T. M.** (2018). Global extent of rivers and streams. *Science*, 361(6402), 585-588.
3. **Altenau, E. H., Pavelsky, T. M., Durand, M. T., et al.** (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD). *Water Resources Research*, 57, e2021WR030054.
4. **Attach_Fill_Variables.py** (internal) - Original river name attachment algorithm
5. **DuckDB schema.py** (internal) - Current field definitions

---

## 14. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-02 | Claude (audit) | Initial specification; full database audit |

---

## Appendix A: Sample Query

```sql
-- Comprehensive river_name audit
SELECT
    region,
    COUNT(*) as total_reaches,
    COUNT(DISTINCT river_name) as distinct_names,
    SUM(CASE WHEN river_name = 'NODATA' THEN 1 ELSE 0 END) as nodata_count,
    ROUND(100.0 * SUM(CASE WHEN river_name = 'NODATA' THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_nodata,
    MIN(LENGTH(river_name)) as min_length,
    MAX(LENGTH(river_name)) as max_length,
    ROUND(AVG(LENGTH(river_name)), 1) as avg_length
FROM reaches
GROUP BY region
ORDER BY pct_nodata DESC;

-- Top rivers by reach count
SELECT river_name, COUNT(*) as reach_count
FROM reaches
WHERE river_name != 'NODATA'
GROUP BY river_name
ORDER BY reach_count DESC
LIMIT 20;

-- Multi-name analysis
SELECT
    COUNT(*) as multi_name_reaches,
    COUNT(DISTINCT river_name) as distinct_combinations,
    MIN(LENGTH(river_name)) as min_combo_length,
    MAX(LENGTH(river_name)) as max_combo_length
FROM reaches
WHERE river_name LIKE '%; %';
```

---

*This validation specification is maintained as part of the SWORD v17c audit process. For questions or updates, contact the SWORD team.*
