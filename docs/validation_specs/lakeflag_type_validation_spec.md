# SWORD Lakeflag and Type Field Validation Specification

## Document Purpose

This specification documents the provenance, meaning, and validation rules for the `lakeflag` and `type` fields in SWORD. It serves as the reference for lint check C004 and guides interpretation of these classification fields.

---

## 1. Field Definitions

### 1.1 lakeflag

**Source:** Global River Widths from Landsat (GRWL) [Allen & Pavelsky, 2018]

**Official Definition (SWORD PDD v17b, Table 3):**
> "GRWL water body identifier for each reach: 0 - river, 1 - lake/reservoir, 2 - canal, 3 - tidally influenced river."

**Derivation Algorithm (from RECONSTRUCTION_SPEC.md):**
```python
# At centerline level: raw GRWL values remapped
# Original -> SWORD mapping:
#   255, 250 -> 0 (river)
#   180, 181, 163 -> 1 (lake/reservoir)
#   126, 125 -> 3 (tidally influenced)
#   86 -> 2 (canal)

# At node level: mode of centerline lakeflag values
node_lakeflag = mode(grwl_lakeflag[node_centerline_points])

# At reach level: mode of node lakeflag values
reach_lakeflag = mode(node_lakeflag[reach_nodes])
```

**Valid Values:**

| Value | Name | Description |
|-------|------|-------------|
| 0 | river | Standard river channel |
| 1 | lake | Lake, reservoir, or pond on river network |
| 2 | canal | Artificial waterway |
| 3 | tidal | Tidally influenced river section |

---

### 1.2 type

**Source:** Encoded in reach_id last digit (SWORD internal classification)

**Official Definition (SWORD PDD v17b, Table 5):**
> "Reach type identifier: 1 - river, 3 - lake on river, 4 - dam or waterfall, 5 - unreliable topology, 6 - ghost reach."

**Derivation:**
The `type` field is the last digit of `reach_id`. The reach_id format is:
```
CBBBBBRRRRT
C = Continent (first digit of Pfafstetter basin code)
B = Remaining Pfafstetter basin codes (6 digits)
R = Reach number within basin (4 digits)
T = Type (1 digit)
```

**Verification (from database):**
```
stored_type | last_digit | count
    1       |     1      | 163,580
    3       |     3      |  25,737
    4       |     4      |  21,372
    5       |     5      |  14,883
    6       |     6      |  23,102
```
All stored `type` values exactly match `reach_id % 10`.

**Valid Values:**

| Value | Name | Description |
|-------|------|-------------|
| 1 | river | Standard river reach |
| 3 | lake_on_river | Lake or reservoir section on river network |
| 4 | dam_waterfall | Reach containing dam, lock, or waterfall |
| 5 | unreliable | Unreliable topology (often deltas, coastal areas) |
| 6 | ghost | Ghost reach (headwater/outlet marker) |

**Note:** Type value 2 does not exist in SWORD (reserved but unused).

---

## 2. Key Differences Between Fields

| Aspect | lakeflag | type |
|--------|----------|------|
| **Source** | GRWL (external) | SWORD internal (reach_id encoding) |
| **What it captures** | Water body classification from Landsat | Network topology role |
| **Primary purpose** | Physical water type | SWOT processing guidance |
| **Can change?** | Yes (via reconstruction) | No (encoded in reach_id) |
| **Independence** | Semi-autonomous from type | Semi-autonomous from lakeflag |

**Critical Insight:** These fields classify different aspects:
- **lakeflag** = "What kind of water body is this?" (from satellite imagery)
- **type** = "What is this reach's role in the network topology?" (from SWORD construction)

---

## 3. Expected Relationships

### 3.1 Cross-Tabulation (v17c Database)

| lakeflag | type | count | pct | Interpretation |
|----------|------|-------|-----|----------------|
| 0 (river) | 1 (river) | 162,605 | 65.39% | **Expected primary combination** |
| 0 (river) | 3 (lake_on_river) | 2,322 | 0.93% | River section through lake (islands?) |
| 0 (river) | 4 (dam) | 19,423 | 7.81% | River at dam location |
| 0 (river) | 5 (unreliable) | 3,410 | 1.37% | Unreliable river topology |
| 0 (river) | 6 (ghost) | 16,534 | 6.65% | Ghost river reaches |
| 1 (lake) | 1 (river) | 240 | 0.10% | **Questionable - investigate** |
| 1 (lake) | 3 (lake_on_river) | 21,410 | 8.61% | **Expected primary combination** |
| 1 (lake) | 4 (dam) | 1,418 | 0.57% | Lake at dam |
| 1 (lake) | 5 (unreliable) | 88 | 0.04% | Unreliable lake topology |
| 1 (lake) | 6 (ghost) | 2,406 | 0.97% | Ghost lake reaches |
| 2 (canal) | 1 (river) | 702 | 0.28% | Canal typed as river |
| 2 (canal) | 3 (lake_on_river) | 19 | 0.01% | Canal in lake section |
| 2 (canal) | 4 (dam) | 288 | 0.12% | Canal at dam/lock |
| 2 (canal) | 5 (unreliable) | 146 | 0.06% | Unreliable canal |
| 2 (canal) | 6 (ghost) | 96 | 0.04% | Ghost canal |
| 3 (tidal) | 1 (river) | 33 | 0.01% | Tidal river section |
| 3 (tidal) | 3 (lake_on_river) | 1,986 | 0.80% | Tidal lake/estuary |
| 3 (tidal) | 4 (dam) | 243 | 0.10% | Tidal at dam |
| 3 (tidal) | 5 (unreliable) | 11,239 | 4.52% | **Expected - tidal areas are topologically complex** |
| 3 (tidal) | 6 (ghost) | 4,066 | 1.64% | Ghost tidal reaches |

---

## 4. Valid vs Questionable Combinations

### 4.1 Expected/Valid Combinations

| lakeflag | type | Rationale |
|----------|------|-----------|
| 0 | 1 | River is a river - primary expected combination |
| 0 | 4 | River at dam/waterfall location |
| 0 | 5 | Unreliable river topology (deltas) |
| 0 | 6 | Ghost reaches marking network endpoints |
| 1 | 3 | Lake identified as lake_on_river - primary expected combination |
| 1 | 4 | Lake at dam (reservoir behind dam) |
| 1 | 6 | Ghost lake reaches |
| 2 | 1 | Canal classified as river type (canals often have river-like topology) |
| 2 | 4 | Canal with lock/dam |
| 3 | 5 | **Tidal + unreliable is expected** - tidal areas have complex, unreliable topology |
| 3 | 6 | Ghost tidal reaches (coastal outlets) |
| ANY | 5 | Type=5 indicates unreliable topology regardless of water body type |
| ANY | 6 | Type=6 indicates ghost reach regardless of water body type |

### 4.2 Questionable Combinations (Require Investigation)

| lakeflag | type | Count | Issue |
|----------|------|-------|-------|
| 1 (lake) | 1 (river) | 240 | Lake by GRWL but river type - potential misclassification |
| 0 (river) | 3 (lake_on_river) | 2,322 | River by GRWL but lake_on_river type - islands in lakes? |
| 3 (tidal) | 1 (river) | 33 | Tidal but river type - should be type=5? |

### 4.3 Analysis of Questionable Cases

**lakeflag=1, type=1 (Lake classified as River type):**
- 240 reaches globally (0.10%)
- Examples show wide reaches (up to 8,452m width) with multiple channels
- Likely explanation: These are wide river sections that GRWL classified as lake-like based on Landsat spectral signature, but SWORD assigned river type based on topology
- **Recommendation:** These may be legitimate - wide braided rivers can appear lake-like

**lakeflag=0, type=3 (River classified as Lake-on-river type):**
- 2,322 reaches (0.93%)
- These are river-like centerlines that pass through lakes (e.g., islands in reservoirs)
- **Recommendation:** Valid - type=3 indicates network role, lakeflag=0 indicates the centerline itself is river-like

---

## 5. Why Tidal (lakeflag=3) Often Has Unreliable Topology (type=5)

**Key Finding:** 64% of tidal reaches have type=5 (unreliable topology)

**Explanation:**
1. Tidal areas are typically estuaries, deltas, and coastal zones
2. These areas have:
   - Multiple channels with complex interconnections
   - Channels that appear/disappear with tides
   - Difficult topology determination from satellite imagery
3. SWORD marks these as type=5 to indicate:
   - Flow direction may be uncertain
   - Neighbor relationships may be incomplete
   - Users should treat these reaches with caution

**This is intentional design, not an error.**

---

## 6. Which Field is Authoritative?

**Neither field is strictly authoritative over the other - they answer different questions:**

| Question | Authoritative Field |
|----------|---------------------|
| "What kind of water body is this physically?" | lakeflag (from GRWL) |
| "How should SWOT process this reach?" | type (from reach_id) |
| "Is this reach's topology reliable?" | type (5=unreliable) |
| "Is this a network endpoint?" | type (6=ghost) |
| "Is this reach tidally influenced?" | lakeflag (3=tidal) |
| "Is this an artificial waterway?" | lakeflag (2=canal) |

**For specific validation purposes:**
- To identify lakes/reservoirs: Use `lakeflag = 1`
- To identify lake reaches in network: Use `type = 3`
- To identify problematic topology: Use `type = 5`
- To identify network endpoints: Use `type = 6` OR `end_reach IN (1,2)`

---

## 7. Recommendations for C004 Check

### 7.1 Current Implementation (Informational)

The current C004 check correctly treats this as INFO severity and reports the cross-tabulation for investigation rather than flagging specific combinations as errors.

### 7.2 Recommended Improvements

1. **Keep INFO severity** - The relationships are complex and context-dependent

2. **Add specific warnings for truly questionable combinations:**
   ```python
   # These specific combinations warrant WARNING level attention:
   # - lakeflag=1 AND type=1: Lake classified as river (240 cases, 0.10%)
   #   May indicate GRWL/SWORD classification mismatch
   ```

3. **Add context in report:**
   - Explain that tidal + unreliable is expected
   - Note that canal + river type is normal
   - Highlight the 240 lake/river cases as the main items to investigate

4. **Consider adding supplementary checks:**
   ```python
   # C005: Wide rivers with lake classification
   # Check if lakeflag=1, type=1 reaches have characteristics
   # that explain the classification (width, n_chan_max)
   ```

### 7.3 Updated Check Definition

```python
# C004 should report three categories:
# 1. EXPECTED: Combinations that are normal and expected
#    - lakeflag=0/type=1, lakeflag=1/type=3, lakeflag=3/type=5, any/type=6
# 2. ACCEPTABLE: Combinations that are unusual but have valid explanations
#    - lakeflag=0/type=3, lakeflag=2/type=1
# 3. INVESTIGATE: Combinations that may indicate issues
#    - lakeflag=1/type=1 (240 cases) - primary concern
#    - lakeflag=3/type=1 (33 cases) - minor concern
```

---

## 8. Summary Statistics (v17c)

### 8.1 lakeflag Distribution

| lakeflag | Name | Count | Percentage |
|----------|------|-------|------------|
| 0 | river | 204,294 | 82.15% |
| 1 | lake | 25,562 | 10.28% |
| 2 | canal | 1,251 | 0.50% |
| 3 | tidal | 17,567 | 7.06% |

### 8.2 type Distribution

| type | Name | Count | Percentage |
|------|------|-------|------------|
| 1 | river | 163,580 | 65.78% |
| 3 | lake_on_river | 25,737 | 10.35% |
| 4 | dam_waterfall | 21,372 | 8.59% |
| 5 | unreliable | 14,883 | 5.98% |
| 6 | ghost | 23,102 | 9.29% |

### 8.3 Agreement Rate

- **Perfect agreement** (lakeflag=0+type=1 OR lakeflag=1+type=3): 184,015 reaches (74.0%)
- **Expected disagreement** (type in 4,5,6): 59,357 reaches (23.9%)
- **Questionable combinations** (lake/river mismatch): ~2,600 reaches (1.0%)

---

## References

1. SWORD Product Description Document v17b (March 2025)
2. Altenau et al. (2021). The SWOT Mission River Database (SWORD). Water Resources Research.
3. Allen & Pavelsky (2018). Global extent of rivers and streams. Science.
4. SWORD RECONSTRUCTION_SPEC.md (internal documentation)
5. SWORD DuckDB schema.py (internal documentation)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-02 | Claude (audit) | Initial specification |
