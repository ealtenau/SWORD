# SWORD v17c Lint Analysis Report

This document summarizes the lint analysis results and provides actionable insights.

## Executive Summary

| Category | Checks | Issues | Key Finding |
|----------|--------|--------|-------------|
| **Topology** | 7 | Low | Delta bifurcations cause most issues |
| **Attributes** | 6 | Medium | `trib_flag` 13% inconsistent (fixable) |
| **Geometry** | 3 | Low | Short reaches cause slope artifacts |
| **Classification** | 4 | Low | ~1.5% lake sandwiches (expected) |

**Overall Assessment:** Database is in good shape. Most "issues" are either:
1. **Edge cases** in complex network regions (deltas, bifurcations)
2. **Stale derived attributes** that need recalculation (trib_flag)
3. **Legitimate physical features** flagged for review (lake sandwiches)

---

## Topology Issues (T0xx)

### T001: dist_out Monotonicity (16 issues, 0.04%)

**Status:** ✅ Acceptable

**Analysis:**
All 16 issues in NA region occur in:
- **Mackenzie River Delta** - Complex braided channel system
- **Mississippi Delta** - Distributary channels
- **Bifurcation points** - Where rivers split

**Root Cause:**
At bifurcations, flow direction assignment may have designated one branch as "main" and another as "tributary". The `dist_out` was calculated based on the main stem path, but topology edges connect to the "other" branch.

**Sample Issues:**
| reach_id | Location | dist_out Increase |
|----------|----------|-------------------|
| 82100800355 | Neyuk/Peel River | 55 km |
| 82213000045 | Mackenzie | 50 km |
| 74300400105 | Atchafalaya | 19 km |

**Recommendation:**
These are structural issues with how SWORD handles bifurcations. For v17c, document as known behavior. For v18, consider marking bifurcation edges explicitly.

---

### T002: path_freq Monotonicity (1,536 issues, 0.64%)

**Status:** ⚠️ Needs Recalculation

**Analysis:**
Path frequency decreases at 0.64% of edges. This is higher than expected and suggests `path_freq` is stale relative to the current topology.

**Recommendation:**
Recalculate `path_freq` using:
```python
workflow.recalculate_path_freq()  # or equivalent
```

---

### T003: facc Monotonicity (7,355 issues, 2.96%)

**Status:** ⚠️ Expected in Deltas

**Analysis:**
The worst cases show facc dropping from millions of km² to near-zero:

| reach_id | Location | facc Drop |
|----------|----------|-----------|
| 74100900025 | Mississippi South Pass | 2.9M → 14 km² |
| 74300400575 | Atchafalaya | 2.9M → 942 km² |
| 82100700035 | Mackenzie | 1.8M → 370 km² |

**Root Cause:**
These are **distributary channels** in deltas. MERIT Hydro's facc follows only one branch at bifurcations, so the "other" branch appears to have much lower accumulation.

**Recommendation:**
- Accept as known behavior for delta regions
- Consider flagging reaches in deltas for special handling
- For v18, investigate bifurcation-aware facc calculation

---

### T004: Orphan Reaches (0 issues)

**Status:** ✅ No Issues

All reaches are connected to the network.

---

### T005: Neighbor Count Consistency (1 issue globally)

**Status:** ✅ Nearly Perfect

Only 1 reach globally has mismatched neighbor counts. This indicates excellent topology table consistency.

---

### T006: Connected Components (246 networks)

**Status:** ✅ Informational

NA region contains 246 distinct networks with no single-reach networks (orphans).

---

### T007: Topology Reciprocity (2 issues globally)

**Status:** ✅ Excellent

Only 2 edges globally lack reciprocal entries. Should be trivial to fix by adding the missing edges.

---

## Attribute Issues (A0xx)

### A001: WSE Monotonicity (1,505 issues, 0.74%)

**Status:** ⚠️ Review Needed

**Analysis:**
Most issues show small increases (measurement noise), but some extreme cases exist:

| reach_id | Location | WSE Increase |
|----------|----------|--------------|
| 85109500031 | Unknown | +385 m |
| 71358000021 | Unknown | +240 m |
| 72360000591 | Joir River | +172 m |
| 81242000391 | Toklat River | +146 m |

**Root Cause:**
1. **Measurement errors** - SWOT uncertainty in narrow/shallow reaches
2. **Temporal mismatch** - WSE from different observation times
3. **Flow direction errors** - Some reaches may have incorrect direction

**Recommendation:**
- Extreme cases (>50m increase) should be flagged for manual review
- Consider filtering by WSE measurement quality flags

---

### A002: Slope Reasonableness (449 issues, 0.22%)

**Status:** ✅ Acceptable

**Analysis:**
All extreme slopes (>100 m/km) occur on **very short reaches**:

| reach_id | Length | Slope (m/km) |
|----------|--------|--------------|
| 73260400481 | 11 m | 11,054 |
| 81242000551 | 32 m | 9,512 |
| 72360000591 | 39 m | 8,891 |

**Root Cause:**
Slope = (wse_up - wse_down) / reach_length. When reach_length is very small, any WSE measurement error gets amplified dramatically.

**Recommendation:**
- Filter slope calculations for reaches < 100m
- Or use smoothed slope over multiple reach segments

---

### A003: Width Trend (1,597 issues, 2.16%)

**Status:** ✅ Informational

Width variability is natural. Flagged cases include:
- Gorges and constrained sections
- Transitions from lakes to rivers
- Anthropogenic narrowing (dams, bridges)

No action needed - for review only.

---

### A004: Attribute Completeness

**Status:** ✅ Complete

No attributes have >5% missing values in the required set:
- dist_out, facc, wse, width, slope
- reach_length, lakeflag, n_rch_up, n_rch_down

---

### A005: trib_flag Consistency (32,615 issues, 13.12%)

**Status:** 🔴 Needs Fix

**Analysis:**
This is the most significant data quality issue found:

| Issue Type | Count | % of Total |
|------------|-------|------------|
| flag=1 but n_rch_up ≤ 1 | ~13,000 | ~5% |
| flag=0 but n_rch_up > 1 | ~19,000 | ~8% |

**Root Cause:**
`trib_flag` was calculated from original SWORD topology, but the topology has been modified (edges added/removed) without updating `trib_flag`.

**Fix:**
```sql
UPDATE reaches SET trib_flag = CASE
    WHEN n_rch_up > 1 THEN 1
    ELSE 0
END;
```

**Recommendation:**
Add this to the v17c release workflow. This is a simple recalculation.

---

### A006: Attribute Outliers (21 issues, 0.01%)

**Status:** ✅ Minimal

Only 21 reaches globally have extreme outlier values. These should be reviewed individually.

---

## Geometry Issues (G0xx)

### G001: Reach Length Bounds (212 issues, 0.09%)

**Status:** ✅ Acceptable

**Analysis:**
212 reaches are too short (<100m). Most have `end_reach=2` which may indicate a special classification.

**Key Insight:**
The shortest reaches correlate with the highest slopes (see A002). This is a geometry → attribute error propagation issue.

**Recommendation:**
Investigate what `end_reach=2` means and whether these reaches should be filtered from slope calculations.

---

### G002: Node Length Consistency (0 issues)

**Status:** ✅ Perfect

All reach lengths match their node length sums within 10% tolerance.

---

### G003: Zero Length Reaches (0 issues)

**Status:** ✅ No Issues

---

## Classification Issues (C0xx)

### C001: Lake Sandwiches (3,006 issues, 1.47%)

**Status:** ✅ Expected

**Analysis:**
Distribution by length:
| Length Bucket | Count | % |
|---------------|-------|---|
| 500m - 2km | 1,242 | 41% |
| 2km - 10km | 1,221 | 41% |
| < 500m | 312 | 10% |
| > 10km | 231 | 8% |

**Key Finding:**
Most lake sandwiches are legitimate features:
- **Connecting channels** between lakes
- **Lake narrows** where width decreases
- **River sections** within lake chains

The longer sandwiches (>10km) are more likely misclassified.

**Recommendation:**
- Short sandwiches (<2km): Accept as legitimate
- Long sandwiches (>10km): Review for potential reclassification

---

### C002-C004: Type/Lakeflag Distribution

**Status:** ✅ Informational

`type` column not present in v17c database - checks skip gracefully.

---

## Recommended Actions

### Priority 1: Must Fix

1. **Recalculate trib_flag** - Simple SQL update, 13% of reaches affected
   ```sql
   UPDATE reaches SET trib_flag = CASE WHEN n_rch_up > 1 THEN 1 ELSE 0 END;
   ```

### Priority 2: Should Consider

2. **Recalculate path_freq** - 0.64% monotonicity violations
3. **Add T007 reciprocal edges** - Only 2 missing edges globally
4. **Filter slope for short reaches** - Exclude reaches < 100m from slope calculations

### Priority 3: Document as Known Behavior

5. **Delta bifurcation issues** - T001, T003 violations are structural
6. **Lake sandwiches** - 1.5% rate is expected and acceptable
7. **WSE increases** - Small increases are measurement noise

### Priority 4: Future Versions (v18)

8. **Bifurcation-aware facc calculation**
9. **Explicit bifurcation edge marking**
10. **WSE quality flags integration**

---

## Issue Overlap

94.6% of problematic reaches have only **one** type of issue. Only 5.4% have two or more issues overlapping. This indicates:
- Issues are relatively independent
- Fixing one type won't cascade to others
- Targeted fixes are appropriate

---

## Regional Comparison

| Region | Reaches | T001 % | A001 % | A005 % | C001 % |
|--------|---------|--------|--------|--------|--------|
| NA | 38,696 | 0.04% | 0.65% | 14.50% | 1.47% |
| (other regions TBD) |

---

## Conclusion

The SWORD v17c database is in good condition. The lint framework identified:
- **1 actionable fix** (trib_flag) affecting 13% of reaches
- **2 minor fixes** (path_freq, reciprocal edges)
- **Structural limitations** in delta regions (expected)
- **Known acceptable variations** (lake sandwiches, width changes)

The most impactful improvement would be recalculating `trib_flag` to match current topology.
