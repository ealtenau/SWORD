# SWORD Lint Check Reference

This document provides detailed documentation for each lint check in the SWORD linting framework.

---

## Table of Contents

- [Topology Checks (T0xx)](#topology-checks-t0xx)
  - [T001: dist_out_monotonicity](#t001-dist_out_monotonicity)
  - [T002: path_freq_monotonicity](#t002-path_freq_monotonicity)
  - [T003: facc_monotonicity](#t003-facc_monotonicity)
  - [T004: orphan_reaches](#t004-orphan_reaches)
  - [T005: neighbor_count_consistency](#t005-neighbor_count_consistency)
  - [T006: connected_components](#t006-connected_components)
  - [T007: topology_reciprocity](#t007-topology_reciprocity)
- [Attribute Checks (A0xx)](#attribute-checks-a0xx)
  - [A001: wse_monotonicity](#a001-wse_monotonicity)
  - [A002: slope_reasonableness](#a002-slope_reasonableness)
  - [A003: width_trend](#a003-width_trend)
  - [A004: attribute_completeness](#a004-attribute_completeness)
  - [A005: trib_flag_consistency](#a005-trib_flag_consistency)
  - [A006: attribute_outliers](#a006-attribute_outliers)
- [Geometry Checks (G0xx)](#geometry-checks-g0xx)
  - [G001: reach_length_bounds](#g001-reach_length_bounds)
  - [G002: node_length_consistency](#g002-node_length_consistency)
  - [G003: zero_length_reaches](#g003-zero_length_reaches)
  - [G013: width_gt_length](#g013-width_gt_length)
  - [G014: duplicate_geometry](#g014-duplicate_geometry)
  - [G015: node_reach_distance](#g015-node_reach_distance)
  - [G016: node_spacing](#g016-node_spacing)
  - [G017: cross_reach_nodes](#g017-cross_reach_nodes)
  - [G018: dist_out_vs_reach_length](#g018-dist_out_vs_reach_length)
  - [G021: reach_overlap](#g021-reach_overlap)
- [Classification Checks (C0xx)](#classification-checks-c0xx)
  - [C001: lake_sandwich](#c001-lake_sandwich)
  - [C002: lakeflag_distribution](#c002-lakeflag_distribution)
  - [C003: type_distribution](#c003-type_distribution)
  - [C004: lakeflag_type_consistency](#c004-lakeflag_type_consistency)
- [Flag Checks (FLxxx)](#flag-checks-flxxx)
  - [FL001: swot_obs_coverage](#fl001-swot_obs_coverage)
  - [FL002: iceflag_values](#fl002-iceflag_values)
  - [FL003: low_slope_flag_consistency](#fl003-low_slope_flag_consistency)
  - [FL004: edit_flag_format](#fl004-edit_flag_format)
- [Network Checks (Nxxx)](#network-checks-nxxx)
  - [N001: main_side_values](#n001-main_side_values)
  - [N002: main_side_stream_order](#n002-main_side_stream_order)
- [SWOT Attribute Checks (A02x)](#swot-attribute-checks-a02x)
  - [A021: wse_obs_vs_wse](#a021-wse_obs_vs_wse)
  - [A024: width_obs_vs_width](#a024-width_obs_vs_width)
  - [A026: slope_obs_nonneg](#a026-slope_obs_nonneg)
  - [A027: slope_obs_extreme](#a027-slope_obs_extreme)

---

## Topology Checks (T0xx)

Topology checks validate the river network structure and flow direction consistency.

### T001: dist_out_monotonicity

| Property | Value |
|----------|-------|
| **Severity** | ERROR |
| **Default Threshold** | 100.0 (meters) |
| **Category** | Topology |

#### What It Checks

Verifies that `dist_out` (distance to outlet) **decreases** as you move downstream. For any reach and its downstream neighbor, the downstream reach should have a smaller `dist_out` value.

#### Why It Matters

`dist_out` is a fundamental topology attribute that encodes the network structure. Water flows from high `dist_out` (headwaters) to low `dist_out` (outlet). Violations indicate:

1. **Flow direction errors** - The topology incorrectly identifies which direction is "downstream"
2. **Corrupted topology** - The `reach_topology` table has incorrect edges
3. **dist_out calculation errors** - The attribute was computed incorrectly

This is marked as ERROR because `dist_out` is used by many downstream calculations and incorrect values propagate errors throughout the database.

#### SQL Logic

```sql
SELECT r1.reach_id, r1.dist_out as dist_out_up, r2.dist_out as dist_out_down
FROM reaches r1
JOIN reach_topology rt ON r1.reach_id = rt.reach_id
JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id
WHERE rt.direction = 'down'
  AND r2.dist_out > r1.dist_out + 100  -- threshold
```

#### Common Causes

1. **Flow direction disagreement** - Topology flow direction differs at bifurcations/deltas
2. **Cycle edges** - Reaches that participate in network cycles may have inconsistent `dist_out`
3. **Manual edits** - Topology was modified without recalculating `dist_out`
4. **Cross-continental connections** - Rare edge cases at region boundaries

#### How to Fix

1. **Recalculate dist_out** - Run `workflow.calculate_dist_out()` after topology changes
2. **Review topology** - Check if the downstream edge is correct using `reach_topology`
3. **Check for cycles** - Use graph analysis to identify cycle participation

#### Interpreting Results

- **0.1% issues**: Acceptable for production, likely edge cases
- **>1% issues**: Indicates systematic problem with topology or calculation
- **Clustered issues**: Often appear in delta regions or complex networks

---

### T002: path_freq_monotonicity

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Default Threshold** | None |
| **Category** | Topology |

#### What It Checks

Verifies that `path_freq` (path traversal frequency) **increases or stays the same** as you move downstream. At confluences, the downstream reach should have `path_freq >= max(upstream path_freqs)`.

#### Why It Matters

`path_freq` indicates how many unique paths from headwaters pass through a reach. It naturally increases downstream as tributaries merge. This attribute is used to:

1. Calculate `stream_order` (log scale of path_freq)
2. Identify main stems vs tributaries
3. Weight reaches by network importance

#### SQL Logic

```sql
SELECT r1.reach_id, r1.path_freq as pf_up, r2.path_freq as pf_down
FROM reaches r1
JOIN reach_topology rt ON r1.reach_id = rt.reach_id
JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id
WHERE rt.direction = 'down'
  AND r2.path_freq < r1.path_freq  -- should never decrease
```

#### Common Causes

1. **Stale path_freq** - Topology changed but path_freq wasn't recalculated
2. **Disconnected components** - Reaches in isolated subnetworks
3. **Artificial channels** - Canals/diversions that don't follow natural flow accumulation

#### How to Fix

1. **Recalculate path_freq** - Run topology recalculation workflow
2. **Check topology edges** - Ensure downstream edges are correct

---

### T003: facc_monotonicity

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Default Threshold** | 5% tolerance |
| **Category** | Topology |

#### What It Checks

Verifies that `facc` (flow accumulation area in km²) **increases** downstream. Downstream reaches should drain a larger area than their upstream neighbors.

#### Why It Matters

Flow accumulation is derived from MERIT Hydro and represents the total upstream drainage area. It's a physical property that should always increase downstream unless:

1. There's a topology error (flow direction wrong)
2. The `facc` values are from a different source with inconsistencies
3. Water is diverted (canals, irrigation)

#### SQL Logic

```sql
SELECT r1.reach_id, r1.facc as facc_up, r2.facc as facc_down
FROM reaches r1
JOIN reach_topology rt ON r1.reach_id = rt.reach_id
JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id
WHERE rt.direction = 'down'
  AND r2.facc < r1.facc * 0.95  -- 5% tolerance
```

#### Common Causes

1. **MERIT Hydro misalignment** - facc sampled from wrong DEM pixel
2. **Bifurcations/deltas** - Flow splits between multiple downstream channels
3. **Anthropogenic features** - Dams, diversions, canals
4. **Lake outlets** - facc may be inconsistent at lake boundaries

#### How to Fix

1. **Re-sample facc from MERIT Hydro** - Using correct reach centroids
2. **Flag for manual review** - Some cases are legitimate (bifurcations)
3. **Use median filtering** - Smooth facc along flow paths

#### Interpreting Results

- **~3% issues globally**: Expected due to DEM/network misalignment
- **High percentage in deltas**: Normal - flow splitting
- **Systematic issues by region**: May indicate MERIT Hydro version mismatch

---

### T004: orphan_reaches

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Default Threshold** | None |
| **Category** | Topology |

#### What It Checks

Finds reaches with `n_rch_up = 0` AND `n_rch_down = 0` - completely disconnected from the network.

#### Why It Matters

Orphan reaches are isolated and cannot participate in network analysis. They may represent:

1. **Data errors** - Topology edges were deleted
2. **Tiny isolated water bodies** - Small ponds incorrectly classified as reaches
3. **Processing artifacts** - Geometry issues during network extraction

#### SQL Logic

```sql
SELECT reach_id, region, river_name, reach_length, width
FROM reaches
WHERE n_rch_up = 0 AND n_rch_down = 0
```

#### Common Causes

1. **Isolated small lakes** - Should be classified as lakes, not reaches
2. **Topology table gaps** - Missing edges in reach_topology
3. **Region boundary artifacts** - Reaches split at boundaries

#### How to Fix

1. **Review classification** - Should these be lakeflag=1?
2. **Add topology edges** - If legitimately connected to network
3. **Delete if artifacts** - Remove invalid geometry

---

### T005: neighbor_count_consistency

| Property | Value |
|----------|-------|
| **Severity** | ERROR |
| **Default Threshold** | None |
| **Category** | Topology |

#### What It Checks

Verifies that `n_rch_up` and `n_rch_down` stored in the `reaches` table match the actual counts in the `reach_topology` table.

#### Why It Matters

These counts are denormalized for performance but must stay synchronized. Mismatches indicate:

1. **Stale cached values** - Topology changed but counts weren't updated
2. **Partial updates** - Only one table was modified
3. **Data corruption** - Inconsistent state

This is ERROR severity because many algorithms rely on these counts.

#### SQL Logic

```sql
WITH actual_counts AS (
    SELECT reach_id, region,
           SUM(CASE WHEN direction = 'up' THEN 1 ELSE 0 END) as actual_up,
           SUM(CASE WHEN direction = 'down' THEN 1 ELSE 0 END) as actual_down
    FROM reach_topology
    GROUP BY reach_id, region
)
SELECT r.reach_id, r.n_rch_up, r.n_rch_down, ac.actual_up, ac.actual_down
FROM reaches r
LEFT JOIN actual_counts ac ON r.reach_id = ac.reach_id
WHERE r.n_rch_up != COALESCE(ac.actual_up, 0)
   OR r.n_rch_down != COALESCE(ac.actual_down, 0)
```

#### How to Fix

```sql
-- Recalculate counts from topology table
UPDATE reaches r SET
    n_rch_up = (SELECT COUNT(*) FROM reach_topology rt
                WHERE rt.reach_id = r.reach_id AND rt.direction = 'up'),
    n_rch_down = (SELECT COUNT(*) FROM reach_topology rt
                  WHERE rt.reach_id = r.reach_id AND rt.direction = 'down')
```

---

### T006: connected_components

| Property | Value |
|----------|-------|
| **Severity** | INFO |
| **Default Threshold** | None |
| **Category** | Topology |

#### What It Checks

Analyzes network connectivity using the `network` field. Reports:
- Total number of distinct networks
- Single-reach networks (most suspicious)

#### Why It Matters

Understanding network fragmentation helps identify:
1. **Isolated subnetworks** that may need connection
2. **Data quality issues** in specific regions
3. **Processing artifacts** from network extraction

#### Interpreting Results

- **Single-reach networks = 0**: Good - all reaches connected to larger networks
- **Many single-reach networks**: May indicate orphaned reaches or tiny isolated features

---

### T007: topology_reciprocity

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Default Threshold** | None |
| **Category** | Topology |

#### What It Checks

Verifies that topology relationships are bidirectional. If reach A has B as a downstream neighbor, then B should have A as an upstream neighbor.

#### Why It Matters

The SWORD topology is stored as directed edges. Reciprocity ensures:
1. **Graph traversal works both directions**
2. **No dangling references**
3. **Consistent network structure**

#### SQL Logic

```sql
-- Find edges where reverse doesn't exist
SELECT rt1.reach_id, rt1.neighbor_reach_id, rt1.direction
FROM reach_topology rt1
LEFT JOIN reach_topology rt2
    ON rt1.reach_id = rt2.neighbor_reach_id
    AND rt1.neighbor_reach_id = rt2.reach_id
    AND rt1.direction != rt2.direction  -- opposite direction
WHERE rt2.reach_id IS NULL
```

#### How to Fix

Add the missing reverse edges to `reach_topology`.

---

## Attribute Checks (A0xx)

Attribute checks validate physical plausibility of reach measurements.

### A001: wse_monotonicity

| Property | Value |
|----------|-------|
| **Severity** | ERROR |
| **Default Threshold** | 0.5 (meters) |
| **Category** | Attributes |

#### What It Checks

Verifies that water surface elevation (WSE) **decreases** downstream. Water flows downhill, so upstream reaches should have higher WSE than downstream reaches.

#### Why It Matters

WSE is a fundamental physical measurement that validates:
1. **Flow direction** - Water flows from high to low elevation
2. **Data quality** - SWOT/satellite measurements are consistent
3. **Slope calculations** - Derived from WSE difference

This is ERROR severity because WSE violations indicate either flow direction errors or measurement problems.

#### SQL Logic

```sql
SELECT r1.reach_id, r1.wse as wse_up, r2.wse as wse_down
FROM reaches r1
JOIN reach_topology rt ON r1.reach_id = rt.reach_id
JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id
WHERE rt.direction = 'down'
  AND r1.lakeflag = 0 AND r2.lakeflag = 0  -- rivers only
  AND r2.wse > r1.wse + 0.5  -- threshold
```

#### Common Causes

1. **Flow direction errors** - Topology has wrong downstream
2. **SWOT measurement uncertainty** - Especially in narrow/shallow reaches
3. **Temporal variability** - WSE from different time periods
4. **Tidal influence** - Near-coast reaches with tidal effects

#### Interpreting Results

- **<1% issues**: Acceptable, likely measurement noise
- **Clustered in region**: May indicate regional flow direction problems
- **Lakes excluded**: Check only applies to river reaches (lakeflag=0)

---

### A002: slope_reasonableness

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Default Threshold** | 100.0 (m/km) |
| **Category** | Attributes |

#### What It Checks

Flags reaches with:
1. **Negative slopes** - Physically impossible for sustained flow
2. **Extremely high slopes** - >100 m/km (10% grade) is unusual for rivers

#### Why It Matters

Slope is derived from WSE difference / reach length. Unreasonable values indicate:
1. **Calculation errors** in slope derivation
2. **WSE measurement errors** propagating to slope
3. **Reach length issues** (very short reaches amplify slope errors)

#### SQL Logic

```sql
SELECT reach_id, slope, reach_length
FROM reaches
WHERE lakeflag = 0  -- rivers only
  AND (slope < 0 OR slope > 100)
```

#### Common Causes

1. **WSE uncertainty** - Small WSE differences on short reaches
2. **Negative from noise** - Measurement uncertainty exceeds true slope
3. **Waterfalls/rapids** - Legitimate high slopes in mountainous terrain

#### Interpreting Results

- **Negative slopes**: Almost always data quality issues
- **High slopes**: May be legitimate in mountain rivers - check geography

---

### A003: width_trend

| Property | Value |
|----------|-------|
| **Severity** | INFO |
| **Default Threshold** | 0.3 (30% ratio) |
| **Category** | Attributes |

#### What It Checks

Flags reaches where downstream width is less than 30% of upstream width. Rivers generally widen downstream as tributaries join.

#### Why It Matters

While width naturally varies, dramatic decreases suggest:
1. **Measurement errors** in width estimation
2. **Classification issues** - Lake/river boundaries
3. **Artificial narrowing** - Dams, canals, constrained reaches

#### SQL Logic

```sql
SELECT r1.reach_id, r1.width as width_up, r2.width as width_down
FROM reaches r1
JOIN reach_topology rt ON r1.reach_id = rt.reach_id
JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id
WHERE rt.direction = 'down'
  AND r1.width > 100  -- only check wider rivers
  AND r2.width < 0.3 * r1.width
```

#### Why INFO Severity

Width variability is natural - rivers narrow through gorges, at dam spillways, etc. This check identifies unusual cases for review but doesn't indicate errors.

---

### A004: attribute_completeness

| Property | Value |
|----------|-------|
| **Severity** | INFO |
| **Default Threshold** | 5% missing |
| **Category** | Attributes |

#### What It Checks

Reports completeness of required attributes:
- dist_out, facc, wse, width, slope
- reach_length, lakeflag, n_rch_up, n_rch_down

Flags attributes with >5% null or -9999 values.

#### Why It Matters

Missing data affects:
1. **Analysis completeness** - Can't compute statistics
2. **Algorithm inputs** - Many functions require these attributes
3. **Export quality** - Missing values propagate to outputs

#### Interpreting Results

The output is a table showing each attribute's missing count and percentage. Review attributes with high missing rates for systematic issues.

---

### A005: trib_flag_consistency

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Default Threshold** | None |
| **Category** | Attributes |

#### What It Checks

Verifies that `trib_flag` matches actual topology:
- `trib_flag = 1` should mean `n_rch_up > 1` (has tributaries)
- `trib_flag = 0` should mean `n_rch_up <= 1` (no tributaries)

#### Why It Matters

`trib_flag` is used to identify confluence points. Inconsistencies indicate:
1. **Stale flag values** - Topology changed but flag wasn't updated
2. **Calculation errors** in original data processing

#### SQL Logic

```sql
SELECT reach_id, trib_flag, n_rch_up
FROM reaches
WHERE (trib_flag = 1 AND n_rch_up <= 1)  -- flag says trib but no upstream
   OR (trib_flag = 0 AND n_rch_up > 1)   -- flag says no trib but has upstream
```

#### How to Fix

```sql
UPDATE reaches SET trib_flag = CASE WHEN n_rch_up > 1 THEN 1 ELSE 0 END
```

---

### A006: attribute_outliers

| Property | Value |
|----------|-------|
| **Severity** | INFO |
| **Default Threshold** | None |
| **Category** | Attributes |

#### What It Checks

Flags reaches with extreme values:
- **width > 50 km** - Wider than the widest rivers
- **wse > 8000 m** - Higher than any river on Earth
- **facc > 10,000,000 km²** - Larger than Amazon basin

#### Why It Matters

Extreme outliers often indicate:
1. **Unit conversion errors** (m vs km)
2. **Data entry errors**
3. **Processing artifacts**

#### Interpreting Results

Any flagged reach should be manually reviewed - these values are almost certainly errors.

---

## Geometry Checks (G0xx)

Geometry checks validate reach spatial properties.

### G001: reach_length_bounds

| Property | Value |
|----------|-------|
| **Severity** | INFO |
| **Default Threshold** | 100m - 50km |
| **Category** | Geometry |

#### What It Checks

Flags reaches outside typical length bounds:
- **Too short**: <100m (excluding `end_reach=1`)
- **Too long**: >50km

#### Why It Matters

- **Short reaches**: May be artifacts or should be merged
- **Long reaches**: May be missing intermediate junctions

#### Special Handling

Reaches with `end_reach=1` are **excluded** from "too short" checks because terminal reaches are expected to be short (they extend to the river terminus).

#### SQL Logic

```sql
SELECT reach_id, reach_length, end_reach
FROM reaches
WHERE (reach_length < 100 AND COALESCE(end_reach, 0) != 1)  -- too short
   OR reach_length > 50000  -- too long
```

---

### G002: node_length_consistency

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Default Threshold** | 0.1 (10% difference) |
| **Category** | Geometry |

#### What It Checks

Verifies that the sum of `node_length` values for a reach approximately equals the reach's `reach_length`.

#### Why It Matters

Nodes are ~200m segments along each reach. Their lengths should sum to the total reach length. Large discrepancies indicate:
1. **Missing nodes**
2. **Geometry inconsistencies**
3. **Coordinate system issues**

#### SQL Logic

```sql
WITH node_sums AS (
    SELECT reach_id, SUM(node_length) as sum_node_length
    FROM nodes
    GROUP BY reach_id
)
SELECT r.reach_id, r.reach_length, ns.sum_node_length
FROM reaches r
JOIN node_sums ns ON r.reach_id = ns.reach_id
WHERE ABS(r.reach_length - ns.sum_node_length) / r.reach_length > 0.1
```

---

### G003: zero_length_reaches

| Property | Value |
|----------|-------|
| **Severity** | INFO |
| **Default Threshold** | None |
| **Category** | Geometry |

#### What It Checks

Finds reaches with zero or negative length - geometry errors.

#### Why It Matters

Zero-length reaches are invalid and indicate:
1. **Duplicate start/end points**
2. **Geometry corruption**
3. **Processing errors**

These should be investigated and either fixed or removed.

---

### G013: width_gt_length

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Category** | Geometry |

Flags non-lake reaches (`lakeflag = 0`) where `width > reach_length`. Lakes are excluded because wide, short lake-reaches are expected.

---

### G014: duplicate_geometry

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Category** | Geometry |

Flags pairs of reaches within the same region whose geometry is identical (`ST_Equals`). Requires spatial extension.

---

### G015: node_reach_distance

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Default Threshold** | 100.0 (meters) |
| **Category** | Geometry |

Flags nodes whose point geometry is more than 100m from their parent reach linestring. Uses approximate Cartesian-to-metre conversion (`ST_Distance * 111000`). Requires spatial extension.

---

### G016: node_spacing

| Property | Value |
|----------|-------|
| **Severity** | INFO |
| **Default Threshold** | 2.0 (ratio) |
| **Category** | Geometry |

Flags nodes whose `node_length` is >2× or <0.5× the mean `node_length` for their reach. Only checks reaches with ≥3 nodes.

---

### G017: cross_reach_nodes

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Default Threshold** | 50.0 (meters) |
| **Category** | Geometry |

Two-pass check: (1) finds nodes >50m from their own reach, (2) checks if any nearby alternative reach is closer. Bbox prefilter (`|Δx| < 0.05°, |Δy| < 0.05°`). LIMIT 5000. Requires spatial extension.

---

### G018: dist_out_vs_reach_length

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Default Threshold** | 0.2 (20% tolerance) |
| **Category** | Geometry |

For each downstream connection, checks `|r1.dist_out - r2.dist_out - r1.reach_length| / r1.reach_length > 0.2`. Large discrepancies suggest dist_out or reach_length errors.

---

> **Removed checks:** G019 (confluence_geometry) and G020 (bifurcation_geometry) were removed as strict subsets of G012 (endpoint_alignment). G012 checks ALL downstream pairs, which includes confluences and bifurcations.

---

### G021: reach_overlap

| Property | Value |
|----------|-------|
| **Severity** | INFO |
| **Category** | Geometry |

Flags pairs of non-connected reaches whose geometries intersect (`ST_Intersects`). Uses bbox prefilter (`|Δx| < 0.15°, |Δy| < 0.15°`) and `NOT EXISTS` on `reach_topology`. LIMIT 5000. Requires spatial extension.

---

## Classification Checks (C0xx)

Classification checks validate lake/river type assignments.

### C001: lake_sandwich

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Default Threshold** | None |
| **Category** | Classification |

#### What It Checks

Finds river reaches (`lakeflag=0`) that are "sandwiched" between lake reaches (`lakeflag=1`) - having at least one lake upstream AND at least one lake downstream.

#### Why It Matters

These reaches may be:
1. **Misclassified lake sections** - Should be lakeflag=1
2. **Lake narrows** - Legitimate river-like sections through lakes
3. **Short connecting channels** between lakes

#### SQL Logic

```sql
WITH river_reaches AS (
    SELECT reach_id, region FROM reaches WHERE lakeflag = 0
),
has_lake_upstream AS (
    SELECT DISTINCT rt.reach_id
    FROM reach_topology rt
    JOIN reaches r ON rt.neighbor_reach_id = r.reach_id
    WHERE rt.direction = 'up' AND r.lakeflag = 1
),
has_lake_downstream AS (
    SELECT DISTINCT rt.reach_id
    FROM reach_topology rt
    JOIN reaches r ON rt.neighbor_reach_id = r.reach_id
    WHERE rt.direction = 'down' AND r.lakeflag = 1
)
SELECT rr.reach_id
FROM river_reaches rr
JOIN has_lake_upstream hu ON rr.reach_id = hu.reach_id
JOIN has_lake_downstream hd ON rr.reach_id = hd.reach_id
```

#### Interpreting Results

- **~1.5% of river reaches**: Expected rate globally
- **Short reaches**: More likely to be misclassified
- **Wide reaches**: May be legitimate lake narrows

#### How to Fix

Manual review required - determine if reach should be reclassified as lake.

---

### C002: lakeflag_distribution

| Property | Value |
|----------|-------|
| **Severity** | INFO |
| **Default Threshold** | None |
| **Category** | Classification |

#### What It Checks

Reports the distribution of `lakeflag` values:
- 0: river
- 1: lake
- 2: canal
- 3: tidal

Flags any unknown values (not 0-3).

#### Why It Matters

Understanding the classification distribution helps:
1. **Validate data quality**
2. **Identify regional differences**
3. **Detect unexpected values**

#### Expected Distribution (approximate)

| lakeflag | Type | Typical % |
|----------|------|-----------|
| 0 | river | 80-85% |
| 1 | lake | 15-18% |
| 2 | canal | <1% |
| 3 | tidal | <1% |

---

### C003: type_distribution

| Property | Value |
|----------|-------|
| **Severity** | INFO |
| **Default Threshold** | None |
| **Category** | Classification |

#### What It Checks

Reports the distribution of the `type` field (if present):
- 1: river
- 2: lake
- 3: tidal river
- 4: artificial (canal/dam)
- 5: unassigned
- 6: unreliable

Reports count of `type=6` (unreliable) reaches.

#### Why It Matters

The `type` field provides more detailed classification than `lakeflag`. Type=6 (unreliable) reaches need review.

#### Note

This check gracefully handles databases where the `type` column doesn't exist.

---

### C004: lakeflag_type_consistency

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Default Threshold** | None |
| **Category** | Classification |

#### What It Checks

Verifies that `lakeflag` and `type` fields are consistent:

| lakeflag | Expected type values |
|----------|---------------------|
| 0 (river) | 1, 3, 5, 6 |
| 1 (lake) | 2, 5, 6 |
| 2 (canal) | 4, 5, 6 |
| 3 (tidal) | 3, 5, 6 |

Type 5 (unassigned) and 6 (unreliable) are allowed for any lakeflag.

#### Why It Matters

Inconsistent classification causes confusion and analysis errors. Both fields should agree on the reach type.

#### Note

This check gracefully handles databases where the `type` column doesn't exist.

---

## Flag Checks (FLxxx)

Flag checks validate iceflag, low_slope_flag, edit_flag, and SWOT observation coverage.

### FL001: swot_obs_coverage

| Property | Value |
|----------|-------|
| **Severity** | INFO |
| **Category** | Flags |

Reports SWOT observation coverage — how many reaches have satellite observations.

### FL002: iceflag_values

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Category** | Flags |

Validates iceflag is in {-9999, 0, 1, 2}:
- -9999: no data
- 0: no ice
- 1: seasonal ice
- 2: permanent ice

### FL003: low_slope_flag_consistency

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Default Threshold** | 1e-4 (m/m) |
| **Category** | Flags |

Checks that low_slope_flag agrees with actual slope values. Flags cases where the flag is set but slope isn't low, or the flag is unset but slope is extremely low (<1e-6 m/m).

### FL004: edit_flag_format

| Property | Value |
|----------|-------|
| **Severity** | INFO |
| **Category** | Flags |

Reports edit_flag distribution. edit_flag is a comma-separated string of edit tags (e.g. 'facc_denoise_v3', 'lake_sandwich').

---

## Network Checks (Nxxx)

Network checks validate main_side and stream_order consistency.

### N001: main_side_values

| Property | Value |
|----------|-------|
| **Severity** | ERROR |
| **Category** | Network |

Validates main_side is in {0, 1, 2}:
- 0: main channel (~95%)
- 1: side channel (~3%)
- 2: secondary outlet (~2%)

### N002: main_side_stream_order

| Property | Value |
|----------|-------|
| **Severity** | ERROR |
| **Category** | Network |

Checks that main channel reaches (main_side=0) have a valid stream_order (not -9999). Side channels and secondary outlets are expected to lack stream_order.

---

## SWOT Attribute Checks (A02x)

These checks validate SWOT observation statistics against reference values. They gracefully skip when SWOT columns are not present in the database.

### A021: wse_obs_vs_wse

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Default Threshold** | 10.0 (meters) |
| **Category** | Attributes |

Checks that SWOT-observed WSE median is close to reference WSE. Large differences suggest measurement issues or temporal variability.

### A024: width_obs_vs_width

| Property | Value |
|----------|-------|
| **Severity** | INFO |
| **Default Threshold** | 3.0 (ratio) |
| **Category** | Attributes |

Checks that SWOT-observed width median is reasonable compared to reference width. Flags reaches where the ratio is outside [1/3, 3].

### A026: slope_obs_nonneg

| Property | Value |
|----------|-------|
| **Severity** | ERROR |
| **Category** | Attributes |

Checks that SWOT-observed slope mean is non-negative. Negative slopes indicate measurement artifacts or processing errors.

### A027: slope_obs_extreme

| Property | Value |
|----------|-------|
| **Severity** | WARNING |
| **Default Threshold** | 50.0 (m/km) |
| **Category** | Attributes |

Checks that SWOT-observed slope mean is below 50 m/km. Extremely high slopes are rare for rivers and likely indicate measurement error.

---

## Threshold Reference

| Check | Parameter | Default | Unit | Description |
|-------|-----------|---------|------|-------------|
| T001 | threshold | 100.0 | meters | Tolerance for dist_out increase |
| A002 | threshold | 100.0 | m/km | Maximum reasonable slope |
| A003 | threshold | 0.3 | ratio | Minimum downstream/upstream width |
| A021 | threshold | 10.0 | meters | Max |wse_obs_median - wse| |
| A024 | threshold | 3.0 | ratio | Max width_obs_median/width ratio |
| A027 | threshold | 50.0 | m/km | Maximum reasonable observed slope |
| FL003 | threshold | 1e-4 | m/m | Slope threshold for low_slope_flag |
| G002 | threshold | 0.1 | ratio | Max difference in node sum vs reach length |
| G015 | threshold | 100.0 | meters | Max node-to-reach distance |
| G016 | threshold | 2.0 | ratio | Max node_length / mean ratio |
| G017 | threshold | 50.0 | meters | Min dist before cross-reach check |
| G018 | threshold | 0.2 | ratio | Max dist_out gap vs reach_length |

Override thresholds via CLI:
```bash
python -m src.sword_duckdb.lint.cli --db sword.duckdb --threshold A002 150
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All checks passed (or only INFO failures) |
| 1 | WARNING-level failures detected |
| 2 | ERROR-level failures detected |

Use `--fail-on-error` or `--fail-on-warning` to control exit behavior.
