# Validation Spec: stream_order, path_segs, path_order

## Summary

| Variable | Source | Units | Official Definition |
|----------|--------|-------|---------------------|
| stream_order | Computed (log transform) | dimensionless integer | "stream order based on the log scale of the path frequency" |
| path_segs | Computed (segment ID) | dimensionless integer | "unique values indicating continuous river segments between river junctions" |
| path_order | Computed (ranking) | dimensionless integer | "unique values representing continuous paths from the river outlet to the headwaters" |

## Dependency Chain

```
reach_topology
     |
     v
path_freq  <---- [T002: monotonicity check]
     |
     +---> stream_order = round(log(path_freq)) + 1
     |
     +---> path_order (ranks by dist_out within same path_freq)
     |
     +---> path_segs (segment ID between junctions)
     |
     +---> main_side (higher path_freq = main channel)
```

---

## 1. stream_order

### Official Definition (v17b PDD)
> "stream order based on the log scale of the path frequency. stream order is calculated for the main network only (see 'main_side' description). stream order is not included for side channels which are given a no data value of -9999."

### Formula
```python
stream_order = round(np.log(path_freq)) + 1  # where path_freq > 0
```

**VERIFIED:** Formula produces exact match with v17b data (100% match rate tested).

### Code Reference
- **File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reconstruction.py`
- **Lines:** 2313-2351
- **Function:** `_reconstruct_reach_stream_order()`

```python
# From reconstruction.py:2347-2348
result_df['stream_order'] = np.round(np.log(result_df['path_freq'])) + 1
result_df['stream_order'] = result_df['stream_order'].astype(int)
```

### AttributeSpec
- **File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reconstruction.py`
- **Lines:** 468-475

```python
"reach.stream_order": AttributeSpec(
    name="reach.stream_order",
    source=SourceDataset.COMPUTED,
    method=DerivationMethod.LOG_TRANSFORM,
    source_columns=[],
    dependencies=["reach.path_freq"],
    description="Stream order: round(log(path_freq)) + 1 where path_freq > 0"
)
```

### Valid Ranges

| Condition | stream_order Value |
|-----------|-------------------|
| path_freq = 1 | 1 |
| path_freq = 2-4 | 2 |
| path_freq = 5-12 | 3 |
| path_freq = 13-33 | 4 |
| path_freq = 34-89 | 5 |
| path_freq = 90-244 | 6 |
| path_freq = 245-664 | 7 |
| path_freq = 665+ | 8+ |
| Side channels (main_side = 1 or 2) | -9999 |
| Invalid/missing path_freq | -9999 |

### v17b Distribution (NA region)

| stream_order | count |
|--------------|-------|
| -9999 | 1,208 |
| 1 | 20,182 |
| 2 | 9,123 |
| 3 | 4,082 |
| 4 | 2,406 |
| 5 | 1,123 |
| 6 | 351 |
| 7 | 173 |
| 8 | 48 |

### Side Channel Handling
From v17b data analysis:
- `main_side = 0` (main channel): 37,488 reaches, ALL have valid stream_order > 0
- `main_side = 1` (side channel): 627 reaches, ALL have stream_order = -9999
- `main_side = 2` (secondary outlet): 581 reaches, ALL have stream_order = -9999

**IMPORTANT:** Side channels (`main_side = 1` or `2`) have `path_freq = -9999`, which propagates to `stream_order = -9999`. This is by design.

### Reactive Recalculation
- **File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reactive.py`
- **Status:** NOT IMPLEMENTED

The reactive system does NOT currently recalculate `stream_order` when `path_freq` changes. This is a gap that should be addressed.

---

## 2. path_order

### Official Definition (v17b PDD)
> "unique values representing continuous paths from the river outlet to the headwaters. Values are unique within level two Pfafstetter basins. The lowest value is always the longest path from outlet to farthest headwater point in a connected river network. Higher path values branch off from the longest path value to other headwater points."

### Algorithm (Current Implementation - POTENTIALLY INCORRECT)
- **File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reconstruction.py`
- **Lines:** 3702-3742
- **Function:** `_reconstruct_reach_path_order()`

```python
# Current implementation ranks by dist_out within path_freq
result_df = self._conn.execute(f"""
    WITH ranked AS (
        SELECT
            reach_id,
            path_freq,
            dist_out,
            ROW_NUMBER() OVER (
                PARTITION BY path_freq
                ORDER BY dist_out ASC
            ) as path_order
        FROM reaches
        WHERE region = ? {where_clause}
    )
    SELECT reach_id, path_order
    FROM ranked
""", params).fetchdf()
```

### Analysis of v17b Data
The current reconstruction code ranks by `dist_out` within `path_freq` groups. However, v17b data shows:

| path_freq | min_path_order | max_path_order | reach_count |
|-----------|----------------|----------------|-------------|
| 1 | 1 | 2,132 | 20,680 |
| 2 | 1 | 2,129 | 4,944 |
| 3 | 1 | 2,122 | 2,521 |
| 4 | 1 | 2,121 | 1,734 |

The path_order values reset to 1 for each path_freq value, which aligns with the current implementation. However, the PDD says "lowest value is always the longest path" - this may need further investigation.

### AttributeSpec
```python
"reach.path_order": AttributeSpec(
    name="reach.path_order",
    source=SourceDataset.COMPUTED,
    method=DerivationMethod.GRAPH_TRAVERSAL,
    source_columns=[],
    dependencies=["reach_topology", "reach.path_freq"],
    description="Path order: 1=longest to N=shortest during pathway construction"
)
```

---

## 3. path_segs

### Official Definition (v17b PDD)
> "unique values indicating continuous river segments between river junctions. Values are unique within level two Pfafstetter basins."

### Algorithm (Current Implementation - INCORRECT)
- **File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reconstruction.py`
- **Lines:** 3744-3782
- **Function:** `_reconstruct_reach_path_segs()`

```python
# INCORRECT: Current implementation counts reaches per path_freq
result_df = self._conn.execute(f"""
    WITH path_counts AS (
        SELECT path_freq, COUNT(*) as seg_count
        FROM reaches
        WHERE region = ?
        GROUP BY path_freq
    )
    SELECT
        r.reach_id,
        COALESCE(pc.seg_count, 1) as path_segs
    FROM reaches r
    LEFT JOIN path_counts pc ON r.path_freq = pc.path_freq
    WHERE r.region = ? {where_clause}
""", [self._region] + params).fetchdf()
```

### CRITICAL FINDING: Implementation Mismatch

**Tested match rate: 0.1% match (43 of 38,069 reaches)**

The current reconstruction code computes `path_segs` as the COUNT of reaches with the same `path_freq`. This is INCORRECT.

From v17b data analysis:
- Same `path_segs` can have different `path_freq` values
- `path_segs` is NOT a count - it's a unique segment identifier
- Reaches with the same `path_segs` form a contiguous path between junctions

### Correct Understanding
From the PDD and data analysis:
1. `path_segs` is an **identifier** for contiguous river segments between junctions
2. Values are **unique within level-2 Pfafstetter basins** (not globally unique)
3. Reaches with the same `path_segs` form a continuous path (no junctions in between)
4. At junctions (`end_reach = 3`), reaches on different branches have different `path_segs`

### Example from v17b (network=3, path_segs=3)
| reach_id | path_segs | path_freq | dist_out | end_reach |
|----------|-----------|-----------|----------|-----------|
| ... | 3 | 1 | 172 | 2 (outlet) |
| ... | 3 | 1 | 1,396 | 0 |
| ... | 3 | 1 | ... | 0 |
(11 reaches total, all path_freq=1, contiguous)

### Proposed Correct Algorithm
```python
def _reconstruct_reach_path_segs_correct():
    """
    Correct algorithm for path_segs:
    1. Build topology graph
    2. Find all junction reaches (end_reach = 3 or n_rch_up > 1 or n_rch_down > 1)
    3. Starting from outlets, traverse upstream
    4. Assign same path_segs to contiguous reaches until hitting a junction
    5. At junctions, start new path_segs for each branch
    """
    pass  # TODO: Implement correct algorithm
```

---

## Failure Modes

### stream_order Failures

| Failure | Detection | Severity |
|---------|-----------|----------|
| Stale data after path_freq change | `stream_order != round(log(path_freq)) + 1` | ERROR |
| Side channel has valid stream_order | `main_side IN (1,2) AND stream_order > 0` | WARNING |
| Main channel missing stream_order | `main_side = 0 AND stream_order = -9999` | ERROR |
| Out of range | `stream_order < 1 AND stream_order != -9999` | ERROR |

### path_segs Failures

| Failure | Detection | Severity |
|---------|-----------|----------|
| Non-contiguous reaches with same path_segs | Topology check | ERROR |
| path_segs not unique within basin | Duplicate check | WARNING |
| Missing path_segs | `path_segs IS NULL OR path_segs = 0` | WARNING |

### path_order Failures

| Failure | Detection | Severity |
|---------|-----------|----------|
| Duplicate path_order within (path_freq, basin) | Unique constraint | ERROR |
| path_order not starting at 1 | `MIN(path_order) != 1` per group | WARNING |

---

## Proposed Consistency Checks

### New Lint Checks

| ID | Severity | Rule | SQL |
|----|----------|------|-----|
| A011 | ERROR | stream_order matches formula | See below |
| A012 | WARNING | Side channels have stream_order = -9999 | See below |
| A013 | WARNING | Main channels have valid stream_order | See below |
| A014 | INFO | path_segs contiguity | Complex topology check |

### A011: stream_order Formula Check
```sql
SELECT reach_id, path_freq, stream_order,
       ROUND(LN(path_freq)) + 1 as expected
FROM reaches
WHERE region = ?
  AND path_freq > 0
  AND path_freq != -9999
  AND stream_order != -9999
  AND stream_order != ROUND(LN(path_freq)) + 1
```

### A012: Side Channel stream_order Check
```sql
SELECT reach_id, main_side, stream_order
FROM reaches
WHERE region = ?
  AND main_side IN (1, 2)  -- side or secondary outlet
  AND stream_order != -9999
```

### A013: Main Channel stream_order Check
```sql
SELECT reach_id, main_side, stream_order, path_freq
FROM reaches
WHERE region = ?
  AND main_side = 0  -- main channel
  AND stream_order = -9999
  AND path_freq > 0
  AND path_freq != -9999
```

---

## Edge Cases

### 1. Disconnected Reaches
- `path_freq = 0` or `-9999` -> `stream_order = -9999`
- `path_segs` may be undefined

### 2. Ghost Reaches (type = 6)
- Placeholder reaches at headwaters/outlets
- May have valid or undefined path variables

### 3. Unreliable Topology (type = 5)
- Path calculations may be incorrect
- Should potentially be excluded from strict validation

### 4. Cross-Basin Reaches
- `path_segs` is unique within level-2 basin only
- Cross-basin reaches need special handling

---

## Reactive System Gaps

The reactive system (`reactive.py`) has the following gaps:

1. **stream_order not recalculated** when `path_freq` changes
2. **path_segs not recalculated** when topology changes
3. **path_order not recalculated** when `dist_out` or `path_freq` changes

### Recommended Fix
Add to `DependencyGraph`:
```python
("reach.path_freq", "reach.stream_order"),  # path_freq -> stream_order
("reach.topology", "reach.path_segs"),      # topology -> path_segs
("reach.dist_out", "reach.path_order"),     # dist_out -> path_order
("reach.path_freq", "reach.path_order"),    # path_freq -> path_order
```

---

## Recommendations

### Immediate Actions

1. **Add A011 as ERROR** - Formula validation is critical
2. **Add A012 and A013 as WARNING** - Side channel handling validation
3. **Document path_segs bug** - Current reconstruction is incorrect

### Code Fixes Needed

1. **reconstruction.py:3744-3782** - `_reconstruct_reach_path_segs()` needs complete rewrite
   - Current: Counts reaches per path_freq (WRONG)
   - Correct: Assigns unique IDs to contiguous segments between junctions

2. **reactive.py** - Add stream_order recalculation trigger
   - When `path_freq` changes, mark `stream_order` dirty

### Future Investigation

1. Verify path_order algorithm against original SWORD construction code
2. Determine if path_segs should be reconstructed at all (may be purely static ID)
3. Test stream_order formula with natural log vs log10 (currently uses natural log)

---

## References

- SWORD Product Description Document v17b, pages 12-13, 17-18, 23-24
- `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reconstruction.py` lines 2313-2351, 3702-3782
- `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reactive.py` lines 615-694
- `/Users/jakegearon/projects/SWORD/docs/validation_specs/path_freq_validation_spec.md`
