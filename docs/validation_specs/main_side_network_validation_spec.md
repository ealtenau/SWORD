# Validation Spec: main_side, network

## Summary

| Variable | Source | Units | Official Definition |
|----------|--------|-------|---------------------|
| main_side | Computed (path_freq analysis) | dimensionless integer | "value indicating whether a node/reach is on the main network (0), side network (1), or is a secondary outlet on the main network (2)" |
| network | Computed (connected components) | dimensionless integer | "unique value for each connected river network. Values are unique within level two Pfafstetter basins" |

**Version Added:** v17 (October 2024)

---

## 1. main_side

### Official Definition (v17b PDD, pages 12-13, 17-18, 23-24)

> "value indicating whether a node is on the main network (0), side network (1), or is a secondary outlet on the main network (2)."

### Valid Values

| Value | Meaning | Description |
|-------|---------|-------------|
| 0 | Main network | Primary channel with calculated stream_order |
| 1 | Side network | Side channel/distributary; stream_order = -9999 |
| 2 | Secondary outlet | Outlet on main network (not primary); stream_order = -9999 |

### Code Reference

- **File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py`
- **Lines:** 3614-3659
- **Function:** `_reconstruct_reach_main_side()`

### AttributeSpec

```python
# Lines 477-484
"reach.main_side": AttributeSpec(
    name="reach.main_side",
    source=SourceDataset.COMPUTED,
    method=DerivationMethod.GRAPH_TRAVERSAL,
    source_columns=[],
    dependencies=["reach_topology", "reach.path_freq"],
    description="Main/side channel: 0=main, 1=side, 2=secondary outlet"
)

# Lines 770-776
"node.main_side": AttributeSpec(
    name="node.main_side",
    source=SourceDataset.INHERITED,
    method=DerivationMethod.INHERITED,
    source_columns=[],
    dependencies=["reach.main_side", "node.reach_id"],
    description="Main/side channel: inherited from parent reach"
)
```

### Current Reconstruction Algorithm

```python
# From reconstruction.py:3645-3658
result_df = self._conn.execute(f"""
    WITH reach_groups AS (
        SELECT reach_id, path_freq, n_rch_up, n_rch_down
        FROM reaches
        WHERE region = ? {where_clause}
    )
    SELECT
        reach_id,
        CASE
            WHEN n_rch_up <= 1 AND n_rch_down <= 1 THEN 1  -- Linear reach = main
            WHEN path_freq >= 1 THEN 1  -- High path_freq = main
            ELSE 2  -- Side channel
        END as main_side
    FROM reach_groups
""", params).fetchdf()
```

**WARNING:** This reconstruction algorithm is INCORRECT. The actual v17b values follow different logic (see "Algorithm Gap" below).

### Derivation Logic (True v17b)

Based on analysis of v17b data, the actual algorithm appears to be:

1. **main_side = 0 (Main network):** Reaches that are part of the primary dendritic tree
   - Stream order is calculated: `stream_order = round(log(path_freq)) + 1`
   - path_freq > 0 (usually >= 1)

2. **main_side = 1 (Side network):** Distributary channels, anabranches, side channels
   - path_freq = -9999 (not calculated)
   - stream_order = -9999 (not applicable)

3. **main_side = 2 (Secondary outlet):** Additional outlets from the main network
   - Typically in deltas or multi-outlet systems
   - path_freq = -9999 (not calculated)
   - stream_order = -9999 (not applicable)

### v17b Distribution (Global)

| main_side | Count | Percentage | stream_order=-9999 |
|-----------|-------|------------|-------------------|
| 0 (main) | 236,151 | 94.96% | 40 (0.02%) |
| 1 (side) | 6,793 | 2.73% | 6,793 (100%) |
| 2 (secondary outlet) | 5,729 | 2.30% | 5,729 (100%) |

### v17b Distribution by Region

| Region | main_side=0 | main_side=1 | main_side=2 |
|--------|-------------|-------------|-------------|
| AF | 20,319 (94.8%) | 683 (3.2%) | 439 (2.0%) |
| AS | 94,175 (94.0%) | 3,546 (3.5%) | 2,464 (2.5%) |
| EU | 29,478 (94.8%) | 838 (2.7%) | 787 (2.5%) |
| NA | 37,488 (96.9%) | 627 (1.6%) | 581 (1.5%) |
| OC | 14,264 (94.5%) | 269 (1.8%) | 556 (3.7%) |
| SA | 40,427 (95.9%) | 830 (2.0%) | 902 (2.1%) |

---

## 2. network

### Official Definition (v17b PDD, pages 13, 19, 25, 27)

> "unique value for each connected river network. Values are unique within level two Pfafstetter basins."

### Code Reference

- **File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py`
- **Lines:** 2493-2565
- **Function:** `_reconstruct_reach_network()`

### AttributeSpec

```python
# Lines 495-502
"reach.network": AttributeSpec(
    name="reach.network",
    source=SourceDataset.COMPUTED,
    method=DerivationMethod.GRAPH_TRAVERSAL,
    source_columns=[],
    dependencies=["reach_topology"],
    description="Connected network ID: groups of hydrologically connected reaches"
)

# Lines 788-794
"node.network": AttributeSpec(
    name="node.network",
    source=SourceDataset.INHERITED,
    method=DerivationMethod.INHERITED,
    source_columns=[],
    dependencies=["reach.network", "node.reach_id"],
    description="Network ID: inherited from parent reach"
)
```

### Reconstruction Algorithm

```python
# From reconstruction.py:2507-2565
# Build adjacency from topology
for _, row in topology_df.iterrows():
    reach = int(row['reach_id'])
    neighbor = int(row['neighbor_reach_id'])
    if reach in adjacency and neighbor in all_reaches:
        adjacency[reach].add(neighbor)
        adjacency[neighbor].add(reach)

# BFS to find connected components
network_ids = {}
visited = set()
current_network = 1

for start_reach in all_reaches:
    if start_reach in visited:
        continue

    # BFS from start_reach
    queue = [start_reach]
    component = []

    while queue:
        reach_id = queue.pop(0)
        if reach_id in visited:
            continue
        visited.add(reach_id)
        component.append(reach_id)

        for neighbor in adjacency.get(reach_id, []):
            if neighbor not in visited:
                queue.append(neighbor)

    # Assign network ID
    for rid in component:
        network_ids[rid] = current_network

    current_network += 1
```

### v17b Distribution

| Region | Unique Networks | Total Reaches |
|--------|-----------------|---------------|
| AF | 79 | 21,441 |
| AS | 246 | 100,185 |
| EU | 103 | 31,103 |
| NA | 105 | 38,696 |
| OC | 211 | 15,089 |
| SA | 53 | 42,159 |
| **Global** | **247** | **248,673** |

**Note:** Network values range from 0 to 1010. The count of 247 unique networks globally is surprising - this may indicate networks span multiple regions or there are cross-region assignments.

---

## 3. Relationship to Other Variables

### main_side Affects stream_order

From the PDD:
> "stream order is calculated for the main network only (see 'main_side' description). stream order is not included for side channels which are given a no data value of -9999."

**Invariant:**
- `main_side IN (1, 2)` implies `stream_order = -9999`
- `main_side = 0` with `path_freq > 0` should have `stream_order = round(log(path_freq)) + 1`

### main_side Relationship to path_freq

| main_side | path_freq behavior |
|-----------|-------------------|
| 0 | Calculated via graph traversal (typically >= 1) |
| 1 | Set to -9999 (not calculated) |
| 2 | Set to -9999 (not calculated) |

### network Identifies Connected Components

All reaches with the same `network` value are topologically connected through the `reach_topology` table.

---

## 4. Failure Modes

### main_side Failures

| Failure | Detection | Count in v17b | Severity |
|---------|-----------|---------------|----------|
| main_side=0 with stream_order=-9999 | Query below | 40 | ERROR |
| main_side=1/2 with stream_order != -9999 | Query below | 0 | ERROR |
| main_side=0 with path_freq <= 0 | Query below | 38 | WARNING |
| main_side outside {0, 1, 2} | Query below | 0 | ERROR |
| NULL main_side | Query below | 0 | ERROR |

### network Failures

| Failure | Detection | Severity |
|---------|-----------|----------|
| NULL network | `network IS NULL` | WARNING |
| network=0 for isolated reaches | Check topology | INFO |
| Disconnected reaches in same network | Verify connectivity | ERROR |

---

## 5. CRITICAL BUG: main_side=0 with stream_order=-9999

### Evidence

v17b contains **40 reaches** where `main_side = 0` (supposedly main channel) but `stream_order = -9999`:

```sql
SELECT reach_id, region, river_name, main_side, stream_order,
       path_freq, n_rch_up, n_rch_down, end_reach, network
FROM reaches
WHERE main_side = 0 AND stream_order = -9999
```

### Analysis of Anomalous Reaches

All 40 reaches are in Asia (AS) region:

| Characteristic | Count |
|---------------|-------|
| path_freq = 0 | 37 |
| path_freq = 2 | 2 |
| path_freq = -9999 | 1 |

Most affected: Kundu River (22 reaches), various small rivers

### Root Cause

These reaches have `path_freq = 0` or invalid path_freq values, which causes:
1. `stream_order = -9999` (cannot compute log(0))
2. Yet `main_side = 0` was assigned (incorrectly marking as main channel)

This represents an **inconsistency** in the original SWORD construction:
- If `path_freq = 0`, the reach was not traversed in the path calculation
- Such reaches should likely have `main_side = 1` (side channel) or special handling

### Proposed Fix

Option A: Reclassify these 40 reaches as `main_side = 1` (side channel)
Option B: Fix `path_freq` calculation to properly traverse these reaches

---

## 6. Algorithm Gap: Reconstruction vs Original

### Current Reconstruction (INCORRECT)

The current `_reconstruct_reach_main_side()` uses:
```python
CASE
    WHEN n_rch_up <= 1 AND n_rch_down <= 1 THEN 1  -- Linear = main
    WHEN path_freq >= 1 THEN 1  -- High path_freq = main
    ELSE 2  -- Side
END
```

This outputs values 1 and 2 only, never 0. This is fundamentally incorrect.

### Original Algorithm (Inferred from v17b data)

Based on data analysis, the true algorithm is likely:

```python
# Pseudocode - NOT actual implementation
def compute_main_side(reach):
    if is_on_primary_dendritic_tree(reach):
        # Path algorithm successfully traversed this reach
        return 0  # main
    elif is_distributary_or_anabranch(reach):
        return 1  # side
    elif is_secondary_outlet(reach):
        return 2  # secondary outlet
    else:
        return 0  # default to main
```

Key insight: `main_side` is determined during the original path traversal algorithm, not from simple topology rules.

### Required Fix

The `_reconstruct_reach_main_side()` function needs a complete rewrite to:
1. Perform proper path traversal from outlets
2. Identify distributaries vs main channels at bifurcations
3. Use facc or path_freq to determine "main" at multi-channel points
4. Mark secondary outlets based on multi-outlet detection

---

## 7. Existing Lint Checks

### Currently Implemented

| Check ID | Name | Covers main_side/network |
|----------|------|--------------------------|
| T004 | orphan_reaches | Uses network for context |
| T006 | connected_components | Validates network field |

### T006: connected_components

**File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/lint/checks/topology.py`
**Lines:** 344-401

This check:
- Counts unique networks
- Identifies single-reach networks
- Reports network statistics

```python
# Query for single-reach networks
SELECT network, region, reach_count, sample_reach_id
FROM network_sizes
WHERE reach_count = 1  -- Single-reach networks
```

---

## 8. Proposed New Lint Checks

### N001: main_side Value Validity

```python
@register_check(
    "N001",
    Category.CLASSIFICATION,
    Severity.ERROR,
    "main_side must be 0, 1, or 2",
)
def check_main_side_validity(conn, region=None):
    query = f"""
    SELECT reach_id, region, main_side, river_name, x, y
    FROM reaches r
    WHERE main_side NOT IN (0, 1, 2)
        OR main_side IS NULL
        {f"AND region = '{region}'" if region else ""}
    """
    issues = conn.execute(query).fetchdf()
    # ... return CheckResult
```

### N002: main_side / stream_order Consistency

```python
@register_check(
    "N002",
    Category.CLASSIFICATION,
    Severity.ERROR,
    "main_side must be consistent with stream_order",
)
def check_main_side_stream_order_consistency(conn, region=None):
    # Side channels MUST have stream_order = -9999
    query1 = f"""
    SELECT reach_id, region, main_side, stream_order, path_freq
    FROM reaches
    WHERE main_side IN (1, 2) AND stream_order != -9999
        {f"AND region = '{region}'" if region else ""}
    """

    # Main channels with valid path_freq should NOT have stream_order = -9999
    query2 = f"""
    SELECT reach_id, region, main_side, stream_order, path_freq
    FROM reaches
    WHERE main_side = 0
        AND stream_order = -9999
        AND path_freq > 0
        AND path_freq != -9999
        {f"AND region = '{region}'" if region else ""}
    """
    # ... return CheckResult
```

### N003: main_side / path_freq Consistency

```python
@register_check(
    "N003",
    Category.CLASSIFICATION,
    Severity.WARNING,
    "main_side should be consistent with path_freq",
)
def check_main_side_path_freq_consistency(conn, region=None):
    # Side/secondary channels should have path_freq = -9999
    query = f"""
    SELECT reach_id, region, main_side, path_freq, stream_order
    FROM reaches
    WHERE main_side IN (1, 2) AND path_freq != -9999
        {f"AND region = '{region}'" if region else ""}
    """
    # ... return CheckResult
```

### N004: network Connectivity Validation

```python
@register_check(
    "N004",
    Category.TOPOLOGY,
    Severity.ERROR,
    "Reaches with same network must be topologically connected",
)
def check_network_connectivity(conn, region=None):
    # Verify that all reaches sharing a network ID form a connected component
    # This requires graph traversal to validate
    pass
```

### N005: main_side=2 Should Be Outlet-Adjacent

```python
@register_check(
    "N005",
    Category.CLASSIFICATION,
    Severity.WARNING,
    "Secondary outlets (main_side=2) should be near actual outlets",
)
def check_secondary_outlet_position(conn, region=None):
    # Secondary outlets should typically have low dist_out
    # or be connected to reaches with end_reach=2
    query = f"""
    SELECT r.reach_id, r.region, r.main_side, r.end_reach, r.dist_out,
           r.n_rch_down, r.river_name
    FROM reaches r
    WHERE r.main_side = 2
        AND r.end_reach NOT IN (2)  -- Not an actual outlet
        AND r.dist_out > 100000  -- More than 100km from outlet
        {f"AND r.region = '{region}'" if region else ""}
    """
    # ... return CheckResult
```

---

## 9. SQL Queries for Validation

### Validate main_side Distribution

```sql
-- Distribution by region
SELECT region, main_side, COUNT(*) as cnt,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY region), 2) as pct
FROM reaches
GROUP BY region, main_side
ORDER BY region, main_side;
```

### Find Inconsistent main_side / stream_order

```sql
-- main_side=0 should have valid stream_order (except edge cases)
SELECT COUNT(*)
FROM reaches
WHERE main_side = 0 AND stream_order = -9999 AND path_freq > 0;

-- main_side=1/2 should have stream_order = -9999
SELECT COUNT(*)
FROM reaches
WHERE main_side IN (1, 2) AND stream_order != -9999;
```

### Validate network Connectivity

```sql
-- Count reaches per network
SELECT network, COUNT(*) as reach_count
FROM reaches
GROUP BY network
ORDER BY reach_count DESC
LIMIT 20;

-- Find isolated reaches (single-reach networks)
SELECT r.reach_id, r.region, r.network, r.river_name,
       r.n_rch_up, r.n_rch_down, r.reach_length
FROM reaches r
JOIN (
    SELECT network, COUNT(*) as cnt
    FROM reaches
    GROUP BY network
    HAVING COUNT(*) = 1
) single ON r.network = single.network;
```

---

## 10. Recommendations

### Immediate Actions

1. **Add N001-N003 lint checks** - Validate main_side consistency
2. **Document 40 anomalous reaches** - Create tracking issue
3. **Review reconstruction code** - Current `_reconstruct_reach_main_side()` is broken

### Code Fixes Required

1. **reconstruction.py:3614-3659** - Complete rewrite of `_reconstruct_reach_main_side()`
   - Current: Uses simple neighbor count logic (produces wrong values)
   - Correct: Should identify main channel via path traversal or facc comparison

2. **Add reactive dependencies:**
   ```python
   ("reach.path_freq", "reach.main_side"),  # path_freq -> main_side
   ("reach.main_side", "reach.stream_order"),  # main_side -> stream_order
   ```

### Data Quality Issue

The 40 reaches with `main_side=0` and `stream_order=-9999` should be investigated:
- Are these truly main channel reaches with broken path_freq?
- Or should they be reclassified as side channels?

---

## 11. References

- SWORD Product Description Document v17b, pages 5 (v17 changes), 12-13 (nodes), 17-18 (reaches), 23-24 (shapefiles), 26-27 (figures)
- `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py` lines 477-502, 770-794, 2493-2565, 3614-3659
- `/Users/jakegearon/projects/SWORD/src/sword_duckdb/lint/checks/topology.py` lines 232-249 (T004), 344-401 (T006)
- `/Users/jakegearon/projects/SWORD/docs/validation_specs/stream_order_path_segs_validation_spec.md`
