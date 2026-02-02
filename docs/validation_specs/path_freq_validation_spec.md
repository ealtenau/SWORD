# Validation Spec: path_freq

## Summary
- **Source:** Computed (graph traversal)
- **Units:** count (dimensionless integer)
- **Official definition (v17b PDD):** "the number of times a reach is traveled to get to any given headwater from the primary outlet."

## Semantic Meaning

`path_freq` represents the number of unique downstream paths that pass through a given reach. It is computed by traversing from each outlet upstream through the network and incrementing a counter for each reach visited. The value indicates network confluence structure:

- **Headwaters:** path_freq = 1 (only one path to outlet)
- **Trunk/main stem:** Higher values indicate more upstream tributaries converging
- **Outlets:** Highest values in network (sum of all upstream paths)

**Key relationship to stream_order:**
```python
stream_order = round(log(path_freq)) + 1  # where path_freq > 0
```

## Code Path

### Primary Implementation
- **File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reconstruction.py`
- **Lines:** 2353-2436
- **Function:** `_reconstruct_reach_path_freq()`

### Algorithm
```python
# Pseudocode from reconstruction.py:2353-2436

1. Build topology maps:
   - upstream_map: reach_id -> list[upstream_reach_ids]
   - downstream_map: reach_id -> list[downstream_reach_ids]

2. Find all outlets (n_rch_down = 0)

3. Initialize path_freq[all_reaches] = 0

4. For each outlet:
   visited = set()
   queue = [outlet_id]

   while queue:
       reach_id = queue.pop(0)  # BFS
       if reach_id in visited:
           continue
       visited.add(reach_id)
       path_freq[reach_id] += 1  # INCREMENT

       for upstream_id in upstream_map[reach_id]:
           if upstream_id not in visited:
               queue.append(upstream_id)
```

### Specification (AttributeSpec)
- **File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reconstruction.py`
- **Lines:** 441-448

```python
"reach.path_freq": AttributeSpec(
    name="reach.path_freq",
    source=SourceDataset.COMPUTED,
    method=DerivationMethod.GRAPH_TRAVERSAL,
    source_columns=[],
    dependencies=["reach_topology"],
    description="Path frequency: flow traversal count from outlet to headwater"
)
```

### Dependencies
Direct dependencies from `reconstruction.py` AttributeSpec:
1. `reach_topology` - The upstream/downstream neighbor relationships

Downstream dependents (attributes that depend on path_freq):
1. `reach.path_order` - Ranking by dist_out within same path_freq group
2. `reach.path_segs` - Count of reaches with same path_freq
3. `reach.stream_order` - `round(log(path_freq)) + 1`
4. `reach.main_side` - At junctions, branch with higher path_freq = main
5. `node.path_freq` - Inherited from parent reach

### Reactive Recalculation
- **File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reactive.py`
- **Lines:** 615-694

The reactive system uses `path_freq` to determine `main_side`:
- At junctions, compares `path_freq` of sibling upstream branches
- Higher `path_freq` = main channel (main_side=1)
- Lower `path_freq` = side channel (main_side=2)

## Expected Behavior

### Monotonicity Rule
**path_freq must be monotonically non-decreasing in the downstream direction.**

At any reach `A` with downstream neighbor `B`:
```
path_freq[B] >= path_freq[A]
```

At confluences where `A` and `C` flow into `B`:
```
path_freq[B] >= max(path_freq[A], path_freq[C])
```

In practice, at a true confluence:
```
path_freq[B] = path_freq[A] + path_freq[C]  # if A and C share no upstream paths
```

### Valid Ranges
| Condition | Expected Value |
|-----------|----------------|
| Headwaters (n_rch_up=0) | 1 |
| Linear reach | Same as upstream |
| Confluence | Sum of unique upstream paths |
| Outlet (n_rch_down=0) | Maximum for network |
| Side channels | Lower than parallel main channel |
| No data | -9999 |

### Relationship to Other Attributes
| Attribute | Relationship |
|-----------|-------------|
| stream_order | `round(log(path_freq)) + 1` |
| main_side | Higher path_freq at junction = main (0) |
| facc | Both increase downstream (correlated, not equal) |
| dist_out | Decreases downstream (inverse relationship) |
| path_order | Ranks paths within same path_freq |
| path_segs | Count of reaches with same path_freq |

## Failure Modes

### 1. Topology Corruption
**Description:** Incorrect upstream/downstream connections cause BFS traversal to miss or double-count reaches.
**Detection:** path_freq decreases downstream
**Check:** T002

### 2. Disconnected Networks
**Description:** Orphan reaches or isolated subnetworks not connected to any outlet.
**Detection:** path_freq = 0 for reaches with valid geometry
**Check:** T004 (orphan_reaches), T006 (connected_components)

### 3. Cycles in Topology
**Description:** Circular references in topology create infinite loops or incorrect counts.
**Detection:** BFS `visited` set prevents infinite loops, but path_freq may be too high or too low.
**Check:** T007 (topology_reciprocity)

### 4. Multiple Outlets (Distributaries/Deltas)
**Description:** Networks with bifurcations have multiple paths to ocean. Current algorithm counts from each outlet independently.
**Detection:** path_freq may be higher than expected in distributary systems.
**Check:** Not currently implemented (see Proposed Checks)

### 5. Incorrect Neighbor Counts
**Description:** `n_rch_up`/`n_rch_down` doesn't match actual topology table entries.
**Detection:** Topology traversal visits wrong reaches.
**Check:** T005 (neighbor_count_consistency)

### 6. Secondary Outlets Not Handled
**Description:** `main_side=2` indicates secondary outlets on main network. These may need special path_freq handling.
**Detection:** Unexpectedly high path_freq values at secondary outlets.
**Check:** Not currently implemented

## Existing Checks

### T002: path_freq_monotonicity
- **File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/lint/checks/topology.py`
- **Lines:** 91-153
- **Severity:** WARNING
- **Description:** Checks that path_freq increases toward outlets

**SQL Logic:**
```sql
-- Find reaches where downstream path_freq < upstream path_freq
SELECT reach_id, pf_up, pf_down, (pf_up - pf_down) as pf_decrease
FROM (
    SELECT r1.reach_id, r1.path_freq as pf_up, r2.path_freq as pf_down
    FROM reaches r1
    JOIN reach_topology rt ON r1.reach_id = rt.reach_id
    JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id
    WHERE rt.direction = 'down'
      AND r1.path_freq > 0 AND r1.path_freq != -9999
      AND r2.path_freq > 0 AND r2.path_freq != -9999
)
WHERE pf_down < pf_up
```

## Proposed Additional Checks

| ID | Severity | Rule | Rationale |
|----|----------|------|-----------|
| T002a | ERROR | Headwaters (n_rch_up=0) must have path_freq >= 1 | Headwaters are visited at least once per outlet path |
| T002b | WARNING | path_freq=0 only valid if reach is disconnected | Zero suggests topology bug or orphan reach |
| T002c | INFO | path_freq consistency with stream_order | `stream_order != round(log(path_freq)) + 1` indicates stale data |
| T002d | WARNING | At confluences, downstream path_freq should equal or exceed sum of unique upstream | Validates proper traversal counting |
| T008 | INFO | Distributary network detection | Flag networks with multiple outlets for manual review |
| T009 | WARNING | Secondary outlet (main_side=2) path_freq validation | Check secondary outlets have consistent path_freq |

### Proposed Check Details

#### T002a: Headwater path_freq validation
```sql
SELECT reach_id, path_freq
FROM reaches
WHERE n_rch_up = 0
  AND (path_freq IS NULL OR path_freq < 1 OR path_freq = -9999)
  AND type NOT IN (5, 6)  -- exclude unreliable topology, ghost
```

#### T002b: Zero path_freq validation
```sql
SELECT r.reach_id, r.path_freq, r.n_rch_up, r.n_rch_down
FROM reaches r
WHERE r.path_freq = 0
  AND (r.n_rch_up > 0 OR r.n_rch_down > 0)  -- connected but zero
```

#### T002c: stream_order consistency
```sql
SELECT reach_id, path_freq, stream_order,
       ROUND(LN(path_freq)) + 1 as expected_stream_order
FROM reaches
WHERE path_freq > 0 AND path_freq != -9999
  AND stream_order != ROUND(LN(path_freq)) + 1
  AND stream_order != -9999
```

#### T002d: Confluence summation check
```sql
-- At true confluences, downstream path_freq should >= sum of upstream
WITH confluences AS (
    SELECT r2.reach_id as downstream_id,
           r2.path_freq as pf_down,
           SUM(r1.path_freq) as sum_pf_up,
           COUNT(*) as n_upstream
    FROM reaches r1
    JOIN reach_topology rt ON r1.reach_id = rt.reach_id
    JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id
    WHERE rt.direction = 'down'
      AND r1.path_freq > 0 AND r2.path_freq > 0
    GROUP BY r2.reach_id, r2.path_freq
    HAVING COUNT(*) > 1  -- true confluences only
)
SELECT * FROM confluences
WHERE pf_down < sum_pf_up  -- path_freq not accounting for all tributaries
```

## Edge Cases

### 1. Bifurcations / Distributaries
In delta regions, a single upstream reach may split into multiple downstream reaches. The current algorithm:
- Counts from EACH outlet upstream
- Each outlet's BFS increments path_freq for its entire upstream network
- Result: Reaches upstream of bifurcation get incremented by each downstream outlet

**Example (distributary with 2 outlets):**
```
        [A]         path_freq=2 (visited from both outlets)
       /   \
     [B]   [C]      path_freq=1 each
      |     |
   outlet  outlet
```

This is correct behavior per the definition ("number of times traveled to get to any given headwater from the primary outlet"). However, it may cause confusion when comparing to single-outlet networks.

### 2. Disconnected Networks
Reaches in isolated subnetworks (no connection to any outlet):
- Will have path_freq=0 after traversal
- Should be flagged by T004 (orphan_reaches) and T006 (connected_components)
- May indicate topology errors or genuinely isolated water bodies

### 3. Side Channels with Separate Outlets
Side channels that have their own outlet (not rejoining main channel):
- Treated as independent networks
- Get their own path_freq starting from 1
- `main_side` classification may still work correctly based on relative path_freq

### 4. Lake Sandwiches
Reaches classified as river (type=1) between two lake reaches (type=3):
- May have topology that doesn't match flow direction
- path_freq calculation unaffected, but interpretation may be misleading
- Covered by C001 (lake_sandwich) check

### 5. Ghost Reaches (type=6)
Placeholder reaches without real geometry:
- Should have path_freq = -9999 (no data)
- Should be excluded from monotonicity checks

### 6. Unreliable Topology (type=5)
Reaches flagged as having questionable topology:
- path_freq may be incorrect
- Should potentially be excluded from strict validation

## Data Quality Considerations

### Known Issues
1. **Lake sandwich problem:** 3,167 reaches (1.55%) globally are rivers between lakes
2. **Region boundaries:** Cross-region reaches may have incomplete topology
3. **Pfafstetter basin boundaries:** path_freq values unique within level-2 basins only

### Validation Thresholds
| Check | Recommended Threshold | Notes |
|-------|----------------------|-------|
| T002 | 0 violations (strict) | Any decrease is topology error |
| T002a | 0 violations | Headwaters must have path_freq >= 1 |
| T002b | 0 violations | Zero path_freq indicates bug |
| T002c | <1% deviation | stream_order may be cached |
| T002d | <5% | Bifurcations complicate summation |

## Recommendations

### Immediate Actions
1. **Keep T002 as WARNING** - Some legitimate edge cases (distributaries) may cause false positives
2. **Add T002a as ERROR** - Headwater validation is strict and critical
3. **Add T002b as WARNING** - Zero detection helps find disconnected reaches

### Future Improvements
1. **Track distributary networks** - Add flag for networks with multiple outlets
2. **Per-network path_freq max** - Store expected maximum for each connected network
3. **Validate against facc** - path_freq and facc should have similar downstream growth patterns
4. **Add path_freq recalculation trigger** - When topology changes, auto-recalculate path_freq

### Integration with Reactive System
The reactive system should:
1. Mark `reach.path_freq` dirty when `reach.topology` changes
2. Cascade to dependents: `stream_order`, `path_order`, `path_segs`, `main_side`
3. Node `path_freq` values should auto-update from parent reach

Currently, `reactive.py` uses `path_freq` for `main_side` calculation but does NOT recalculate `path_freq` itself. This is a gap that should be addressed.

## Test Cases

### Unit Test Requirements
```python
def test_path_freq_headwater():
    """Headwaters should have path_freq >= 1."""

def test_path_freq_monotonic():
    """path_freq should not decrease downstream."""

def test_path_freq_confluence():
    """At confluence, downstream >= max(upstream)."""

def test_path_freq_distributary():
    """Upstream of bifurcation, path_freq = sum of outlet paths."""

def test_path_freq_disconnected():
    """Disconnected reaches should have path_freq = 0 or -9999."""

def test_stream_order_consistency():
    """stream_order should equal round(log(path_freq)) + 1."""
```

### Integration Test Queries
```sql
-- Count violations by region
SELECT region, COUNT(*) as violations
FROM (
    SELECT r1.reach_id, r1.region
    FROM reaches r1
    JOIN reach_topology rt ON r1.reach_id = rt.reach_id AND r1.region = rt.region
    JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id AND rt.region = r2.region
    WHERE rt.direction = 'down'
      AND r1.path_freq > r2.path_freq
      AND r1.path_freq != -9999 AND r2.path_freq != -9999
)
GROUP BY region
ORDER BY violations DESC;
```

## References

- SWORD Product Description Document v17b, pages 12-13, 17-18, 23-24
- `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reconstruction.py` lines 2353-2436
- `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/lint/checks/topology.py` lines 91-153
- `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reactive.py` lines 615-694
