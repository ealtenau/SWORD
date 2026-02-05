# Validation Specification: Reach Neighbor ID Arrays

**Version:** 1.0
**Date:** 2025-02-02
**Author:** Variable Audit System

---

## 1. Overview

This document specifies the source, computation, validation rules, and edge cases for the reach neighbor ID arrays in SWORD v17c. These 8 variables form the [4,N] upstream and downstream neighbor matrices used for network topology traversal.

| Variable | Description | Units | Type | Table |
|----------|-------------|-------|------|-------|
| `rch_id_up_1` | Primary upstream neighbor reach ID (rank 0) | reach_id | BIGINT | reach_topology |
| `rch_id_up_2` | Secondary upstream neighbor (rank 1) | reach_id | BIGINT | reach_topology |
| `rch_id_up_3` | Tertiary upstream neighbor (rank 2) | reach_id | BIGINT | reach_topology |
| `rch_id_up_4` | Quaternary upstream neighbor (rank 3) | reach_id | BIGINT | reach_topology |
| `rch_id_dn_1` | Primary downstream neighbor reach ID (rank 0) | reach_id | BIGINT | reach_topology |
| `rch_id_dn_2` | Secondary downstream neighbor (rank 1) | reach_id | BIGINT | reach_topology |
| `rch_id_dn_3` | Tertiary downstream neighbor (rank 2) | reach_id | BIGINT | reach_topology |
| `rch_id_dn_4` | Quaternary downstream neighbor (rank 3) | reach_id | BIGINT | reach_topology |

**Note:** These are NOT explicit columns in the reaches table. Instead, they are reconstructed on-demand from the `reach_topology` table as [4,N] numpy arrays for backward compatibility with the original SWORD NetCDF interface.

---

## 2. Representation and Storage

### 2.1 Python Interface (SWORD Class)

Users access these variables as [4,N] arrays:
```python
from sword_duckdb import SWORD

sword = SWORD('data/duckdb/sword_v17c.duckdb', 'NA')

# Access as [4,N] arrays
upstream = sword.reaches.rch_id_up     # shape: (4, 248674) for all NA reaches
downstream = sword.reaches.rch_id_down # shape: (4, 248674)

# Access individual neighbors
rank_0_upstream = upstream[0, :]       # Primary upstream (rch_id_up_1)
rank_1_upstream = upstream[1, :]       # Secondary upstream (rch_id_up_2)
rank_2_upstream = upstream[2, :]       # Tertiary upstream (rch_id_up_3)
rank_3_upstream = upstream[3, :]       # Quaternary upstream (rch_id_up_4)
```

### 2.2 Database Storage (DuckDB)

Stored in normalized table: `reach_topology`

**Schema** (`schema.py`, lines 303-316):
```sql
CREATE TABLE reach_topology (
    reach_id BIGINT NOT NULL,
    region VARCHAR(2) NOT NULL,
    direction VARCHAR(4) NOT NULL,  -- 'up' or 'down'
    neighbor_rank TINYINT NOT NULL, -- 0-3 (corresponds to array rank)
    neighbor_reach_id BIGINT NOT NULL,

    PRIMARY KEY (reach_id, region, direction, neighbor_rank)
);
```

### 2.3 Reconstruction Algorithm

**Class:** `SWORD._reconstruct_reach_topology()` in `sword_class.py`, lines 165-200

```python
def _reconstruct_reach_topology(self, direction: str) -> np.ndarray:
    """
    Reconstruct [4,N] topology array from reach_topology table.

    Parameters:
        direction: 'up' for upstream, 'down' for downstream

    Returns:
        [4,N] array of neighbor reach IDs
    """
    n = len(self._reaches_df)
    arr = np.zeros((4, n), dtype=np.int64)  # Initialize with 0 (no neighbor)

    reach_ids = self._reaches_df['reach_id'].values
    reach_idx = {rid: i for i, rid in enumerate(reach_ids)}

    # Query all neighbors for this direction
    topo_df = self._db.query("""
        SELECT reach_id, neighbor_rank, neighbor_reach_id
        FROM reach_topology
        WHERE direction = ? AND region = ?
    """, [direction, self.region])

    # Populate array
    for _, row in topo_df.iterrows():
        idx = reach_idx.get(row['reach_id'])
        if idx is not None:
            rank = row['neighbor_rank']
            if 0 <= rank <= 3:
                arr[rank, idx] = row['neighbor_reach_id']

    return arr
```

**Key properties:**
- Array is initialized with zeros (represents "no neighbor")
- Sparsity is preserved: reaches with < 4 neighbors have zero(s) in trailing positions
- Order matters: neighbor_rank determines position in the array (0=primary, 1=secondary, etc.)
- Cross-region references are allowed (neighbor_reach_id may not exist in same region)

---

## 3. Data Source and Semantics

### 3.1 Official Definition (PDD v17b)

> "rch_id_up: SWORD reach IDs of upstream neighbor reaches. Multiple upstream reaches arise at tributary confluences." (Table 3)
> "rch_id_down: SWORD reach IDs of downstream neighbor reaches. Multiple downstream reaches arise at distributaries." (Table 3)

**Original Format (NetCDF):** [4,N] arrays (4 = max neighbors in SWORD design)

### 3.2 Source Data and Lineage

| Aspect | Details |
|--------|---------|
| **Primary source** | GRWL centerline topology |
| **Derivation** | Flow direction analysis on centerline network |
| **Version** | v17b (original SWORD release) |
| **Reconstruction** | reach_topology table (normalized from NetCDF arrays) |

### 3.3 Semantics and Ranking

**Upstream neighbors** (`direction = 'up'`):
- Reaches whose flow direction points toward this reach
- Rank 0: Primary tributary (usually largest facc)
- Rank 1-3: Secondary/tertiary tributaries
- Used by `path_freq` calculation (count traversals to outlets)

**Downstream neighbors** (`direction = 'down'`):
- Reaches toward which this reach's flow points
- Rank 0: Primary channel (continues main stem)
- Rank 1-3: Secondary channels (distributaries)
- Used by `dist_out` calculation (BFS from outlets)

**Ranking criterion:** Historically based on centerline connectivity, but not explicitly documented. Likely ordered by:
1. Flow accumulation (facc) for tributaries
2. Continuity of main channel for distributaries
3. Spatial proximity for tied cases

---

## 4. Database Consistency Invariants

### 4.1 Critical Invariant: n_rch_up/down Consistency

**MUST hold:** `n_rch_up[i]` = COUNT of non-zero values in `upstream[0:4, i]`

```sql
-- This should return 0 rows if consistent:
SELECT r.reach_id, r.n_rch_up, r.n_rch_down,
       COALESCE(up.cnt, 0) as actual_up,
       COALESCE(down.cnt, 0) as actual_down
FROM reaches r
LEFT JOIN (
    SELECT reach_id, COUNT(*) as cnt
    FROM reach_topology
    WHERE direction = 'up'
    GROUP BY reach_id
) up ON r.reach_id = up.reach_id
LEFT JOIN (
    SELECT reach_id, COUNT(*) as cnt
    FROM reach_topology
    WHERE direction = 'down'
    GROUP BY reach_id
) down ON r.reach_id = down.reach_id
WHERE r.n_rch_up != COALESCE(up.cnt, 0)
   OR r.n_rch_down != COALESCE(down.cnt, 0);
```

### 4.2 Reciprocity Invariant

**MUST hold:** If A has downstream neighbor B, then B must have upstream neighbor A

```sql
-- Upstream reciprocity: if R has upstream neighbor U, U has downstream neighbor R
SELECT DISTINCT
    t1.reach_id as reach_with_upstream,
    t1.neighbor_reach_id as upstream_reach,
    'missing_downstream' as issue
FROM reach_topology t1
WHERE t1.direction = 'up'
  AND t1.region = ?
  AND NOT EXISTS (
      SELECT 1 FROM reach_topology t2
      WHERE t2.reach_id = t1.neighbor_reach_id
        AND t2.direction = 'down'
        AND t2.neighbor_reach_id = t1.reach_id
        AND t2.region = ?
  );
```

### 4.3 Value Constraints

| Constraint | Rule | Rationale |
|-----------|------|-----------|
| Non-negative | All reach IDs >= 0 (actually > 0 valid) | Reach IDs are positive integers |
| Sparse | Each reach has ≤ 4 neighbors per direction | NetCDF array design limit |
| Ordered | Ranks are 0, 1, 2, 3 (no gaps unless sparse) | Array indexing |
| Non-self | `neighbor_reach_id != reach_id` | No self-loops in hydrology |
| Nullability | 0 represents "no neighbor" | Initialized value for sparse matrix |

---

## 5. Valid Ranges and Distributions

### 5.1 Population Statistics

Based on v17b global SWORD database (248,674 reaches across 6 regions):

| Metric | Upstream | Downstream | Notes |
|--------|----------|------------|-------|
| Reaches with 0 neighbors | ~45% (headwaters) | ~2% (outlets) | Normal |
| Reaches with 1 neighbor | ~50% (mainstem) | ~80% (mainstem) | Most common |
| Reaches with 2 neighbors | ~4% (confluences) | ~15% (bifurcations) | Local distributions |
| Reaches with 3-4 neighbors | <1% | <3% | Major confluences/deltas |

### 5.2 Typical Cases

| Case | n_rch_up | n_rch_dn | end_reach | Notes |
|------|----------|----------|-----------|-------|
| Headwater | 0 | 1 | 1 | Source reach |
| Mainstem | 1 | 1 | 0 | Typical channel |
| Confluence | 2-4 | 1 | 3 | Where tributaries meet |
| Distributary | 1 | 2-4 | 0 | Delta/braided section |
| Outlet | ≥1 | 0 | 2 | Mouth of river |
| Orphan | 0 | 0 | N/A | Disconnected (error) |

### 5.3 Edge Cases

| Edge Case | Expected Behavior | Detection |
|-----------|-------------------|-----------|
| Cross-region neighbor | Valid (e.g., Danube crosses EU/AS boundary) | Check `reach_topology` for different region |
| Ghost reach (type=6) | May have unusual topology | Not filtered from topology |
| Lake reach (lakeflag=1) | Normal topology (still part of network) | No special handling |
| Deleted reach | Stale neighbor_reach_id references | Referential integrity check |
| Duplicate edges | Multiple entries for same (reach_id, direction, rank) | PK prevents this |

---

## 6. Existing Lint Checks

**Status:** These variables are NOT currently validated by the lint system.

### 6.1 Related Checks (Indirect)

| Check ID | Name | Severity | Coverage |
|----------|------|----------|----------|
| T004 | orphan_reaches | WARNING | Detects n_rch_up=0 AND n_rch_down=0 |
| T005 | neighbor_count_consistency | ERROR | Validates n_rch_up/down match topology table |
| T007 | topology_reciprocity | WARNING | Checks bidirectional consistency |
| A010 | end_reach_consistency | (proposed) | Validates n_rch_* vs end_reach |

### 6.2 Why No Direct Checks?

The neighbor ID arrays themselves are **not directly validated** because:
1. **Reconstruction-only:** These are derived arrays, not stored columns
2. **Lazy evaluation:** Reconstructed on-demand via `_reconstruct_reach_topology()`
3. **Coverage by indirect checks:** T005 validates that n_rch_up/down counts match reach_topology

---

## 7. Failure Modes and Detection

### 7.1 Common Failure Scenarios

| Failure Mode | Cause | Detection | Impact |
|--------------|-------|-----------|--------|
| Missing neighbor entry | Incomplete reach_topology data | T007 reciprocity check | Routing algorithms fail silently |
| Stale neighbor_reach_id | Referenced reach was deleted | Referential integrity check | Invalid graph traversal |
| Incorrect rank order | Rank values not 0-3 | Schema validation (TINYINT bounds) | Array indexing errors |
| Null reach_id in rank | Sparse array not filled properly | Non-zero in early ranks (algorithm check) | Interpretation ambiguity |
| Cross-region mismatch | Neighbor has different region | Cross-join analysis | Some queries may miss edges |

### 7.2 Detection Queries

**Query 1: Find reaches with mismatched neighbor counts**
```sql
SELECT r.reach_id, r.region, r.n_rch_up,
       (SELECT COUNT(*) FROM reach_topology WHERE reach_id = r.reach_id AND direction = 'up') as actual_up
FROM reaches r
WHERE r.n_rch_up != (
    SELECT COUNT(*) FROM reach_topology WHERE reach_id = r.reach_id AND direction = 'up'
);
```

**Query 2: Find missing reciprocal edges**
```sql
SELECT t1.reach_id, t1.neighbor_reach_id, t1.direction
FROM reach_topology t1
WHERE NOT EXISTS (
    SELECT 1 FROM reach_topology t2
    WHERE t2.reach_id = t1.neighbor_reach_id
      AND t2.neighbor_reach_id = t1.reach_id
      AND t2.direction = (CASE WHEN t1.direction = 'up' THEN 'down' ELSE 'up' END)
);
```

**Query 3: Find invalid neighbor ranks**
```sql
SELECT reach_id, direction, COUNT(*) as neighbor_count
FROM reach_topology
WHERE neighbor_rank NOT IN (0, 1, 2, 3)
GROUP BY reach_id, direction;
```

**Query 4: Find non-contiguous ranks (rank gaps)**
```sql
WITH ranked_neighbors AS (
    SELECT reach_id, direction,
           neighbor_rank,
           ROW_NUMBER() OVER (PARTITION BY reach_id, direction ORDER BY neighbor_rank) as expected_rank
    FROM reach_topology
    WHERE neighbor_rank IS NOT NULL
)
SELECT reach_id, direction, neighbor_rank, expected_rank
FROM ranked_neighbors
WHERE neighbor_rank != expected_rank - 1;  -- expect 0, 1, 2, 3 in order
```

---

## 8. Proposed Lint Checks

### 8.1 Proposed: T013 - Neighbor ID Reciprocity

**Check ID:** T013
**Category:** Topology
**Severity:** ERROR
**Description:** Upstream/downstream neighbor relationships must be bidirectional

```python
@register_check("T013", Category.TOPOLOGY, Severity.ERROR,
                "Upstream/downstream neighbors must be reciprocal")
def check_neighbor_reciprocity(conn, region=None):
    """
    Validates that if reach A lists reach B as upstream neighbor,
    reach B lists reach A as downstream neighbor.

    Non-reciprocal edges indicate incomplete topology data.
    """
    query = """
    WITH all_edges AS (
        SELECT reach_id, neighbor_reach_id, direction FROM reach_topology
        UNION ALL
        SELECT neighbor_reach_id, reach_id,
               CASE WHEN direction = 'up' THEN 'down' ELSE 'up' END
        FROM reach_topology
    ),
    reciprocal_check AS (
        SELECT a.reach_id, a.neighbor_reach_id, a.direction,
               COUNT(b.reach_id) as reciprocal_count
        FROM reach_topology a
        LEFT JOIN reach_topology b ON a.reach_id = b.neighbor_reach_id
                                   AND a.neighbor_reach_id = b.reach_id
                                   AND b.direction = CASE WHEN a.direction = 'up' THEN 'down' ELSE 'up' END
        GROUP BY a.reach_id, a.neighbor_reach_id, a.direction
    )
    SELECT reach_id, neighbor_reach_id, direction
    FROM reciprocal_check
    WHERE reciprocal_count = 0;
    """
```

### 8.2 Proposed: T014 - Neighbor Rank Contiguity

**Check ID:** T014
**Category:** Topology
**Severity:** WARNING
**Description:** Neighbor ranks should be contiguous (0, 1, 2, ...) without gaps

```python
@register_check("T014", Category.TOPOLOGY, Severity.WARNING,
                "Neighbor ranks must be contiguous (no gaps)")
def check_neighbor_rank_contiguity(conn, region=None):
    """
    For each (reach_id, direction), ranks should be 0, 1, 2, ... without gaps.
    A gap (e.g., ranks 0, 1, 3 missing 2) indicates data corruption or
    inconsistent normalization.
    """
    query = """
    WITH rank_check AS (
        SELECT reach_id, direction,
               MIN(neighbor_rank) as min_rank,
               MAX(neighbor_rank) as max_rank,
               COUNT(*) as neighbor_count
        FROM reach_topology
        GROUP BY reach_id, direction
    )
    SELECT reach_id, direction, neighbor_count,
           max_rank - min_rank + 1 as expected_count
    FROM rank_check
    WHERE (max_rank - min_rank + 1) != neighbor_count
       OR min_rank != 0;
    """
```

### 8.3 Proposed: T015 - Neighbor ID Validity

**Check ID:** T015
**Category:** Topology
**Severity:** WARNING
**Description:** All neighbor reach IDs should exist in reaches table

```python
@register_check("T015", Category.TOPOLOGY, Severity.WARNING,
                "All neighbor reach IDs must exist in reaches table")
def check_neighbor_id_validity(conn, region=None):
    """
    Detects stale neighbor_reach_id references (references to deleted reaches).
    May occur if reaches are deleted without topology cleanup.
    """
    query = """
    SELECT t.reach_id, t.neighbor_reach_id, t.direction, t.region
    FROM reach_topology t
    LEFT JOIN reaches r ON t.neighbor_reach_id = r.reach_id
    WHERE r.reach_id IS NULL;
    """
```

### 8.4 Proposed: T016 - No Self-Loop Neighbors

**Check ID:** T016
**Category:** Topology
**Severity:** ERROR
**Description:** A reach should never be its own neighbor

```python
@register_check("T016", Category.TOPOLOGY, Severity.ERROR,
                "Reaches cannot be their own neighbors (no self-loops)")
def check_self_loops(conn, region=None):
    """
    Self-loops violate hydrology and indicate data corruption.
    Should never occur in properly constructed topology.
    """
    query = """
    SELECT reach_id, direction, neighbor_rank
    FROM reach_topology
    WHERE reach_id = neighbor_reach_id;
    """
```

---

## 9. Code References

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| Schema: reach_topology | schema.py | 303-316 | Table definition |
| Reconstruct: rch_id_up/down | sword_class.py | 165-200 | Array reconstruction |
| View access: rch_id_up | views.py | 964-971 | ReachesView property |
| View access: rch_id_down | views.py | 974-981 | ReachesView property |
| Lint: T004 orphans | lint/checks/topology.py | 223-268 | Orphan detection |
| Lint: T005 count consistency | lint/checks/topology.py | 271-334 | n_rch_* validation |
| Lint: T007 reciprocity | lint/checks/topology.py | 404-479 | Edge bidirectionality |
| Test: rch_id arrays | test_sword_class.py | 140-146 | Shape validation |

---

## 10. Summary

### 10.1 Key Characteristics

| Aspect | Details |
|--------|---------|
| **Storage** | Normalized reach_topology table (not explicit columns) |
| **Interface** | [4,N] numpy arrays reconstructed on-demand |
| **Values** | Reach IDs (BIGINT), 0 = no neighbor |
| **Sparsity** | Reaches with < 4 neighbors have trailing zeros |
| **Coverage** | Indirectly validated by T004, T005, T007 |
| **Modification** | Via reach_topology table (not direct array manipulation) |

### 10.2 Validation Gaps

**Currently uncovered:**
1. **Bidirectional reciprocity** - T007 checks but not comprehensive
2. **Rank contiguity** - No check for rank gaps
3. **Neighbor validity** - No check for deleted reaches
4. **Self-loops** - No check for reach referencing itself

**Proposed additions:** T013, T014, T015, T016

### 10.3 Usage in SWORD Pipeline

These neighbor IDs are critical for:
1. **dist_out calculation** - BFS from outlets via downstream neighbors
2. **path_freq calculation** - Upstream traversal via upstream neighbors
3. **stream_order calculation** - Network aggregation
4. **Network connectivity analysis** - Finding connected components
5. **Topology visualization** - Building network graphs

**Corruption of these variables breaks:** All distance/order/connectivity calculations.

---

## 11. Implementation Notes

### 11.1 Backward Compatibility

The [4,N] array format is maintained for compatibility with the original SWORD NetCDF interface. Users should NOT:
- Assume array indices correspond to reach indices
- Modify arrays directly and expect database changes
- Mix array access with reach_topology table access

**Correct usage:**
```python
# Get all neighbors of reach 123
reach_idx = np.where(sword.reaches.id == 123)[0][0]
upstream_neighbors = sword.reaches.rch_id_up[:, reach_idx]
upstream_neighbors = upstream_neighbors[upstream_neighbors > 0]  # Remove zeros
```

### 11.2 Performance Considerations

- **Reconstruction is O(n_edges):** Iterates reach_topology table once
- **Array is sparse:** Most entries are zeros (acceptable for hydrology)
- **Caching:** Arrays are cached after first access in SWORD instance
- **Memory:** [4, 248K] = 8M entries × 8 bytes = 64 MB per direction

### 11.3 Migration from NetCDF

Original NetCDF SWORD stored rch_id_up/down as explicit variables. DuckDB version normalizes to reach_topology table for:
- **Flexibility:** Variable neighbor counts without padding
- **Querying:** SQL-native topology traversal
- **Maintenance:** Single source of truth (not duplicated in reaches table)

---

## 12. Related Issues

| GitHub Issue | Status | Description |
|--------------|--------|-------------|
| #14 | Needs investigation | "Fix facc using MERIT Hydro" - may affect topology |
| #107 | Fixed | "main_side reconstruction was using wrong value mapping" |
| #102 | Fixed | "trib_flag reconstruction was using wrong definition" |

---

## 13. Testing

### 13.1 Unit Tests

```python
def test_rch_id_up_shape():
    sword = SWORD('test_minimal.duckdb', 'NA')
    assert sword.reaches.rch_id_up.shape[0] == 4
    assert sword.reaches.rch_id_up.shape[1] == len(sword.reaches)
    assert sword.reaches.rch_id_up.dtype == np.int64

def test_rch_id_reconstruction():
    # Verify reconstruction matches reach_topology table
    sword = SWORD('sword_v17c.duckdb', 'NA')

    # Sample check
    reach_id = sword.reaches.id[0]
    reach_idx = 0

    upstream_from_array = sword.reaches.rch_id_up[:, reach_idx]
    upstream_from_array = upstream_from_array[upstream_from_array > 0]

    upstream_from_db = conn.execute("""
        SELECT neighbor_reach_id FROM reach_topology
        WHERE reach_id = ? AND direction = 'up'
        ORDER BY neighbor_rank
    """, [reach_id]).fetchall()

    assert set(upstream_from_array) == set([r[0] for r in upstream_from_db])

def test_neighbor_count_consistency():
    # Verify n_rch_up/down match array non-zeros
    sword = SWORD('sword_v17c.duckdb', 'NA')

    for i in range(len(sword.reaches)):
        upstream = sword.reaches.rch_id_up[:, i]
        actual_up = np.count_nonzero(upstream)
        assert actual_up == sword.reaches.n_rch_up[i]
```

### 13.2 Integration Tests

- **Full region load:** Verify all reach_topology entries reconstruct correctly
- **Cross-region references:** Confirm neighbors can be in different regions
- **Reciprocity validation:** Run T007 over full database
- **Performance:** Time reconstruction for large regions (should be < 1s)

---

## 14. Appendix: SQL Queries for Analysis

### A.1 Neighbor Count Distribution

```sql
SELECT 'upstream' as direction,
       COUNT(*) as num_reaches,
       SUM(CASE WHEN n_rch_up = 0 THEN 1 ELSE 0 END) as zero_neighbors,
       SUM(CASE WHEN n_rch_up = 1 THEN 1 ELSE 0 END) as one_neighbor,
       SUM(CASE WHEN n_rch_up = 2 THEN 1 ELSE 0 END) as two_neighbors,
       SUM(CASE WHEN n_rch_up >= 3 THEN 1 ELSE 0 END) as three_plus_neighbors
FROM reaches
UNION ALL
SELECT 'downstream',
       COUNT(*),
       SUM(CASE WHEN n_rch_down = 0 THEN 1 ELSE 0 END),
       SUM(CASE WHEN n_rch_down = 1 THEN 1 ELSE 0 END),
       SUM(CASE WHEN n_rch_down = 2 THEN 1 ELSE 0 END),
       SUM(CASE WHEN n_rch_down >= 3 THEN 1 ELSE 0 END)
FROM reaches;
```

### A.2 Verify Array Reconstruction

```sql
-- Check that array reconstruction matches DB
WITH array_reconstruction AS (
    SELECT reach_id,
           COUNT(CASE WHEN neighbor_rank = 0 THEN 1 END) as has_rank_0,
           COUNT(*) as neighbor_count
    FROM reach_topology
    WHERE direction = 'up'
    GROUP BY reach_id
)
SELECT r.reach_id, r.n_rch_up, ar.neighbor_count,
       CASE WHEN r.n_rch_up = ar.neighbor_count THEN 'MATCH' ELSE 'MISMATCH' END
FROM reaches r
LEFT JOIN array_reconstruction ar ON r.reach_id = ar.reach_id
WHERE (r.n_rch_up != ar.neighbor_count OR ar.neighbor_count IS NULL);
```

### A.3 Find Problem Reaches

```sql
-- Reaches with unusual topology
SELECT reach_id, n_rch_up, n_rch_down, end_reach, type
FROM reaches
WHERE (n_rch_up > 4 OR n_rch_down > 4)  -- Exceeds array bounds
   OR (n_rch_up = 0 AND n_rch_down = 0 AND type != 6)  -- Orphan non-ghost
   OR (n_rch_up > 0 AND end_reach = 1)  -- Headwater with upstream
   OR (n_rch_down > 0 AND end_reach = 2);  -- Outlet with downstream
```

