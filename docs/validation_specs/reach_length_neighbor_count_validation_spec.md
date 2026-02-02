# Validation Specification: reach_length, n_rch_up, n_rch_down

**Version:** 1.0
**Date:** 2025-02-02
**Author:** Variable Audit System

---

## 1. Overview

This document specifies the source, computation, validation rules, and edge cases for three SWORD reach attributes:

| Variable | Description | Units | Table |
|----------|-------------|-------|-------|
| `reach_length` | Length of reach measured along centerline | meters | reaches |
| `n_rch_up` | Number of upstream neighbor reaches | none | reaches |
| `n_rch_down` | Number of downstream neighbor reaches | none | reaches |

---

## 2. reach_length

### 2.1 Official Definition (PDD v17b)

> "reach length measured along the high-resolution centerline points" (Table 3, Table 5)

**Units:** meters
**Dimensions:** [number of reaches]

### 2.2 Source Data

| Source | Dataset | Resolution |
|--------|---------|------------|
| Primary | GRWL centerlines | 30m |
| Derived from | Centerline x,y coordinates | ~30m spacing |

### 2.3 Algorithm

**Original Construction** (`Reach_Definition_Tools_v11.py`):
- Sum of Euclidean distances between consecutive centerline points
- Centerlines ordered by `cl_id` within each reach

**Reconstruction** (`reconstruction.py`, lines 1724-1804):
```python
def _reconstruct_reach_length(self, reach_ids, force, dry_run):
    # 1. Get centerlines ordered by cl_id within each reach
    cl_df = conn.execute("""
        SELECT reach_id, cl_id, x, y
        FROM centerlines
        WHERE region = ?
        ORDER BY reach_id, cl_id
    """)

    # 2. For each reach, sum distances
    for reach_id in cl_df['reach_id'].unique():
        reach_cl = cl_df[cl_df['reach_id'] == reach_id].sort_values('cl_id')
        x, y = reach_cl['x'].values, reach_cl['y'].values

        # 3. Convert degrees to meters (latitude-dependent)
        dx = np.diff(x)
        dy = np.diff(y)
        lat_mid = np.mean(y)
        meters_per_deg_lon = 111320 * np.cos(np.radians(lat_mid))
        meters_per_deg_lat = 110540

        dx_m = dx * meters_per_deg_lon
        dy_m = dy * meters_per_deg_lat

        # 4. Sum Euclidean distances
        total_length = np.sum(np.sqrt(dx_m**2 + dy_m**2))
```

### 2.4 Relationship to node_length

**Critical Constraint (v17 onwards):**
> "Corrected node lengths to match reach lengths when summed" (PDD v17)
> "Updates to reach and node lengths... to correct bug in node length calculation" (PDD v17b, <2% globally)

**Invariant:** `SUM(node_length) WHERE reach_id = R` should approximately equal `reach_length` for reach R.

### 2.5 Valid Ranges

| Range | Meaning | Source |
|-------|---------|--------|
| 100m - 50km | Normal range | G001 lint check |
| < 100m | Short reach (often end_reach=1 headwaters) | G001 |
| > 50km | Unusually long (may indicate missing junctions) | G001 |
| Median: 10.3 km | Global typical value | PDD Table 6 |
| Mean: 9.8 km | Global average | PDD Table 6 |

**Statistics by Type (PDD Table 6):**
- 60% of reaches: 10-20 km
- 23.6% of reaches: < 5 km
- 0.02% of reaches: > 20 km

### 2.6 Existing Lint Checks

| Check ID | Name | Severity | Description |
|----------|------|----------|-------------|
| G001 | reach_length_bounds | INFO | Flags <100m (excl end_reach=1) or >50km |
| G002 | node_length_consistency | WARNING | Flags SUM(node_length) differs from reach_length by >10% |
| G003 | zero_length_reaches | INFO | Flags reach_length <= 0 or NULL |

### 2.7 Edge Cases

| Edge Case | Expected Behavior | Check Coverage |
|-----------|-------------------|----------------|
| `end_reach = 1` (headwater) | May have short length (<100m) | G001 excludes from "too short" |
| Ghost reaches (type=6) | May have unusual length | Not explicitly handled |
| Single centerline point | Length = 0 (geometry error) | G003 |
| Lake reaches (lakeflag=1) | Often shorter than river reaches | G001 notes but doesn't filter |
| Cross-antimeridian reaches | Potential coordinate wrap issue | Not checked |

### 2.8 Failure Modes

| Failure | Cause | Detection |
|---------|-------|-----------|
| Length = 0 | Single centerline point, geometry error | G003 |
| Length mismatch with nodes | Node boundary errors, v17b bug | G002 |
| Unreasonably long | Missing junction points | G001 (>50km) |
| Unreasonably short | Fragmented reach definition | G001 (<100m) |
| NULL length | Missing centerline data | G003 |

### 2.9 Dependencies

**reach_length is used by:**
- `dist_out` calculation (accumulates reach lengths upstream)
- `slope` calculation (elevation change / length)
- Routing algorithms
- SWOT data aggregation

---

## 3. n_rch_up and n_rch_down

### 3.1 Official Definition (PDD v17b)

> "n_rch_up: number of upstream reaches for each reach" (Table 3, Table 5)
> "n_rch_down: number of downstream reaches for each reach"

**Units:** none (count)
**Dimensions:** [number of reaches]

### 3.2 Source Data

| Source | Dataset | Notes |
|--------|---------|-------|
| Derived from | reach_topology table | Aggregation of neighbor relationships |
| Original | rch_id_up, rch_id_dn arrays | [4, N] arrays in NetCDF |

### 3.3 Algorithm

**Reconstruction** (`reconstruction.py`, lines 2173-2221):

```python
def _reconstruct_reach_n_rch_up(self, reach_ids, force, dry_run):
    result_df = conn.execute("""
        SELECT t.reach_id, COUNT(*) as n_rch_up
        FROM reach_topology t
        WHERE t.region = ? AND t.direction = 'up'
        GROUP BY t.reach_id
    """)

def _reconstruct_reach_n_rch_down(self, reach_ids, force, dry_run):
    result_df = conn.execute("""
        SELECT t.reach_id, COUNT(*) as n_rch_down
        FROM reach_topology t
        WHERE t.region = ? AND t.direction = 'down'
        GROUP BY t.reach_id
    """)
```

### 3.4 Database Schema

**reaches table** (`schema.py`, lines 235-236):
```sql
n_rch_up INTEGER,      -- number of upstream neighbors
n_rch_down INTEGER,    -- number of downstream neighbors
```

**reach_topology table** (`schema.py`, lines 303-316):
```sql
CREATE TABLE reach_topology (
    reach_id BIGINT NOT NULL,
    region VARCHAR(2) NOT NULL,
    direction VARCHAR(4) NOT NULL,  -- 'up' or 'down'
    neighbor_rank TINYINT NOT NULL, -- 0-3
    neighbor_reach_id BIGINT NOT NULL,
    PRIMARY KEY (reach_id, region, direction, neighbor_rank)
);
```

### 3.5 Valid Ranges

| Value | Meaning | Valid For |
|-------|---------|-----------|
| 0 | No neighbors in direction | Headwaters (up), Outlets (down) |
| 1 | Single neighbor | Most reaches |
| 2-4 | Multiple neighbors | Confluences (up), Distributaries (down) |
| > 4 | Unusual | Should not occur (max 4 in NetCDF arrays) |

### 3.6 Consistency Invariants

**Critical Invariant:** `n_rch_up` and `n_rch_down` MUST match actual counts from `reach_topology` table.

```sql
-- This should return 0 rows if consistent:
SELECT r.reach_id, r.n_rch_up, r.n_rch_down,
       COALESCE(up.cnt, 0) as actual_up,
       COALESCE(down.cnt, 0) as actual_down
FROM reaches r
LEFT JOIN (
    SELECT reach_id, COUNT(*) as cnt
    FROM reach_topology WHERE direction = 'up'
    GROUP BY reach_id
) up ON r.reach_id = up.reach_id
LEFT JOIN (
    SELECT reach_id, COUNT(*) as cnt
    FROM reach_topology WHERE direction = 'down'
    GROUP BY reach_id
) down ON r.reach_id = down.reach_id
WHERE r.n_rch_up != COALESCE(up.cnt, 0)
   OR r.n_rch_down != COALESCE(down.cnt, 0);
```

### 3.7 Existing Lint Checks

| Check ID | Name | Severity | Description |
|----------|------|----------|-------------|
| T005 | neighbor_count_consistency | ERROR | n_rch_up/down must match reach_topology counts |
| T004 | orphan_reaches | WARNING | Reaches with n_rch_up=0 AND n_rch_down=0 |
| T007 | topology_reciprocity | WARNING | If A has downstream B, B should have upstream A |

### 3.8 Relationship to end_reach

| end_reach | Meaning | Expected n_rch_up | Expected n_rch_down |
|-----------|---------|-------------------|---------------------|
| 0 | Normal mainstem | >= 1 | >= 1 |
| 1 | Headwater | 0 | >= 1 |
| 2 | Outlet | >= 1 | 0 |
| 3 | Junction | >= 2 | >= 1 |

### 3.9 Edge Cases

| Edge Case | Expected Behavior | Check Coverage |
|-----------|-------------------|----------------|
| Orphan reach | n_rch_up=0 AND n_rch_down=0 | T004 |
| Ghost reach (type=6) | May have unusual topology | Not explicitly handled |
| Unreliable topology (type=5) | May have incorrect counts | Excluded from T010, T011 |
| Cross-region reaches | Neighbors in different region | Schema allows, may cause issues |
| Deleted neighbor | Stale count if not updated | T005 detects |

### 3.10 Failure Modes

| Failure | Cause | Detection |
|---------|-------|-----------|
| Count mismatch | Stale data after topology edit | T005 |
| Missing reciprocal | Incomplete topology update | T007 |
| Orphan non-ghost | Disconnected network | T004 |
| n_rch_down > 0 for outlet | end_reach inconsistency | Not checked (proposed A010) |
| n_rch_up > 0 for headwater | end_reach inconsistency | Not checked (proposed) |

### 3.11 Dependencies

**n_rch_up/n_rch_down are used by:**
- Network traversal algorithms
- `dist_out` BFS calculation (finds outlets via n_rch_down=0)
- `path_freq` calculation
- Junction/headwater/outlet identification

---

## 4. Proposed Additional Checks

### 4.1 Proposed: T012 - end_reach vs neighbor count consistency

**Rationale:** end_reach values should be consistent with n_rch_up/n_rch_down.

```python
@register_check("T012", Category.TOPOLOGY, Severity.WARNING,
                "end_reach must be consistent with neighbor counts")
def check_end_reach_neighbor_consistency(conn, region=None):
    query = """
    SELECT reach_id, region, end_reach, n_rch_up, n_rch_down,
           CASE
               WHEN end_reach = 1 AND n_rch_up > 0 THEN 'headwater_has_upstream'
               WHEN end_reach = 2 AND n_rch_down > 0 THEN 'outlet_has_downstream'
               WHEN end_reach = 3 AND n_rch_up < 2 THEN 'junction_few_upstream'
           END as issue
    FROM reaches
    WHERE end_reach IN (1, 2, 3)
      AND (
          (end_reach = 1 AND n_rch_up > 0) OR
          (end_reach = 2 AND n_rch_down > 0) OR
          (end_reach = 3 AND n_rch_up < 2)
      )
    """
```

### 4.2 Proposed: G004 - reach_length vs n_nodes consistency

**Rationale:** reach_length should approximately equal n_nodes * 200m (node spacing).

```python
@register_check("G004", Category.GEOMETRY, Severity.INFO,
                "reach_length should be consistent with node count")
def check_length_node_count_consistency(conn, region=None, threshold=0.5):
    # Expected: reach_length ≈ n_nodes * 200m
    # Flag if ratio is outside [0.5, 2.0] (very loose bounds)
    query = """
    SELECT reach_id, reach_length, n_nodes,
           reach_length / (n_nodes * 200.0) as length_ratio
    FROM reaches
    WHERE n_nodes > 0 AND reach_length > 0
      AND (reach_length < n_nodes * 100 OR reach_length > n_nodes * 400)
    """
```

---

## 5. Code References

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| Schema: reaches.reach_length | schema.py | 211 | Column definition |
| Schema: reaches.n_rch_up | schema.py | 235 | Column definition |
| Schema: reaches.n_rch_down | schema.py | 236 | Column definition |
| Schema: reach_topology | schema.py | 303-316 | Topology table |
| Reconstruct: reach_length | reconstruction.py | 1724-1804 | Centerline sum algorithm |
| Reconstruct: n_rch_up | reconstruction.py | 2173-2196 | Topology count |
| Reconstruct: n_rch_down | reconstruction.py | 2198-2221 | Topology count |
| Lint: G001 length_bounds | lint/checks/geometry.py | 20-85 | Length range check |
| Lint: G002 node_consistency | lint/checks/geometry.py | 88-156 | Node sum vs reach length |
| Lint: G003 zero_length | lint/checks/geometry.py | 159-205 | Zero/negative length |
| Lint: T004 orphan | lint/checks/topology.py | 223-268 | No neighbors |
| Lint: T005 neighbor_count | lint/checks/topology.py | 271-334 | Count consistency |
| Lint: T007 reciprocity | lint/checks/topology.py | 404-479 | Bidirectional edges |

---

## 6. Summary

### 6.1 reach_length

- **Source:** Computed from centerline geometry (sum of point-to-point distances)
- **Key constraint:** SUM(node_length) should equal reach_length
- **Typical range:** 5-20 km (60% of reaches are 10-20 km)
- **Coverage:** Well-covered by G001, G002, G003

### 6.2 n_rch_up / n_rch_down

- **Source:** Derived from reach_topology table (COUNT by direction)
- **Key constraint:** Must match actual topology table counts (T005)
- **Typical range:** 0-4 (max 4 neighbors stored in NetCDF arrays)
- **Coverage:** Well-covered by T004, T005, T007
- **Gap:** end_reach consistency not validated (proposed T012)

### 6.3 Validation Assessment

The existing lint checks provide **good coverage** for these variables:
- **reach_length:** 3 checks (G001, G002, G003) cover bounds, node consistency, and zero values
- **n_rch_up/n_rch_down:** 3 checks (T004, T005, T007) cover orphans, count consistency, and reciprocity

**Minor gap:** end_reach vs neighbor count consistency (proposed T012)
