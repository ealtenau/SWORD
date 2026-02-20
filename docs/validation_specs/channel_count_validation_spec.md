# Validation Spec: Channel Count Variables (n_chan_max, n_chan_mod, n_nodes)

## Summary

| Variable | Level | Source | Units | Official Definition (PDD v17b) |
|----------|-------|--------|-------|-------------------------------|
| `n_chan_max` | Node | GRWL | none | "maximum number of channels for each node" |
| `n_chan_mod` | Node | GRWL | none | "mode of the number of channels for each node" |
| `n_chan_max` | Reach | Computed | none | "maximum number of channels for each reach" |
| `n_chan_mod` | Reach | Computed | none | "mode of the number of channels for each reach" |
| `n_nodes` | Reach | Computed | none | "number of nodes associated with each reach" |

## 1. n_chan_max and n_chan_mod

### 1.1 Source Dataset

**Primary Source:** Global River Widths from Landsat (GRWL) [Allen & Pavelsky, 2018]

From PDD v17b Table 1:
> "Provides river centerline locations at 30 m resolution and associated width, water body type, and **number of channels** attributes."

The `nchan` attribute in GRWL represents the number of distinct water channels detected at each 30m centerline point using Landsat imagery.

### 1.2 Derivation Method

**Node Level:**
- `n_chan_max`: Maximum `nchan` value across all centerline points within the node
- `n_chan_mod`: Statistical mode (most frequent) `nchan` value across centerline points

**Reach Level:**
- `n_chan_max`: Maximum of all node `n_chan_max` values within the reach
- `n_chan_mod`: Mode of all node `n_chan_mod` values within the reach

### 1.3 Current Reconstruction Code

**File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py`

#### AttributeSpec Definitions (Lines 329-355, 640-656)
```python
# Node level
"node.n_chan_max": AttributeSpec(
    name="node.n_chan_max",
    source=SourceDataset.GRWL,
    method=DerivationMethod.MAX,
    source_columns=["nchan"],
    dependencies=["centerline.nchan", "centerline.node_id"],
    description="Node max channel count: np.max(nchan[node_centerlines])"
)

"node.n_chan_mod": AttributeSpec(
    name="node.n_chan_mod",
    source=SourceDataset.GRWL,
    method=DerivationMethod.MODE,
    source_columns=["nchan"],
    dependencies=["centerline.nchan", "centerline.node_id"],
    description="Node channel count mode"
)

# Reach level
"reach.n_chan_max": AttributeSpec(
    name="reach.n_chan_max",
    source=SourceDataset.GRWL,
    method=DerivationMethod.MAX,
    source_columns=["nchan"],
    dependencies=["centerline.nchan"],
    description="Maximum number of channels: np.max(nchan[reach_centerlines])"
)

"reach.n_chan_mod": AttributeSpec(
    name="reach.n_chan_mod",
    source=SourceDataset.GRWL,
    method=DerivationMethod.MODE,
    source_columns=["nchan"],
    dependencies=["centerline.nchan"],
    description="Mode of channel count: most frequent nchan value"
)
```

#### Algorithm from RECONSTRUCTION_SPEC.md (Section 5.5)
```python
# From original SWORD construction code (Reach_Definition_Tools_v11.py)
# n_chan_max: Maximum number of channels
node_n_chan_max = np.max(grwl_nchan[node_points])
reach_n_chan_max = np.max(node_n_chan_max[reach_nodes])

# n_chan_mod: Mode (most frequent) number of channels
from scipy.stats import mode
node_n_chan_mod = mode(grwl_nchan[node_points])[0][0]
reach_n_chan_mod = mode(node_n_chan_mod[reach_nodes])[0][0]
```

#### Reconstruction Methods (Lines 2254-2307, 3463-3519)

**Node Reconstruction:**
```python
def _reconstruct_node_n_chan_max(self, ...):
    result_df = self._conn.execute(f"""
        SELECT c.node_id, MAX(c.n_chan_max) as n_chan_max
        FROM centerlines c
        WHERE c.region = ? {where_clause}
        GROUP BY c.node_id
    """, params).fetchdf()

def _reconstruct_node_n_chan_mod(self, ...):
    result_df = self._conn.execute(f"""
        SELECT c.node_id, MODE(c.n_chan_max) as n_chan_mod
        FROM centerlines c
        WHERE c.region = ? {where_clause}
        GROUP BY c.node_id
    """, params).fetchdf()
```

**Reach Reconstruction:**
```python
def _reconstruct_reach_n_chan_max(self, ...):
    result_df = self._conn.execute(f"""
        SELECT n.reach_id, MAX(n.n_chan_max) as n_chan_max
        FROM nodes n
        WHERE n.region = ? {where_clause}
        GROUP BY n.reach_id
    """, params).fetchdf()

def _reconstruct_reach_n_chan_mod(self, ...):
    # Uses windowed mode calculation
    result_df = self._conn.execute(f"""
        SELECT reach_id, n_chan_mod
        FROM (
            SELECT n.reach_id, n.n_chan_mod, COUNT(*) as cnt,
                   ROW_NUMBER() OVER (PARTITION BY n.reach_id ORDER BY COUNT(*) DESC) as rn
            FROM nodes n
            WHERE n.region = ? {where_clause}
            GROUP BY n.reach_id, n.n_chan_mod
        )
        WHERE rn = 1
    """, params).fetchdf()
```

### 1.4 Valid Ranges and Statistics

**v17b Database Analysis:**

| Level | Variable | Min | Max | Mean | Median | Notes |
|-------|----------|-----|-----|------|--------|-------|
| Node | n_chan_max | 1 | 44 | 1.24 | 1 | 87.4% single-channel |
| Node | n_chan_mod | 1 | 20 | 1.11 | 1 | 91.2% single-channel |
| Reach | n_chan_max | 1 | 36 | 1.94 | 1 | 51.2% single-channel |
| Reach | n_chan_mod | 1 | 20 | 1.06 | 1 | 94.7% single-channel |

**Distribution by Lakeflag (Nodes):**
| lakeflag | Meaning | avg n_chan_max | max n_chan_max | count |
|----------|---------|----------------|----------------|-------|
| 0 | River | 1.21 | 36 | 9,634,597 |
| 1 | Lake/Reservoir | 1.37 | 44 | 843,010 |
| 2 | Canal | 1.04 | 8 | 38,149 |
| 3 | Tidal | 1.21 | 26 | 596,698 |

**Multi-Channel Reaches by Lakeflag:**
| lakeflag | Total | Multi-channel | Percentage |
|----------|-------|---------------|------------|
| 0 (River) | 204,458 | 98,275 | 48.1% |
| 1 (Lake) | 25,397 | 14,226 | 56.0% |
| 2 (Canal) | 1,251 | 343 | 27.4% |
| 3 (Tidal) | 17,567 | 8,401 | 47.8% |

### 1.5 Constraints

| Constraint | Expression | Rationale |
|------------|------------|-----------|
| Minimum value | `n_chan_max >= 1` | Must have at least one channel |
| Minimum value | `n_chan_mod >= 1` | Must have at least one channel |
| Mode <= Max | `n_chan_mod <= n_chan_max` | Mode cannot exceed maximum |
| Non-null | `n_chan_max IS NOT NULL` | Required attribute |
| Non-null | `n_chan_mod IS NOT NULL` | Required attribute |

**v17b Constraint Validation:**
- Violations of `n_chan_max >= 1`: 0
- Violations of `n_chan_mod >= 1`: 0
- Violations of `n_chan_mod > n_chan_max`: 0 (nodes), 0 (reaches)
- NULL values: 0 for both at node and reach level

---

## 2. n_nodes

### 2.1 Official Definition

From PDD v17b Table 3:
> "number of nodes associated with each reach"

### 2.2 Source and Derivation

**Source:** Computed from topology (count of nodes per reach)

**Algorithm:**
```python
n_nodes = COUNT(DISTINCT node_id WHERE node.reach_id = reach.reach_id)
```

### 2.3 Relationship to Node Spacing

From RECONSTRUCTION_SPEC.md Section 2.1:
```python
# Nodes are created at ~200m intervals along centerlines
node_len = 200  # meters (parameter)
divs = np.round(reach_length / node_len)
divs_dist = reach_length / divs
```

**Expected relationship:** `n_nodes ~ reach_length / 200`

**v17b Validation:**
| n_nodes | avg_reach_length | avg_node_spacing |
|---------|------------------|------------------|
| 1 | 194m | 194m |
| 5 | 1,013m | 203m |
| 10 | 1,992m | 199m |
| 50 | 9,953m | 199m |
| 100 | 19,843m | 198m |

### 2.4 Current Reconstruction Code

**File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py`

**AttributeSpec (Lines 207-213):**
```python
"reach.n_nodes": AttributeSpec(
    name="reach.n_nodes",
    source=SourceDataset.COMPUTED,
    method=DerivationMethod.COUNT,
    source_columns=[],
    dependencies=["node.reach_id"],
    description="Number of nodes in reach: len(unique(node_id[reach]))"
)
```

**Reconstruction Method (Lines 2148-2171):**
```python
def _reconstruct_reach_n_nodes(self, ...):
    result_df = self._conn.execute(f"""
        SELECT n.reach_id, COUNT(DISTINCT n.node_id) as n_nodes
        FROM nodes n
        WHERE n.region = ? {where_clause}
        GROUP BY n.reach_id
    """, params).fetchdf()
```

### 2.5 Valid Ranges and Statistics

**v17b Database Analysis:**
| Metric | Value |
|--------|-------|
| Min | 1 |
| Max | 100 |
| Mean | 44.69 |
| Median | 51 |

**Distribution:**
| n_nodes | Count | Percentage |
|---------|-------|------------|
| 1 | 25,732 | 10.35% |
| 2 | 18,491 | 7.44% |
| 3-10 | 15,584 | 6.27% |
| 11-50 | 63,235 | 25.43% |
| 51-90 | 101,017 | 40.62% |
| 91-99 | 22,522 | 9.06% |
| 100 | 1,092 | 0.44% |

**Maximum by Region:**
All regions have max n_nodes = 100 (hard limit from node creation algorithm)

### 2.6 Constraints

| Constraint | Expression | Rationale |
|------------|------------|-----------|
| Minimum | `n_nodes >= 1` | Every reach must have at least one node |
| Maximum | `n_nodes <= 100` | Design limit (~20km reaches max) |
| Non-null | `n_nodes IS NOT NULL` | Required attribute |
| Count match | `n_nodes = COUNT(nodes WHERE reach_id)` | Must match actual node count |
| Length consistency | `reach_length / n_nodes ~ 200m` | Expected node spacing |

**v17b Constraint Validation:**
- Violations of `n_nodes >= 1`: 0
- Violations of `n_nodes = 0`: 0
- Violations of n_nodes vs COUNT(nodes): 0 (perfect match)
- NULL values: 0

---

## 3. Failure Modes

### 3.1 n_chan_max / n_chan_mod Failure Modes

| # | Mode | Description | Severity | Impact |
|---|------|-------------|----------|--------|
| 1 | **Zero channels** | n_chan_max = 0 or n_chan_mod = 0 | ERROR | Physically impossible |
| 2 | **Mode > Max** | n_chan_mod > n_chan_max | ERROR | Statistical impossibility |
| 3 | **NULL values** | Missing channel count data | WARNING | Incomplete data |
| 4 | **Extreme values** | n_chan_max > 50 (arbitrary high) | INFO | Unusual, needs verification |
| 5 | **Source mismatch** | Node n_chan_max > centerline max | ERROR | Aggregation bug |
| 6 | **Reach-node mismatch** | reach.n_chan_max != MAX(node.n_chan_max) | ERROR | Aggregation bug |
| 7 | **Inconsistent mode** | reach.n_chan_mod != MODE(node.n_chan_mod) | WARNING | Tie-breaking may differ |

### 3.2 n_nodes Failure Modes

| # | Mode | Description | Severity | Impact |
|---|------|-------------|----------|--------|
| 1 | **Zero nodes** | n_nodes = 0 | ERROR | Invalid reach |
| 2 | **Count mismatch** | n_nodes != COUNT(actual nodes) | ERROR | Data corruption |
| 3 | **NULL value** | n_nodes IS NULL | ERROR | Missing required field |
| 4 | **Spacing anomaly** | reach_length / n_nodes << 100 or >> 300 | WARNING | Node creation issue |
| 5 | **Exceeds max** | n_nodes > 100 | WARNING | May indicate unusual reach |
| 6 | **Single-node long reach** | n_nodes = 1 AND reach_length > 1000m | INFO | Edge case worth review |

---

## 4. Existing Lint Checks

**Currently: NO dedicated lint checks exist for n_chan_max, n_chan_mod, or n_nodes.**

The following existing checks provide tangential coverage:
- **A004** (`attribute_completeness`): Checks NULL counts but does not include channel count fields
- **G002** (`node_length_consistency`): Validates sum(node_length) vs reach_length (indirectly related to n_nodes)

---

## 5. Proposed New Lint Checks

### 5.1 A011: Channel Count Validity
```python
@register_check(
    "A011",
    Category.ATTRIBUTES,
    Severity.ERROR,
    "n_chan_max and n_chan_mod must be >= 1 and mod <= max",
)
def check_channel_count_validity(conn, region=None, threshold=None):
    """
    Check channel count constraints:
    - n_chan_max >= 1
    - n_chan_mod >= 1
    - n_chan_mod <= n_chan_max
    """
    # SQL for nodes
    node_query = """
    SELECT node_id, region, n_chan_max, n_chan_mod,
           CASE
               WHEN n_chan_max < 1 THEN 'zero_max'
               WHEN n_chan_mod < 1 THEN 'zero_mod'
               WHEN n_chan_mod > n_chan_max THEN 'mod_exceeds_max'
           END as issue_type
    FROM nodes
    WHERE (n_chan_max < 1 OR n_chan_mod < 1 OR n_chan_mod > n_chan_max)
        AND region = COALESCE(?, region)
    """
    # Similar for reaches
```

### 5.2 A012: n_nodes Consistency
```python
@register_check(
    "A012",
    Category.ATTRIBUTES,
    Severity.ERROR,
    "n_nodes must match actual node count",
)
def check_n_nodes_consistency(conn, region=None, threshold=None):
    """
    Check that n_nodes equals COUNT(nodes) for each reach.
    """
    query = """
    WITH node_counts AS (
        SELECT reach_id, COUNT(*) as actual_count
        FROM nodes
        WHERE region = COALESCE(?, region)
        GROUP BY reach_id
    )
    SELECT r.reach_id, r.region, r.n_nodes, nc.actual_count
    FROM reaches r
    JOIN node_counts nc ON r.reach_id = nc.reach_id AND r.region = nc.region
    WHERE r.n_nodes != nc.actual_count
    """
```

### 5.3 A013: Channel Count Aggregation Consistency
```python
@register_check(
    "A013",
    Category.ATTRIBUTES,
    Severity.WARNING,
    "Reach n_chan_max should equal MAX(node.n_chan_max)",
)
def check_reach_channel_aggregation(conn, region=None, threshold=None):
    """
    Verify reach-level channel counts are correct aggregations of node values.
    """
    query = """
    WITH node_agg AS (
        SELECT reach_id, MAX(n_chan_max) as node_max
        FROM nodes
        WHERE region = COALESCE(?, region)
        GROUP BY reach_id
    )
    SELECT r.reach_id, r.region, r.n_chan_max as reach_max, na.node_max
    FROM reaches r
    JOIN node_agg na ON r.reach_id = na.reach_id
    WHERE r.n_chan_max != na.node_max
    """
```

### 5.4 A014: Node Spacing Anomalies
```python
@register_check(
    "A014",
    Category.ATTRIBUTES,
    Severity.INFO,
    "Node spacing should be approximately 200m",
    default_threshold=0.5,  # Allow 50% deviation (100-300m)
)
def check_node_spacing(conn, region=None, threshold=0.5):
    """
    Check that average node spacing is within expected range.
    Expected: 200m per node (100-300m acceptable)
    """
    query = """
    SELECT reach_id, region, reach_length, n_nodes,
           reach_length / n_nodes as avg_spacing
    FROM reaches
    WHERE n_nodes > 0
        AND (reach_length / n_nodes < 200 * (1 - ?)
             OR reach_length / n_nodes > 200 * (1 + ?))
        AND region = COALESCE(?, region)
    """
```

### 5.5 A015: Extreme Channel Counts
```python
@register_check(
    "A015",
    Category.ATTRIBUTES,
    Severity.INFO,
    "Flag unusually high channel counts for review",
    default_threshold=20,
)
def check_extreme_channel_counts(conn, region=None, threshold=20):
    """
    Flag nodes/reaches with very high channel counts.
    These may be valid (e.g., braided rivers) but warrant review.
    """
    query = """
    SELECT node_id, region, n_chan_max, reach_id, lakeflag
    FROM nodes
    WHERE n_chan_max > ?
        AND region = COALESCE(?, region)
    ORDER BY n_chan_max DESC
    """
```

---

## 6. Priority for Implementation

| Priority | Check ID | Severity | Rationale |
|----------|----------|----------|-----------|
| 1 | A011 | ERROR | Catches physically impossible values |
| 2 | A012 | ERROR | Validates critical reach attribute |
| 3 | A013 | WARNING | Ensures aggregation correctness |
| 4 | A014 | INFO | Quality metric for node creation |
| 5 | A015 | INFO | Identifies unusual features for review |

---

## 7. SQL Validation Queries

### Check n_chan constraints
```sql
-- Zero or negative channel counts (should return 0)
SELECT COUNT(*) FROM nodes WHERE n_chan_max < 1 OR n_chan_mod < 1;
SELECT COUNT(*) FROM reaches WHERE n_chan_max < 1 OR n_chan_mod < 1;

-- Mode exceeds max (should return 0)
SELECT COUNT(*) FROM nodes WHERE n_chan_mod > n_chan_max;
SELECT COUNT(*) FROM reaches WHERE n_chan_mod > n_chan_max;
```

### Check n_nodes consistency
```sql
-- Mismatch between n_nodes and actual count (should return 0)
WITH node_counts AS (
    SELECT reach_id, COUNT(*) as actual_count
    FROM nodes
    GROUP BY reach_id
)
SELECT COUNT(*)
FROM reaches r
JOIN node_counts nc ON r.reach_id = nc.reach_id
WHERE r.n_nodes != nc.actual_count;
```

### Check reach-node aggregation
```sql
-- Reach n_chan_max mismatch (should return 0)
WITH node_agg AS (
    SELECT reach_id, MAX(n_chan_max) as max_nchan
    FROM nodes
    GROUP BY reach_id
)
SELECT COUNT(*)
FROM reaches r
JOIN node_agg na ON r.reach_id = na.reach_id
WHERE r.n_chan_max != na.max_nchan;
```

---

## 8. Edge Cases

### 8.1 Single-Node Reaches
- **Count:** 25,732 reaches (10.35%)
- **Typical cause:** Short reaches (avg 194m) at headwaters, outlets, or junctions
- **Handling:** Valid - n_nodes = 1 is acceptable for short reaches

### 8.2 Maximum-Node Reaches (n_nodes = 100)
- **Count:** 1,092 reaches (0.44%)
- **Typical cause:** Long reaches (~20km) that hit the node count cap
- **Handling:** Valid but may indicate need for reach splitting

### 8.3 Lakes and Multi-Channel Features
- Lakes (lakeflag=1) have slightly higher multi-channel rates (56% vs 48% for rivers)
- This is expected behavior - lakes often have islands creating multiple channels
- No special handling needed

### 8.4 Ghost Reaches (type=6)
- Not present in current v17b DuckDB schema (`type` column not in tables)
- Filtering by `lakeflag` and `end_reach` may be needed instead

---

## 9. Known Issues

### 9.1 Centerline nchan Column Missing
**Issue:** The DuckDB schema for `centerlines` does not include `nchan` or `n_chan_max` columns.

**Columns present:** `cl_id, geom, node_id, reach_id, region, version, x, y`

**Impact:** The reconstruction code references `c.n_chan_max` from centerlines, which would fail.

**Root cause:** The nchan values may be stored differently in DuckDB vs NetCDF format, or the reconstruction code was written for a planned schema that wasn't implemented.

**Recommendation:** Either:
1. Add nchan to centerlines schema when importing from NetCDF
2. Modify reconstruction to read from nodes (current values are correct)

### 9.2 Mode Tie-Breaking
**Issue:** When multiple values tie for mode, different SQL implementations may return different results.

**Example:** If a reach has 25 nodes with n_chan_mod=1 and 25 with n_chan_mod=2, the mode could be either.

**Current handling:** The reconstruction code uses `ROW_NUMBER() ... ORDER BY COUNT(*) DESC` which selects arbitrarily among ties.

**Recommendation:** Add secondary sort criteria (e.g., lowest value wins) for deterministic results.

---

## 10. Recommendations

1. **Add A011 check immediately** - validates basic constraints, zero implementation risk
2. **Add A012 check** - validates n_nodes integrity, critical for downstream calculations
3. **Investigate centerline nchan** - determine if source data exists in DuckDB or needs import
4. **Add mode tie-breaking** - ensure deterministic behavior in reconstruction code
5. **Document expected ranges** - update PDD to include typical/max values for channel counts

---

## 11. References

- SWORD Product Description Document v17b, Tables 3-5
- Allen, G. H., & Pavelsky, T. M. (2018). Global extent of rivers and streams. Science, 361(6402), 585-588.
- RECONSTRUCTION_SPEC.md Section 5.5 (Number of Channels)
- `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py`
