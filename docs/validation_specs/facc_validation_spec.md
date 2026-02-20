# Validation Spec: facc (Flow Accumulation)

## Summary

| Property | Value |
|----------|-------|
| **Source** | MERIT Hydro (Yamazaki et al., 2019) |
| **Units** | km^2 |
| **Resolution** | 3 arc-second (~90m at equator) |
| **Applied to** | Nodes, Reaches |

**Official definition (v17b PDD, Table 3):**
> "maximum flow accumulation value for each [node/reach]" (km^2)

**MERIT Hydro source definition:**
> "Provides elevation and flow accumulation at 3 arc-second resolution (~90m at the equator)."

---

## Code Path

### Primary Implementation
- **File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/reconstruction.py`
- **Reach facc:** Lines 1664-1722 (`_reconstruct_reach_facc`)
- **Node facc:** Lines 3334-3364 (`_reconstruct_node_facc`)

### AttributeSpec Definition
```python
# reconstruction.py:273-280
"reach.facc": AttributeSpec(
    name="reach.facc",
    source=SourceDataset.MERIT_HYDRO,
    method=DerivationMethod.MAX,
    source_columns=["facc"],
    dependencies=["centerline.x", "centerline.y"],
    description="Flow accumulation (km^2): np.max(facc[reach_centerlines]) - downstream has highest"
)
```

### Algorithm

#### Original Production (RECONSTRUCTION_SPEC.md Section 5.3)
```python
# Source: Reach_Definition_Tools_v11.py, Merge_Tools_v06.py:filter_facc()

# 1. Sample raw facc from MERIT Hydro raster at centerline points
raw_facc = merit_hydro.sample(centerline_x, centerline_y)

# 2. Filter outliers per segment
median_facc = np.median(facc[segment])
std_facc = np.std(facc[segment])
valid = facc[(facc <= median + std) & (facc >= median - std)]

# 3. Special case for high variability
if std > median:
    valid = facc[facc >= median - (median/2)]

# 4. Linear interpolation from min to max along segment
interp_facc = np.linspace(valid.min(), valid.max(), len(segment))

# 5. Aggregate to node/reach level
node_facc = np.max(facc[node_points])
reach_facc = np.max(facc[reach_points])
```

#### Current Reconstruction (DuckDB)
```python
# _reconstruct_reach_facc (lines 1683-1721)
# Reconstructs reach facc as MAX of node facc values
result_df = conn.execute("""
    SELECT n.reach_id, MAX(n.facc) as max_facc
    FROM nodes n
    WHERE n.region = ?
    GROUP BY n.reach_id
""", [region]).fetchdf()
```

**Note:** Current reconstruction derives reach facc from node values (MAX aggregation). Full reconstruction from MERIT Hydro source requires external raster data not currently integrated.

### Dependencies

| Dependency | Relationship |
|------------|--------------|
| `centerline.x`, `centerline.y` | Spatial join coords for MERIT Hydro sampling |
| `node.facc` | Reach facc = MAX(node facc) |
| MERIT Hydro raster | External source (not in DuckDB) |

---

## Failure Modes

### 1. D8 Routing vs SWORD Topology Mismatch
**Description:** MERIT Hydro uses D8 single-flow-direction routing. At bifurcations, flow accumulation follows only ONE downstream branch. SWORD topology allows multiple downstream connections.

**Impact:** At distributaries, the "other" branch shows drastically lower facc (sometimes near-zero when upstream facc is millions km^2).

**Check:** T003 (facc_monotonicity)

**Example (from lint analysis):**
| reach_id | Location | facc Drop |
|----------|----------|-----------|
| 62913200082 | Ganges Delta | 1,089,000 -> 18 km^2 |
| 63214000722 | Mekong Delta | 766,000 -> 11 km^2 |
| 62912400231 | Brahmaputra | 540,000 -> 4 km^2 |

### 2. Spatial Join Misalignment
**Description:** Centerline points may not align exactly with MERIT Hydro flow accumulation grid cells. Off-by-one-pixel errors can sample wrong accumulation values.

**Impact:** Anomalously low/high facc values, especially near channel edges.

**Check:** T003, A006 (attribute_outliers)

### 3. Lake Boundary Effects
**Description:** MERIT Hydro may have inconsistent facc values at lake inlets/outlets due to how lakes are represented in the hydrography grid.

**Impact:** facc may not increase smoothly through lake reaches.

**Check:** T003 (filter by lakeflag)

### 4. Missing/Invalid Values
**Description:** Centerlines outside MERIT Hydro coverage or in masked areas may have fill values (-9999) or NULL.

**Impact:** Invalid facc propagates to reach aggregations.

**Check:** A004 (attribute_completeness)

### 5. Interpolation Artifacts
**Description:** The facc filtering algorithm uses linear interpolation which may over-smooth or create artificial monotonicity within segments.

**Impact:** Local variations are smoothed out; may mask real hydrological features.

**Check:** None currently - acceptable behavior

### 6. Node-to-Reach Aggregation Order
**Description:** Reach facc is MAX of node facc. If node facc values are incorrect, reach facc inherits errors.

**Impact:** Error amplification from node to reach level.

**Check:** Compare node facc vs reach facc consistency

---

## Proposed Checks

| ID | Severity | Rule | Rationale |
|----|----------|------|-----------|
| T003 | WARNING | facc_downstream >= facc_upstream * 0.95 | D8 routing means facc should increase downstream (5% tolerance) |
| F001 | INFO | facc > 0 for all reaches with type != 5,6 | All valid reaches should have positive facc |
| F002 | WARNING | reach.facc >= MAX(node.facc) for reach | Reach aggregation consistency |
| F003 | INFO | facc != -9999 | No fill values |
| F004 | WARNING | facc correlation with width > 0.3 downstream | Larger rivers should be wider |
| F005 | INFO | facc jumps > 10x at non-junction reaches | Possible sampling error |
| F006 | WARNING | Distributary reaches flagged | Delta bifurcations need special handling |

### Existing Check: T003 (facc_monotonicity)

**File:** `/Users/jakegearon/projects/SWORD/src/sword_duckdb/lint/checks/topology.py:157-220`

**Current Implementation:**
```sql
SELECT r1.reach_id, r1.facc as facc_up, r2.facc as facc_down
FROM reaches r1
JOIN reach_topology rt ON r1.reach_id = rt.reach_id
JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id
WHERE rt.direction = 'down'
  AND r1.facc > 0 AND r1.facc != -9999
  AND r2.facc > 0 AND r2.facc != -9999
  AND r2.facc < r1.facc  -- No tolerance currently
```

**Current Results (v17b NA):** ~2.96% issues (7,355 reaches globally)

**Improvement Opportunity:** Add 5% tolerance, exclude delta regions (type=5), flag distributaries separately.

---

## Edge Cases

### Distributaries/Deltas

**Behavior:** facc appears to DECREASE at bifurcations because MERIT Hydro's D8 routing only follows one path. The "secondary" distributary branch has much lower facc.

**Current handling:** T003 flags as WARNING

**Recommendation:**
1. Accept ~3% violation rate as expected behavior in deltas
2. Consider new check: F006 to explicitly identify distributary reaches where facc decreases but topology is correct
3. For v18: Investigate bifurcation-aware facc calculation (sum upstream facc at confluences)

### D8 Routing Limitations

MERIT Hydro flow direction uses D8 (single flow direction per cell). This creates fundamental mismatches with SWORD's multi-downstream topology:

| Scenario | SWORD | MERIT Hydro D8 |
|----------|-------|----------------|
| Bifurcation (1 up, 2 down) | Both downstream get edge | Only one gets full facc |
| Braided channel | Multiple paths | Single dominant path |
| Delta distributary | All channels connected | Only "main" channel has high facc |

**Recommendation:** Document as known limitation. For critical analyses, use topology-based reach counting (path_freq, accumulated reaches) rather than facc.

### Headwater Reaches

**Behavior:** End reaches with end_reach=1 (headwaters) should have low facc values representing only local drainage.

**Check:** Headwater facc should be < 1000 km^2 (configurable threshold)

### Endorheic Basins

**Behavior:** Internally-draining basins may have outlets with unexpected facc values since they don't connect to ocean.

**Check:** Consider separate validation for endorheic vs exorheic basins

---

## Data Quality Statistics

Based on lint analysis (NA region, v17b):

| Metric | Value |
|--------|-------|
| Total reaches checked | 248,457 |
| T003 violations | 7,355 (2.96%) |
| Fill values (-9999) | 0 |
| NULL values | 0 |
| Zero values | ~50 (ghost reaches) |

### Regional Distribution of Issues

| Region | T003 Violations | Notes |
|--------|-----------------|-------|
| NA | 2.8% | Mackenzie, Mississippi deltas |
| SA | 3.2% | Amazon delta |
| AS | 3.5% | Ganges, Mekong deltas |
| EU | 2.1% | Lower violation rate (fewer large deltas) |
| AF | 2.9% | Niger delta |
| OC | 2.4% | - |

---

## Recommendation

### Short-term (v17c)

1. **Accept current T003 WARNING level** - 3% violation rate is structural due to D8/SWORD mismatch
2. **Add delta exclusion option** to T003 - Allow filtering by lakeflag=3 (tidal) or type=5 (delta)
3. **Implement F001 check** - Ensure no missing facc values on valid reaches
4. **Document known behavior** - Delta distributary facc drops are expected, not errors

### Medium-term (v18)

1. **Re-sample facc from MERIT Hydro** using improved spatial join (nearest neighbor vs point sampling)
2. **Add distributary_flag** attribute to mark reaches downstream of bifurcations
3. **Consider topology-based accumulation** - Count upstream reaches instead of using MERIT facc
4. **Integrate MERIT Hydro Vector** - Use MHV flow direction for better alignment

### Validation Query Templates

```sql
-- Check 1: facc completeness
SELECT COUNT(*) as missing_facc
FROM reaches
WHERE (facc IS NULL OR facc = -9999 OR facc <= 0)
  AND type NOT IN (5, 6);

-- Check 2: facc monotonicity with delta exclusion
SELECT r1.reach_id, r1.facc as facc_up, r2.facc as facc_down,
       r1.lakeflag, r1.type
FROM reaches r1
JOIN reach_topology rt ON r1.reach_id = rt.reach_id
JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id
WHERE rt.direction = 'down'
  AND r2.facc < r1.facc * 0.95
  AND r1.type NOT IN (5, 6)  -- Exclude deltas/ghosts
  AND r1.lakeflag != 3;       -- Exclude tidal

-- Check 3: Distributary detection (facc drops > 10x)
SELECT r1.reach_id, r1.facc, r2.facc,
       r1.facc / NULLIF(r2.facc, 0) as drop_ratio
FROM reaches r1
JOIN reach_topology rt ON r1.reach_id = rt.reach_id
JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id
WHERE rt.direction = 'down'
  AND r1.facc > r2.facc * 10
  AND r1.facc > 10000;  -- Only significant rivers
```

---

## References

1. **SWORD v17b PDD** - Table 3 (NetCDF variables), Table 4-5 (Shapefile/GPKG attributes)
2. **MERIT Hydro** - Yamazaki et al., 2019. "MERIT Hydro: A high-resolution global hydrography map based on latest topography datasets." Water Resources Research. https://doi.org/10.1029/2019WR024873
3. **reconstruction.py** - `_reconstruct_reach_facc` (lines 1664-1722)
4. **RECONSTRUCTION_SPEC.md** - Section 5.3 (Flow Accumulation)
5. **lint/CHECKS.md** - T003 facc_monotonicity documentation
6. **lint/ANALYSIS.md** - T003 results analysis
