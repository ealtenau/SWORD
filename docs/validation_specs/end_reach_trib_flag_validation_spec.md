# Validation Specification: end_reach and trib_flag

## Overview

This document provides a comprehensive audit of the `end_reach` (end_rch) and `trib_flag` variables in SWORD, including their provenance, derivation logic, consistency rules, and validation checks.

---

## 1. end_reach (end_rch)

### 1.1 Official Definition

**Source:** SWORD Product Description Document v17b (pages 14, 19, 25, 27)

> **end_rch**: value indicating whether a reach is a headwater (1), outlet (2), or junction (3) reach. A value of 0 means it is a normal main stem river reach.

### 1.2 Valid Values

| Value | Meaning | Topological Implication |
|-------|---------|------------------------|
| 0 | Normal main stem reach | Has both upstream AND downstream neighbors |
| 1 | Headwater | No upstream neighbors (n_rch_up = 0) |
| 2 | Outlet | No downstream neighbors (n_rch_down = 0) |
| 3 | Junction | Multiple upstream neighbors (n_rch_up > 1) |

### 1.3 Source and Derivation

**Source Dataset:** COMPUTED (derived from topology)

**Derivation Method:** GRAPH_TRAVERSAL

**Code Path:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reconstruction.py`

**Function:** `_reconstruct_reach_end_reach` (lines 2438-2491)

```python
def _reconstruct_reach_end_reach(self, ...):
    """
    Reconstruct end_reach type from topology.
    Values: 0=main, 1=headwater, 2=outlet, 3=junction
    """
    # Get topology counts
    topology_df = self._conn.execute("""
        SELECT
            r.reach_id,
            COALESCE(r.n_rch_up, 0) as n_rch_up,
            COALESCE(r.n_rch_down, 0) as n_rch_down
        FROM reaches r
        WHERE r.region = ?
    """, [self._region]).fetchdf()

    def classify_reach(row):
        n_up = row['n_rch_up']
        n_down = row['n_rch_down']
        if n_up == 0:
            return 1  # headwater
        elif n_down == 0:
            return 2  # outlet
        elif n_up > 1:
            return 3  # junction
        else:
            return 0  # main

    topology_df['end_reach'] = topology_df.apply(classify_reach, axis=1)
```

### 1.4 Schema Definition

**File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/schema.py` (line 157, 258)

```python
# In NODES_TABLE (line 157):
end_reach INTEGER,           -- end_rch: 0=main, 1=headwater, 2=outlet, 3=junction

# In REACHES_TABLE (line 258):
end_reach INTEGER,           -- end_rch: 0=main, 1=headwater, 2=outlet, 3=junction
```

### 1.5 Attribute Specification

**File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reconstruction.py` (lines 486-492)

```python
"reach.end_reach": AttributeSpec(
    name="reach.end_reach",
    source=SourceDataset.COMPUTED,
    method=DerivationMethod.GRAPH_TRAVERSAL,
    source_columns=[],
    dependencies=["reach_topology"],
    description="End reach type: 0=main, 1=headwater, 2=outlet, 3=junction"
),
```

### 1.6 Node Inheritance

Nodes inherit `end_reach` from their parent reach.

**File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reconstruction.py` (lines 779-785)

```python
"node.end_reach": AttributeSpec(
    name="node.end_reach",
    source=SourceDataset.INHERITED,
    method=DerivationMethod.INHERITED,
    source_columns=[],
    dependencies=["reach.end_reach", "node.reach_id"],
    description="End reach type: inherited from parent reach"
),
```

### 1.7 Consistency Rules

| Rule | Description | Expected Behavior |
|------|-------------|-------------------|
| R1 | end_reach=1 implies n_rch_up=0 | Headwaters have no upstream |
| R2 | end_reach=2 implies n_rch_down=0 | Outlets have no downstream |
| R3 | n_rch_up=0 implies end_reach=1 | All reaches with no upstream should be headwaters |
| R4 | n_rch_down=0 implies end_reach=2 | All reaches with no downstream should be outlets |
| R5 | n_rch_up > 1 implies end_reach=3 | Junctions have multiple upstream |
| R6 | end_reach=0 implies n_rch_up >= 1 AND n_rch_down >= 1 | Main reaches have both neighbors |

### 1.8 Current Lint Check

**File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/lint/checks/attributes.py`

**Check ID:** A010 - `check_end_reach_consistency` (lines 529-592)

**Severity:** WARNING

**Current Logic:**
```python
query = f"""
SELECT
    r.reach_id, r.region, r.river_name, r.x, r.y,
    r.end_reach, r.n_rch_up, r.n_rch_down,
    CASE
        WHEN r.end_reach = 1 AND r.n_rch_up > 0 THEN 'marked_headwater_but_has_upstream'
        WHEN r.end_reach = 2 AND r.n_rch_down > 0 THEN 'marked_outlet_but_has_downstream'
        WHEN r.end_reach = 0 AND r.n_rch_up = 0 THEN 'unmarked_headwater'
        WHEN r.end_reach = 0 AND r.n_rch_down = 0 THEN 'unmarked_outlet'
    END as issue_type
FROM reaches r
WHERE (
    (r.end_reach = 1 AND r.n_rch_up > 0) OR
    (r.end_reach = 2 AND r.n_rch_down > 0) OR
    (r.end_reach = 0 AND r.n_rch_up = 0 AND r.n_rch_down > 0) OR
    (r.end_reach = 0 AND r.n_rch_down = 0 AND r.n_rch_up > 0)
)
    AND r.type NOT IN (5, 6)
    {where_clause}
```

**Note:** Check excludes type=5 (unreliable topology) and type=6 (ghost reaches).

### 1.9 Usage in Other Checks

**G001 (reach_length_bounds):** Excludes end_reach=1 from "too short" check because headwater reaches are expected to be short.

```python
# From geometry.py lines 43-56:
CASE
    WHEN r.reach_length < 100 AND COALESCE(r.end_reach, 0) != 1 THEN 'too_short'
    WHEN r.reach_length > 50000 THEN 'too_long'
END as issue_type
```

---

## 2. trib_flag (Tributary Flag)

### 2.1 Official Definition

**Source:** SWORD Product Description Document v17b (pages 13, 18, 24, 26)

> **trib_flag**: binary flag indicating if a large tributary not represented in SWORD is entering a reach. 0 - no tributary, 1 - tributary.

**Historical Note:** Added in Release v15 (February 2023):
> Added a tributary flag ("trib_flag") to the reach and node attributes. The tributary flag indicates whether a larger river identified in MERIT Hydro-Vector, but not in SWORD, is entering a reach or node.

### 2.2 Valid Values

| Value | Meaning |
|-------|---------|
| 0 | No unmapped tributary entering |
| 1 | Large unmapped tributary entering |

### 2.3 Important Clarification

**trib_flag is NOT about SWORD topology (n_rch_up count).** It indicates external tributaries from MERIT Hydro-Vector that are NOT represented in SWORD but contribute flow to a reach/node.

This distinction is critical:
- `n_rch_up > 1` = Junction in SWORD topology (multiple SWORD reaches converge)
- `trib_flag = 1` = External tributary not in SWORD (detected via facc jumps from MERIT Hydro)

### 2.4 Source and Derivation

**Source Dataset:** MERIT_HYDRO (MERIT Hydro-Vector)

**Derivation Method:** SPATIAL_PROXIMITY

**Original Algorithm:** `Add_Trib_Flag.py`

**RECONSTRUCTION_SPEC.md Reference:**
```python
# From /Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/RECONSTRUCTION_SPEC.md
# Section 6.1 Tributary Flag (trib_flag)

# cKDTree proximity search
# Find nodes within 0.003 degrees (~333m at equator)

kdt = sp.cKDTree(all_node_coords)
distances, indices = kdt.query(query_coords, k=10)

# Node is a tributary junction if:
# 1. Multiple reaches converge within threshold distance
# 2. Upstream reach count > 1

trib_flag = 1  # tributary junction
```

### 2.5 Attribute Specification

**File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reconstruction.py` (lines 431-437)

```python
"reach.trib_flag": AttributeSpec(
    name="reach.trib_flag",
    source=SourceDataset.MERIT_HYDRO,
    method=DerivationMethod.SPATIAL_PROXIMITY,
    source_columns=["stream_order"],
    dependencies=["reach.geom"],
    description="Tributary flag: cKDTree proximity (k=10, <=0.003 deg) to MERIT stream_order>=3 with sword_flag=0"
),
```

### 2.6 Current Implementation (Stub - FIXED 2026-02-02)

**Status:** The reconstruction is now a **stub that preserves existing values**.

The wrong implementation (`n_rch_up > 1`) was removed. Full reconstruction requires external MERIT Hydro-Vector (MHV) data files that are not included in the repository.

**File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reconstruction.py`

**Function (reach/node):** `_reconstruct_reach_trib_flag` and `_reconstruct_node_trib_flag`

```python
def _reconstruct_reach_trib_flag(self, ...):
    """
    STUB: Preserve existing trib_flag values.

    trib_flag indicates EXTERNAL tributaries from MERIT Hydro-Vector (MHV)
    that are NOT represented in SWORD but contribute flow to a reach.

    IMPORTANT: trib_flag != (n_rch_up > 1)
    - n_rch_up > 1 = SWORD junction (multiple SWORD reaches converge)
    - trib_flag = 1 = External tributary from MHV enters here
    """
    logger.warning("trib_flag requires external MHV data - preserving existing values")
    return {"status": "skipped", "reason": "requires MHV data", "updated": 0}
```

**Evidence the old approach was wrong (from v17b data):**
- 15,133 reaches have `trib_flag=1` but `n_rch_up <= 1` (true external tribs, NOT junctions)
- 17,482 reaches have `n_rch_up > 1` but `trib_flag=0` (junctions WITHOUT external trib)
- Only 836 reaches have BOTH

**Future:** To properly reconstruct trib_flag, obtain MHV data files from `data/inputs/MHV_SWORD/` and implement the spatial proximity algorithm from `Add_Trib_Flag.py`.

### 2.7 Schema Definition

**File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/schema.py`

```python
# In NODES_TABLE (line 149):
trib_flag INTEGER,           -- 0=no tributary, 1=tributary

# In REACHES_TABLE (line 250):
trib_flag INTEGER,           -- 0=no tributary, 1=tributary
```

### 2.8 Current Lint Check

**File:** `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/lint/checks/attributes.py`

**Check ID:** A005 - `check_trib_flag_distribution` (lines 235-294)

**Severity:** INFO (informational only)

**Purpose:** Reports distribution of trib_flag values.

```python
def check_trib_flag_distribution(conn, region=None, threshold=None):
    """
    Report trib_flag distribution.

    trib_flag indicates UNMAPPED tributaries (rivers not in SWORD topology
    but contributing flow, detected via facc jumps from MERIT Hydro).
    - 0 = no unmapped tributary
    - 1 = unmapped tributary entering

    This is NOT about n_rch_up count - it's about external flow sources.
    """
```

**Note:** The docstring correctly describes the original intent, but the simplified reconstruction conflates this with junction detection.

---

## 3. Ghost Reaches and Type=6

### 3.1 Impact on end_reach

Ghost reaches (type=6) are markers for headwater/outlet endpoints that were added in Beta v0.2 (December 2019) to improve topology representation.

**From RECONSTRUCTION_SPEC.md:**
```python
# Ghost Reaches (Headwater/Outlet Markers)
# Source: Reach_Definition_Tools_v11.py:ghost_reaches() (lines 5845-5951)

# 1. Identify ghost nodes:
# Spatial query: find 10 nearest neighbors within 500m
kdt = sp.cKDTree(all_points)
distances, indices = kdt.query(endpoints, k=10, distance_upper_bound=500)

# 2. Threshold: 2nd nearest neighbor >= 60m indicates isolated endpoint

# 3. Filter: if 4th nearest < 100m, remove from ghost list
# (too many neighbors = not really isolated)

# 4. Ghost reaches get type='6' (vs type='1' for rivers)
ghost_type = 6
```

**Validation Implication:** Ghost reaches (type=6) should be excluded from end_reach consistency checks because they are synthetic markers, not real river segments.

### 3.2 Impact on trib_flag

Ghost reaches should have trib_flag=0 since they are not real flow-carrying segments.

---

## 4. Failure Modes

| Failure Mode | Description | Likely Cause | Severity |
|--------------|-------------|--------------|----------|
| FM1 | end_reach=1 but n_rch_up > 0 | Topology update without end_reach recalc | ERROR |
| FM2 | end_reach=2 but n_rch_down > 0 | Topology update without end_reach recalc | ERROR |
| FM3 | end_reach=0 but n_rch_up=0 | Missing headwater classification | WARNING |
| FM4 | end_reach=0 but n_rch_down=0 | Missing outlet classification | WARNING |
| FM5 | end_reach=3 but n_rch_up <= 1 | Stale junction classification | WARNING |
| FM6 | n_rch_up > 1 but end_reach != 3 | Missing junction classification | WARNING |
| FM7 | trib_flag conflation | Using n_rch_up > 1 instead of MERIT Hydro | DATA_QUALITY |
| FM8 | Ghost reach with end_reach != 0 | Ghost misclassification | INFO |

---

## 5. Proposed Additional Checks

### 5.1 A011: Junction Classification Consistency

**Rationale:** Current A010 doesn't check that junctions (n_rch_up > 1) are properly marked.

```python
@register_check(
    "A011",
    Category.ATTRIBUTES,
    Severity.WARNING,
    "Junction reaches (n_rch_up > 1) should have end_reach=3",
)
def check_junction_classification(conn, region=None, threshold=None):
    """
    Check that junction reaches are properly classified.
    """
    query = f"""
    SELECT
        r.reach_id, r.region, r.river_name, r.x, r.y,
        r.end_reach, r.n_rch_up, r.n_rch_down
    FROM reaches r
    WHERE r.n_rch_up > 1
        AND r.end_reach != 3
        AND r.type NOT IN (5, 6)
        {where_clause}
    """
```

### 5.2 A012: trib_flag Source Validation

**Rationale:** Document that trib_flag reconstruction is approximate.

```python
@register_check(
    "A012",
    Category.ATTRIBUTES,
    Severity.INFO,
    "trib_flag may be approximate if MERIT Hydro-Vector not used",
)
def check_trib_flag_provenance(conn, region=None, threshold=None):
    """
    Check if trib_flag values appear to be from MERIT Hydro or approximation.

    If trib_flag distribution matches n_rch_up > 1 exactly, it's likely
    the simplified approximation rather than true MERIT Hydro detection.
    """
```

### 5.3 A013: Ghost Reach Classification

**Rationale:** Verify ghost reaches don't have conflicting classifications.

```python
@register_check(
    "A013",
    Category.ATTRIBUTES,
    Severity.INFO,
    "Ghost reaches (type=6) should have expected attribute patterns",
)
def check_ghost_reach_classification(conn, region=None, threshold=None):
    """
    Check that ghost reaches have consistent attributes.

    Ghost reaches are synthetic markers and should have:
    - trib_flag = 0 (not real flow paths)
    - Consistent end_reach classification
    """
```

---

## 6. Summary

### end_reach

| Aspect | Details |
|--------|---------|
| Source | Computed from topology (n_rch_up, n_rch_down) |
| Values | 0=main, 1=headwater, 2=outlet, 3=junction |
| Derivation | Graph traversal of reach_topology table |
| Code | `reconstruction.py:_reconstruct_reach_end_reach` (lines 2438-2491) |
| Lint Check | A010 (WARNING) - partial coverage |
| Gap | Missing junction (n_rch_up > 1) validation |

### trib_flag

| Aspect | Details |
|--------|---------|
| Source | MERIT Hydro-Vector (original) / Approximated (current) |
| Values | 0=no unmapped tributary, 1=unmapped tributary |
| Derivation | Spatial proximity to MHV streams not in SWORD |
| Code | `reconstruction.py:_reconstruct_reach_trib_flag` (lines 3784-3815) |
| Lint Check | A005 (INFO) - distribution report only |
| Gap | Current reconstruction conflates with junction detection |

---

## 7. References

1. SWORD Product Description Document v17b (March 2025)
2. `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/reconstruction.py`
3. `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/schema.py`
4. `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/RECONSTRUCTION_SPEC.md`
5. `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/lint/checks/attributes.py`
6. `/Users/jakegearon/projects/SWORD/src/updates/sword_duckdb/lint/checks/geometry.py`
