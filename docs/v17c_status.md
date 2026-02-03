# v17c Status Report

*Prepared: February 2026 | Deadline: March 31, 2026*

## Summary

| Component | Status | Details |
|-----------|--------|---------|
| Lake Typology | 🟡 In Progress | 3,167 sandwiches identified, 165 fixed |
| facc Fixes | 🟡 In Progress | 7,355 T003 violations, fix exists |
| SWOT Integration | 🟢 Working | 72.6% coverage, slopes WIP |
| v17c Topology | 🟢 Working | hydro_dist, mainstem complete |
| PostgreSQL Backend | 🟢 Complete | Ready for testing |

---

## Lake Typology

### Current State
- **C001 lake_sandwich**: 3,167 identified, 165 reclassified (high-confidence)
- **Remaining**: 393 medium-confidence + 1,142 low-confidence need review
- **C004 lakeflag/type**: 240 questionable cases (semi-independent fields)
- **Island detection**: Architecture exists (type=3), no dedicated lint check

### Critical Issue: Type Column Missing

| Database | type column | Source |
|----------|-------------|--------|
| v17b GPKG | ✅ Present | Original UNC release |
| v17c DuckDB | ❌ Missing | Never copied |

**Action Required**: Copy `type` from v17b GPKG → v17c DuckDB

This blocks the lake sandwich GUI from fixing type/lakeflag mismatches.

### Routing Issue Example

**Reach 62270000143** (Amazon, SA):
- lakeflag=1, type=3 (classified as lake-on-river)
- width=4,560m
- Upstream/downstream: both lakeflag=0, type=1 (normal river)

**Problem**: Mid-channel bar creates false lake classification, disrupts routing algorithms.

**Question for meeting**: How many similar cases exist? Systematic fix approach?

---

## facc Fixes

### Current State
- **T003 lint violations**: 7,355 reaches (2.96%), mostly in deltas
- **Root cause**: D8 flow direction mismatch with actual hydrology
- **fix_facc_violations()**: Implemented, uses upstream tracing
- **facc_quality flag**: Schema ready, workflow exists
- **Issue #14**: MERIT Hydro re-sampling NOT implemented

### ML/Statistical Model Sketch

**Goal**: Identify erroneous facc values for imputation

| Approach | Effort | Interpretability | Expected Perf |
|----------|--------|------------------|---------------|
| Regression (expected from upstream) | Low | High | P:85% R:60% |
| Ratio by stream_order | Low | High | P:80% R:50% |
| Random Forest classifier | Medium | High (SHAP) | P:88% R:75% |
| Isolation Forest (unsupervised) | Medium | Low | P:70% R:85% |

**Recommended: Regression baseline → Random Forest**

**Phase 1 - Regression:**
```
expected_facc = sum(upstream_facc) + local_contrib(width, slope, length)
anomaly_score = |actual - expected| / expected
flag if score > 2.0 (100% deviation)
```

**Phase 2 - Random Forest:**
- Features: facc, width, slope, reach_length, n_rch_up, stream_order, path_freq, dist_out, facc/width ratio, upstream_facc_median/std
- Target: Binary (corrupted/not)
- Training: T003 violations as weak negatives, consistent reaches as positives
- Validation: Does model reduce T003 violations after fixing?

**Current code**: None. fix_facc_violations() uses single threshold (facc/width > 5000).

### Concrete Examples: Tributary Misrouting

These reaches are tributaries but have mainstem-level facc (D8 routes Amazon/Parana flow through them):

| reach_id | facc (km²) | width (m) | facc/width | stream_order |
|----------|------------|-----------|------------|--------------|
| 64231000301 | 2.2M | 63 | **35,239** | 2 |
| 62236100011 | 5.2M | 228 | **22,811** | 2 |
| 62238000021 | 5.2M | 3,336 | 1,559 | 2 |
| 64231000291 | 2.2M | 2,261 | 982 | 2 |
| 62255000451 | 4.5M | 8,427 | 528 | 1 |

**Two failure modes identified:**

**1. Entry points** (where bad facc enters tributary):
| reach_id | upstream_facc | actual_facc | facc_ratio |
|----------|---------------|-------------|------------|
| 62238000021 | 1,628 | 5,199,400 | **3194x** |
| 62236100011 | 25,730 | 5,200,884 | **202x** |

**2. Propagation** (inherited bad facc, ratio ≈ 1.0):
- 64231000301, 62255000451, 64231000291

**Detection strategy:**
- Entry: `facc_jump / expected_local_contrib >> threshold`
- Propagation: `facc/width >> regional_norm_for_stream_order`

**Question for meeting**: Start ML now (regression baseline) or defer to v18?

---

## SWOT Integration

### Coverage
- **Reaches with observations**: 180,266 / 248,373 (72.6%)
- **WSE/width obs**: Working, excellent agreement with v17b

### Slopes
- **reach_slope_obs.py**: Provides reach-level slopes via OLS ✅
- **Negative slopes**: 24,129 reaches (documented, physically plausible in some cases)
- **Decision**: Using reach-level slopes only (reach_slope_obs.py), section-level slopes not needed

---

## v17c Topology Additions

### Working
| Attribute | Status | Notes |
|-----------|--------|-------|
| hydro_dist_out | ✅ | Dijkstra-based |
| hydro_dist_hw | ✅ | Max distance from headwater |
| best_headwater | ✅ | Width-prioritized |
| best_outlet | ✅ | Width-prioritized |
| is_mainstem_edge | ✅ | Continuous paths |
| v17c_sections | ✅ | 42,607 junction-to-junction segments |

### Known Issues
- **#69**: Mississippi/Ohio mainstem algorithm bug at Cairo confluence
- **#111**: 1,210 hydro_dist_out monotonicity violations (closed, documented as expected)

---

## PostgreSQL/DuckDB Architecture

**Status**: Complete, production-ready, awaiting testing

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SWORD Data Workflow                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐                                           │
│  │   PostgreSQL     │  ← Master DB (multi-user QGIS editing)    │
│  │   + PostGIS      │                                           │
│  │                  │                                           │
│  │  • Advisory locks│  ← Region-level locking                   │
│  │  • sword_ops log │  ← All changes tracked                    │
│  │  • synced flag   │  ← Marks what's pushed to DuckDB          │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           │ workflow.sync_to_duckdb()                           │
│           │ (incremental: only changed rows)                    │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │   DuckDB Copy    │  ← Local analysis/processing              │
│  │  sword_v17c.db   │                                           │
│  │                  │                                           │
│  │  • Fast queries  │                                           │
│  │  • Pipeline runs │                                           │
│  │  • Export source │                                           │
│  └──────────────────┘                                           │
│           │                                                     │
│           │ workflow.export()                                   │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │   Exports        │                                           │
│  │  • NetCDF        │  ← For UNC/JPL distribution               │
│  │  • GeoPackage    │  ← For GIS users                          │
│  │  • Parquet       │  ← For cloud analytics                    │
│  └──────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Components
- Backend abstraction layer (`backends/base.py`, `duckdb.py`, `postgres.py`)
- Connection manager (`sword_db.py`) - auto-detects backend
- PostgreSQL schema (`scripts/create_postgres_schema.sql`, 900 lines)
- DuckDB → PostgreSQL export
- PostgreSQL → DuckDB incremental sync
- Region-level advisory locking

### Related Issues
- #77, #78, #80, #120: Implemented (code exists, issues open for testing)

**Question for meeting**: Ready for undergrad QGIS editing? Timeline?

---

## GitHub Milestone: v17c-april-2026

| Status | Count |
|--------|-------|
| Closed | 17 |
| Open | ~30 |

### Priority Open Issues

| Priority | Issue | Title |
|----------|-------|-------|
| P0 | #87 | Topology errors |
| P0 | #101 | Error-level lint |
| P1 | #83 | Mainstem lint |
| P1 | #78, #77 | PostgreSQL |
| P1 | #72, #71, #70 | Validation |

---

## Meeting Discussion Points

1. **Lake routing**: How many cases like 62270000143? Systematic approach?
2. **PostgreSQL timeline**: When to deploy for undergrad QGIS editing?
3. **April deadline**: What's must-have vs nice-to-have?
