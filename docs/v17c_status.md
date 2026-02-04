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

### Type Column Status ✅ RESOLVED

| Database | type column | Status |
|----------|-------------|--------|
| v17b DuckDB | ✅ Present | 248,673 reaches (copied from GPKGs 2026-02-03) |
| v17c DuckDB | ✅ Present | 248,674 reaches |
| PostgreSQL v17c | ✅ Present | 248,674 reaches (synced 2026-02-03) |

**Distribution:** type=1 (river): 65.8%, type=3 (lake): 10.3%, type=4 (canal): 8.6%, type=5 (tidal): 6.0%, type=6 (reservoir): 9.3%

**TODO (later):** Add type to validation pipeline lint checks.

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
- **Detection pipeline**: `src/updates/sword_duckdb/facc_detection/` ✅ Complete
- **GeoJSON exports**: `output/facc_detection/` for QGIS review
- **554 anomalies detected** globally (run on v17b pristine reference)
- **Correction pipeline**: Exists, awaiting visual validation before applying to v17c

### Detection Pipeline (2026-02-04)

**Run command:**
```bash
python -m src.updates.sword_duckdb.facc_detection.cli \
    --db data/duckdb/sword_v17b.duckdb \
    --all \
    --export-geojson \
    --output-dir output/facc_detection/
```

**Detection rules in `detect_hybrid()`:**

| Rule | Criteria | Count | Description |
|------|----------|-------|-------------|
| entry_point | facc_jump > 10 AND ratio_to_median > 50 | 424 | Bad facc enters network |
| jump_entry | path_freq invalid AND facc_jump > 20 AND FWR > 500 | 102 | D8 error with missing metadata |
| junction_extreme | FWR > 15000 AND end_reach = 3 AND facc_jump > 10 | 15 | Extreme at junctions |
| headwater_extreme | n_rch_up = 0 AND facc > 500K AND FWR > 5000 | 13 | Impossible headwater facc |

**Output files:**
- `entry_point.geojson` (424 reaches)
- `jump_entry.geojson` (102 reaches)
- `junction_extreme.geojson` (15 reaches)
- `headwater_extreme.geojson` (13 reaches)
- `all_anomalies.geojson` (554 total)
- `detection_summary.json`

### Seed Reaches (17 confirmed bad)

| reach_id | Region | Mode | Notes |
|----------|--------|------|-------|
| 64231000301 | SA | propagation | FWR=35,239 |
| 62236100011 | SA | entry | FWR=22,811 |
| 62238000021 | SA | entry | |
| 64231000291 | SA | propagation | |
| 62255000451 | SA | propagation | Amazon side channel |
| 17211100181 | SA | entry | |
| 13261100101 | SA | entry | |
| 13214000011 | SA | entry | |
| 13212000011 | SA | entry | |
| 62210000705 | SA | misrouted | facc should be on 62210000055/45/35 |
| 62233000095 | SA | entry | |
| 28315000523 | EU | propagation | |
| 28315000751 | EU | propagation | inherited from nearby lake |
| 28315000783 | EU | entry | |
| 31251000111 | AF | entry | |
| 31248100141 | AF | jump_entry | side channel inherited downstream facc |
| 32257000231 | AF | entry | |

**Seed detection:** 11/17 detected directly, 6 are propagation cases (inherit bad facc from upstream)

### Next Steps

1. **Visual review** in QGIS using exported GeoJSONs
2. **Identify false positives** from visual review
3. **Apply corrections** to v17c using correction pipeline
4. **Re-run T003 lint** to verify monotonicity restored

---

## SWOT Integration

### Coverage
- **Reaches with observations**: 180,266 / 248,373 (72.6%)
- **WSE/width obs**: Working, excellent agreement with v17b

### Slopes
- **reach_swot_obs.py**: Provides reach-level slope/wse/width via OLS ✅ (renamed from reach_slope_obs.py)
- **Negative slopes**: 24,129 reaches (documented, physically plausible in some cases)
- **Decision**: Using reach-level observations only (reach_swot_obs.py), section-level not needed

### Updated Coverage (2026-02-03)
| Region | Slope | WSE/Width |
|--------|-------|-----------|
| NA | 82.4% | 67.8% |
| SA | 86.2% | 82.8% |
| EU | 76.2% | 78.4% |
| AF | 87.0% | 84.6% |
| AS | 79.6% | 83.1% |
| OC | 77.3% | 78.6% |

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
