# SWORD v17c/v18 Roadmap

## Overview

| Version | Deadline | Scope | Issues |
|---------|----------|-------|--------|
| **v17c** | 1-2 months (JPL) | Attributes only, no geometry changes | 39 |
| **v18** | 6+ months | Geometry updates, new reaches, ID changes | 25 |

---

## v17c Scope (JPL Delivery)

### What's IN v17c
- **Topology fixes** - facc corrections from MERIT Hydro, new hydro_dist_out
- **Lake/type fixes** - Island reclassification, lakeflag updates
- **SWOT observation stats** - WSE, width, slope (mean, median, std, range, n_obs)
- **20+ new pipeline columns** - hydro_dist_out, hydro_dist_hw, is_mainstem_edge, subnetwork_id, etc.

### What's NOT in v17c
- No geometry changes
- No reach ID changes
- No dtype optimization

### v17c Milestones

| Milestone | Issues | Description |
|-----------|--------|-------------|
| v17c-verify | 7 | Verify pipeline outputs before use |
| v17c-topology | 6 | facc fix, stream_order, path_freq recalc |
| v17c-lake-type | 4 | Island/lake misclassification fixes |
| v17c-pipeline | 8 | Import 20+ new columns (6 regions) |
| v17c-swot | 4 | SWOT observation statistics |
| v17c-schema | 1 | Essential new columns |
| v17c-export | 6 | DuckDB, GPKG, NetCDF, Parquet exports |
| v17c-docs | 3 | Release notes, data dictionary |

### v17c New Columns

| Column | Description |
|--------|-------------|
| hydro_dist_out | Hydrological distance to outlet via main channel |
| hydro_dist_hw | Hydrological distance to headwater |
| is_mainstem_edge | Boolean mainstem flag |
| rch_id_up_main | Main upstream reach ID |
| rch_id_dn_main | Main downstream reach ID |
| subnetwork_id | Subnetwork identifier |
| main_path_id | Main path identifier |
| best_headwater | Chosen headwater reach |
| best_outlet | Chosen outlet reach |
| dist_out_suspect | Flag for 241 monotonicity violations |
| wse_mean/median/std/range | SWOT WSE statistics |
| width_mean/median/std/range | SWOT width statistics |
| slope_mean/median/std/range | SWOT slope statistics |
| n_obs | Number of SWOT observations |

### v17c Deliverables

| File | Format |
|------|--------|
| sword_v17c.duckdb | DuckDB |
| {region}_v17c.gpkg × 6 | GeoPackage |
| {region}_v17c.nc × 6 | NetCDF |
| {region}_v17c.parquet × 6 | Parquet (new) |
| CHECKSUMS.txt | SHA256 hashes |
| v17c_release_notes.md | Documentation |
| data_dictionary_v17c.md | Column definitions |

---

## v18 Scope (Future)

### What's IN v18
- **Centerline updates** - Sentinel-2 derived water masks
- **Reach modifications** - Merge short reaches, add MERIT rivers (~30m width)
- **Source data refresh** - Latest MERIT Hydro, updated GROD
- **ID mapping** - old_reach_id → new_reach_id tracking

### v18 Milestones

| Milestone | Issues | Description |
|-----------|--------|-------------|
| v18-planning | 5 | Scope, ID mapping design, legacy code |
| v18-sources | 5 | MERIT Hydro, GRWL v2, GROD updates |
| v18-imagery | 6 | Sentinel-2 centerline pipeline |
| v18-reach-mod | 6 | Merge/add reaches |
| v18-export | 3 | Exports + ID mapping file |

### v18 Key Challenges

| Challenge | Approach |
|-----------|----------|
| Narrow rivers (<50m) | Lower vote threshold from 4/6 |
| Braided/anastomosing | Multi-thread handling TBD |
| River migration | Skeleton + SWORD-guided pathfinding |
| Backward compatibility | reach_id_mapping.csv (M:N support) |

---

## Tracking

- **Issues**: https://github.com/ealtenau/SWORD/issues
- **Project Board**: https://github.com/users/jameshgrn/projects/1
- **v17c Branch**: `v17c-updates`
- **v18 Branch**: `v18-planning`

---

## Key Decisions

| Topic | Decision |
|-------|----------|
| dist_out | Keep legacy + add hydro_dist_out (both) |
| 241 violations | Flag, don't fix |
| facc source | MERIT Hydro as ground truth |
| Pipeline | Verify outputs BEFORE import |
| Exports | Legacy (NetCDF, GPKG) + new (Parquet) |
| v18 imagery | Sentinel-2 |
| v18 new reaches | MERIT Hydro source |
| v18 ID changes | Mapping table for backward compat |

---

## Timeline

```
NOW ─────────────────────────────────────────────────────> 6+ months
 │
 ├── v17c-verify (FIRST)
 ├── v17c-topology
 ├── v17c-lake-type
 ├── v17c-pipeline
 ├── v17c-swot
 ├── v17c-export
 └── v17c-docs
      │
      └── JPL DELIVERY (1-2 months)
                │
                ├── v18-planning
                ├── v18-sources
                ├── v18-imagery
                ├── v18-reach-mod
                └── v18-export
                     │
                     └── v18 RELEASE (6+ months)
```

---

## Unresolved Questions

1. **Estuary/tidal approach** - How to fix rivers extending too far into estuaries?
2. **facc threshold** - What defines "suspect" facc needing correction?
3. **Legacy reach code** - Need to locate in repo for v18 reuse
