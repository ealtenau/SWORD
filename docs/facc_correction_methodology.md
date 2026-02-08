# Facc Correction Methodology

## Executive Summary

SWORD's flow accumulation (facc) values come from MERIT Hydro's D8 algorithm, which routes flow to a single downstream raster cell. This creates systematic errors when mapped onto SWORD's vector network:

1. **Anomalous reaches** (~1,725): D8 routes full upstream facc down wrong branch at bifurcations, giving side channels continental-scale drainage areas. Detected via FWR ratios, corrected with RF regressor.
2. **Conservation violations** (~18,000 junctions): At confluences, facc < sum(upstream facc) because D8 followed one branch and ignored the other. Fixed by enforcing junction floor.
3. **Bifurcation surplus** (~2,500 children): D8 clones full parent facc to ALL children instead of splitting. Fixed by width-proportional sharing with downstream cascade.
4. **Single-link noise** (~8,000 links): facc randomly drops 2-50% going downstream on 1:1 reaches due to raster-vector misalignment. **Not yet corrected** — fixing cascades through downstream chains and inflates totals by ~270%.

**Current state after all corrections:**
- 77% of reaches unchanged from v17b
- 26 junction deficits > 1 km² remain (down from thousands)
- 7,959 single-link drops remain (3.9% of all 1:1 links, median 16% drop)
- Global facc sum within 0.07% of v17b original

**Rollback:** Every correction has a CSV with original values. Full rollback to v17b possible via `scripts/rollback_facc_conservation.py`.

---

## Phase 1: Anomaly Detection and RF Correction

### Problem

~1,725 reaches had facc values 10-1000x too high. D8 routed full upstream facc down the wrong branch at bifurcations.

### Detection (11 rules, 1,725 anomalies)

| Rule | Count | Criteria |
|------|-------|----------|
| fwr_drop | 815 | FWR drops >5x going downstream, FWR > 500 |
| entry_point | 466 | facc jumps >10x, ratio_to_median > 40 |
| extreme_fwr | 200 | FWR > 15,000 (physically impossible) |
| jump_entry | 99 | path_freq <= 0, facc jump > 20x, FWR > 500 |
| upstream_fwr_spike | 40 | upstream FWR/this FWR > 10, facc > 100K |
| impossible_headwater | 30 | path_freq <= 2, facc > 1M, FWR > 5000 |
| invalid_side_channel | 27 | path_freq=-9999, main_side=1, facc > 200K |
| high_ratio | 17 | ratio_to_median > 500 |
| side_channel_misroute | 15 | main_side=1, fwr_drop > 20, facc > 100K |
| facc_sum_inflation | 12 | facc > 3x sum(upstream) at junction |
| headwater_extreme | 4 | headwater, facc > 500K, FWR > 5000 |

Key metrics:
- **FWR** (Flow-Width Ratio) = facc/width. Normal: 50-100. Anomalous: 5,000-60,000.
- **ratio_to_median** = (facc/path_freq) / regional_median. Normal: 0.5-2. Anomalous: 40-1000+.

### Correction (RF Regressor)

Trained Random Forest on ~247K clean reaches to predict facc from network position.

| Model | R² | Median Error | Top Feature | Use |
|-------|-----|-------------|-------------|-----|
| Standard (2-hop facc) | 0.98 | 0.3% | max_2hop_upstream_facc (64%) | Primary |
| No-facc (topology only) | 0.79 | 32.8% | hydro_dist_hw (56.6%) | Sanity check |

Results: Median facc reduced 14x (68,637 -> 4,933 km²). 3 false positives identified, 4 false negatives added as seeds.

### Files

```
src/updates/sword_duckdb/facc_detection/
  detect.py             # FaccDetector, rule-based detection
  correct.py            # FaccCorrector (regression)
  rf_regressor.py       # RF model classes
  rf_features.py        # 91-feature extraction
output/facc_detection/
  rf_regressor_*.joblib # Trained models
  rf_*_predictions.csv  # Correction predictions
```

---

## Phase 2: Conservation Corrections (Passes 1-3)

### Problem

After Phase 1 fixed the gross anomalies, the facc schema still violated mass conservation at thousands of junctions. D8 assigns each reach an independent facc from the raster, so values don't necessarily sum up across SWORD's vector topology.

Three distinct failure modes:

| Mode | Description | Cause |
|------|-------------|-------|
| **Junction deficit** | facc < sum(upstream facc) at confluences | D8 followed one branch, ignored tributary |
| **Bifurcation surplus** | Child gets full parent facc instead of share | D8 can only route to one cell |
| **Cascade** | Surplus/deficit propagates through downstream chain | Inherited from upstream error |

### Pass 1: Equal-Split Propagation

**Commit:** `8e58f62` | **Date:** 2026-02-07 | **Corrections:** 23,610

At bifurcations where D8 starved one child, raise the starved child to `parent_facc / n_children`. Only raises, never lowers.

| Region | Corrections | Median Delta |
|--------|-------------|-------------|
| NA | 3,976 | +7,371 km² |
| SA | 3,706 | +5,413 km² |
| EU | 3,272 | +5,700 km² |
| AF | 2,165 | +5,367 km² |
| AS | 9,116 | +3,133 km² |
| OC | 1,375 | +1,899 km² |

Tags: `edit_flag='facc_conservation_p1'`, `facc_quality='conservation_corrected_p1'`

### Pass 2: Junction Floor

**Commit:** `8d7a6c7` | **Date:** 2026-02-07 | **Corrections:** 9,170

At junctions (n_rch_up >= 2), enforce `facc >= sum(upstream_corrected_facc)`. Only raises, never lowers. Applied only at junctions, no cascade through single-upstream chains.

| Region | Corrections | Pct Increase |
|--------|-------------|-------------|
| NA | 1,402 | +1.5% |
| SA | 1,724 | +5.4% |
| EU | 932 | +0.8% |
| AF | 696 | +1.3% |
| AS | 3,961 | +4.7% |
| OC | 455 | +1.4% |

Result: 0 F006 lint violations remaining (was thousands).

Tags: `edit_flag='facc_conservation_p2'`, `facc_quality='conservation_corrected_p2'`

### Pass 3: Bifurcation Surplus + Cascade

**Commit:** `37139a8` | **Date:** 2026-02-07 | **Corrections:** 30,409

Single topological-order walk (headwater -> outlet):
1. At bifurcations (n_dn >= 2): if child_facc > expected_share * 1.5, lower to width-proportional share
2. Cascade: propagate lowering through single-upstream chains where child >> corrected parent
3. Junction floor: re-enforce facc >= sum(upstream) for any new deficits

| Region | Bifurc | Cascade | Junction | Net Change |
|--------|--------|---------|----------|-----------|
| NA | 308 | 4,178 | 329 | -5.8% |
| SA | 291 | 3,720 | 442 | -5.3% |
| EU | 317 | 3,010 | 144 | -6.5% |
| AF | 199 | 2,346 | 139 | -16.3% |
| AS | 1,149 | 10,977 | 883 | -7.0% |
| OC | 228 | 1,672 | 93 | -7.3% |

Tags: `edit_flag='facc_conservation_p3'`, `facc_quality='topology_derived'`

### Combined Effect (P1 + P2 + P3)

| Metric | Value |
|--------|-------|
| Total reaches | 248,673 |
| Unchanged from v17b | 191,285 (77%) |
| Modified | 57,388 (23%) |
| Global facc sum vs v17b | -0.07% |
| Changes < 2x | ~45,000 |
| Changes 2-10x | ~4,100 |
| Changes 10-100x | ~4,750 |
| Changes > 100x | 3,540 (1.4%) |

The 3,540 reaches with >100x change are mostly D8-starved channels (v17b facc < 1,000 km²) downstream of major rivers. Their new values are consistent with their width and network position. Median original facc for these: 115 km².

### Per-Region Comparison (v17b vs v17c)

| Region | v17b (B km²) | v17c (B km²) | Change |
|--------|-------------|-------------|--------|
| NA | 2.3 | 2.2 | -2.5% |
| SA | 3.0 | 3.9 | +29.2% |
| EU | 1.0 | 1.0 | -1.7% |
| AF | 2.7 | 2.3 | -16.0% |
| AS | 4.9 | 4.6 | -7.0% |
| OC | 0.4 | 0.4 | -6.8% |
| **Global** | **14.35 T** | **14.34 T** | **-0.07%** |

SA +29%: P1/P2 raised starved Amazon distributaries significantly; P3 lowered surplus branches but raises outweighed lowering. The individual reach values are order-of-magnitude correct given width/position context.

Note: Total facc sum (~14T km²) is ~96x Earth's land area. This is expected — facc is cumulative drainage area, so each reach's value includes all upstream drainage. Summing across reaches double-counts.

---

## Remaining Issues

### 1. Junction Deficits (26 remaining)

26 junctions where facc < sum(upstream) by >1 km². All are edge cases at topology irregularities. Fixable with a targeted junction floor pass.

### 2. Bifurcation Imbalance (551 over, 952 under)

At 2,910 bifurcations:
- 1,407 (48%): sum(children)/parent between 0.9x-1.1x (good)
- 551 (19%): sum(children) > 1.1x parent (D8 duplication residual)
- 952 (33%): sum(children) < 0.9x parent (D8 starved one child)

Median ratio: 1.000. The imbalance is mostly in the tails.

### 3. Single-Link Drops (7,959 reaches) — NOT YET FIXED

**This is the largest remaining issue.** 3.9% of all 1:1 links (parent has n_dn=1, child has n_rch_up=1) show facc decreasing downstream. This is physically impossible on a non-bifurcating channel.

| Drop Size | Count |
|-----------|-------|
| < 10 km² | 160 |
| 10-100 km² | 611 |
| 100-1K km² | 2,192 |
| 1K-10K km² | 2,780 |
| 10K-100K km² | 1,702 |
| > 100K km² | 514 |

| Drop as % of Parent | Count |
|---------------------|-------|
| < 1% | 1,165 |
| 1-5% | 1,232 |
| 5-20% | 1,974 |
| 20-50% | 2,949 |
| > 50% | 639 |

By region: NA 1,330 / SA 1,399 / EU 835 / AF 682 / AS 3,253 / OC 460

**Root cause:** MERIT D8 assigns each raster cell an independent flow accumulation. When SWORD's vector reaches map to slightly different raster cells than their topological neighbors, facc can drop. This is a raster-vector alignment issue, not a topology error.

**Why not fixed:** Enforcing `child_facc >= parent_facc` at single-dn links (monotonicity floor) cascades through every downstream chain. A single junction raise propagates through hundreds of downstream 1:1 reaches. Testing this on NA showed +272% total facc increase — clearly too aggressive.

**Potential approaches:**
- Accept as noise (median drop is 16%, mostly small reaches)
- Re-extract facc from MERIT raster at affected reach locations
- Use width-based facc estimation for the 639 severe (>50%) drops only
- Clamp drops to max X% decrease per link (soft monotonicity)

---

## Rollback

Every correction is recorded in CSV with original values:

```
output/facc_detection/
  facc_conservation_p1_{region}.csv   # 6 files, P1 originals
  facc_conservation_p2_{region}.csv   # 6 files, P2 originals
  facc_conservation_p3_{region}.csv   # 6 files, P3 originals
```

Rollback script:
```bash
# Dry run
python scripts/rollback_facc_conservation.py --db data/duckdb/sword_v17c.duckdb

# Roll back all passes
python scripts/rollback_facc_conservation.py --db data/duckdb/sword_v17c.duckdb --apply

# Roll back only P3
python scripts/rollback_facc_conservation.py --db data/duckdb/sword_v17c.duckdb --apply --passes 3

# Single region
python scripts/rollback_facc_conservation.py --db data/duckdb/sword_v17c.duckdb --apply --region SA
```

After rollback, sync to Postgres:
```bash
python scripts/update_facc_postgres.py --duckdb data/duckdb/sword_v17c.duckdb \
    --pg "$SWORD_POSTGRES_URL" --target-table reaches
```

---

## Files Reference

### Phase 1 (Anomaly Detection + RF)
```
src/updates/sword_duckdb/facc_detection/
  detect.py, correct.py, rf_regressor.py, rf_features.py
  features.py, evaluate.py, cli.py, reach_acc.py
```

### Phase 2 (Conservation Passes)
```
src/updates/sword_duckdb/facc_detection/
  correct_conservation.py       # Pass 1: equal-split propagation
  correct_conservation_p2.py    # Pass 2: junction floor
  correct_conservation_p3.py    # Pass 3: bifurcation surplus + cascade
  correct_topological.py        # Shared RTREE-safe DB update utility
```

### Rollback + Sync
```
scripts/
  rollback_facc_conservation.py  # Restore original facc from CSVs
  update_facc_postgres.py        # Push DuckDB facc to PostgreSQL
```

### Output
```
output/facc_detection/
  facc_conservation_p{1,2,3}_{region}.csv          # Rollback CSVs (18 files)
  facc_conservation_p{1,2,3}_summary_{region}.json # Summary stats (18 files)
  rf_regressor_*.joblib                             # RF models
  rf_*_predictions.csv                              # RF predictions
```
