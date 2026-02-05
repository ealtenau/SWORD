# FACC Error Detection and Correction

## The Problem

SWORD reach geometries come from GRWL (satellite-derived centerlines). Flow accumulation (facc) comes from MERIT Hydro (DEM-derived flow network). **These are independent datasets with different geometries.**

When SWORD reaches are assigned facc values from MERIT Hydro:
1. **Spatial mismatch:** SWORD centerlines don't perfectly align with MERIT flow paths
2. **Topology differences:** SWORD may have channels that MERIT doesn't recognize (or vice versa)
3. **D8 routing artifacts:** At bifurcations, MERIT's D8 algorithm routes ALL flow down one branch

**Result:** ~1,725 reaches have facc values 10-1000x too high - they got assigned facc from the wrong MERIT Hydro cell or inherited flow meant for a different channel.

---

## Detection: How We Found Them

### Key Metrics

**1. Flow-Width Ratio (FWR)**

```
FWR = facc / width
```

| Channel Type | Normal FWR |
|--------------|------------|
| Main channel | ~44 |
| Side channel | ~7 |
| Secondary outlet | ~1 |

Anomalous reaches have FWR of 2,000-75,000.

---

**2. Ratio to Median (facc vs network position)**

```
facc_per_reach = facc / path_freq
ratio_to_median = facc_per_reach / regional_median
```

`path_freq` is a traversal count that increases toward outlets - essentially "how many upstream reaches flow into this point."

- **Normal:** ratio_to_median = 0.5-2.0
- **Anomaly:** ratio_to_median = 40-1000+

This catches reaches with facc way too high for their network position.

---

**3. Upstream Sum Comparison**

```
facc vs SUM(upstream_facc)
```

At confluences, a reach's facc should roughly equal the sum of its upstream tributaries. If `facc > 3× upstream_sum`, the reach grabbed facc from the wrong MERIT cell.

### Detection Rules

**1. fwr_drop (815 detections)**

![fwr_drop schematic](../output/facc_detection/figures/schematic_fwr_drop.png)

*Logic:* If a reach has bad facc but its downstream neighbor has correct facc, the FWR will drop dramatically at that boundary.

```
FWR_current / FWR_downstream > 5
```

*Why it works:* Bad facc doesn't propagate everywhere - when flow rejoins the correct MERIT path downstream, FWR returns to normal. This rule catches the downstream edge of corrupted segments.

---

**2. entry_point (466 detections)**

![entry_point schematic](../output/facc_detection/figures/schematic_entry_point.png)

*Logic:* Bad facc "enters" a reach when facc suddenly jumps AND the resulting FWR is way above regional norms.

```
facc_current / facc_upstream > 10   (sudden 10x jump)
AND
FWR_current / regional_median_FWR > 40   (40x above peers)
```

*Why it works:* The facc jump alone isn't enough - a tributary joining a mainstem also causes a facc jump. The difference:

| Scenario | facc jump? | FWR after? |
|----------|------------|------------|
| Tributary → mainstem (legit) | Yes | Normal for mainstem width |
| Bad facc enters side channel | Yes | 40x+ above normal |

The **ratio_to_median** filter is key: when a mainstem receives tributary flow, its FWR stays normal because both facc AND width are large. When a narrow side channel (50-200m) gets mainstem facc (millions km²), the FWR explodes to 40-1000x above peers of similar stream order.

---

**3. extreme_fwr (200 detections)**

![extreme_fwr schematic](../output/facc_detection/figures/schematic_extreme_fwr.png)

*Logic:* Some FWR values are simply impossible - no river has 15,000+ km² drainage per meter of width.

```
FWR > 15,000
```

*Why it works:* Even the Amazon's FWR is ~2,000. Values above 15,000 indicate a narrow channel (50-200m) with drainage area of a continental river (millions of km²). Physically impossible.

---

**4. jump_entry (99 detections)**

![jump_entry schematic](../output/facc_detection/figures/schematic_jump_entry.png)

*Logic:* Disconnected side channels (path_freq = -9999) that somehow have high facc.

```
path_freq <= 0   (disconnected from main network)
AND facc_jump > 5   (significant facc present)
AND facc > 1000
```

*Why it works:* If a reach is topologically disconnected (path_freq invalid), it shouldn't have accumulated significant drainage. High facc here means it grabbed a value from an unrelated MERIT cell.

---

**5. facc_sum_inflation (45 detections)**

![facc_sum_inflation schematic](../output/facc_detection/figures/schematic_facc_sum_inflation.png)

*Logic:* At confluences, facc should equal the sum of upstream tributaries. Large inflation indicates D8 error.

```
n_rch_up >= 2   (confluence)
AND facc > 3 × SUM(upstream_facc)
```

*Why it works:* Mass balance. If two tributaries with facc = 100K and 50K meet, the downstream reach should have ~150K. If it has 500K+, the reach grabbed facc from a different MERIT cell.

---

### Detection Summary

![Detection Rules Breakdown](../output/facc_detection/figures/fig4_detection_rules.png)

| Rule | Count | % |
|------|-------|---|
| fwr_drop | 815 | 47% |
| entry_point | 466 | 27% |
| extreme_fwr | 200 | 12% |
| jump_entry | 99 | 6% |
| facc_sum_inflation | 45 | 3% |
| Other | 100 | 6% |
| **Total** | **1,725** | 100% |

---

## Correction: How We Fixed Them

### RF Regressor Approach

Train Random Forest on **clean reaches** to predict what facc SHOULD be based on network position.

**Training:** 247,000 clean reaches (all non-anomalous)
**Target:** Predict facc from 59 topology/position features
**Excludes:** Any feature derived FROM facc (to avoid circularity)

![RF Correction Logic](../output/facc_detection/figures/fig5_correction_logic.png)

### Why This Works

Facc = cumulative drainage area. Drainage accumulates as you move downstream. Therefore:

```
farther from headwater → more accumulated drainage → higher facc
```

The RF model learns: **"at this network position, facc should be approximately X"**

When it sees an anomalous reach at `hydro_dist_hw = 150 km` with `facc = 2,500,000 km²`, it knows that's wrong because other reaches at similar positions have facc ~5,000 km².

### Model Performance

| Metric | Value |
|--------|-------|
| R² | 0.77 |
| Median % error | 35% |
| Training samples | 247,000 |
| Features used | 59 |

### Top Predictive Features

![Feature Importance](../output/facc_detection/figures/fig1_feature_importance.png)

| Feature | Importance | Description |
|---------|------------|-------------|
| `hydro_dist_hw` | **56%** | Dijkstra distance from nearest headwater |
| `path_freq` | 6% | Network traversal count (increases toward outlets) |
| `main_side` | 5% | Channel type: 0=main, 1=side, 2=secondary |
| `log_path_freq` | 5% | Log-transformed path_freq |
| `main_path_id` | 3% | Mainstem identifier |
| `lakeflag` | 2% | Lake/river/canal/tidal classification |
| `wse` | 1% | Water surface elevation |
| `hydro_dist_out` | 1% | Dijkstra distance to outlet |

**Key insight:** `hydro_dist_hw` alone explains 56% of variance - network position is the dominant predictor.

### Why hydro_dist_hw works (not dist_out)

| Variable | Algorithm | Direction |
|----------|-----------|-----------|
| `dist_out` (v17b) | BFS upstream from outlets | Decreases downstream |
| `hydro_dist_hw` (v17c) | Dijkstra downstream from headwaters | **Increases downstream** |

**hydro_dist_hw computation:**
1. Build directed graph from SWORD topology (nodes = reaches, edges = flow)
2. Identify headwaters (reaches with no upstream neighbors)
3. For each headwater, run Dijkstra downstream using `reach_length` as edge weights
4. For each reach, keep the **maximum distance from any headwater** that can reach it

**Why it predicts facc:**
- facc **accumulates downstream** from headwaters
- `hydro_dist_hw` **increases downstream** from headwaters
- They move in the **same direction**

`dist_out` moves opposite to facc, so it's less intuitive for prediction.

### Feature Categories (59 total)

| Category | Count | Examples |
|----------|-------|----------|
| Position/Topology | 10 | hydro_dist_hw, path_freq, dist_out |
| Channel Classification | 13 | main_side, lakeflag, is_mainstem |
| Width metrics | 8 | width, max_width, width_ratio_to_dn |
| SWOT observations | 14 | wse_obs_median, width_obs_mean |
| Other | 14 | wse, slope, reach_length, network |

### Anomalies vs Clean Reaches

![Hydro Dist vs FACC](../output/facc_detection/figures/fig2_hydro_dist_vs_facc.png)

Anomalies (red) have facc values 10-1000x higher than clean reaches at the same network position.

### Correction Results

![Before/After FWR](../output/facc_detection/figures/fig3_fwr_before_after.png)

| Metric | Before | After |
|--------|--------|-------|
| Median facc | 68,637 km² | 4,933 km² |
| Median FWR | ~1,000 | ~50 |
| Reduction | - | **14x** |

---

## Example Corrections

| reach_id | River | Before | After | Reduction |
|----------|-------|--------|-------|-----------|
| 62210000705 | Amazon side | 5,885,793 | 907 | 6,500x |
| 28311300405 | Niger delta | 3,591,099 | 276 | 13,000x |
| 14631000181 | Ngalanka | 378,700 | 1,733 | 219x |

---

## Validation

- **39 seed reaches** (known bad) used for validation
- **36/39 (92%)** detected by rules
- **3 missed** - propagation patterns too subtle for any method

### Known Issues

**3 false positives** (wrongly corrected):
- 77250000153 - mainstem reach
- 74300400575, 74300400565 - incorrectly flipped

**4 false negatives** (missed, now added as seeds):
- 62293100143, 62293100156, 62253000321, 62235900101

---

## Summary

| Step | Method | Result |
|------|--------|--------|
| **Detection** | Rule-based (FWR ratios) | 1,725 anomalies found |
| **Correction** | RF regressor (network position) | Median 14x reduction |
| **Validation** | 39 seed reaches | 92% recall |
