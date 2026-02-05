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

*Logic:* If a reach has bad facc but its downstream neighbor has correct facc, the FWR will drop dramatically at that boundary.

```
FWR_current / FWR_downstream > 5
AND FWR > 500
```

*Why it works:* Bad facc doesn't propagate everywhere - when flow rejoins the correct MERIT path downstream, FWR returns to normal. This rule catches the downstream edge of corrupted segments.

---

**2. entry_point (466 detections)**

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

*Logic:* Some FWR values are simply impossible - no river has 15,000+ km² drainage per meter of width.

```
FWR > 15,000
```

*Why it works:* Even the Amazon's FWR is ~2,000. Values above 15,000 indicate a narrow channel (50-200m) with drainage area of a continental river (millions of km²). Physically impossible.

---

**4. jump_entry (99 detections)**

*Logic:* Disconnected side channels (path_freq = -9999) that somehow have high facc.

```
path_freq <= 0   (disconnected from main network)
AND facc_jump > 20
AND FWR > 500
```

*Why it works:* If a reach is topologically disconnected (path_freq invalid), it shouldn't have accumulated significant drainage. High facc here means it grabbed a value from an unrelated MERIT cell.

---

**5. upstream_fwr_spike (40 detections)**

*Logic:* Catches propagated bad facc by looking upstream.

```
upstream_FWR / this_FWR > 10
AND facc > 100,000
```

*Why it works:* When bad facc enters upstream, it can propagate down. If an upstream reach has 10x higher FWR than its downstream neighbor, the bad facc started upstream.

---

**6. impossible_headwater (30 detections)**

*Logic:* Near-headwater reaches (path_freq ≤ 2) shouldn't have continental-scale drainage.

```
path_freq <= 2
AND facc > 1,000,000
AND (fwr_drop > 2 OR FWR > 5000)
```

*Why it works:* path_freq=1-2 means near the headwater. These reaches couldn't have accumulated millions of km² of drainage. The FWR check excludes delta distributaries.

---

**7. invalid_side_channel (27 detections)**

*Logic:* Side channels with invalid topology and high facc.

```
path_freq = -9999   (disconnected)
AND main_side = 1   (marked as side channel)
AND facc > 200,000
AND fwr_drop > 3
```

*Why it works:* A disconnected side channel shouldn't have significant facc. These grabbed MERIT cell values from the adjacent mainstem.

---

**8. high_ratio (17 detections)**

*Logic:* Extreme ratio_to_median without other triggers.

```
ratio_to_median > 500
AND (fwr_drop > 2 OR no downstream neighbor)
```

*Why it works:* When facc is 500x above what's expected for the network position, it's wrong. The fwr_drop check excludes legitimate multi-channel rivers (Ob, etc.).

---

**9. side_channel_misroute (15 detections)**

*Logic:* Side channels with mainstem facc values.

```
main_side = 1   (side channel)
AND fwr_drop > 20
AND facc > 100,000
```

*Why it works:* Side channels shouldn't have FWR that drops 20x downstream. This indicates the side channel got routed through a mainstem MERIT cell.

---

**10. facc_sum_inflation (12 detections)**

*Logic:* At confluences, facc should equal the sum of upstream tributaries. Large inflation indicates D8 error.

```
n_rch_up >= 2   (confluence)
AND upstream_facc_sum > 50,000
AND facc > 3 × SUM(upstream_facc)
```

*Why it works:* Mass balance. If two tributaries with facc = 100K and 50K meet, the downstream reach should have ~150K. If it has 500K+, the reach grabbed facc from a different MERIT cell.

---

**11. headwater_extreme (4 detections)**

*Logic:* True headwaters (no upstream) with impossible facc.

```
n_rch_up = 0   (true headwater)
AND facc > 500,000
AND FWR > 5,000
```

*Why it works:* A reach with no upstream neighbors is a true entry point - it can't have accumulated 500K+ km² of drainage. This is the most unambiguous signal.

---

### Detection Summary

| Rule | Count | % |
|------|-------|---|
| fwr_drop | 815 | 47.2% |
| entry_point | 466 | 27.0% |
| extreme_fwr | 200 | 11.6% |
| jump_entry | 99 | 5.7% |
| upstream_fwr_spike | 40 | 2.3% |
| impossible_headwater | 30 | 1.7% |
| invalid_side_channel | 27 | 1.6% |
| high_ratio | 17 | 1.0% |
| side_channel_misroute | 15 | 0.9% |
| facc_sum_inflation | 12 | 0.7% |
| headwater_extreme | 4 | 0.2% |
| **Total** | **1,725** | 100% |

---

## Correction: How We Fixed Them

### RF Regressor Approach

Train Random Forest on **clean reaches** to predict what facc SHOULD be based on network position.

**Training:** 247,000 clean reaches (all non-anomalous)
**Target:** Predict facc from 59 topology/position features
**Excludes:** Any feature derived FROM facc (to avoid circularity)

![RF Correction Logic](../output/facc_detection/figures/fig4_correction_logic.png)

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

![Model Validation](../output/facc_detection/figures/fig5_model_validation.png)

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
- **5 missed** by rule-based detection (see below)

**Missed seeds** (propagation patterns too subtle for any method):
- 22513000171
- 44581100665
- 44581100675
- 34211700241
- 34211101775

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
