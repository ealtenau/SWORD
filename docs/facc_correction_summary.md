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

### Key Metric: Flow-Width Ratio (FWR)

```
FWR = facc / width
```

| Channel Type | Normal FWR |
|--------------|------------|
| Main channel | ~44 |
| Side channel | ~7 |
| Secondary outlet | ~1 |

Anomalous reaches have FWR of 2,000-75,000.

### Detection Rules

**1. fwr_drop (815 detections)**

*Logic:* If a reach has bad facc but its downstream neighbor has correct facc, the FWR will drop dramatically at that boundary.

```
FWR_current / FWR_downstream > 5
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
AND facc_jump > 5   (significant facc present)
AND facc > 1000
```

*Why it works:* If a reach is topologically disconnected (path_freq invalid), it shouldn't have accumulated significant drainage. High facc here means it grabbed a value from an unrelated MERIT cell.

---

| Rule | Count |
|------|-------|
| fwr_drop | 815 |
| entry_point | 466 |
| extreme_fwr | 200 |
| jump_entry | 99 |
| Other | 145 |
| **Total** | **1,725** |

---

## Correction: How We Fixed Them

### RF Regressor Approach

Train Random Forest on **clean reaches** to predict what facc SHOULD be based on network position.

**Training:** 247,000 clean reaches
**Target:** Predict facc from topology features

### Model Performance

| Metric | Value |
|--------|-------|
| R² | 0.77 |
| Median % error | 35% |

### Top Predictive Features

| Feature | Importance | Meaning |
|---------|------------|---------|
| `hydro_dist_hw` | 56% | Distance from headwater |
| `path_freq` | 6% | Network traversal count |
| `main_side` | 5% | Main vs side channel |

**Key insight:** Distance from headwater predicts facc because drainage area accumulates downstream.

### Correction Results

| Metric | Before | After |
|--------|--------|-------|
| Median facc | 68,637 km² | 4,933 km² |
| Median FWR | ~1,000 | ~50 |

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
