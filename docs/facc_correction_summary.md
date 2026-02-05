# FACC Error Detection and Correction

## The Problem

SWORD uses MERIT Hydro flow accumulation (facc), which is derived from D8 routing. At river bifurcations, D8 picks ONE downstream branch - the other branch inherits incorrect (inflated) facc values.

**Result:** ~1,725 reaches have facc values 10-1000x too high.

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

| Rule | Count | Logic |
|------|-------|-------|
| **fwr_drop** | 815 | FWR drops >5x at downstream boundary |
| **entry_point** | 466 | facc jumps >10x AND FWR >40x regional median |
| **extreme_fwr** | 200 | FWR > 15,000 |
| **jump_entry** | 99 | Disconnected reach (path_freq invalid) + high facc jump |
| Other rules | 145 | Various edge cases |
| **Total** | **1,725** | |

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
