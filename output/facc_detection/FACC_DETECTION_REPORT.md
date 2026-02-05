# FACC Anomaly Detection and Correction Report

**Date:** 2026-02-05
**Database:** sword_v17c.duckdb
**Total Anomalies Detected:** 1,725 reaches

---

## 1. Problem Statement

Flow accumulation (facc) values in SWORD are derived from MERIT Hydro using D8 flow direction. At river bifurcations, D8 routing picks ONE downstream branch, causing the other branch to inherit incorrect (often inflated) facc values.

**Two failure modes:**
1. **Entry Points:** Bad facc ENTERS a tributary with massive jump (200x-3000x)
2. **Propagation:** Inherited bad facc continues downstream with ratio ~1.0

---

## 2. Variable Definitions

### 2.1 Original Variables (from SWORD v17b)

| Variable | Description | Units |
|----------|-------------|-------|
| `facc` | Flow accumulation from MERIT Hydro D8 routing | km² |
| `width` | River width from GRWL/SWOT | meters |
| `dist_out` | Distance to outlet along network | meters |
| `path_freq` | Traversal count - increases toward outlets | count |
| `stream_order` | Log scale of path_freq: `round(log(path_freq)) + 1` | integer |
| `main_side` | Channel classification: 0=main, 1=side, 2=secondary outlet | integer |
| `end_reach` | Position: 0=middle, 1=headwater, 2=outlet, 3=junction | integer |
| `n_rch_up` | Number of upstream neighbors | count |
| `n_rch_down` | Number of downstream neighbors | count |

### 2.2 v17c Variables (computed by us)

| Variable | Description | Units |
|----------|-------------|-------|
| `hydro_dist_hw` | Dijkstra distance FROM nearest headwater | meters |
| `hydro_dist_out` | Dijkstra distance TO nearest outlet | meters |
| `pathlen_hw` | Cumulative path length from headwater | meters |
| `pathlen_out` | Cumulative path length to outlet | meters |
| `is_mainstem_edge` | TRUE if on mainstem path | boolean |

---

## 3. Derived Ratios and Metrics

### 3.1 Flow-Width Ratio (FWR)

**Definition:**
```
FWR = facc / width
```

**Interpretation:**
- Measures drainage area per unit channel width
- Normal range: 1-100 for most rivers
- Values >1000 indicate potential anomaly
- Values >10000 almost certainly anomalous

**Typical values by channel type:**
| main_side | Median FWR | Description |
|-----------|------------|-------------|
| 0 (main) | 44 | Main channel |
| 1 (side) | 7 | Side channel |
| 2 (secondary) | 1 | Secondary outlet |

### 3.2 FWR Drop Ratio

**Definition:**
```
fwr_drop_ratio = FWR_current / FWR_downstream
```

**Calculation:**
```sql
fwr_drop_ratio = (r.facc / r.width) / (dn.facc / dn.width)
```
where `dn` is the downstream neighbor.

**Interpretation:**
- Normal: ~1.0 (FWR stable or slightly increasing downstream)
- Anomaly signal: >5 (FWR drops dramatically at anomaly boundary)
- This is the **#1 detection rule** (catches 815 anomalies)

### 3.3 FACC Jump Ratio

**Definition:**
```
facc_jump_ratio = facc_current / facc_upstream
```

**Calculation:**
```sql
facc_jump_ratio = r.facc / MAX(upstream.facc)
```

**Interpretation:**
- Normal: 1.0-2.0 (facc increases gradually)
- Entry point signal: >10 (sudden massive increase)
- Combined with `ratio_to_median > 40` → entry_point rule (catches 466)

### 3.4 Ratio to Regional Median

**Definition:**
```
ratio_to_median = FWR_current / FWR_regional_median
```

**Calculation:**
```sql
-- Regional median FWR per stream_order
WITH regional_stats AS (
    SELECT region, stream_order,
           PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY facc/width) as median_fwr
    FROM reaches WHERE width > 0
    GROUP BY region, stream_order
)
ratio_to_median = (r.facc / r.width) / rs.median_fwr
```

**Interpretation:**
- Normal: 0.5-2.0
- Anomaly signal: >40 (40x higher than regional peers)

### 3.5 Width Ratio to Downstream

**Definition:**
```
width_ratio_to_dn = width_current / width_downstream
```

**Interpretation:**
- Normal: ~1.0 (width gradually increases downstream)
- Anomaly context: Side channels often narrower than main stem

### 3.6 Upstream FACC Sum

**Definition:**
```
upstream_facc_sum = SUM(facc of all upstream neighbors)
```

**Interpretation:**
- What facc SHOULD be (approximately)
- If current facc >> upstream_facc_sum, likely anomalous

---

## 4. Detection Rules (detect_hybrid)

### Rule Performance Summary

| Rule | Detections | Description |
|------|------------|-------------|
| fwr_drop | 815 | FWR drops >5x at downstream boundary |
| entry_point | 466 | facc_jump >10 AND ratio_to_median >40 |
| extreme_fwr | 200 | FWR >15,000 |
| jump_entry | 99 | Invalid path_freq + high facc jump |
| fwr_sum_check | 72 | FWR > 5x sum of upstream FWRs |
| facc_sum_inflation | 45 | facc > 10x upstream sum |
| propagation | 28 | Downstream of known anomaly |

### Rule Definitions

**1. fwr_drop (most effective)**
```sql
fwr_drop_ratio > 5.0
AND facc_width_ratio > 100
```

**2. entry_point**
```sql
facc_jump_ratio > 10
AND ratio_to_median > 40
```

**3. extreme_fwr**
```sql
facc_width_ratio > 15000
```

**4. jump_entry**
```sql
path_freq <= 0
AND facc_jump_ratio > 5
AND facc > 1000
```

---

## 5. RF Regressor for Correction

### 5.1 Approach

Train Random Forest regressor on **clean reaches** (non-anomalous) to predict what facc SHOULD be based on network position.

### 5.2 Model Performance

| Metric | Value |
|--------|-------|
| R² (log space) | 0.787 |
| R² (original) | 0.773 |
| MAE | 16,203 km² |
| Median AE | 1,237 km² |
| Median % error | 34.8% |
| P90 % error | 141% |

### 5.3 Top Features for Prediction

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | hydro_dist_hw | 56.4% | Distance from headwater (Dijkstra) |
| 2 | path_freq | 5.7% | Network traversal count |
| 3 | main_side | 5.1% | Channel type (main/side/secondary) |
| 4 | log_path_freq | 4.5% | Log of path_freq |
| 5 | main_path_id | 2.8% | Mainstem identifier |
| 6 | lakeflag | 2.2% | Lake/river classification |
| 7 | wse | 1.4% | Water surface elevation |
| 8 | hydro_dist_out | 1.4% | Distance to outlet |

**Key insight:** Network position (`hydro_dist_hw`) is the dominant predictor - facc accumulates as you move downstream from headwaters.

### 5.4 Correction Results

| Metric | Before | After |
|--------|--------|-------|
| Median facc (anomalies) | 68,637 km² | 4,933 km² |
| Median FWR (anomalies) | ~1,000 | ~50 |

---

## 6. Validation

### 6.1 Seed Reaches (39 known anomalies)

| Status | Count | Rate |
|--------|-------|------|
| Detected by rules | 36 | 92.3% |
| Missed (propagation) | 3 | 7.7% |

**Missed seeds** (uncatchable - propagation patterns too subtle):
- 44581100675
- 44581100665
- 14631000181

### 6.2 Known Issues with RF Regressor

**False Positives (wrongly corrected):**
- 77250000153 - mainstem reach
- 74300400575, 74300400565 - incorrectly flipped

**False Negatives (should be added as seeds):**
- 62293100143
- 62293100156
- 62253000321
- 62235900101

---

## 7. Files Reference

```
output/facc_detection/
├── all_anomalies.geojson         # 1,725 detected anomalies
├── rf_features.csv               # 77 features, 248K reaches
├── rf_feature_importance.csv     # Classifier feature rankings
├── rf_regressor.joblib           # Trained regressor model (110 MB)
├── rf_regressor_importance.csv   # Regressor feature rankings
├── rf_regressor_predictions.csv  # Predictions for anomalies
└── FACC_DETECTION_REPORT.md      # This report
```

---

## 8. Database Updates

### 8.1 DuckDB (sword_v17c.duckdb)
- **Batch ID:** 2
- **Table:** `facc_corrections`
- **Reaches updated:** 1,725

### 8.2 PostgreSQL (sword_v17c)
- **Batch ID:** 1
- **Table:** `facc_corrections`
- **Reaches updated:** 1,725

### Rollback Command
```sql
-- DuckDB
UPDATE reaches SET facc = c.old_facc
FROM facc_corrections c
WHERE reaches.reach_id = c.reach_id AND c.batch_id = 2;

-- PostgreSQL
UPDATE reaches SET facc = c.old_facc
FROM facc_corrections c
WHERE reaches.reach_id = c.reach_id AND c.batch_id = 1;
```

---

## 9. Recommendations

1. **Add new seeds:** 62293100143, 62293100156, 62253000321, 62235900101
2. **Review false positives:** 77250000153, 74300400575, 74300400565
3. **Manual review:** Large corrections (>100x) warrant visual inspection
4. **Consider upstream constraint:** Use `min(predicted_facc, upstream_facc_sum * 1.1)` for conservative corrections

---

*Generated: 2026-02-05*
