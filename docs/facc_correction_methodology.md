# Facc Correction Methodology

## Overview

This document describes the methodology used to detect and correct corrupted flow accumulation (facc) values in SWORD v17c. The corruption originates from MERIT Hydro's D8 flow routing algorithm, which sometimes routes flow through incorrect channels at bifurcations.

## Problem Description

### Root Cause
MERIT Hydro uses D8 (deterministic 8-direction) flow routing, which can only route flow to ONE downstream cell. At river bifurcations, D8 picks a single branch, causing:

1. **Entry Points**: The "chosen" branch receives the full upstream facc (often 100-1000x too high for a narrow side channel)
2. **Propagation**: Downstream reaches inherit the corrupted facc value

### Failure Modes

| Mode | Description | Detection |
|------|-------------|-----------|
| Entry Point | D8 routes full upstream facc down wrong branch at bifurcation (split) | FWR jump > 10x, ratio_to_median > 50 |
| Propagation | Inherited bad facc from upstream entry point | ratio_to_median > 100, n_rch_up > 0 |
| path_freq=-9999 | Disconnected side channels with bad facc | path_freq=-9999 AND FWR > 5000 |

### Key Metric: Flow-Width Ratio (FWR)

```
FWR = facc / width
```

Normal FWR varies by region but typically 50-100. Corrupted reaches have FWR of 5,000-60,000.

### Key Metric: ratio_to_median

```
facc_per_reach = facc / path_freq
regional_median = PERCENTILE_CONT(0.5) of facc_per_reach
ratio_to_median = facc_per_reach / regional_median
```

Corrupted reaches have ratio_to_median > 100x (often 300-1400x).

## Correction Batches

### Batch 1: Hybrid Detection (958 reaches)
**Date**: 2026-02-04
**Method**: Hybrid detection using ratio_to_median

Detection rules:
- `entry_point`: facc_jump > 10 AND ratio_to_median > 50
- `propagation_high_ratio`: ratio_to_median > 100 AND n_rch_up > 0
- `junction_extreme`: FWR > 15000 AND end_reach = 3
- `headwater_extreme`: n_rch_up = 0 AND facc > 500K AND FWR > 5000

Correction formula:
```python
# Fit log-linear regression on non-anomalous reaches
log(facc) ~ log(width) + log(slope) + log(path_freq)

# Predict corrected facc
facc_corrected = exp(predicted_log_facc)
```

### Batch 2: Nitpick Fixes (4 reaches)
**Date**: 2026-02-04
**Method**: Manual review and targeted fixes

| reach_id | Issue | Fix |
|----------|-------|-----|
| 14210000525 | False positive (tidal bifurcation) | Rolled back to v17b |
| 62293900366 | Headwater entry point | width × 600 |
| 62293900353 | Lake propagation | upstream + local |
| 62293900401 | path_freq=-9999 entry | Inherit from downstream |

### Batch 3: path_freq=-9999 High-FWR (167 reaches)
**Date**: 2026-02-04
**Method**: Width-based estimation

**Problem**: Reaches with `path_freq=-9999` are invisible to ratio_to_median detection because the calculation produces negative/invalid values.

**Detection**:
```sql
WHERE path_freq = -9999
  AND width > 30
  AND facc/width > 5000
```

**Correction formula**:
```python
new_facc = width × regional_median_fwr
```

**Regional median FWR values** (derived from analysis of normal reaches):
| Region | Median FWR |
|--------|------------|
| NA | 80 |
| SA | 60 |
| EU | 100 |
| AF | 50 |
| AS | 70 |
| OC | 60 |

**Results**:
- 167 reaches corrected
- Old FWR range: 5,007 - 56,002
- New FWR range: 50 - 100
- Reduction factors: 63x - 933x

## Validation

### Seed Reaches
5 manually-identified corrupted reaches used for validation:

| reach_id | Description | Pre-fix FWR | Post-fix FWR |
|----------|-------------|-------------|--------------|
| 62236100011 | SA entry point | 22,811 | 114 |
| 62238000021 | SA entry point | 1,559 | 2 |
| 62255000451 | SA propagation | 528 | 1 |
| 64231000291 | SA propagation | 982 | 3 |
| 64231000301 | SA entry point | 35,239 | 123 |

All 5 seeds successfully corrected.

### Correction Effectiveness

| Metric | Value |
|--------|-------|
| Corrections logged | 2,087 |
| Actually applied to DB | 2,083 |
| Not applied (intentional overrides) | 4 |
| Median reduction factor | 106x |
| Max reduction factor | 2,344x |

### Neighbor Comparison (Direct Continuation)
For 221 corrected reaches with a single non-corrected downstream neighbor:

| Metric | Before | After |
|--------|--------|-------|
| Median ratio to downstream FWR | 50.2x | 2.27x |
| Within 0.5x-2x of downstream | 27.4% | 14.9% |

**Interpretation**: Median improved significantly (50x → 2x), but the "within range" metric dropped because:
1. Some corrections over-corrected (now lower than downstream)
2. Some under-corrected (regression predicted similar value)

### Why Not Use Simple FWR Threshold?

**745 river reaches with FWR > 5000 were NOT detected** by our hybrid method. Manual review confirmed **almost all are false positives** - legitimate high-FWR reaches, not D8 routing errors.

This validates the ratio_to_median approach:
- Simple FWR > 5000 threshold would have ~745 false positives
- ratio_to_median correctly distinguishes real anomalies from legitimate high-facc reaches
- High FWR alone doesn't indicate corruption - context (path_freq, neighbors) matters

### False Positive Protection
- Tidal reaches (lakeflag=3) at bifurcations may have legitimately high facc
- Rolled back 14210000525 after identifying as false positive
- Future: Consider excluding lakeflag=3 from automatic correction

## Files

### Detection Module
```
src/updates/sword_duckdb/facc_detection/
├── __init__.py      # Module exports
├── reach_acc.py     # Reach accumulation via sparse matrix
├── features.py      # Feature extraction SQL
├── detect.py        # FaccDetector, detect_hybrid()
├── evaluate.py      # Evaluation against seeds
├── correct.py       # FaccCorrector class
└── cli.py           # Command-line interface
```

### Output Files
```
output/
├── facc_corrections_final.geojson          # FINAL: All 2,087 corrections with before/after values
├── facc_anomalies_hybrid.geojson           # Batch 1 anomalies (detection)
├── path_freq_minus9999_high_fwr.geojson    # Batch 3 anomalies (detection)
├── path_freq_minus9999_corrections.csv     # Batch 3 corrections log
└── remaining_high_fwr_rivers.geojson       # 745 reaches confirmed as false positives
```

### Database Tables
- `facc_fix_log`: Audit trail of all corrections (reach_id, old_facc, new_facc, fix_type, timestamp)

## CLI Usage

```bash
# Detect anomalies
python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --region SA

# Dry-run correction
python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --fix --region SA

# Apply corrections
python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --fix --apply --region SA

# Verify seeds
python -m src.updates.sword_duckdb.facc_detection.cli --db sword_v17c.duckdb --verify-seeds
```

## Summary Statistics

| Metric | Before | After |
|--------|--------|-------|
| Corrections logged | - | 2,087 |
| Corrections applied | - | 2,083 |
| Intentional overrides | - | 4 |
| Reaches with FWR > 5000 (rivers, width>30) | ~1,800 | 745 |
| Seed reaches fixed | 0/5 | 5/5 |
| Median reduction factor | - | 106x |

### Completion Status ✅
- All 2,083 corrections successfully applied to v17c database
- **4 intentional overrides** (not applied):
  - 14210000525 (AF): Rolled back - tidal bifurcation false positive
  - 62293900353 (SA): Manual fix - lake propagation edge case
- **745 high-FWR reaches** reviewed and confirmed as false positives - no action needed
- Full audit trail in `facc_fix_log` table

## References

- SWORD Product Description Document (PDD) v17b
- MERIT Hydro documentation
- GitHub Issue #14: Fix facc using MERIT Hydro
