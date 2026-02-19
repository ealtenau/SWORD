# SWORD Validation Plan

## Current State (v17b)

| Check | Issues | % | Priority |
|-------|--------|---|----------|
| path_freq Consistency | 1,536 | 0.64% | ✅ Fixed in v17c |
| Width Monotonicity | 1,597 | 2.15% | Medium |
| WSE Gradient | 1,506 | 0.74% | High (data quality) |
| Lake Sandwiches | 3,167 | 1.55% | Medium |
| dist_out Monotonicity | 0 | 0.00% | ✅ OK |
| Orphaned Reaches | 0 | 0.00% | ✅ OK |

## Validation Module

**Location:** `src/updates/sword_duckdb/validation.py`

**Usage:**
```bash
# Run all checks
python -m updates.sword_duckdb.validation data/duckdb/sword_v17b.duckdb

# Run specific check on region
python -m updates.sword_duckdb.validation data/duckdb/sword_v17b.duckdb -r NA -c wse_gradient

# Export issues to CSV
python -m updates.sword_duckdb.validation data/duckdb/sword_v17b.duckdb -o issues.csv
```

## Future: CI/CD Integration

### Phase 1: Baseline Establishment
- [x] Create validation module
- [x] Run on v17b to establish baselines
- [x] Compare v17b vs v17c
- [ ] Document expected issue counts per region

### Phase 2: Pytest Wrapper
```
tests/
  test_topology_validation.py
  conftest.py
```

Baseline thresholds (v17b):
```python
BASELINES = {
    'wse_gradient': 1506,
    'width_monotonicity': 1597,
    'path_freq_consistency': 1536,
    'lake_sandwiches': 3167,
    'dist_out_monotonicity': 0,
    'orphaned_reaches': 0,
}
```

### Phase 3: GitHub Actions
```yaml
# .github/workflows/validate.yml
on: [pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/test_topology_validation.py
```

### Phase 4: Continuous Improvement
- Track issue counts over time
- Ratchet down baselines as fixes are merged
- Alert on regressions

## Checks to Add

1. **SWOT Observation Validation** - compare observed vs predicted WSE
2. **Cross-continent Consistency** - same river shouldn't have gaps at borders  
3. **Temporal Stability** - changes between versions should be explainable
4. **External Benchmark** - compare mainstem to regional datasets (NHD, EU-Hydro, etc.)

## Issue Resolution Workflow

```
1. Run validation
2. Export issues to CSV
3. Prioritize by:
   - Impact (affects routing? classification?)
   - Ease of fix (data error vs topology error)
   - Coverage (one river vs global pattern)
4. Create fix
5. Re-run validation to confirm improvement
6. Update baseline if improved
```
