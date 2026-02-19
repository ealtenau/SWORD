# compare-v17b

Compare v17c region against v17b baseline to identify changes.

## Usage
```
/compare-v17b NA
/compare-v17b NA --attribute dist_out
/compare-v17b all --summary-only
```

## Arguments
- `region`: Region code (NA, SA, EU, AF, AS, OC) or "all"
- `--attribute`: Focus on specific attribute comparison
- `--summary-only`: Only show high-level stats, not individual changes

## Instructions

When the user invokes this skill:

1. **Load both databases**
   ```python
   import duckdb

   v17b = duckdb.connect('data/duckdb/sword_v17b.duckdb', read_only=True)
   v17c = duckdb.connect('data/duckdb/sword_v17c.duckdb', read_only=True)
   ```

2. **Compare schema**
   - New columns in v17c (expected: hydro_dist_out, is_mainstem_edge, etc.)
   - Removed columns (should be none)
   - Type changes (should be none)

3. **Compare row counts**
   | Table | v17b | v17c | Delta |
   |-------|------|------|-------|
   | reaches | X | Y | +/- |
   | nodes | X | Y | +/- |
   | centerlines | X | Y | +/- |

4. **Compare attribute statistics** (for each numeric column)
   - Mean, median, std
   - Min, max
   - % NULL
   - Flag significant differences (>5% change in mean)

5. **Identify changed reaches**
   ```sql
   SELECT reach_id,
          v17b.dist_out as dist_out_v17b,
          v17c.dist_out as dist_out_v17c,
          v17c.dist_out - v17b.dist_out as delta
   FROM v17b.reaches
   JOIN v17c.reaches USING (reach_id)
   WHERE ABS(delta) > threshold
   ```

6. **Categorize changes**
   - Topology fixes (dist_out, path_freq changes)
   - Attribute corrections (facc, wse, width)
   - New attributes (v17c columns)
   - Reclassifications (lakeflag, type changes)

7. **Generate report**
   ```markdown
   # v17b vs v17c Comparison: {REGION}

   ## Summary
   - Reaches: X (v17b) → Y (v17c), +/- Z
   - New columns: [list]
   - Significant changes: N reaches

   ## Topology Changes
   | reach_id | Change Type | Before | After |
   ...

   ## Attribute Changes
   ...
   ```

## Output Files

- `output/compare_v17b_v17c_{region}.md` - Human-readable report
- `output/compare_v17b_v17c_{region}.json` - Machine-readable changes

## Important

- v17b database is READ-ONLY - never modify it
- Large diffs (>10k changes) suggest possible issues
- Changes should align with known fixes (topology_reviewer logs)
