# validate-region

Run full lint validation suite on a SWORD region.

## Usage
```
/validate-region NA
/validate-region all
```

## Arguments
- `region`: Region code (NA, SA, EU, AF, AS, OC) or "all" for all regions

## Instructions

When the user invokes this skill:

1. **Validate the region argument**
   - Must be one of: NA, SA, EU, AF, AS, OC, all
   - If invalid, show error and list valid options

2. **Run the lint CLI**
   ```bash
   python -m src.sword_duckdb.lint.cli \
     --db data/duckdb/sword_v17c.duckdb \
     --region {REGION} \
     --format markdown
   ```

   For "all" regions:
   ```bash
   for region in NA SA EU AF AS OC; do
     python -m src.sword_duckdb.lint.cli \
       --db data/duckdb/sword_v17c.duckdb \
       --region $region \
       --format json \
       -o output/lint_report_${region}.json
   done
   ```

3. **Summarize results**
   - Count by severity (ERROR, WARNING, INFO)
   - List ERROR checks that failed
   - Highlight any new issues vs baseline

4. **Save report**
   - JSON: `output/lint_report_{region}.json`
   - Summary: `output/lint_summary_{region}.md`

## Check Categories

| Prefix | Category | Description |
|--------|----------|-------------|
| T | Topology | Flow direction, connectivity, reciprocity |
| A | Attributes | WSE, slope, width trends |
| G | Geometry | Reach/node lengths |
| C | Classification | Lakeflag, type consistency |

## Release Criteria

**Must pass (0 violations):** T001, T005, T007, A001

**Should pass (<0.1%):** T002, T003, T004, A002, A005, G002, C001, C004
