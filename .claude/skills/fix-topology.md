# fix-topology

Interactive topology correction workflow for a specific reach.

## Usage
```
/fix-topology 23456789011
/fix-topology 23456789011 --show-context
```

## Arguments
- `reach_id`: 11-digit SWORD reach ID (CBBBBBRRRRT format)
- `--show-context`: Show extended neighborhood context

## Instructions

When the user invokes this skill:

1. **Validate reach_id format**
   - Must be 11 digits
   - First digit (continent): 1-9
   - Last digit (type): 1-6
   - If invalid, show format error

2. **Load reach context from v17c database**
   ```python
   from sword_duckdb import SWORDWorkflow

   workflow = SWORDWorkflow(user_id="fix-topology-skill")
   sword = workflow.load('data/duckdb/sword_v17c.duckdb', region)
   reach = sword.get_reach(reach_id)
   ```

3. **Display reach information**
   - Basic: reach_id, region, lakeflag, type
   - Topology: n_rch_up, n_rch_down, rch_id_up, rch_id_dn
   - Attributes: dist_out, facc, wse, width, slope
   - v17c: is_mainstem_edge, hydro_dist_out, best_headwater, best_outlet

4. **Run targeted lint checks**
   - T001 (dist_out) for this reach
   - T003 (facc) for this reach
   - A001 (wse) for this reach
   - Show which checks pass/fail

5. **Suggest fixes based on failures**

   | Check Failed | Suggested Fix |
   |--------------|---------------|
   | T001 (dist_out) | Recalculate dist_out or check flow direction |
   | T003 (facc) | Check MERIT Hydro join, manual facc correction |
   | T005 (neighbor count) | Verify topology edges |
   | T007 (reciprocity) | Add missing reverse edge |

6. **Apply fix with provenance (if user confirms)**
   ```python
   workflow.modify_reach(reach_id,
       facc=corrected_value,
       reason="Manual correction via /fix-topology skill")
   workflow.close()
   ```

7. **Validate after fix**
   - Re-run the failed checks
   - Confirm fix resolved the issue

## Important

- All modifications are logged with provenance
- User must confirm before applying changes
- Use deploy/reviewer/app.py GUI for bulk/visual fixes
