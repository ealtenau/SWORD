#!/usr/bin/env python3
"""
Reimport Lint Fixes from CSV
============================
Reimport fixes exported from topology_reviewer.py into another database.

Usage:
    python reimport_fixes.py --db data/duckdb/sword_v17c.duckdb --csv c001_fixes_NA_20250127.csv
"""

import argparse
import pandas as pd
from src.sword_duckdb import SWORDWorkflow


def main():
    parser = argparse.ArgumentParser(description="Reimport lint fixes from CSV")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--csv", required=True, help="Path to CSV file with fixes")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without applying")
    args = parser.parse_args()

    # Load fixes
    fixes = pd.read_csv(args.csv)
    print(f"Loaded {len(fixes)} records from {args.csv}")

    # Filter to actual fixes (not skips or undos)
    actual_fixes = fixes[(fixes['action'] == 'fix') & (fixes['undone'] == False)]
    print(f"Found {len(actual_fixes)} fixes to apply (excluding skips and undone)")

    if args.dry_run:
        print("\nDry run - would apply:")
        for _, fix in actual_fixes.iterrows():
            print(f"  {fix['check_id']}: reach {fix['reach_id']} ({fix['region']}): {fix['column_changed']} = {fix['new_value']}")
        return

    # Group by region
    for region in actual_fixes['region'].unique():
        region_fixes = actual_fixes[actual_fixes['region'] == region]
        print(f"\nApplying {len(region_fixes)} fixes to region {region}...")

        workflow = SWORDWorkflow(user_id="reimport")
        sword = workflow.load(args.db, region)

        with workflow.transaction(f"Reimport {len(region_fixes)} fixes from {args.csv}"):
            for _, fix in region_fixes.iterrows():
                # Build kwargs for modify_reach
                kwargs = {}
                if fix['column_changed'] and pd.notna(fix['column_changed']):
                    # Convert value to appropriate type
                    col = fix['column_changed']
                    val = fix['new_value']
                    if col in ['lakeflag', 'type']:
                        val = int(val)
                    kwargs[col] = val

                if kwargs:
                    reason = f"{fix['check_id']}: reimported from {args.csv}"
                    if pd.notna(fix.get('notes')) and fix['notes']:
                        reason += f" - {fix['notes']}"

                    workflow.modify_reach(
                        fix['reach_id'],
                        reason=reason,
                        **kwargs
                    )
                    print(f"  Applied: reach {fix['reach_id']}: {kwargs}")

        workflow.close()
        print(f"Region {region} complete")

    print("\nDone!")


if __name__ == "__main__":
    main()
