#!/usr/bin/env python3
"""
Generate SWORD v17c by running topology recalculations on v17b.

This script:
1. Loads each region from v17c (copy of v17b)
2. Runs topology recalculations
3. Saves results back to database
"""

import sys
sys.path.insert(0, '/Users/jakegearon/projects/SWORD/src')

from pathlib import Path
from updates.sword_duckdb import SWORDWorkflow

DB_PATH = Path('/Users/jakegearon/projects/SWORD/data/duckdb/sword_v17c.duckdb')
REGIONS = ['NA', 'SA', 'EU', 'AF', 'AS', 'OC']


def recalculate_region_topology(region: str, workflow: SWORDWorkflow = None):
    """Run all topology recalculations for a region."""
    print(f"\n{'='*60}")
    print(f"Processing region: {region}")
    print(f"{'='*60}")

    # Initialize workflow if not provided
    if workflow is None:
        workflow = SWORDWorkflow(user_id="v17c_generation")

    # Load region
    print(f"Loading {region}...")
    sword = workflow.load(str(DB_PATH), region)
    print(f"  Loaded {len(sword.reaches)} reaches, {len(sword.nodes)} nodes")

    results = {}

    # 1. Skip dist_out recalculation - preserve v17b values
    # v17c uses separate hydro_dist_out/hydro_dist_hw computed by assign_attribute.py
    print("\n1. Skipping dist_out recalculation (preserving v17b values)")
    print("   v17c uses hydro_dist_out/hydro_dist_hw from assign_attribute.py")
    results['dist_out'] = {'skipped': True, 'reason': 'v17c uses hydro_dist_out/hydro_dist_hw'}

    # 2. Recalculate stream_order
    print("\n2. Recalculating stream_order...")
    try:
        so_result = workflow.recalculate_stream_order(
            update_nodes=True,
            update_reaches=True,
            reason="v17c topology recalculation"
        )
        results['stream_order'] = so_result
        print(f"   stream_order: {so_result.get('reaches_updated', 0)} reaches updated")
    except Exception as e:
        print(f"   ERROR: {e}")
        results['stream_order'] = {'error': str(e)}

    # 3. Recalculate path_segs
    print("\n3. Recalculating path_segs...")
    try:
        ps_result = workflow.recalculate_path_segs(
            reason="v17c topology recalculation"
        )
        results['path_segs'] = ps_result
        print(f"   path_segs: {ps_result.get('reaches_updated', 0)} reaches updated")
    except Exception as e:
        print(f"   ERROR: {e}")
        results['path_segs'] = {'error': str(e)}

    # 4. Recalculate sinuosity
    print("\n4. Recalculating sinuosity...")
    try:
        sin_result = workflow.recalculate_sinuosity(
            reason="v17c topology recalculation"
        )
        results['sinuosity'] = sin_result
        print(f"   sinuosity: updated")
    except Exception as e:
        print(f"   ERROR: {e}")
        results['sinuosity'] = {'error': str(e)}

    # Close the connection
    workflow.close()

    print(f"\n{region} complete!")
    return results


def main():
    """Run topology recalculation for all regions or specified region."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', '-r', help='Specific region to process')
    parser.add_argument('--all', '-a', action='store_true', help='Process all regions')
    args = parser.parse_args()

    if args.region:
        regions = [args.region.upper()]
    elif args.all:
        regions = REGIONS
    else:
        # Default to NA for testing
        regions = ['NA']

    print("SWORD v17c Topology Recalculation")
    print(f"Database: {DB_PATH}")
    print(f"Regions: {regions}")

    all_results = {}
    for region in regions:
        all_results[region] = recalculate_region_topology(region)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for region, results in all_results.items():
        print(f"\n{region}:")
        for attr, result in results.items():
            if 'error' in result:
                print(f"  {attr}: ERROR - {result['error']}")
            else:
                print(f"  {attr}: OK")


if __name__ == '__main__':
    main()
