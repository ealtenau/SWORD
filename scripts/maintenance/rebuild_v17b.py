#!/usr/bin/env python3
"""
Rebuild SWORD v17b DuckDB from source NetCDF files.

⚠️  IMPORTANT: v17b is the READ-ONLY REFERENCE DATABASE
    - NEVER modify v17b directly in production
    - All edits/fixes should go to v17c
    - Only run this script if v17b was accidentally corrupted
    - v17b exists for comparison with v17c changes

Usage:
    python rebuild_v17b.py                    # Rebuild all regions
    python rebuild_v17b.py --regions NA SA    # Rebuild specific regions
    python rebuild_v17b.py --dry-run          # Show what would be done

This creates a fresh sword_v17b.duckdb from the source NetCDF files.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, 'src')

from sword_duckdb.sword_db import SWORDDatabase
from sword_duckdb.migrations import migrate_region, build_all_geometry
from sword_duckdb.schema import create_schema

# Paths
NC_DIR = Path('data/netcdf')
DB_PATH = Path('data/duckdb/sword_v17b.duckdb')
BACKUP_DIR = Path('data/duckdb/backups')

REGIONS = ['NA', 'SA', 'EU', 'AF', 'AS', 'OC']
VERSION = 'v17b'


def backup_existing(db_path: Path) -> Path:
    """Create timestamped backup of existing database."""
    if not db_path.exists():
        return None

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = BACKUP_DIR / f"sword_v17b_backup_{timestamp}.duckdb"

    print(f"Backing up existing database to: {backup_path}")
    import shutil
    shutil.copy2(db_path, backup_path)

    return backup_path


def rebuild_database(regions: list = None, dry_run: bool = False, skip_backup: bool = False):
    """Rebuild the v17b database from NetCDF sources."""

    if regions is None:
        regions = REGIONS

    print("=" * 60)
    print("SWORD v17b Database Rebuild")
    print("=" * 60)
    print(f"NetCDF source: {NC_DIR}")
    print(f"Target DB: {DB_PATH}")
    print(f"Regions: {regions}")
    print(f"Version: {VERSION}")
    print()

    # Check source files exist
    missing = []
    for region in regions:
        nc_path = NC_DIR / f"{region.lower()}_sword_{VERSION}.nc"
        if not nc_path.exists():
            missing.append(str(nc_path))

    if missing:
        print("ERROR: Missing source NetCDF files:")
        for f in missing:
            print(f"  - {f}")
        return False

    if dry_run:
        print("\n[DRY RUN] Would rebuild database with above settings")
        return True

    # Backup existing
    if not skip_backup and DB_PATH.exists():
        backup_path = backup_existing(DB_PATH)
        if backup_path:
            print(f"Backup created: {backup_path}")

    # Delete existing database
    if DB_PATH.exists():
        print(f"Removing existing database: {DB_PATH}")
        os.remove(DB_PATH)

    # Create fresh database
    print(f"\nCreating fresh database: {DB_PATH}")
    db = SWORDDatabase(str(DB_PATH))

    # Create schema
    print("Creating tables...")
    conn = db.connect()
    create_schema(conn)

    # Migrate each region
    all_stats = {}
    for region in regions:
        nc_path = NC_DIR / f"{region.lower()}_sword_{VERSION}.nc"
        print(f"\n--- Migrating {region} from {nc_path.name} ---")

        stats = migrate_region(
            str(nc_path),
            db,
            region,
            VERSION,
            verbose=True,
            build_geometry=False  # Do geometry separately
        )
        all_stats[region] = stats

    # Build geometry
    print("\n--- Building geometry ---")
    build_all_geometry(db, regions, verbose=True)

    # Summary
    print("\n" + "=" * 60)
    print("REBUILD COMPLETE")
    print("=" * 60)

    for region, stats in all_stats.items():
        print(f"\n{region}:")
        for table, count in stats.get('row_counts', {}).items():
            print(f"  {table}: {count:,}")

    # Final counts
    print("\nFinal database totals:")
    for table in ['centerlines', 'nodes', 'reaches', 'reach_topology']:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {count:,}")
        except Exception as e:
            print(f"  {table}: error - {e}")

    db.close()
    print(f"\nDatabase saved: {DB_PATH}")
    print(f"Size: {DB_PATH.stat().st_size / 1e9:.2f} GB")

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rebuild SWORD v17b database from NetCDF')
    parser.add_argument('--regions', nargs='+', choices=REGIONS,
                        help='Specific regions to rebuild (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--skip-backup', action='store_true',
                        help='Skip backup of existing database')

    args = parser.parse_args()

    success = rebuild_database(
        regions=args.regions,
        dry_run=args.dry_run,
        skip_backup=args.skip_backup
    )

    sys.exit(0 if success else 1)
