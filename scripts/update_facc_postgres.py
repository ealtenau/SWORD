#!/usr/bin/env python3
"""
Update facc column in PostgreSQL from DuckDB v17c corrections.

Usage:
    python scripts/update_facc_postgres.py \
        --duckdb data/duckdb/sword_v17c.duckdb \
        --pg "postgresql://user:pass@host:5432/sword_db" \
        --target-table sword_reaches_v17c

    # Dry run
    python scripts/update_facc_postgres.py \
        --duckdb data/duckdb/sword_v17c.duckdb \
        --pg "..." --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

import duckdb

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    print("Error: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)


def update_facc(
    duckdb_path: str,
    pg_conn_str: str,
    target_table: str = "sword_reaches_v17c",
    source_table: str = "sword_reaches_v17b",
    dry_run: bool = False,
    batch_size: int = 10000,
):
    """Update facc column in PostgreSQL from DuckDB."""

    # Connect to DuckDB
    print(f"Connecting to DuckDB: {duckdb_path}")
    duck_conn = duckdb.connect(duckdb_path, read_only=True)

    # Get all reach_id and facc from DuckDB
    print("Extracting facc values from DuckDB...")
    facc_df = duck_conn.execute("""
        SELECT reach_id, facc
        FROM reaches
        ORDER BY reach_id
    """).fetchdf()

    print(f"  Found {len(facc_df):,} reaches")
    duck_conn.close()

    if dry_run:
        print("\n[DRY RUN] Would update:")
        print(f"  Target table: {target_table}")
        print(f"  Rows to update: {len(facc_df):,}")
        print(f"\nSample values:")
        print(facc_df.head(10).to_string(index=False))
        return

    # Connect to PostgreSQL
    print(f"\nConnecting to PostgreSQL...")
    pg_conn = psycopg2.connect(pg_conn_str)
    pg_cursor = pg_conn.cursor()

    # Check if target table exists
    pg_cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = %s
        )
    """, [target_table])

    table_exists = pg_cursor.fetchone()[0]

    if not table_exists:
        print(f"  Target table '{target_table}' doesn't exist, creating from {source_table}...")
        pg_cursor.execute(f"""
            CREATE TABLE {target_table} AS
            SELECT * FROM {source_table}
        """)
        pg_conn.commit()
        print(f"  Created {target_table}")

        # Add primary key
        pg_cursor.execute(f"""
            ALTER TABLE {target_table} ADD PRIMARY KEY (reach_id)
        """)
        pg_conn.commit()
        print(f"  Added primary key on reach_id")

    # Create temp table for updates
    print("Creating temp table for facc updates...")
    pg_cursor.execute("""
        CREATE TEMP TABLE _facc_updates (
            reach_id BIGINT PRIMARY KEY,
            new_facc DOUBLE PRECISION
        )
    """)

    # Batch insert into temp table
    print(f"Inserting {len(facc_df):,} facc values (batch size: {batch_size})...")
    data = list(zip(facc_df['reach_id'].astype(int), facc_df['facc'].astype(float)))

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        execute_values(
            pg_cursor,
            "INSERT INTO _facc_updates (reach_id, new_facc) VALUES %s",
            batch,
            page_size=batch_size
        )
        print(f"  Inserted {min(i + batch_size, len(data)):,} / {len(data):,}", end='\r')

    print()

    # Update target table
    print(f"Updating {target_table}.facc from temp table...")
    pg_cursor.execute(f"""
        UPDATE {target_table} t
        SET facc = u.new_facc
        FROM _facc_updates u
        WHERE t.reach_id = u.reach_id
    """)

    updated = pg_cursor.rowcount
    print(f"  Updated {updated:,} rows")

    # Commit
    pg_conn.commit()
    print("Committed changes")

    # Verify
    pg_cursor.execute(f"""
        SELECT COUNT(*) as n,
               AVG(facc) as avg_facc,
               MIN(facc) as min_facc,
               MAX(facc) as max_facc
        FROM {target_table}
    """)
    stats = pg_cursor.fetchone()
    print(f"\nVerification - {target_table}:")
    print(f"  Rows: {stats[0]:,}")
    print(f"  Avg facc: {stats[1]:,.0f} km²")
    print(f"  Min facc: {stats[2]:,.0f} km²")
    print(f"  Max facc: {stats[3]:,.0f} km²")

    pg_cursor.close()
    pg_conn.close()
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Update facc column in PostgreSQL from DuckDB v17c"
    )
    parser.add_argument('--duckdb', required=True,
                        help='Path to DuckDB database')
    parser.add_argument('--pg', required=True,
                        help='PostgreSQL connection string')
    parser.add_argument('--target-table', default='sword_reaches_v17c',
                        help='Target table to update (default: sword_reaches_v17c)')
    parser.add_argument('--source-table', default='sword_reaches_v17b',
                        help='Source table to copy from if target missing (default: sword_reaches_v17b)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--batch-size', type=int, default=10000,
                        help='Batch size for inserts (default: 10000)')

    args = parser.parse_args()

    # Support env var for pg connection
    pg_conn = args.pg
    if pg_conn == "$SWORD_POSTGRES_URL" or pg_conn.startswith("$"):
        env_var = pg_conn.lstrip("$")
        pg_conn = os.environ.get(env_var)
        if not pg_conn:
            print(f"Error: Environment variable {env_var} not set")
            sys.exit(1)

    update_facc(
        duckdb_path=args.duckdb,
        pg_conn_str=pg_conn,
        target_table=args.target_table,
        source_table=args.source_table,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
