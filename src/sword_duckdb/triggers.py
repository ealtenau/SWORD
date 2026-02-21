# -*- coding: utf-8 -*-
"""
SWORD PostgreSQL Triggers
=========================

PostgreSQL trigger functions for automatic recalculation when SWORD data
is edited in QGIS. Uses plpython3u for Python-based recalculation logic.

Workflow:
1. User exports SWORD data to PostgreSQL using export_to_postgres()
2. User edits data in QGIS
3. Triggers fire on UPDATE/INSERT/DELETE
4. Triggers mark affected entities as "dirty" in a change tracking table
5. User syncs changes back to DuckDB using sync_from_postgres()
6. Reactive system recalculates derived attributes

This module provides:
- SQL to create trigger functions and change tracking tables
- Helper functions to install/remove triggers
- Change detection queries

Example Usage:
    from sword_duckdb.triggers import install_triggers, get_pending_changes

    # After export_to_postgres()
    install_triggers(pg_connection_string, prefix="na_")

    # After QGIS editing session
    changes = get_pending_changes(pg_connection_string, prefix="na_")
    print(f"Modified: {changes['reaches']} reaches, {changes['nodes']} nodes")

NOTE: Git-like versioning has been implemented in workflow.py (v1.3.0):
1. Edit history with timestamps and user info - DONE (provenance.py)
2. Ability to revert to previous versions - DONE (workflow.snapshot/restore_snapshot)
3. Diff between any two versions - FUTURE (not yet implemented)
4. Branch/merge support for concurrent editing sessions - FUTURE (not yet implemented)
See DEVELOPMENT_STATUS.md for tracking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# SQL to create the change tracking table
CHANGE_TRACKING_TABLE = """
CREATE TABLE IF NOT EXISTS {prefix}sword_changes (
    change_id SERIAL PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    entity_id BIGINT NOT NULL,
    region VARCHAR(2) NOT NULL,
    change_type VARCHAR(10) NOT NULL,  -- 'INSERT', 'UPDATE', 'DELETE'
    changed_columns TEXT[],             -- Which columns were modified
    old_values JSONB,                   -- Previous values (for UPDATE/DELETE)
    new_values JSONB,                   -- New values (for INSERT/UPDATE)
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    synced BOOLEAN DEFAULT FALSE,       -- Has this change been synced to DuckDB?
    synced_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_{prefix}changes_synced ON {prefix}sword_changes(synced);
CREATE INDEX IF NOT EXISTS idx_{prefix}changes_table ON {prefix}sword_changes(table_name);
CREATE INDEX IF NOT EXISTS idx_{prefix}changes_entity ON {prefix}sword_changes(entity_id, table_name);
"""

# Trigger function for reaches table
REACHES_TRIGGER_FUNCTION = """
CREATE OR REPLACE FUNCTION {prefix}reaches_change_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO {prefix}sword_changes (table_name, entity_id, region, change_type, new_values)
        VALUES ('reaches', NEW.reach_id, NEW.region, 'INSERT',
                jsonb_build_object(
                    'reach_id', NEW.reach_id,
                    'dist_out', NEW.dist_out,
                    'facc', NEW.facc,
                    'wse', NEW.wse,
                    'width', NEW.width,
                    'slope', NEW.slope,
                    'x', NEW.x,
                    'y', NEW.y
                ));
        RETURN NEW;

    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO {prefix}sword_changes (table_name, entity_id, region, change_type, old_values, new_values, changed_columns)
        VALUES ('reaches', NEW.reach_id, NEW.region, 'UPDATE',
                jsonb_build_object(
                    'dist_out', OLD.dist_out,
                    'facc', OLD.facc,
                    'wse', OLD.wse,
                    'width', OLD.width,
                    'slope', OLD.slope,
                    'x', OLD.x,
                    'y', OLD.y
                ),
                jsonb_build_object(
                    'dist_out', NEW.dist_out,
                    'facc', NEW.facc,
                    'wse', NEW.wse,
                    'width', NEW.width,
                    'slope', NEW.slope,
                    'x', NEW.x,
                    'y', NEW.y
                ),
                ARRAY(
                    SELECT col FROM (VALUES
                        (CASE WHEN OLD.dist_out IS DISTINCT FROM NEW.dist_out THEN 'dist_out' END),
                        (CASE WHEN OLD.facc IS DISTINCT FROM NEW.facc THEN 'facc' END),
                        (CASE WHEN OLD.wse IS DISTINCT FROM NEW.wse THEN 'wse' END),
                        (CASE WHEN OLD.width IS DISTINCT FROM NEW.width THEN 'width' END),
                        (CASE WHEN OLD.slope IS DISTINCT FROM NEW.slope THEN 'slope' END),
                        (CASE WHEN OLD.x IS DISTINCT FROM NEW.x THEN 'x' END),
                        (CASE WHEN OLD.y IS DISTINCT FROM NEW.y THEN 'y' END)
                    ) AS t(col) WHERE col IS NOT NULL
                ));
        RETURN NEW;

    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO {prefix}sword_changes (table_name, entity_id, region, change_type, old_values)
        VALUES ('reaches', OLD.reach_id, OLD.region, 'DELETE',
                jsonb_build_object(
                    'reach_id', OLD.reach_id,
                    'dist_out', OLD.dist_out,
                    'facc', OLD.facc
                ));
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;
"""

# Trigger function for nodes table
NODES_TRIGGER_FUNCTION = """
CREATE OR REPLACE FUNCTION {prefix}nodes_change_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO {prefix}sword_changes (table_name, entity_id, region, change_type, new_values)
        VALUES ('nodes', NEW.node_id, NEW.region, 'INSERT',
                jsonb_build_object(
                    'node_id', NEW.node_id,
                    'reach_id', NEW.reach_id,
                    'dist_out', NEW.dist_out,
                    'facc', NEW.facc,
                    'wse', NEW.wse,
                    'width', NEW.width,
                    'x', NEW.x,
                    'y', NEW.y
                ));
        RETURN NEW;

    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO {prefix}sword_changes (table_name, entity_id, region, change_type, old_values, new_values, changed_columns)
        VALUES ('nodes', NEW.node_id, NEW.region, 'UPDATE',
                jsonb_build_object(
                    'dist_out', OLD.dist_out,
                    'facc', OLD.facc,
                    'wse', OLD.wse,
                    'width', OLD.width,
                    'x', OLD.x,
                    'y', OLD.y
                ),
                jsonb_build_object(
                    'dist_out', NEW.dist_out,
                    'facc', NEW.facc,
                    'wse', NEW.wse,
                    'width', NEW.width,
                    'x', NEW.x,
                    'y', NEW.y
                ),
                ARRAY(
                    SELECT col FROM (VALUES
                        (CASE WHEN OLD.dist_out IS DISTINCT FROM NEW.dist_out THEN 'dist_out' END),
                        (CASE WHEN OLD.facc IS DISTINCT FROM NEW.facc THEN 'facc' END),
                        (CASE WHEN OLD.wse IS DISTINCT FROM NEW.wse THEN 'wse' END),
                        (CASE WHEN OLD.width IS DISTINCT FROM NEW.width THEN 'width' END),
                        (CASE WHEN OLD.x IS DISTINCT FROM NEW.x THEN 'x' END),
                        (CASE WHEN OLD.y IS DISTINCT FROM NEW.y THEN 'y' END)
                    ) AS t(col) WHERE col IS NOT NULL
                ));
        RETURN NEW;

    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO {prefix}sword_changes (table_name, entity_id, region, change_type, old_values)
        VALUES ('nodes', OLD.node_id, OLD.region, 'DELETE',
                jsonb_build_object(
                    'node_id', OLD.node_id,
                    'reach_id', OLD.reach_id
                ));
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;
"""

# SQL to create triggers on the tables
CREATE_TRIGGERS = """
-- Reaches trigger
DROP TRIGGER IF EXISTS {prefix}reaches_change_trigger ON {prefix}reaches;
CREATE TRIGGER {prefix}reaches_change_trigger
    AFTER INSERT OR UPDATE OR DELETE ON {prefix}reaches
    FOR EACH ROW EXECUTE FUNCTION {prefix}reaches_change_trigger();

-- Nodes trigger
DROP TRIGGER IF EXISTS {prefix}nodes_change_trigger ON {prefix}nodes;
CREATE TRIGGER {prefix}nodes_change_trigger
    AFTER INSERT OR UPDATE OR DELETE ON {prefix}nodes
    FOR EACH ROW EXECUTE FUNCTION {prefix}nodes_change_trigger();
"""

# SQL to remove all triggers
DROP_TRIGGERS = """
DROP TRIGGER IF EXISTS {prefix}reaches_change_trigger ON {prefix}reaches;
DROP TRIGGER IF EXISTS {prefix}nodes_change_trigger ON {prefix}nodes;
DROP FUNCTION IF EXISTS {prefix}reaches_change_trigger();
DROP FUNCTION IF EXISTS {prefix}nodes_change_trigger();
DROP TABLE IF EXISTS {prefix}sword_changes;
"""


def _get_pg_connection(connection_string: str) -> Any:
    """Get a PostgreSQL connection using psycopg2."""
    try:
        import psycopg2
    except ImportError:
        raise ImportError(
            "psycopg2 is required for PostgreSQL triggers. "
            "Install with: pip install psycopg2-binary"
        )
    return psycopg2.connect(connection_string)


def install_triggers(
    connection_string: str, prefix: str = "", verbose: bool = True
) -> bool:
    """
    Install change tracking triggers on SWORD tables in PostgreSQL.

    Parameters
    ----------
    connection_string : str
        PostgreSQL connection string
    prefix : str, optional
        Table name prefix (e.g., "na_" for "na_reaches")
    verbose : bool, optional
        Print progress messages

    Returns
    -------
    bool
        True if triggers were installed successfully

    Example
    -------
    >>> install_triggers("postgresql://user:pass@localhost/sword_edit", prefix="na_")
    True
    """
    conn = _get_pg_connection(connection_string)
    cursor = conn.cursor()

    try:
        # Create change tracking table
        if verbose:
            logger.info(f"Creating change tracking table: {prefix}sword_changes")
        cursor.execute(CHANGE_TRACKING_TABLE.format(prefix=prefix))
        conn.commit()

        # Create trigger functions
        if verbose:
            logger.info("Creating trigger functions...")
        cursor.execute(REACHES_TRIGGER_FUNCTION.format(prefix=prefix))
        cursor.execute(NODES_TRIGGER_FUNCTION.format(prefix=prefix))
        conn.commit()

        # Create triggers
        if verbose:
            logger.info("Installing triggers on tables...")
        cursor.execute(CREATE_TRIGGERS.format(prefix=prefix))
        conn.commit()

        if verbose:
            logger.info("Triggers installed successfully!")
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"Error installing triggers: {e}")
        return False

    finally:
        cursor.close()
        conn.close()


def remove_triggers(
    connection_string: str, prefix: str = "", verbose: bool = True
) -> bool:
    """
    Remove change tracking triggers from SWORD tables.

    Parameters
    ----------
    connection_string : str
        PostgreSQL connection string
    prefix : str, optional
        Table name prefix
    verbose : bool, optional
        Print progress messages

    Returns
    -------
    bool
        True if triggers were removed successfully
    """
    conn = _get_pg_connection(connection_string)
    cursor = conn.cursor()

    try:
        if verbose:
            logger.info("Removing triggers...")
        cursor.execute(DROP_TRIGGERS.format(prefix=prefix))
        conn.commit()

        if verbose:
            logger.info("Triggers removed successfully!")
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"Error removing triggers: {e}")
        return False

    finally:
        cursor.close()
        conn.close()


def get_pending_changes(
    connection_string: str, prefix: str = "", table: str = None
) -> Dict[str, int]:
    """
    Get count of pending (unsynced) changes by table.

    Parameters
    ----------
    connection_string : str
        PostgreSQL connection string
    prefix : str, optional
        Table name prefix
    table : str, optional
        Filter to specific table ('reaches' or 'nodes')

    Returns
    -------
    dict
        Dictionary with table names and pending change counts
    """
    conn = _get_pg_connection(connection_string)
    cursor = conn.cursor()

    try:
        where_clause = "WHERE synced = FALSE"
        if table:
            where_clause += f" AND table_name = '{table}'"

        cursor.execute(f"""
            SELECT table_name, COUNT(*)
            FROM {prefix}sword_changes
            {where_clause}
            GROUP BY table_name
        """)

        results = {row[0]: row[1] for row in cursor.fetchall()}
        return results

    finally:
        cursor.close()
        conn.close()


def get_changed_entities(
    connection_string: str, prefix: str = "", table: str = "reaches", since: str = None
) -> List[Dict]:
    """
    Get list of changed entities with their changes.

    Parameters
    ----------
    connection_string : str
        PostgreSQL connection string
    prefix : str, optional
        Table name prefix
    table : str
        Table to query ('reaches' or 'nodes')
    since : str, optional
        Only return changes after this timestamp (ISO format)

    Returns
    -------
    list of dict
        List of change records with entity_id, change_type, changed_columns, etc.
    """
    conn = _get_pg_connection(connection_string)
    cursor = conn.cursor()

    try:
        where_clause = f"WHERE synced = FALSE AND table_name = '{table}'"
        if since:
            where_clause += f" AND changed_at > '{since}'"

        cursor.execute(f"""
            SELECT
                change_id,
                entity_id,
                region,
                change_type,
                changed_columns,
                old_values,
                new_values,
                changed_at
            FROM {prefix}sword_changes
            {where_clause}
            ORDER BY changed_at
        """)

        columns = [
            "change_id",
            "entity_id",
            "region",
            "change_type",
            "changed_columns",
            "old_values",
            "new_values",
            "changed_at",
        ]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    finally:
        cursor.close()
        conn.close()


def mark_changes_synced(
    connection_string: str,
    prefix: str = "",
    change_ids: List[int] = None,
    all_changes: bool = False,
) -> int:
    """
    Mark changes as synced after importing to DuckDB.

    Parameters
    ----------
    connection_string : str
        PostgreSQL connection string
    prefix : str, optional
        Table name prefix
    change_ids : list of int, optional
        Specific change IDs to mark as synced
    all_changes : bool, optional
        If True, mark all unsynced changes as synced

    Returns
    -------
    int
        Number of changes marked as synced
    """
    conn = _get_pg_connection(connection_string)
    cursor = conn.cursor()

    try:
        if all_changes:
            cursor.execute(f"""
                UPDATE {prefix}sword_changes
                SET synced = TRUE, synced_at = CURRENT_TIMESTAMP
                WHERE synced = FALSE
            """)
        elif change_ids:
            cursor.execute(
                f"""
                UPDATE {prefix}sword_changes
                SET synced = TRUE, synced_at = CURRENT_TIMESTAMP
                WHERE change_id = ANY(%s)
            """,
                (change_ids,),
            )
        else:
            return 0

        count = cursor.rowcount
        conn.commit()
        return count

    finally:
        cursor.close()
        conn.close()


def clear_change_history(
    connection_string: str,
    prefix: str = "",
    synced_only: bool = True,
    older_than_days: int = None,
) -> int:
    """
    Clear change history records.

    Parameters
    ----------
    connection_string : str
        PostgreSQL connection string
    prefix : str, optional
        Table name prefix
    synced_only : bool, optional
        Only delete synced changes (default True for safety)
    older_than_days : int, optional
        Only delete records older than this many days

    Returns
    -------
    int
        Number of records deleted
    """
    conn = _get_pg_connection(connection_string)
    cursor = conn.cursor()

    try:
        where_clauses = []
        if synced_only:
            where_clauses.append("synced = TRUE")
        if older_than_days:
            where_clauses.append(
                f"changed_at < CURRENT_TIMESTAMP - INTERVAL '{older_than_days} days'"
            )

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        cursor.execute(f"""
            DELETE FROM {prefix}sword_changes
            WHERE {where_clause}
        """)

        count = cursor.rowcount
        conn.commit()
        return count

    finally:
        cursor.close()
        conn.close()


def get_trigger_sql(prefix: str = "") -> str:
    """
    Get the complete SQL for creating triggers (for manual installation).

    Parameters
    ----------
    prefix : str, optional
        Table name prefix

    Returns
    -------
    str
        Complete SQL script for creating change tracking infrastructure
    """
    sql_parts = [
        "-- SWORD Change Tracking Infrastructure",
        "-- Run this SQL in PostgreSQL after exporting SWORD data",
        "",
        "-- 1. Create change tracking table",
        CHANGE_TRACKING_TABLE.format(prefix=prefix),
        "",
        "-- 2. Create trigger functions",
        REACHES_TRIGGER_FUNCTION.format(prefix=prefix),
        NODES_TRIGGER_FUNCTION.format(prefix=prefix),
        "",
        "-- 3. Create triggers",
        CREATE_TRIGGERS.format(prefix=prefix),
    ]
    return "\n".join(sql_parts)
