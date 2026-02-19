# -*- coding: utf-8 -*-
"""
SWORD Provenance Logging System
================================

This module provides comprehensive provenance tracking for all SWORD database operations.
It captures WHO made changes, WHAT changed, WHEN it happened, and WHY.

Key Features:
    - Operation logging with context managers for automatic status tracking
    - Value snapshots for rollback capability
    - Entity history queries
    - Source data lineage tracking

Example Usage:
    from sword_duckdb.provenance import ProvenanceLogger

    logger = ProvenanceLogger(db_connection)

    # Log an operation
    with logger.operation('UPDATE', 'reaches', [123, 456], reason='Fix elevations'):
        # ... make changes ...
        logger.log_value_change(op_id, 'reaches', 123, 'wse', 44.2, 45.5)

    # Query history
    history = logger.get_entity_history('reach', 123)
"""

from __future__ import annotations

import getpass
import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from .sword_db import SWORDDatabase

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations that can be logged."""
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    RECALCULATE = "RECALCULATE"
    IMPORT = "IMPORT"
    EXPORT = "EXPORT"
    RECONSTRUCT = "RECONSTRUCT"
    BATCH = "BATCH"
    SNAPSHOT = "SNAPSHOT"
    RESTORE = "RESTORE"


class OperationStatus(Enum):
    """Status of an operation."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


class ProvenanceLogger:
    """
    Logs all SWORD operations with full provenance tracking.

    This class provides:
    - Operation lifecycle management (start, complete, fail)
    - Value change tracking for rollback
    - History queries by entity or time range
    - Rollback capability using stored snapshots

    Parameters
    ----------
    db : SWORDDatabase
        Database object with execute() and query() methods.
        Works with both DuckDB and PostgreSQL backends.
    user_id : str, optional
        User identifier. Defaults to system username.
    session_id : str, optional
        Session identifier. Auto-generated if not provided.
    enabled : bool, optional
        Whether to actually log operations. Defaults to True.

    Attributes
    ----------
    user_id : str
        Current user identifier
    session_id : str
        Current session identifier
    enabled : bool
        Whether logging is active

    Example
    -------
    >>> logger = ProvenanceLogger(conn, user_id="jake")
    >>> with logger.operation('UPDATE', 'reaches', [123], reason='Fix wse'):
    ...     # make changes
    ...     pass
    """

    def __init__(
        self,
        db: 'SWORDDatabase',
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        enabled: bool = True,
    ):
        self._db = db
        self.user_id = user_id or getpass.getuser()
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.enabled = enabled
        self._operation_stack: List[int] = []  # Stack of active operation IDs
        self._next_operation_id: Optional[int] = None
        self._next_snapshot_id: Optional[int] = None

        # Initialize ID counters
        self._init_id_counters()

    def _init_id_counters(self) -> None:
        """Initialize operation and snapshot ID counters from database."""
        try:
            result = self._db.execute(
                "SELECT COALESCE(MAX(operation_id), 0) FROM sword_operations"
            ).fetchone()
            self._next_operation_id = (result[0] or 0) + 1
        except Exception:
            # Table may not exist yet
            self._next_operation_id = 1

        try:
            result = self._db.execute(
                "SELECT COALESCE(MAX(snapshot_id), 0) FROM sword_value_snapshots"
            ).fetchone()
            self._next_snapshot_id = (result[0] or 0) + 1
        except Exception:
            self._next_snapshot_id = 1

    def _get_next_operation_id(self) -> int:
        """Get the next available operation ID."""
        op_id = self._next_operation_id
        self._next_operation_id += 1
        return op_id

    def _get_next_snapshot_id(self) -> int:
        """Get the next available snapshot ID."""
        snap_id = self._next_snapshot_id
        self._next_snapshot_id += 1
        return snap_id

    @property
    def current_operation_id(self) -> Optional[int]:
        """Get the current active operation ID, if any."""
        return self._operation_stack[-1] if self._operation_stack else None

    @contextmanager
    def operation(
        self,
        op_type: Union[str, OperationType],
        table_name: Optional[str] = None,
        entity_ids: Optional[List[int]] = None,
        region: Optional[str] = None,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        affected_columns: Optional[List[str]] = None,
    ) -> Generator[int, None, None]:
        """
        Context manager for logging an operation with automatic status tracking.

        The operation is marked as IN_PROGRESS when entering the context and
        COMPLETED when exiting successfully, or FAILED if an exception occurs.

        Parameters
        ----------
        op_type : str or OperationType
            Type of operation (CREATE, UPDATE, DELETE, etc.)
        table_name : str, optional
            Target table name (reaches, nodes, centerlines)
        entity_ids : list of int, optional
            List of affected entity IDs
        region : str, optional
            Region code (NA, EU, AS, etc.)
        reason : str, optional
            User-provided reason for the operation
        details : dict, optional
            Additional operation-specific parameters
        affected_columns : list of str, optional
            Which columns are being modified

        Yields
        ------
        int
            The operation ID for this operation

        Example
        -------
        >>> with logger.operation('UPDATE', 'reaches', [123], reason='Fix wse') as op_id:
        ...     logger.log_value_change(op_id, 'reaches', 123, 'wse', 44.2, 45.5)
        ...     # ... make database changes ...
        """
        if not self.enabled:
            yield None
            return

        # Convert enum to string if needed
        if isinstance(op_type, OperationType):
            op_type = op_type.value

        # Create the operation record
        op_id = self._start_operation(
            op_type=op_type,
            table_name=table_name,
            entity_ids=entity_ids,
            region=region,
            reason=reason,
            details=details,
            affected_columns=affected_columns,
        )

        self._operation_stack.append(op_id)

        try:
            yield op_id
            self._complete_operation(op_id)
        except Exception as e:
            self._fail_operation(op_id, str(e))
            raise
        finally:
            self._operation_stack.pop()

    def _start_operation(
        self,
        op_type: str,
        table_name: Optional[str],
        entity_ids: Optional[List[int]],
        region: Optional[str],
        reason: Optional[str],
        details: Optional[Dict[str, Any]],
        affected_columns: Optional[List[str]],
    ) -> int:
        """Start a new operation and return its ID."""
        op_id = self._get_next_operation_id()

        # Get parent operation if nested
        parent_op_id = self._operation_stack[-1] if self._operation_stack else None

        # Convert to lists for array parameters (works with both DuckDB and PostgreSQL)
        entity_ids_list = list(entity_ids) if entity_ids else None
        affected_cols_list = list(affected_columns) if affected_columns else None

        # Insert operation record (use parameterized arrays)
        sql = """
            INSERT INTO sword_operations (
                operation_id, operation_type, table_name, entity_ids, region,
                user_id, session_id, started_at,
                operation_details, affected_columns,
                reason, source_operation_id, status
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, CURRENT_TIMESTAMP,
                ?, ?,
                ?, ?, 'IN_PROGRESS'
            )
        """

        self._db.execute(sql, [
            op_id,
            op_type,
            table_name,
            entity_ids_list,
            region,
            self.user_id,
            self.session_id,
            json.dumps(details) if details else None,
            affected_cols_list,
            reason,
            parent_op_id,
        ])

        logger.debug(f"Started operation {op_id}: {op_type} on {table_name}")
        return op_id

    def _complete_operation(self, operation_id: int) -> None:
        """Mark an operation as completed."""
        self._db.execute("""
            UPDATE sword_operations
            SET status = 'COMPLETED', completed_at = CURRENT_TIMESTAMP
            WHERE operation_id = ?
        """, [operation_id])
        logger.debug(f"Completed operation {operation_id}")

    def _fail_operation(self, operation_id: int, error_message: str) -> None:
        """Mark an operation as failed."""
        self._db.execute("""
            UPDATE sword_operations
            SET status = 'FAILED', completed_at = CURRENT_TIMESTAMP, error_message = ?
            WHERE operation_id = ?
        """, [error_message, operation_id])
        logger.warning(f"Operation {operation_id} failed: {error_message}")

    def log_value_change(
        self,
        operation_id: int,
        table_name: str,
        entity_id: int,
        column_name: str,
        old_value: Any,
        new_value: Any,
        data_type: Optional[str] = None,
    ) -> int:
        """
        Log an individual value change for rollback capability.

        Parameters
        ----------
        operation_id : int
            The operation ID this change belongs to
        table_name : str
            Table being modified
        entity_id : int
            Entity being modified (reach_id, node_id, or cl_id)
        column_name : str
            Column being modified
        old_value : any
            Value before the change
        new_value : any
            Value after the change
        data_type : str, optional
            Data type for proper restoration (float, int, str, etc.)

        Returns
        -------
        int
            The snapshot ID
        """
        if not self.enabled:
            return -1

        snapshot_id = self._get_next_snapshot_id()

        # Detect data type if not provided
        if data_type is None:
            if isinstance(old_value, float) or isinstance(new_value, float):
                data_type = "float"
            elif isinstance(old_value, int) or isinstance(new_value, int):
                data_type = "int"
            elif isinstance(old_value, str) or isinstance(new_value, str):
                data_type = "str"
            else:
                data_type = "json"

        self._db.execute("""
            INSERT INTO sword_value_snapshots (
                snapshot_id, operation_id, table_name, entity_id, column_name,
                old_value, new_value, data_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            snapshot_id,
            operation_id,
            table_name,
            entity_id,
            column_name,
            json.dumps(old_value) if old_value is not None else None,
            json.dumps(new_value) if new_value is not None else None,
            data_type,
        ])

        return snapshot_id

    def log_value_changes_batch(
        self,
        operation_id: int,
        table_name: str,
        entity_ids: List[int],
        column_name: str,
        old_values: List[Any],
        new_values: List[Any],
        data_type: Optional[str] = None,
    ) -> List[int]:
        """
        Log multiple value changes efficiently.

        Parameters
        ----------
        operation_id : int
            The operation ID these changes belong to
        table_name : str
            Table being modified
        entity_ids : list of int
            Entities being modified
        column_name : str
            Column being modified
        old_values : list
            Values before changes
        new_values : list
            Values after changes
        data_type : str, optional
            Data type for restoration

        Returns
        -------
        list of int
            The snapshot IDs
        """
        if not self.enabled:
            return []

        snapshot_ids = []
        for entity_id, old_val, new_val in zip(entity_ids, old_values, new_values):
            snap_id = self.log_value_change(
                operation_id, table_name, entity_id, column_name,
                old_val, new_val, data_type
            )
            snapshot_ids.append(snap_id)

        return snapshot_ids

    def get_entity_history(
        self,
        entity_type: str,
        entity_id: int,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get complete modification history for an entity.

        Parameters
        ----------
        entity_type : str
            Type of entity (reach, node, centerline)
        entity_id : int
            Entity ID
        limit : int, optional
            Maximum number of records to return

        Returns
        -------
        list of dict
            History records with operation details and value changes
        """
        # Map entity type to table name
        table_map = {
            'reach': 'reaches',
            'node': 'nodes',
            'centerline': 'centerlines',
        }
        table_name = table_map.get(entity_type, entity_type)

        # Get operations that affected this entity
        sql = """
            SELECT
                o.operation_id,
                o.operation_type,
                o.table_name,
                o.user_id,
                o.session_id,
                o.started_at,
                o.completed_at,
                o.reason,
                o.status,
                o.affected_columns
            FROM sword_operations o
            WHERE o.table_name = ?
              AND list_contains(o.entity_ids, ?)
            ORDER BY o.started_at DESC
            LIMIT ?
        """

        results = self._db.execute(sql, [table_name, entity_id, limit]).fetchall()

        history = []
        for row in results:
            op_id = row[0]

            # Get value changes for this operation and entity
            changes_sql = """
                SELECT column_name, old_value, new_value, data_type
                FROM sword_value_snapshots
                WHERE operation_id = ? AND table_name = ? AND entity_id = ?
                ORDER BY snapshot_id
            """
            changes = self._db.execute(
                changes_sql, [op_id, table_name, entity_id]
            ).fetchall()

            value_changes = []
            for change in changes:
                value_changes.append({
                    'column': change[0],
                    'old_value': json.loads(change[1]) if change[1] else None,
                    'new_value': json.loads(change[2]) if change[2] else None,
                    'data_type': change[3],
                })

            history.append({
                'operation_id': op_id,
                'operation_type': row[1],
                'table_name': row[2],
                'user_id': row[3],
                'session_id': row[4],
                'started_at': row[5],
                'completed_at': row[6],
                'reason': row[7],
                'status': row[8],
                'affected_columns': row[9],
                'value_changes': value_changes,
            })

        return history

    def get_operation_history(
        self,
        since: Optional[datetime] = None,
        table_name: Optional[str] = None,
        region: Optional[str] = None,
        operation_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get operation history with optional filters.

        Parameters
        ----------
        since : datetime, optional
            Only return operations after this time
        table_name : str, optional
            Filter by table
        region : str, optional
            Filter by region
        operation_type : str, optional
            Filter by operation type
        limit : int, optional
            Maximum records to return

        Returns
        -------
        list of dict
            Operation records
        """
        conditions = []
        params = []

        if since:
            conditions.append("started_at >= ?")
            params.append(since)
        if table_name:
            conditions.append("table_name = ?")
            params.append(table_name)
        if region:
            conditions.append("region = ?")
            params.append(region)
        if operation_type:
            conditions.append("operation_type = ?")
            params.append(operation_type)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        sql = f"""
            SELECT
                operation_id, operation_type, table_name, entity_ids, region,
                user_id, session_id, started_at, completed_at,
                reason, status, error_message
            FROM sword_operations
            {where_clause}
            ORDER BY started_at DESC
            LIMIT ?
        """
        params.append(limit)

        results = self._db.execute(sql, params).fetchall()

        return [
            {
                'operation_id': r[0],
                'operation_type': r[1],
                'table_name': r[2],
                'entity_ids': r[3],
                'region': r[4],
                'user_id': r[5],
                'session_id': r[6],
                'started_at': r[7],
                'completed_at': r[8],
                'reason': r[9],
                'status': r[10],
                'error_message': r[11],
            }
            for r in results
        ]

    def rollback_operation(self, operation_id: int) -> int:
        """
        Rollback a specific operation using stored snapshots.

        This restores all values that were changed during the operation
        to their previous state.

        Parameters
        ----------
        operation_id : int
            The operation ID to rollback

        Returns
        -------
        int
            Number of values restored

        Raises
        ------
        ValueError
            If the operation cannot be rolled back
        """
        # Check operation status
        result = self._db.execute("""
            SELECT status, table_name FROM sword_operations WHERE operation_id = ?
        """, [operation_id]).fetchone()

        if not result:
            raise ValueError(f"Operation {operation_id} not found")

        status, table_name = result
        if status == OperationStatus.ROLLED_BACK.value:
            raise ValueError(f"Operation {operation_id} already rolled back")

        # Get all snapshots for this operation
        snapshots = self._db.execute("""
            SELECT snapshot_id, table_name, entity_id, column_name, old_value, data_type
            FROM sword_value_snapshots
            WHERE operation_id = ?
            ORDER BY snapshot_id DESC
        """, [operation_id]).fetchall()

        if not snapshots:
            logger.warning(f"No snapshots found for operation {operation_id}")
            return 0

        # Restore values (in reverse order of changes)
        restored = 0
        for snap in snapshots:
            snap_id, tbl, entity_id, column, old_val_json, dtype = snap

            if old_val_json is None:
                continue

            old_value = json.loads(old_val_json)

            # Determine the ID column for this table
            id_col_map = {
                'reaches': 'reach_id',
                'nodes': 'node_id',
                'centerlines': 'cl_id',
            }
            id_col = id_col_map.get(tbl, 'id')

            # Update the value
            try:
                self._db.execute(f"""
                    UPDATE {tbl} SET {column} = ? WHERE {id_col} = ?
                """, [old_value, entity_id])
                restored += 1
            except Exception as e:
                logger.error(f"Failed to restore {tbl}.{column} for {entity_id}: {e}")

        # Mark operation as rolled back
        self._db.execute("""
            UPDATE sword_operations
            SET status = 'ROLLED_BACK'
            WHERE operation_id = ?
        """, [operation_id])

        logger.info(f"Rolled back operation {operation_id}, restored {restored} values")
        return restored

    def get_lineage(
        self,
        entity_type: str,
        entity_id: int,
        region: str,
    ) -> List[Dict[str, Any]]:
        """
        Get source data lineage for an entity.

        Shows which source datasets contributed to each attribute
        of the entity.

        Parameters
        ----------
        entity_type : str
            Type of entity (reach, node, centerline)
        entity_id : int
            Entity ID
        region : str
            Region code

        Returns
        -------
        list of dict
            Lineage records showing source attribution
        """
        results = self._db.execute("""
            SELECT
                lineage_id, source_dataset, source_id, source_version,
                attribute_name, derivation_method, created_at
            FROM sword_source_lineage
            WHERE entity_type = ? AND entity_id = ? AND region = ?
            ORDER BY attribute_name
        """, [entity_type, entity_id, region]).fetchall()

        return [
            {
                'lineage_id': r[0],
                'source_dataset': r[1],
                'source_id': r[2],
                'source_version': r[3],
                'attribute_name': r[4],
                'derivation_method': r[5],
                'created_at': r[6],
            }
            for r in results
        ]

    def record_lineage(
        self,
        entity_type: str,
        entity_id: int,
        region: str,
        source_dataset: str,
        attribute_name: str,
        derivation_method: str = "direct",
        source_id: Optional[str] = None,
        source_version: Optional[str] = None,
    ) -> int:
        """
        Record source data lineage for an entity attribute.

        Parameters
        ----------
        entity_type : str
            Type of entity (reach, node, centerline)
        entity_id : int
            Entity ID
        region : str
            Region code
        source_dataset : str
            Source dataset name (GRWL, MERIT_HYDRO, etc.)
        attribute_name : str
            Attribute that came from this source
        derivation_method : str, optional
            How the value was derived (direct, interpolated, aggregated, computed)
        source_id : str, optional
            ID in the source dataset
        source_version : str, optional
            Version of the source dataset

        Returns
        -------
        int
            The lineage ID
        """
        # Get next lineage ID
        result = self._db.execute(
            "SELECT COALESCE(MAX(lineage_id), 0) + 1 FROM sword_source_lineage"
        ).fetchone()
        lineage_id = result[0]

        self._db.execute("""
            INSERT INTO sword_source_lineage (
                lineage_id, entity_type, entity_id, region,
                source_dataset, source_id, source_version,
                attribute_name, derivation_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            lineage_id, entity_type, entity_id, region,
            source_dataset, source_id, source_version,
            attribute_name, derivation_method,
        ])

        return lineage_id

    def get_max_operation_id(self) -> int:
        """
        Get the highest completed operation ID.

        Used by the snapshot system to mark a reference point in history.

        Returns
        -------
        int
            The maximum operation_id with status='COMPLETED', or 0 if none exist.
        """
        result = self._db.execute("""
            SELECT COALESCE(MAX(operation_id), 0)
            FROM sword_operations
            WHERE status = 'COMPLETED'
        """).fetchone()
        return result[0] if result else 0

    def get_operations_after(
        self,
        operation_id: int,
        include_in_progress: bool = False,
        exclude_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all operations after a specific operation ID.

        Used by the restore system to find operations that need to be rolled back.

        Parameters
        ----------
        operation_id : int
            The reference operation ID. Returns operations with ID > this value.
        include_in_progress : bool, optional
            If True, include IN_PROGRESS operations. Default False.
        exclude_types : list of str, optional
            Operation types to exclude (e.g., ['SNAPSHOT', 'RESTORE']).
            These are metadata operations that shouldn't be rolled back.

        Returns
        -------
        list of dict
            Operations in reverse chronological order (most recent first),
            suitable for rollback processing.
        """
        status_filter = "IN ('COMPLETED')"
        if include_in_progress:
            status_filter = "IN ('COMPLETED', 'IN_PROGRESS')"

        # Build type exclusion clause
        type_exclusion = ""
        query_params = [operation_id]
        if exclude_types:
            placeholders = ", ".join(["?" for _ in exclude_types])
            type_exclusion = f" AND operation_type NOT IN ({placeholders})"
            query_params.extend(exclude_types)

        results = self._db.execute(f"""
            SELECT
                operation_id, operation_type, table_name, entity_ids, region,
                user_id, session_id, started_at, completed_at,
                reason, status
            FROM sword_operations
            WHERE operation_id > ?
              AND status {status_filter}
              {type_exclusion}
            ORDER BY operation_id DESC
        """, query_params).fetchall()

        return [
            {
                'operation_id': r[0],
                'operation_type': r[1],
                'table_name': r[2],
                'entity_ids': r[3],
                'region': r[4],
                'user_id': r[5],
                'session_id': r[6],
                'started_at': r[7],
                'completed_at': r[8],
                'reason': r[9],
                'status': r[10],
            }
            for r in results
        ]
