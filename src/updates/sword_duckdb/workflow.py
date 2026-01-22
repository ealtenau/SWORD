# -*- coding: utf-8 -*-
"""
SWORD Workflow Orchestration
============================

This module provides high-level workflow orchestration for SWORD database operations.
It ties together loading, modification tracking, provenance logging, reactive
recalculation, and export into a unified workflow.

SWORDWorkflow is the RECOMMENDED entry point for all SWORD operations.

Example Usage:
    from sword_duckdb import SWORDWorkflow

    # Initialize workflow with provenance tracking
    workflow = SWORDWorkflow(user_id="jake")

    # Load a region
    sword = workflow.load('data/duckdb/sword_v17b.duckdb', 'NA')

    # Modify with transaction (automatic rollback on error)
    with workflow.transaction("Fix elevation errors"):
        workflow.modify_reach(123, wse=45.5, reason="Corrected from field data")

    # Or batch modifications
    with workflow.batch_modify():
        sword.reaches.wse[mask] = new_values
        sword.nodes.x[node_mask] = new_x_values

    # Query history
    history = workflow.get_history(entity_type='reach', entity_id=123)

    # Export to various formats
    workflow.export(formats=['geopackage'], output_dir='outputs/')
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from .sword_class import SWORD
    from .reactive import SWORDReactive
    from .provenance import ProvenanceLogger
    from .reconstruction import ReconstructionEngine
    from .imagery import ImageryPipeline

logger = logging.getLogger(__name__)


class SWORDWorkflow:
    """
    High-level workflow orchestrator for SWORD database operations.

    This is the RECOMMENDED entry point for all SWORD operations. It provides:
    - Loading SWORD databases
    - Full provenance tracking (who, what, when, why)
    - Transaction support with automatic rollback
    - Reactive recalculation of derived attributes
    - Export to various formats
    - QGIS/PostgreSQL integration

    Parameters
    ----------
    user_id : str, optional
        User identifier for provenance logging. Defaults to system username.
    enable_provenance : bool, optional
        Whether to enable provenance logging. Defaults to True.

    Attributes
    ----------
    sword : SWORD
        The loaded SWORD instance (None until load() is called)
    reactive : SWORDReactive
        The reactive system for change tracking
    provenance : ProvenanceLogger
        The provenance logger for operation tracking

    Example
    -------
    >>> workflow = SWORDWorkflow(user_id="jake")
    >>> sword = workflow.load('sword_v17b.duckdb', 'NA')
    >>> with workflow.transaction("Fix wse"):
    ...     workflow.modify_reach(123, wse=45.5)
    >>> workflow.close()
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        enable_provenance: bool = True,
    ):
        """Initialize the workflow orchestrator."""
        self._sword: Optional['SWORD'] = None
        self._reactive: Optional['SWORDReactive'] = None
        self._provenance: Optional['ProvenanceLogger'] = None
        self._reconstruction: Optional['ReconstructionEngine'] = None
        self._imagery: Optional['ImageryPipeline'] = None
        self._user_id = user_id
        self._enable_provenance = enable_provenance
        self._in_batch: bool = False
        self._in_transaction: bool = False
        self._current_transaction_op_id: Optional[int] = None
        self._pending_changes: int = 0
        self._db_path: Optional[Path] = None
        self._region: Optional[str] = None

    @property
    def sword(self) -> Optional['SWORD']:
        """Get the loaded SWORD instance."""
        return self._sword

    @property
    def reactive(self) -> Optional['SWORDReactive']:
        """Get the reactive system instance."""
        return self._reactive

    @property
    def provenance(self) -> Optional['ProvenanceLogger']:
        """Get the provenance logger instance."""
        return self._provenance

    @property
    def reconstruction(self) -> Optional['ReconstructionEngine']:
        """Get the reconstruction engine instance."""
        return self._reconstruction

    @property
    def imagery(self) -> Optional['ImageryPipeline']:
        """
        Get the imagery pipeline instance (lazy-loaded).

        Returns None if no SWORD database is loaded.
        """
        if self._sword is None:
            return None

        if self._imagery is None:
            from .imagery import ImageryPipeline

            self._imagery = ImageryPipeline(
                sword=self._sword,
                db_conn=self._sword.db.conn if hasattr(self._sword, 'db') else None,
            )

        return self._imagery

    @property
    def is_loaded(self) -> bool:
        """Check if a SWORD database is currently loaded."""
        return self._sword is not None

    @property
    def has_pending_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        if self._reactive is None:
            return False
        return len(self._reactive._dirty_attrs) > 0

    @property
    def region(self) -> Optional[str]:
        """Get the current region code."""
        return self._region

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    def load(
        self,
        db_path: Union[str, Path],
        region: str,
        reason: str = "Load database",
    ) -> 'SWORD':
        """
        Load a SWORD database and initialize all subsystems.

        Parameters
        ----------
        db_path : str or Path
            Path to the DuckDB database file
        region : str
            Region code (e.g., 'NA', 'EU', 'AS')
        reason : str, optional
            Reason for loading (logged to provenance)

        Returns
        -------
        SWORD
            The loaded SWORD instance with reactive tracking enabled

        Raises
        ------
        FileNotFoundError
            If the database file doesn't exist
        ValueError
            If a database is already loaded (call close() first)

        Example
        -------
        >>> workflow = SWORDWorkflow()
        >>> sword = workflow.load('sword_v17b.duckdb', 'NA')
        >>> print(f"Loaded {len(sword.reaches)} reaches")
        """
        if self._sword is not None:
            raise ValueError(
                "A database is already loaded. Call close() before loading another."
            )

        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        # Import here to avoid circular imports
        from .sword_class import SWORD
        from .reactive import SWORDReactive
        from .provenance import ProvenanceLogger
        from .schema import create_provenance_tables
        from .reconstruction import ReconstructionEngine

        logger.info(f"Loading SWORD database: {db_path}, region: {region}")

        # Load the SWORD database
        self._sword = SWORD(str(db_path), region)
        self._db_path = db_path
        self._region = region

        # Initialize reactive system
        self._reactive = SWORDReactive(self._sword)
        self._sword.set_reactive(self._reactive)

        # Initialize provenance system
        if self._enable_provenance:
            # Ensure provenance tables exist
            try:
                create_provenance_tables(self._sword.db.conn)
            except Exception as e:
                # Tables may already exist
                logger.debug(f"Provenance tables check: {e}")

            self._provenance = ProvenanceLogger(
                self._sword.db.conn,
                user_id=self._user_id,
                enabled=self._enable_provenance,
            )

            # Log the load operation
            with self._provenance.operation(
                'IMPORT', None, None, region, reason=reason
            ):
                pass  # Just log the operation

        # Initialize reconstruction engine
        self._reconstruction = ReconstructionEngine(
            self._sword,
            provenance=self._provenance,
        )

        logger.info(
            f"Loaded {len(self._sword.reaches)} reaches, "
            f"{len(self._sword.nodes)} nodes"
        )

        return self._sword

    def close(self, save: bool = True) -> None:
        """
        Close the current SWORD database connection.

        Parameters
        ----------
        save : bool, optional
            If True and there are pending changes, commit them first.
            If False, pending changes are discarded.
        """
        if self._sword is not None:
            if self.has_pending_changes:
                if save:
                    logger.info("Committing pending changes before close")
                    self.commit()
                else:
                    logger.warning(
                        "Closing with uncommitted changes. "
                        "Changes will be lost."
                    )

            self._sword.close()
            self._sword = None
            self._reactive = None
            self._provenance = None
            self._reconstruction = None
            self._imagery = None
            self._db_path = None
            self._region = None
            logger.info("SWORD database closed")

    # =========================================================================
    # TRANSACTION METHODS
    # =========================================================================

    @contextmanager
    def transaction(
        self,
        reason: str = None,
    ) -> Generator[int, None, None]:
        """
        Context manager for atomic operations with automatic rollback on error.

        All modifications within a transaction are logged together. If an
        exception occurs, all changes are automatically rolled back.

        Parameters
        ----------
        reason : str, optional
            Description of what this transaction is doing

        Yields
        ------
        int
            The operation ID for this transaction

        Example
        -------
        >>> with workflow.transaction("Fix delta reaches"):
        ...     workflow.modify_reach(123, dist_out=1234.5)
        ...     workflow.modify_reach(456, dist_out=5678.9)
        ... # All changes committed atomically

        >>> with workflow.transaction("This will fail"):
        ...     workflow.modify_reach(123, dist_out=999)
        ...     raise ValueError("Something went wrong")
        ... # All changes rolled back
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        if self._in_transaction:
            raise RuntimeError("Already in a transaction. Nesting not supported.")

        self._in_transaction = True

        # Start provenance tracking
        op_id = None
        if self._provenance and self._enable_provenance:
            # We'll use the operation context manager
            with self._provenance.operation(
                'BATCH',
                table_name=None,
                entity_ids=None,
                region=self._region,
                reason=reason,
            ) as op_id:
                self._current_transaction_op_id = op_id
                try:
                    yield op_id
                    # On success, recalculate if there are dirty attrs
                    if self.has_pending_changes:
                        self._reactive.recalculate()
                except Exception:
                    # On failure, rollback
                    if op_id is not None:
                        try:
                            self._provenance.rollback_operation(op_id)
                            logger.info(f"Transaction {op_id} rolled back")
                        except Exception as rollback_err:
                            logger.error(f"Rollback failed: {rollback_err}")
                    raise
                finally:
                    self._in_transaction = False
                    self._current_transaction_op_id = None
        else:
            # No provenance, just track transaction state
            try:
                yield None
                if self.has_pending_changes:
                    self._reactive.recalculate()
            finally:
                self._in_transaction = False

    @contextmanager
    def batch_modify(self) -> Generator[None, None, None]:
        """
        Context manager for batch modifications without transaction semantics.

        Modifications are tracked and recalculation is deferred until exit.
        Unlike transaction(), errors do NOT trigger automatic rollback.

        Example
        -------
        >>> with workflow.batch_modify():
        ...     sword.reaches.wse[0] = 100.0
        ...     sword.reaches.wse[1] = 101.0
        ...     sword.reaches.slope[0] = 0.001
        ... # Recalculation runs once at exit
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        self._in_batch = True
        logger.debug("Entering batch modification mode")

        try:
            yield
        finally:
            self._in_batch = False
            logger.debug("Exiting batch modification mode")
            if self.has_pending_changes:
                self.commit()

    def commit(self) -> Dict[str, Any]:
        """
        Commit pending changes by running reactive recalculation.

        Returns
        -------
        dict
            Statistics about what was recalculated

        Raises
        ------
        RuntimeError
            If no database is loaded

        Example
        -------
        >>> sword.reaches.dist_out[0] = 1234.5
        >>> stats = workflow.commit()
        >>> print(f"Recalculated: {stats}")
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        if not self.has_pending_changes:
            logger.info("No pending changes to commit")
            return {"recalculated": 0}

        logger.info("Committing changes - running reactive recalculation")

        dirty_attrs = list(self._reactive._dirty_attrs)
        dirty_count = len(dirty_attrs)

        # Log the recalculation operation
        if self._provenance and self._enable_provenance:
            with self._provenance.operation(
                'RECALCULATE',
                table_name=None,
                entity_ids=None,
                region=self._region,
                reason="Commit pending changes",
                affected_columns=dirty_attrs,
            ):
                self._reactive.recalculate()
        else:
            self._reactive.recalculate()

        logger.info(f"Recalculated {dirty_count} dirty attributes")

        return {
            "dirty_attributes": dirty_count,
            "attributes_affected": dirty_attrs,
        }

    def rollback(self, operation_id: int) -> int:
        """
        Rollback a specific operation to restore previous values.

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
        RuntimeError
            If provenance is not enabled
        ValueError
            If the operation cannot be rolled back
        """
        if not self._provenance:
            raise RuntimeError("Provenance tracking is not enabled")

        return self._provenance.rollback_operation(operation_id)

    # =========================================================================
    # MODIFICATION METHODS
    # =========================================================================

    def modify_reach(
        self,
        reach_id: int,
        reason: str = None,
        **attributes,
    ) -> None:
        """
        Modify reach attributes with provenance logging.

        Parameters
        ----------
        reach_id : int
            The reach ID to modify
        reason : str, optional
            Why this modification is being made
        **attributes : dict
            Attribute name/value pairs to modify (e.g., wse=45.5, dist_out=1234)

        Example
        -------
        >>> workflow.modify_reach(123, wse=45.5, reason="Corrected from field data")
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        # Find the index for this reach_id
        reach_idx = np.where(self._sword.reaches.id == reach_id)[0]
        if len(reach_idx) == 0:
            raise ValueError(f"Reach {reach_id} not found")
        reach_idx = reach_idx[0]

        # Get operation ID (from transaction or create new one)
        op_id = self._current_transaction_op_id

        if self._provenance and self._enable_provenance and op_id is None:
            with self._provenance.operation(
                'UPDATE',
                table_name='reaches',
                entity_ids=[reach_id],
                region=self._region,
                reason=reason,
                affected_columns=list(attributes.keys()),
            ) as op_id:
                self._apply_reach_modifications(reach_idx, reach_id, op_id, attributes)
        else:
            self._apply_reach_modifications(reach_idx, reach_id, op_id, attributes)

    def _apply_reach_modifications(
        self,
        reach_idx: int,
        reach_id: int,
        op_id: Optional[int],
        attributes: Dict[str, Any],
    ) -> None:
        """Apply modifications to a reach, logging value changes."""
        for attr_name, new_value in attributes.items():
            # Get the writable array for this attribute
            if not hasattr(self._sword.reaches, attr_name):
                raise ValueError(f"Unknown reach attribute: {attr_name}")

            arr = getattr(self._sword.reaches, attr_name)
            old_value = float(arr[reach_idx])

            # Log the change if provenance enabled
            if self._provenance and op_id is not None:
                self._provenance.log_value_change(
                    op_id, 'reaches', reach_id, attr_name, old_value, new_value
                )

            # Apply the change
            arr[reach_idx] = new_value

    def modify_node(
        self,
        node_id: int,
        reason: str = None,
        **attributes,
    ) -> None:
        """
        Modify node attributes with provenance logging.

        Parameters
        ----------
        node_id : int
            The node ID to modify
        reason : str, optional
            Why this modification is being made
        **attributes : dict
            Attribute name/value pairs to modify

        Example
        -------
        >>> workflow.modify_node(456, wse=42.0, width=150.0, reason="Updated from lidar")
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        # Find the index for this node_id
        node_idx = np.where(self._sword.nodes.id == node_id)[0]
        if len(node_idx) == 0:
            raise ValueError(f"Node {node_id} not found")
        node_idx = node_idx[0]

        op_id = self._current_transaction_op_id

        if self._provenance and self._enable_provenance and op_id is None:
            with self._provenance.operation(
                'UPDATE',
                table_name='nodes',
                entity_ids=[node_id],
                region=self._region,
                reason=reason,
                affected_columns=list(attributes.keys()),
            ) as op_id:
                self._apply_node_modifications(node_idx, node_id, op_id, attributes)
        else:
            self._apply_node_modifications(node_idx, node_id, op_id, attributes)

    def _apply_node_modifications(
        self,
        node_idx: int,
        node_id: int,
        op_id: Optional[int],
        attributes: Dict[str, Any],
    ) -> None:
        """Apply modifications to a node, logging value changes."""
        for attr_name, new_value in attributes.items():
            if not hasattr(self._sword.nodes, attr_name):
                raise ValueError(f"Unknown node attribute: {attr_name}")

            arr = getattr(self._sword.nodes, attr_name)
            old_value = float(arr[node_idx])

            if self._provenance and op_id is not None:
                self._provenance.log_value_change(
                    op_id, 'nodes', node_id, attr_name, old_value, new_value
                )

            arr[node_idx] = new_value

    def bulk_modify(
        self,
        entity_type: str,
        entity_ids: List[int],
        attributes: Dict[str, np.ndarray],
        reason: str = None,
    ) -> int:
        """
        Efficiently modify multiple entities at once.

        Parameters
        ----------
        entity_type : str
            Type of entity ('reach', 'node', 'centerline')
        entity_ids : list of int
            Entity IDs to modify
        attributes : dict
            Mapping of attribute name to array of new values
        reason : str, optional
            Why this modification is being made

        Returns
        -------
        int
            Number of entities modified

        Example
        -------
        >>> workflow.bulk_modify(
        ...     'reach',
        ...     reach_ids=[123, 456, 789],
        ...     attributes={'wse': np.array([45.0, 46.0, 47.0])},
        ...     reason="Batch elevation correction"
        ... )
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        # Get the view for this entity type
        view_map = {
            'reach': (self._sword.reaches, 'reaches'),
            'node': (self._sword.nodes, 'nodes'),
            'centerline': (self._sword.centerlines, 'centerlines'),
        }

        if entity_type not in view_map:
            raise ValueError(f"Unknown entity type: {entity_type}")

        view, table_name = view_map[entity_type]

        # Find indices for all entity IDs
        entity_ids_arr = np.array(entity_ids)
        indices = np.searchsorted(view.id, entity_ids_arr)

        # Verify all IDs exist
        valid = (indices < len(view.id)) & (view.id[indices] == entity_ids_arr)
        if not np.all(valid):
            missing = entity_ids_arr[~valid]
            raise ValueError(f"Entity IDs not found: {missing[:5]}...")

        op_id = self._current_transaction_op_id

        if self._provenance and self._enable_provenance and op_id is None:
            with self._provenance.operation(
                'UPDATE',
                table_name=table_name,
                entity_ids=list(entity_ids),
                region=self._region,
                reason=reason,
                affected_columns=list(attributes.keys()),
            ) as op_id:
                self._apply_bulk_modifications(
                    view, table_name, indices, entity_ids, op_id, attributes
                )
        else:
            self._apply_bulk_modifications(
                view, table_name, indices, entity_ids, op_id, attributes
            )

        return len(entity_ids)

    def _apply_bulk_modifications(
        self,
        view,
        table_name: str,
        indices: np.ndarray,
        entity_ids: List[int],
        op_id: Optional[int],
        attributes: Dict[str, np.ndarray],
    ) -> None:
        """Apply bulk modifications with optional provenance logging."""
        for attr_name, new_values in attributes.items():
            if not hasattr(view, attr_name):
                raise ValueError(f"Unknown attribute: {attr_name}")

            arr = getattr(view, attr_name)
            old_values = arr[indices].copy()

            # Log batch changes
            if self._provenance and op_id is not None:
                self._provenance.log_value_changes_batch(
                    op_id, table_name, entity_ids, attr_name,
                    list(old_values), list(new_values)
                )

            # Apply changes
            arr[indices] = new_values

    # =========================================================================
    # PROVENANCE QUERY METHODS
    # =========================================================================

    def get_history(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[int] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get operation history with optional filters.

        Parameters
        ----------
        entity_type : str, optional
            Filter by entity type ('reach', 'node', 'centerline')
        entity_id : int, optional
            Filter by specific entity ID
        since : datetime, optional
            Only return operations after this time
        limit : int, optional
            Maximum records to return

        Returns
        -------
        list of dict
            Operation history records

        Example
        -------
        >>> history = workflow.get_history(entity_type='reach', entity_id=123)
        >>> for op in history:
        ...     print(f"{op['started_at']}: {op['operation_type']} - {op['reason']}")
        """
        if not self._provenance:
            raise RuntimeError("Provenance tracking is not enabled")

        if entity_id is not None and entity_type is not None:
            return self._provenance.get_entity_history(
                entity_type, entity_id, limit=limit
            )
        else:
            # Map entity_type to table_name
            table_name = None
            if entity_type:
                table_map = {
                    'reach': 'reaches',
                    'node': 'nodes',
                    'centerline': 'centerlines',
                }
                table_name = table_map.get(entity_type, entity_type)

            return self._provenance.get_operation_history(
                since=since,
                table_name=table_name,
                region=self._region,
                limit=limit,
            )

    def get_lineage(
        self,
        entity_type: str,
        entity_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Get source data lineage for an entity.

        Shows which source datasets contributed to each attribute.

        Parameters
        ----------
        entity_type : str
            Type of entity ('reach', 'node', 'centerline')
        entity_id : int
            Entity ID

        Returns
        -------
        list of dict
            Lineage records showing source attribution
        """
        if not self._provenance:
            raise RuntimeError("Provenance tracking is not enabled")

        return self._provenance.get_lineage(entity_type, entity_id, self._region)

    # =========================================================================
    # SNAPSHOT VERSIONING METHODS
    # =========================================================================

    def snapshot(
        self,
        name: str,
        description: Optional[str] = None,
        auto: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a named snapshot of the current database state.

        Snapshots are like git tags - they mark a specific point in the
        operation history that you can later restore to.

        Parameters
        ----------
        name : str
            Unique snapshot name (alphanumeric, hyphens, underscores only).
            Maximum 100 characters.
        description : str, optional
            Human-readable description of the snapshot.
        auto : bool, optional
            If True, marks this as an auto-snapshot (for internal use).

        Returns
        -------
        dict
            Snapshot metadata including:
            - 'snapshot_id': int
            - 'name': str
            - 'operation_id_max': int - Reference point in operation history
            - 'created_at': datetime
            - 'reach_count', 'node_count', 'centerline_count': int

        Raises
        ------
        ValueError
            If snapshot name already exists or is invalid.
        RuntimeError
            If no database is loaded or provenance is disabled.

        Example
        -------
        >>> workflow.snapshot("before-bulk-edit", description="Clean state")
        >>> # ... make changes ...
        >>> workflow.restore_snapshot("before-bulk-edit")  # Revert changes
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")
        if not self._provenance or not self._enable_provenance:
            raise RuntimeError("Provenance tracking must be enabled for snapshots")

        # Validate name
        self._validate_snapshot_name(name)

        conn = self._sword.db.conn

        # Check uniqueness
        if self._snapshot_exists(name):
            raise ValueError(f"Snapshot '{name}' already exists")

        # Commit pending changes
        if self.has_pending_changes:
            self.commit()

        # Get current max operation_id
        max_op_id = self._provenance.get_max_operation_id()

        # Get entity counts
        reach_count = len(self._sword.reaches)
        node_count = len(self._sword.nodes)
        centerline_count = conn.execute(
            "SELECT COUNT(*) FROM centerlines WHERE region = ?", [self._region]
        ).fetchone()[0]

        # Get next snapshot ID
        result = conn.execute(
            "SELECT COALESCE(MAX(snapshot_id), 0) + 1 FROM sword_snapshots"
        ).fetchone()
        snapshot_id = result[0]

        # Insert snapshot
        conn.execute("""
            INSERT INTO sword_snapshots (
                snapshot_id, name, description, operation_id_max,
                created_by, session_id,
                reach_count, node_count, centerline_count,
                is_auto_snapshot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            snapshot_id, name, description, max_op_id,
            self._provenance.user_id, self._provenance.session_id,
            reach_count, node_count, centerline_count,
            auto
        ])

        # Log as operation
        with self._provenance.operation(
            'SNAPSHOT', None, None, self._region,
            reason=f"Created snapshot: {name}",
            details={'snapshot_name': name, 'snapshot_id': snapshot_id}
        ):
            pass

        created_at = datetime.now()
        logger.info(f"Created snapshot '{name}' at operation_id {max_op_id}")

        return {
            'snapshot_id': snapshot_id,
            'name': name,
            'operation_id_max': max_op_id,
            'created_at': created_at,
            'reach_count': reach_count,
            'node_count': node_count,
            'centerline_count': centerline_count,
        }

    def list_snapshots(
        self,
        include_auto: bool = False,
        limit: int = 100,
    ):
        """
        List all available snapshots.

        Parameters
        ----------
        include_auto : bool, optional
            If True, include auto-created snapshots. Default False.
        limit : int, optional
            Maximum number of snapshots to return.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns: name, created_at, description,
            operation_id_max, reach_count, node_count, centerline_count,
            created_by

        Raises
        ------
        RuntimeError
            If no database is loaded.

        Example
        -------
        >>> snapshots = workflow.list_snapshots()
        >>> print(snapshots[['name', 'created_at', 'description']])
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        import pandas as pd

        conn = self._sword.db.conn

        auto_filter = "" if include_auto else "WHERE is_auto_snapshot = FALSE"

        result = conn.execute(f"""
            SELECT
                name, created_at, description, operation_id_max,
                reach_count, node_count, centerline_count, created_by
            FROM sword_snapshots
            {auto_filter}
            ORDER BY created_at DESC
            LIMIT ?
        """, [limit]).fetchdf()

        return result

    def delete_snapshot(self, name: str) -> bool:
        """
        Delete a snapshot by name.

        Note: This only removes the snapshot metadata. It does NOT restore
        any data or affect the operation history.

        Parameters
        ----------
        name : str
            Name of the snapshot to delete.

        Returns
        -------
        bool
            True if snapshot was deleted, False if not found.

        Example
        -------
        >>> workflow.delete_snapshot("old-snapshot")
        True
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        conn = self._sword.db.conn

        # Check if exists
        if not self._snapshot_exists(name):
            return False

        conn.execute(
            "DELETE FROM sword_snapshots WHERE name = ?", [name]
        )

        logger.info(f"Deleted snapshot '{name}'")
        return True

    def restore_snapshot(
        self,
        name: str,
        dry_run: bool = False,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Restore the database to a named snapshot state.

        This rolls back all operations that occurred after the snapshot,
        in reverse chronological order.

        Parameters
        ----------
        name : str
            Name of the snapshot to restore to.
        dry_run : bool, optional
            If True, return what would be done without actually doing it.
        reason : str, optional
            Reason for the restore (logged to provenance).

        Returns
        -------
        dict
            Restore results including:
            - 'operations_rolled_back': int
            - 'values_restored': int
            - 'snapshot_name': str
            - 'restored_to_operation_id': int

        Raises
        ------
        ValueError
            If snapshot name doesn't exist.
        RuntimeError
            If database not loaded or provenance disabled.

        Example
        -------
        >>> # Preview what would be restored
        >>> result = workflow.restore_snapshot("before-edit", dry_run=True)
        >>> print(f"Would rollback {result['operations_to_rollback']} operations")
        >>>
        >>> # Actually restore
        >>> result = workflow.restore_snapshot("before-edit")
        >>> print(f"Rolled back {result['operations_rolled_back']} operations")
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")
        if not self._provenance:
            raise RuntimeError("Provenance tracking must be enabled for restore")

        conn = self._sword.db.conn

        # Look up snapshot
        result = conn.execute(
            "SELECT snapshot_id, operation_id_max, created_at FROM sword_snapshots WHERE name = ?",
            [name]
        ).fetchone()

        if not result:
            raise ValueError(f"Snapshot '{name}' not found")

        snapshot_id, target_op_id, snapshot_created = result

        # Delegate to internal method
        results = self._rollback_to_operation_id(
            target_op_id,
            reason=reason or f"Restore to snapshot: {name}",
            dry_run=dry_run
        )

        results['snapshot_name'] = name
        results['snapshot_created_at'] = snapshot_created
        return results

    def restore_to_timestamp(
        self,
        timestamp: Union[str, datetime],
        dry_run: bool = False,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Restore the database to its state at a specific timestamp.

        Parameters
        ----------
        timestamp : str or datetime
            The target timestamp. Strings are parsed as ISO format.
            Example: "2024-01-15 10:30:00"
        dry_run : bool, optional
            If True, return what would be done without actually doing it.
        reason : str, optional
            Reason for the restore (logged to provenance).

        Returns
        -------
        dict
            Restore results including:
            - 'operations_rolled_back': int
            - 'values_restored': int
            - 'restored_to_timestamp': datetime
            - 'target_operation_id': int

        Raises
        ------
        ValueError
            If timestamp is invalid or before first operation.
        RuntimeError
            If database not loaded or provenance disabled.

        Example
        -------
        >>> result = workflow.restore_to_timestamp("2024-01-15 10:30:00")
        >>> print(f"Rolled back {result['operations_rolled_back']} operations")
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")
        if not self._provenance:
            raise RuntimeError("Provenance tracking must be enabled for restore")

        # Parse timestamp if string
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        # Check if timestamp is in the future
        if timestamp > datetime.now():
            raise ValueError("Cannot restore to a future timestamp")

        conn = self._sword.db.conn

        # Find the last operation completed before timestamp
        result = conn.execute("""
            SELECT MAX(operation_id) FROM sword_operations
            WHERE completed_at <= ? AND status = 'COMPLETED'
        """, [timestamp]).fetchone()

        target_op_id = result[0] if result and result[0] else 0

        # Delegate to internal method
        results = self._rollback_to_operation_id(
            target_op_id,
            reason=reason or f"Restore to timestamp: {timestamp}",
            dry_run=dry_run
        )

        results['restored_to_timestamp'] = timestamp
        return results

    def _validate_snapshot_name(self, name: str) -> None:
        """Validate snapshot name format."""
        import re

        if not name:
            raise ValueError("Snapshot name cannot be empty")
        if len(name) > 100:
            raise ValueError("Snapshot name too long (max 100 characters)")
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            raise ValueError(
                "Snapshot name can only contain letters, numbers, hyphens, and underscores"
            )

    def _snapshot_exists(self, name: str) -> bool:
        """Check if a snapshot with this name exists."""
        conn = self._sword.db.conn
        result = conn.execute(
            "SELECT COUNT(*) FROM sword_snapshots WHERE name = ?", [name]
        ).fetchone()
        return result[0] > 0

    def _rollback_to_operation_id(
        self,
        target_op_id: int,
        reason: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Internal method to rollback all operations after a target operation.

        Used by both restore_snapshot and restore_to_timestamp.
        """
        # Get operations to rollback (excluding SNAPSHOT and RESTORE which are metadata operations)
        operations = self._provenance.get_operations_after(
            target_op_id,
            exclude_types=['SNAPSHOT', 'RESTORE']
        )

        if dry_run:
            return {
                'dry_run': True,
                'operations_to_rollback': len(operations),
                'operation_ids': [op['operation_id'] for op in operations],
                'restored_to_operation_id': target_op_id,
            }

        if len(operations) == 0:
            logger.info("Already at target state, nothing to rollback")
            return {
                'operations_rolled_back': 0,
                'values_restored': 0,
                'restored_to_operation_id': target_op_id,
            }

        # Track statistics
        total_values_restored = 0
        operations_rolled_back = 0

        # Log the RESTORE operation
        with self._provenance.operation(
            'RESTORE', None, None, self._region,
            reason=reason,
            details={
                'target_operation_id': target_op_id,
                'operations_to_rollback': len(operations)
            }
        ):
            # Rollback each operation in reverse order (already sorted by get_operations_after)
            for op in operations:
                op_id = op['operation_id']
                try:
                    values_restored = self._provenance.rollback_operation(op_id)
                    total_values_restored += values_restored
                    operations_rolled_back += 1
                except ValueError as e:
                    # Operation already rolled back or no snapshots
                    logger.warning(f"Could not rollback operation {op_id}: {e}")

        logger.info(
            f"Rolled back {operations_rolled_back} operations, "
            f"restored {total_values_restored} values"
        )

        return {
            'operations_rolled_back': operations_rolled_back,
            'values_restored': total_values_restored,
            'restored_to_operation_id': target_op_id,
        }

    # =========================================================================
    # RECONSTRUCTION METHODS
    # =========================================================================

    def reconstruct(
        self,
        attribute: str,
        entity_ids: Optional[List[int]] = None,
        force: bool = False,
        reason: str = None,
    ) -> Dict[str, Any]:
        """
        Reconstruct an attribute from source data.

        Parameters
        ----------
        attribute : str
            Full attribute name (e.g., "reach.wse", "node.facc", "reach.dist_out")
        entity_ids : list of int, optional
            Specific entities to reconstruct. If None, reconstructs all.
        force : bool, optional
            If True, reconstruct even if values exist
        reason : str, optional
            Reason for reconstruction (logged to provenance)

        Returns
        -------
        dict
            Reconstruction results including:
            - 'reconstructed': number of values reconstructed
            - 'entity_ids': list of affected entity IDs
            - 'attribute': the attribute name

        Example
        -------
        >>> result = workflow.reconstruct('reach.dist_out', reach_ids=[123, 456])
        >>> print(f"Reconstructed {result['reconstructed']} values")
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        if not self._reconstruction:
            raise RuntimeError("Reconstruction engine not initialized")

        return self._reconstruction.reconstruct(
            attribute,
            entity_ids=entity_ids,
            force=force,
            reason=reason,
        )

    def reconstruct_from_centerlines(
        self,
        attributes: Optional[List[str]] = None,
        reach_ids: Optional[List[int]] = None,
        reason: str = None,
    ) -> Dict[str, Any]:
        """
        Reconstruct derived attributes from centerline geometry.

        This reconstructs attributes that are computed from the centerline
        points (wse, slope, length, facc).

        Parameters
        ----------
        attributes : list of str, optional
            Attributes to reconstruct. Defaults to all centerline-derived.
        reach_ids : list of int, optional
            Specific reaches. If None, all reaches.
        reason : str, optional
            Reason for reconstruction

        Returns
        -------
        dict
            Results for each attribute reconstructed

        Example
        -------
        >>> results = workflow.reconstruct_from_centerlines(['reach.wse', 'reach.slope'])
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        if not self._reconstruction:
            raise RuntimeError("Reconstruction engine not initialized")

        return self._reconstruction.reconstruct_from_centerlines(
            attributes=attributes,
            reach_ids=reach_ids,
            reason=reason,
        )

    def validate_reconstruction(
        self,
        attribute: str,
        entity_ids: Optional[List[int]] = None,
        tolerance: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Validate reconstruction against existing values.

        Compares reconstructed values to current values to verify
        reconstruction accuracy.

        Parameters
        ----------
        attribute : str
            Attribute to validate
        entity_ids : list of int, optional
            Specific entities to validate
        tolerance : float, optional
            Relative tolerance for comparison (default 1%)

        Returns
        -------
        dict
            Validation report including:
            - 'passed': bool - whether validation passed
            - 'total': number of values compared
            - 'within_tolerance': number within tolerance
            - 'max_difference': maximum relative difference
            - 'failures': list of entity IDs that failed

        Example
        -------
        >>> report = workflow.validate_reconstruction('reach.wse', tolerance=0.01)
        >>> if report['passed']:
        ...     print("All values within 1% tolerance")
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        if not self._reconstruction:
            raise RuntimeError("Reconstruction engine not initialized")

        return self._reconstruction.validate(
            attribute,
            entity_ids=entity_ids,
            tolerance=tolerance,
        )

    def get_source_info(self, attribute: str) -> Optional[Dict[str, Any]]:
        """
        Get source information for an attribute.

        Shows which source dataset and derivation method was used
        to construct the attribute.

        Parameters
        ----------
        attribute : str
            Full attribute name (e.g., "reach.wse")

        Returns
        -------
        dict or None
            Source specification including:
            - 'source': source dataset name
            - 'method': derivation method
            - 'description': human-readable description

        Example
        -------
        >>> info = workflow.get_source_info('reach.wse')
        >>> print(f"Source: {info['source']}, Method: {info['method']}")
        """
        if not self._reconstruction:
            return None

        spec = self._reconstruction.get_source_info(attribute)
        if not spec:
            return None

        return {
            'attribute': spec.name,
            'source': spec.source.value,
            'method': spec.method.value,
            'source_columns': spec.source_columns,
            'dependencies': spec.dependencies,
            'description': spec.description,
        }

    def list_reconstructable_attributes(self) -> List[str]:
        """
        List all attributes that can be reconstructed.

        Returns
        -------
        list of str
            Attribute names that have reconstruction functions

        Example
        -------
        >>> attrs = workflow.list_reconstructable_attributes()
        >>> print(f"Can reconstruct: {attrs}")
        """
        if not self._reconstruction:
            return []

        return self._reconstruction.list_reconstructable_attributes()

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================

    def export(
        self,
        formats: List[str],
        output_dir: Union[str, Path],
        version: Optional[str] = None,
        overwrite: bool = False,
        reason: str = None,
    ) -> Dict[str, Path]:
        """
        Export the SWORD database to various formats.

        Parameters
        ----------
        formats : list of str
            Export formats. Supported: 'geopackage', 'geoparquet', 'postgres'
        output_dir : str or Path
            Directory to write exported files
        version : str, optional
            Version string for filenames (e.g., 'v18')
        overwrite : bool, optional
            If True, overwrite existing files
        reason : str, optional
            Reason for export (logged to provenance)

        Returns
        -------
        dict
            Mapping of format names to output file paths
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        if self.has_pending_changes:
            logger.warning(
                "Exporting with uncommitted changes. "
                "Consider calling commit() first."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        version_str = f"_{version}" if version else ""
        results = {}

        from . import export

        # Log export operation
        if self._provenance and self._enable_provenance:
            with self._provenance.operation(
                'EXPORT',
                table_name=None,
                entity_ids=None,
                region=self._region,
                reason=reason or f"Export to {formats}",
                details={'formats': formats, 'output_dir': str(output_dir)},
            ):
                results = self._do_export(
                    formats, output_dir, version_str, overwrite, export
                )
        else:
            results = self._do_export(
                formats, output_dir, version_str, overwrite, export
            )

        logger.info(f"Export complete: {list(results.keys())}")
        return results

    def _do_export(
        self,
        formats: List[str],
        output_dir: Path,
        version_str: str,
        overwrite: bool,
        export_module,
    ) -> Dict[str, Path]:
        """Perform the actual export operations."""
        results = {}

        for fmt in formats:
            fmt_lower = fmt.lower()

            if fmt_lower == 'geopackage':
                output_path = output_dir / f"sword_{self._region}{version_str}.gpkg"
                if output_path.exists() and not overwrite:
                    raise FileExistsError(
                        f"File exists: {output_path}. Use overwrite=True to replace."
                    )
                logger.info(f"Exporting to GeoPackage: {output_path}")
                export_module.export_to_geopackage(
                    self._sword.db,
                    str(output_path),
                    self._region,
                )
                results['geopackage'] = output_path

            elif fmt_lower == 'geoparquet':
                output_path = output_dir / f"sword_{self._region}{version_str}.parquet"
                if output_path.exists() and not overwrite:
                    raise FileExistsError(
                        f"File exists: {output_path}. Use overwrite=True to replace."
                    )
                logger.info(f"Exporting to GeoParquet: {output_path}")
                export_module.export_to_geoparquet(
                    self._sword.db,
                    str(output_path),
                    self._region,
                )
                results['geoparquet'] = output_path

            elif fmt_lower == 'postgres':
                logger.info("Exporting to PostgreSQL")
                export_module.export_to_postgres(self._sword.db, self._region)
                results['postgres'] = None

            else:
                raise ValueError(
                    f"Unsupported export format: {fmt}. "
                    f"Supported: geopackage, geoparquet, postgres"
                )

        return results

    # =========================================================================
    # REACH OPERATIONS (Priority 1 - QGIS Workflow)
    # =========================================================================

    def break_reach(
        self,
        reach_id: int,
        break_cl_id: int,
        reason: str = None,
    ) -> Dict[str, Any]:
        """
        Break a reach at a specified centerline point.

        This is a Priority 1 operation for the QGIS editing workflow. It splits
        a single reach into two reaches at the specified centerline ID.

        Parameters
        ----------
        reach_id : int
            The reach ID to break
        break_cl_id : int
            The centerline ID where the break should occur. This becomes
            the first centerline of the new downstream reach.
        reason : str, optional
            Reason for the break (logged to provenance)

        Returns
        -------
        dict
            Operation results including:
            - 'original_reach': int - The original reach ID
            - 'new_reach': int - The newly created reach ID
            - 'break_cl_id': int - Where the break occurred
            - 'success': bool - Whether operation succeeded

        Raises
        ------
        RuntimeError
            If no database is loaded
        ValueError
            If reach_id or break_cl_id is invalid

        Example
        -------
        >>> with workflow.transaction("Break reach at tributary"):
        ...     result = workflow.break_reach(72140300041, 10823001)
        ...     print(f"Created new reach: {result['new_reach']}")

        Notes
        -----
        Algorithm from legacy break_reaches_post_topo.py:
        1. Validate cl_id is within reach
        2. Split centerline arrays at cl_id
        3. Generate new reach ID (basin + max_rch + 1 + type)
        4. Generate new node IDs (divide by 200m)
        5. Recalculate attributes for both new reaches
        6. Update topology (neighbors now point to 2 reaches)
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        # Log operation if provenance enabled
        if self._provenance and self._enable_provenance:
            with self._provenance.operation(
                'BREAK_REACH',
                table_name='reaches',
                entity_ids=[reach_id],
                region=self._region,
                reason=reason or f"Break reach {reach_id} at cl_id {break_cl_id}",
                details={'break_cl_id': break_cl_id},
            ):
                self._sword.break_reaches(
                    np.array([reach_id]),
                    np.array([break_cl_id]),
                    verbose=True
                )
        else:
            self._sword.break_reaches(
                np.array([reach_id]),
                np.array([break_cl_id]),
                verbose=True
            )

        # Get the new reach ID (will be max reach ID in basin after break)
        # This assumes the new reach was just created
        return {
            'original_reach': reach_id,
            'break_cl_id': break_cl_id,
            'success': True,
        }

    def delete_reach(
        self,
        reach_id: int,
        cascade: bool = True,
        reason: str = None,
    ) -> Dict[str, Any]:
        """
        Delete a reach with optional cascade to centerlines and nodes.

        Parameters
        ----------
        reach_id : int
            The reach ID to delete
        cascade : bool, optional
            If True (default), delete associated centerlines and nodes.
            If False, only delete the reach record.
        reason : str, optional
            Reason for deletion (logged to provenance)

        Returns
        -------
        dict
            Operation results including:
            - 'deleted_reach': int - The deleted reach ID
            - 'cascade': bool - Whether cascade was used
            - 'success': bool - Whether operation succeeded

        Example
        -------
        >>> workflow.delete_reach(72140300041, reason="Duplicate reach")
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        if self._provenance and self._enable_provenance:
            with self._provenance.operation(
                'DELETE',
                table_name='reaches',
                entity_ids=[reach_id],
                region=self._region,
                reason=reason or f"Delete reach {reach_id}",
                details={'cascade': cascade},
            ):
                if cascade:
                    self._sword.delete_data([reach_id])
                else:
                    self._sword.delete_rchs([reach_id])
        else:
            if cascade:
                self._sword.delete_data([reach_id])
            else:
                self._sword.delete_rchs([reach_id])

        return {
            'deleted_reach': reach_id,
            'cascade': cascade,
            'success': True,
        }

    def delete_reaches(
        self,
        reach_ids: List[int],
        cascade: bool = True,
        reason: str = None,
    ) -> Dict[str, Any]:
        """
        Delete multiple reaches with optional cascade.

        Parameters
        ----------
        reach_ids : list of int
            The reach IDs to delete
        cascade : bool, optional
            If True (default), delete associated centerlines and nodes.
        reason : str, optional
            Reason for deletion (logged to provenance)

        Returns
        -------
        dict
            Operation results
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        if self._provenance and self._enable_provenance:
            with self._provenance.operation(
                'DELETE',
                table_name='reaches',
                entity_ids=list(reach_ids),
                region=self._region,
                reason=reason or f"Delete {len(reach_ids)} reaches",
                details={'cascade': cascade},
            ):
                if cascade:
                    self._sword.delete_data(reach_ids)
                else:
                    self._sword.delete_rchs(reach_ids)
        else:
            if cascade:
                self._sword.delete_data(reach_ids)
            else:
                self._sword.delete_rchs(reach_ids)

        return {
            'deleted_count': len(reach_ids),
            'cascade': cascade,
            'success': True,
        }

    def merge_reach(
        self,
        source_reach_id: int,
        target_reach_id: int,
        reason: str = None,
    ) -> Dict[str, Any]:
        """
        Merge a source reach into a target reach.

        This is a Priority 4 operation for reach aggregation. It combines two
        adjacent reaches by moving all centerlines and nodes from the source
        into the target, recalculating attributes, and deleting the source.

        Parameters
        ----------
        source_reach_id : int
            The reach ID to merge (will be deleted after merge)
        target_reach_id : int
            The reach ID to merge into (will be preserved and expanded)
        reason : str, optional
            Reason for the merge (logged to provenance)

        Returns
        -------
        dict
            Operation results including:
            - 'source_reach': int - The merged (deleted) reach ID
            - 'target_reach': int - The expanded reach ID
            - 'merged_nodes': int - Number of nodes merged
            - 'merged_centerlines': int - Number of centerlines merged
            - 'success': bool - Whether operation succeeded

        Raises
        ------
        RuntimeError
            If no database is loaded
        ValueError
            If reaches are not adjacent or cannot be merged

        Example
        -------
        >>> with workflow.transaction("Merge short reach"):
        ...     result = workflow.merge_reach(72140300061, 72140300041)
        ...     print(f"Merged {result['merged_nodes']} nodes")

        Notes
        -----
        Algorithm from legacy aggregate_1node_rchs.py:
        1. Validate source and target are topologically adjacent
        2. Reassign centerlines from source to target
        3. Update node IDs and reach assignments
        4. Recalculate reach attributes:
           - wse, wth: median of nodes
           - wse_var, wth_var: max of nodes
           - nchan_max, grod: max of nodes
           - nchan_mod, lakeflag: mode of nodes
           - slope: linear regression of wse vs dist_out
        5. Update topology (target inherits source's neighbors)
        6. Delete source reach
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        # Log operation if provenance enabled
        if self._provenance and self._enable_provenance:
            with self._provenance.operation(
                'MERGE_REACH',
                table_name='reaches',
                entity_ids=[source_reach_id, target_reach_id],
                region=self._region,
                reason=reason or f"Merge reach {source_reach_id} into {target_reach_id}",
                details={
                    'source_reach': source_reach_id,
                    'target_reach': target_reach_id
                },
            ):
                result = self._sword.merge_reaches(
                    source_reach_id,
                    target_reach_id,
                    verbose=True
                )
        else:
            result = self._sword.merge_reaches(
                source_reach_id,
                target_reach_id,
                verbose=True
            )

        return result

    def merge_reaches(
        self,
        merge_pairs: List[tuple],
        reason: str = None,
    ) -> Dict[str, Any]:
        """
        Merge multiple reach pairs.

        Parameters
        ----------
        merge_pairs : list of tuple
            List of (source_reach_id, target_reach_id) tuples to merge
        reason : str, optional
            Reason for the merges (logged to provenance)

        Returns
        -------
        dict
            Operation results including total counts

        Example
        -------
        >>> pairs = [(72140300061, 72140300041), (72140300071, 72140300051)]
        >>> result = workflow.merge_reaches(pairs, reason="Aggregate short reaches")
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        total_nodes = 0
        total_centerlines = 0
        merged_count = 0

        for source_id, target_id in merge_pairs:
            try:
                result = self.merge_reach(
                    source_id, target_id,
                    reason=reason or f"Batch merge: {source_id} -> {target_id}"
                )
                total_nodes += result.get('merged_nodes', 0)
                total_centerlines += result.get('merged_centerlines', 0)
                merged_count += 1
            except ValueError as e:
                logger.warning(f"Could not merge {source_id} -> {target_id}: {e}")
                continue

        return {
            'merged_count': merged_count,
            'total_nodes': total_nodes,
            'total_centerlines': total_centerlines,
            'success': merged_count > 0,
        }

    def append_reaches(
        self,
        centerlines,
        nodes,
        reaches,
        validate_ids: bool = True,
        reason: str = None,
    ) -> Dict[str, Any]:
        """
        Append new reaches with their centerlines and nodes.

        Parameters
        ----------
        centerlines : object
            Centerlines data object with cl_id, x, y, reach_id, node_id
        nodes : object
            Nodes data object matching NodesView structure
        reaches : object
            Reaches data object matching ReachesView structure
        validate_ids : bool, optional
            If True (default), validate ID formats before appending.
        reason : str, optional
            Reason for append (logged to provenance)

        Returns
        -------
        dict
            Operation results including counts of appended entities

        Raises
        ------
        ValueError
            If validate_ids is True and IDs fail validation

        Notes
        -----
        ID Formats:
        - Reach ID: CBBBBBRRRRT (11 digits)
        - Node ID: CBBBBBRRRRNNNT (14 digits)
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        reach_ids = list(reaches.id) if hasattr(reaches, 'id') else []

        if self._provenance and self._enable_provenance:
            with self._provenance.operation(
                'INSERT',
                table_name='reaches',
                entity_ids=reach_ids,
                region=self._region,
                reason=reason or f"Append {len(reach_ids)} new reaches",
            ):
                self._sword.append_data(
                    centerlines, nodes, reaches,
                    validate_ids=validate_ids
                )
        else:
            self._sword.append_data(
                centerlines, nodes, reaches,
                validate_ids=validate_ids
            )

        return {
            'appended_centerlines': len(centerlines.cl_id) if hasattr(centerlines, 'cl_id') else 0,
            'appended_nodes': len(nodes.id) if hasattr(nodes, 'id') else 0,
            'appended_reaches': len(reach_ids),
            'success': True,
        }

    # =========================================================================
    # GHOST REACH OPERATIONS (Priority 3)
    # =========================================================================

    def create_ghost_reach(
        self,
        reach_id: int,
        position: str = 'auto',
        reason: str = None,
    ) -> Dict[str, Any]:
        """
        Create a ghost reach at the headwater or outlet of an existing reach.

        Ghost reaches (type 6) are placeholder reaches used to mark network
        endpoints. This method creates a new ghost reach by extracting the
        first or last node from an existing reach and assigning it to a new
        ghost reach with proper SWORD ID format.

        Parameters
        ----------
        reach_id : int
            The reach ID to split a ghost reach from.
        position : str, optional
            Where to create the ghost reach:
            - 'headwater': Create at upstream end (takes last node by ID)
            - 'outlet': Create at downstream end (takes first node by ID)
            - 'auto': Automatically determine based on topology (default)
        reason : str, optional
            Reason for creating the ghost reach (logged to provenance)

        Returns
        -------
        dict
            Operation results including:
            - 'success': bool - Whether operation succeeded
            - 'original_reach': int - The original reach ID
            - 'ghost_reach_id': int - The new ghost reach ID
            - 'ghost_node_id': int - The new ghost node ID
            - 'position': str - Where the ghost was created

        Raises
        ------
        RuntimeError
            If no database is loaded
        ValueError
            If position='auto' but reach has both up and down neighbors,
            or if the reach has only one node and can't be split.

        Example
        -------
        >>> # Create ghost at headwater of a reach with no upstream
        >>> result = workflow.create_ghost_reach(72140300041)
        >>> print(f"Created ghost reach: {result['ghost_reach_id']}")
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        if self._provenance and self._enable_provenance:
            with self._provenance.operation(
                'CREATE_GHOST_REACH',
                table_name='reaches',
                entity_ids=[reach_id],
                region=self._region,
                reason=reason or f"Create ghost reach at {position} of {reach_id}",
                details={'position': position},
            ):
                return self._sword.create_ghost_reach(
                    reach_id,
                    position=position,
                    verbose=True
                )
        else:
            return self._sword.create_ghost_reach(
                reach_id,
                position=position,
                verbose=True
            )

    def find_missing_ghost_reaches(self) -> Dict[str, Any]:
        """
        Find reaches that should have ghost reaches but don't.

        Ghost reaches are typically needed at:
        - Headwaters: Non-ghost reaches (type != 6) with no upstream neighbors
        - Outlets: Non-ghost reaches (type != 6) with no downstream neighbors

        Returns
        -------
        dict
            Dictionary containing:
            - 'missing_headwaters': list of reach IDs needing headwater ghosts
            - 'missing_outlets': list of reach IDs needing outlet ghosts
            - 'total_missing': int total count

        Raises
        ------
        RuntimeError
            If no database is loaded

        Example
        -------
        >>> missing = workflow.find_missing_ghost_reaches()
        >>> print(f"Found {missing['total_missing']} reaches needing ghost reaches")
        >>> for rid in missing['missing_headwaters'][:5]:
        ...     workflow.create_ghost_reach(rid, position='headwater')
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        return self._sword.find_missing_ghost_reaches()

    def find_incorrect_ghost_reaches(self) -> Dict[str, Any]:
        """
        Find ghost reaches that are incorrectly labeled.

        A ghost reach (type 6) should only have neighbors in ONE direction
        (either upstream OR downstream, not both). Ghost reaches with both
        are likely mislabeled and should be a different type.

        Returns
        -------
        dict
            Dictionary containing:
            - 'incorrect_ghost_reaches': list of dicts with reach_id and suggested_type
            - 'total_incorrect': int

        Raises
        ------
        RuntimeError
            If no database is loaded

        Example
        -------
        >>> incorrect = workflow.find_incorrect_ghost_reaches()
        >>> for item in incorrect['incorrect_ghost_reaches']:
        ...     print(f"Reach {item['reach_id']} should be type {item['suggested_type']}")
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        return self._sword.find_incorrect_ghost_reaches()

    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================

    def check_topology(
        self,
        verbose: int = 1,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """
        Check topological consistency of the loaded database.

        Parameters
        ----------
        verbose : int, optional
            Output verbosity (0=silent, 1=errors, 2=all)
        return_details : bool, optional
            If True, include detailed error messages

        Returns
        -------
        dict
            Topology check results

        Example
        -------
        >>> results = workflow.check_topology()
        >>> if not results['passed']:
        ...     print(f"Issues in {len(results['reaches_with_issues'])} reaches")
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        return self._sword.check_topo_consistency(
            verbose=verbose,
            return_details=return_details
        )

    def check_node_lengths(
        self,
        verbose: int = 1,
        threshold: float = 1000.0,
    ) -> Dict[str, Any]:
        """
        Check for abnormal node lengths.

        Parameters
        ----------
        verbose : int, optional
            Output verbosity
        threshold : float, optional
            Length threshold in meters (default 1000m)

        Returns
        -------
        dict
            Node length check results
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        return self._sword.check_node_lengths(
            verbose=verbose,
            long_threshold=threshold
        )

    def validate_ids(
        self,
        reach_ids: List[int] = None,
        node_ids: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Validate that IDs follow SWORD format conventions.

        Parameters
        ----------
        reach_ids : list of int, optional
            Reach IDs to validate. If None, validates all loaded reaches.
        node_ids : list of int, optional
            Node IDs to validate. If None, validates all loaded nodes.

        Returns
        -------
        dict
            Validation results including invalid IDs
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        invalid_reaches = []
        invalid_nodes = []

        # Validate reach IDs
        rids = reach_ids if reach_ids is not None else self._sword.reaches.id
        for rid in rids:
            if not self._sword.validate_reach_id(int(rid)):
                invalid_reaches.append(rid)

        # Validate node IDs
        nids = node_ids if node_ids is not None else self._sword.nodes.id
        for nid in nids:
            if not self._sword.validate_node_id(int(nid)):
                invalid_nodes.append(nid)

        return {
            'passed': len(invalid_reaches) == 0 and len(invalid_nodes) == 0,
            'total_reaches_checked': len(rids),
            'total_nodes_checked': len(nids),
            'invalid_reaches': invalid_reaches,
            'invalid_nodes': invalid_nodes,
        }

    # =========================================================================
    # NETWORK ANALYSIS (Priority 5)
    # =========================================================================

    def calculate_dist_out(
        self,
        update_nodes: bool = True,
        reason: str = None,
    ) -> Dict[str, Any]:
        """
        Calculate distance from outlet (dist_out) using topology BFS.

        This is a Priority 5 operation for network analysis. It traverses the
        river network from outlets upstream, computing cumulative reach lengths
        as the distance from the nearest outlet.

        Parameters
        ----------
        update_nodes : bool, optional
            If True (default), also update node-level dist_out values.
        reason : str, optional
            Reason for the recalculation (logged to provenance)

        Returns
        -------
        dict
            Operation results including:
            - 'success': bool - Whether all reaches were computed
            - 'reaches_updated': int - Number of reaches updated
            - 'nodes_updated': int - Number of nodes updated
            - 'outlets_found': int - Number of outlet reaches
            - 'unfilled_reaches': list - Reaches that couldn't be computed

        Example
        -------
        >>> result = workflow.calculate_dist_out()
        >>> print(f"Updated {result['reaches_updated']} reaches")

        Notes
        -----
        Algorithm from legacy dist_out_from_topo.py:
        - For outlets (n_rch_down == 0): dist_out = reach_length
        - For non-outlets: dist_out = reach_length + max(downstream dist_out)
        - Node dist_out: cumsum(node_lengths) + (reach_dist_out - reach_length)
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        # Log operation if provenance enabled
        if self._provenance and self._enable_provenance:
            with self._provenance.operation(
                'CALCULATE_DIST_OUT',
                table_name='reaches',
                entity_ids=[],  # Affects all reaches
                region=self._region,
                reason=reason or "Recalculate dist_out from topology",
                details={'update_nodes': update_nodes},
            ):
                result = self._sword.calculate_dist_out_from_topology(
                    update_nodes=update_nodes,
                    verbose=True
                )
        else:
            result = self._sword.calculate_dist_out_from_topology(
                update_nodes=update_nodes,
                verbose=True
            )

        return result

    def recalculate_network_attributes(
        self,
        attributes: List[str] = None,
        reason: str = None,
    ) -> Dict[str, Any]:
        """
        Recalculate network-derived attributes.

        Parameters
        ----------
        attributes : list of str, optional
            Specific attributes to recalculate. If None, recalculates all:
            - 'dist_out': Distance from outlet (reaches and nodes)
            Default is ['dist_out'].
        reason : str, optional
            Reason for recalculation (logged to provenance)

        Returns
        -------
        dict
            Combined results from all recalculations

        Example
        -------
        >>> result = workflow.recalculate_network_attributes(['dist_out'])
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        if attributes is None:
            attributes = ['dist_out']

        results = {
            'attributes_requested': attributes,
            'attributes_updated': [],
            'success': True,
        }

        for attr in attributes:
            if attr == 'dist_out':
                dist_result = self.calculate_dist_out(
                    update_nodes=True,
                    reason=reason or f"Recalculate {attr}"
                )
                results['dist_out'] = dist_result
                if dist_result['success']:
                    results['attributes_updated'].append('dist_out')
                else:
                    results['success'] = False
            elif attr == 'stream_order':
                so_result = self.recalculate_stream_order(
                    reason=reason or f"Recalculate {attr}"
                )
                results['stream_order'] = so_result
                results['attributes_updated'].append('stream_order')
            elif attr == 'path_segs':
                ps_result = self.recalculate_path_segs(
                    reason=reason or f"Recalculate {attr}"
                )
                results['path_segs'] = ps_result
                results['attributes_updated'].append('path_segs')
            else:
                logger.warning(f"Unknown attribute '{attr}' - skipping")

        return results

    def recalculate_stream_order(
        self,
        update_nodes: bool = True,
        update_reaches: bool = True,
        reason: str = None,
    ) -> Dict[str, Any]:
        """
        Recalculate stream_order from path_freq using the legacy formula.

        Stream order is calculated as: round(ln(path_freq)) + 1
        This provides a log-based stream ordering where higher values indicate
        larger/more connected streams.

        Parameters
        ----------
        update_nodes : bool, optional
            If True (default), update stream_order in nodes table.
        update_reaches : bool, optional
            If True (default), update stream_order in reaches table (mode of node values).
        reason : str, optional
            Reason for the recalculation (logged to provenance).

        Returns
        -------
        dict
            Operation results including nodes_updated, reaches_updated, etc.

        Example
        -------
        >>> result = workflow.recalculate_stream_order()
        >>> print(f"Updated {result['nodes_updated']} nodes")

        Notes
        -----
        Algorithm from legacy stream_order.py:
        - For nodes with path_freq > 0: stream_order = round(ln(path_freq)) + 1
        - For nodes with path_freq <= 0: stream_order = -9999 (nodata)
        - Reach values are the mode of node values
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        # Log operation if provenance enabled
        if self._provenance and self._enable_provenance:
            with self._provenance.operation(
                'RECALCULATE_STREAM_ORDER',
                table_name='nodes',
                entity_ids=[],  # Affects all nodes
                region=self._region,
                reason=reason or "Recalculate stream_order from path_freq",
                details={'update_nodes': update_nodes, 'update_reaches': update_reaches},
            ):
                return self._sword.recalculate_stream_order(
                    update_nodes=update_nodes,
                    update_reaches=update_reaches,
                    verbose=True
                )
        else:
            return self._sword.recalculate_stream_order(
                update_nodes=update_nodes,
                update_reaches=update_reaches,
                verbose=True
            )

    def recalculate_path_segs(
        self,
        update_nodes: bool = True,
        update_reaches: bool = True,
        reason: str = None,
    ) -> Dict[str, Any]:
        """
        Recalculate path_segs (path segments) from path_order and path_freq.

        Path segments are unique IDs assigned to river segments between junctions.
        Each unique combination of (path_order, path_freq) gets a unique segment ID.

        Parameters
        ----------
        update_nodes : bool, optional
            If True (default), update path_segs in nodes table.
        update_reaches : bool, optional
            If True (default), update path_segs in reaches table (mode of node values).
        reason : str, optional
            Reason for the recalculation (logged to provenance).

        Returns
        -------
        dict
            Operation results including nodes_updated, reaches_updated, total_segments.

        Example
        -------
        >>> result = workflow.recalculate_path_segs()
        >>> print(f"Created {result['total_segments']} unique segments")

        Notes
        -----
        Algorithm from legacy stream_order.py find_path_segs function:
        1. For each unique path_order value (sorted):
           - Find all nodes with that path_order
           - For each unique path_freq value within those nodes:
             - Assign a unique segment ID (incrementing counter)
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        # Log operation if provenance enabled
        if self._provenance and self._enable_provenance:
            with self._provenance.operation(
                'RECALCULATE_PATH_SEGS',
                table_name='nodes',
                entity_ids=[],  # Affects all nodes
                region=self._region,
                reason=reason or "Recalculate path_segs from path_order and path_freq",
                details={'update_nodes': update_nodes, 'update_reaches': update_reaches},
            ):
                return self._sword.recalculate_path_segs(
                    update_nodes=update_nodes,
                    update_reaches=update_reaches,
                    verbose=True
                )
        else:
            return self._sword.recalculate_path_segs(
                update_nodes=update_nodes,
                update_reaches=update_reaches,
                verbose=True
            )

    def recalculate_sinuosity(
        self,
        reach_ids: Optional[List[int]] = None,
        min_reach_len_factor: float = 1.0,
        smoothing_span: int = 5,
        reason: str = None,
    ) -> Dict[str, Any]:
        """
        Recalculate sinuosity from centerline geometry using the legacy MATLAB algorithm.

        Sinuosity measures how winding a river is, calculated as the ratio of
        the actual channel length to the straight-line distance between endpoints.

        Parameters
        ----------
        reach_ids : list of int, optional
            Specific reaches to recalculate. If None, recalculates all reaches.
        min_reach_len_factor : float, optional
            Multiplier for minimum reach length based on width. Default 1.0.
        smoothing_span : int, optional
            Number of points for moving average smoothing. Default 5.
        reason : str, optional
            Reason for the recalculation (logged to provenance).

        Returns
        -------
        dict
            Operation results including:
            - 'reaches_processed': int - Number of reaches processed
            - 'reaches_updated': int - Number of reaches with changed sinuosity
            - 'mean_sinuosity': float - Mean sinuosity across all reaches
            - 'reach_sinuosities': dict - {reach_id: sinuosity} for each reach

        Example
        -------
        >>> result = workflow.recalculate_sinuosity()
        >>> print(f"Mean sinuosity: {result['mean_sinuosity']:.2f}")

        Notes
        -----
        Algorithm (ported from legacy MATLAB code):
        1. Project centerline coordinates to UTM for accurate distance measurements
        2. Smooth coordinates with moving average to remove pixel-level noise
        3. Detect inflection points (where river curvature changes direction)
        4. Merge short reaches based on similarity to neighbors
        5. Calculate sinuosity as arc_length / straight_line_distance for each bend

        Sinuosity interpretation:
        - 1.0 = perfectly straight
        - 1.0-1.5 = nearly straight
        - 1.5-2.0 = sinuous
        - >2.0 = highly sinuous/meandering
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        # Log operation if provenance enabled
        if self._provenance and self._enable_provenance:
            with self._provenance.operation(
                'RECALCULATE_SINUOSITY',
                table_name='reaches',
                entity_ids=reach_ids or [],  # Affects specified or all reaches
                region=self._region,
                reason=reason or "Recalculate sinuosity from centerline geometry",
                details={
                    'reach_ids': reach_ids,
                    'min_reach_len_factor': min_reach_len_factor,
                    'smoothing_span': smoothing_span,
                },
            ):
                return self._sword.recalculate_sinuosity(
                    reach_ids=reach_ids,
                    update_database=True,
                    min_reach_len_factor=min_reach_len_factor,
                    smoothing_span=smoothing_span,
                    verbose=True
                )
        else:
            return self._sword.recalculate_sinuosity(
                reach_ids=reach_ids,
                update_database=True,
                min_reach_len_factor=min_reach_len_factor,
                smoothing_span=smoothing_span,
                verbose=True
            )

    def recalculate_trib_flag(
        self,
        mhv_data_dir: str,
        distance_threshold: float = 0.003,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Recalculate trib_flag for nodes and reaches using MERIT Hydro Vector data.

        trib_flag marks nodes/reaches that are near MHV river segments that are
        NOT already in SWORD (i.e., smaller tributaries). This helps identify
        locations where smaller streams join SWORD rivers.

        Parameters
        ----------
        mhv_data_dir : str
            Path to MHV_SWORD data directory containing gpkg/{region}/ subfolders.
            Each subfolder should have files like: mhv_sword_hb71_pts_v18.gpkg
        distance_threshold : float, optional
            Maximum distance in degrees to consider a match. Default 0.003 (~333m).
        reason : str, optional
            Reason for recalculation (logged to provenance).

        Returns
        -------
        dict
            Results including:
            - 'nodes_flagged': int - Number of nodes with trib_flag=1
            - 'reaches_flagged': int - Number of reaches with trib_flag=1
            - 'mhv_files_processed': int - Number of MHV files used
            - 'total_mhv_points': int - Total MHV points checked

        Raises
        ------
        RuntimeError
            If no database is loaded.
        FileNotFoundError
            If MHV data directory doesn't exist.

        Example
        -------
        >>> result = workflow.recalculate_trib_flag(
        ...     mhv_data_dir='/Volumes/SWORD_DATA/data/MHV_SWORD',
        ...     reason="Update tributary flags from MHV v18"
        ... )
        >>> print(f"Flagged {result['nodes_flagged']} nodes as tributaries")

        Notes
        -----
        Algorithm from legacy Add_Trib_Flag.py:
        1. Load MHV points filtered by sword_flag=0 (not in SWORD)
        2. Build KD-tree from MHV coordinates
        3. Find SWORD nodes within distance_threshold
        4. Flag matching nodes and their reaches with trib_flag=1
        """
        import glob
        from pathlib import Path
        from scipy import spatial as sp
        import numpy as np
        import geopandas as gpd

        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        mhv_path = Path(mhv_data_dir)
        region_path = mhv_path / 'gpkg' / self._region
        if not region_path.exists():
            raise FileNotFoundError(
                f"MHV data not found for region {self._region}: {region_path}"
            )

        # Get all gpkg files for this region (exclude macOS ._ metadata files)
        all_files = sorted(glob.glob(str(region_path / '*.gpkg')))
        gpkg_files = [f for f in all_files if not Path(f).name.startswith('._')]
        if not gpkg_files:
            raise FileNotFoundError(f"No gpkg files found in {region_path}")

        # Build mapping from basin code (hbXX) to file path
        basin_to_file = {}
        for f in gpkg_files:
            # Extract basin code from filename like "mhv_sword_hb71_pts_v18.gpkg"
            name = Path(f).stem
            if '_hb' in name:
                # Extract the 2-digit basin code after 'hb'
                idx = name.index('_hb') + 3
                basin_code = name[idx:idx+2]
                basin_to_file[basin_code] = f

        logger.info(f"Found {len(gpkg_files)} MHV files for region {self._region}")
        logger.info(f"Basin codes available: {sorted(basin_to_file.keys())}")

        # Get SWORD node coordinates
        conn = self._sword.db.conn
        nodes_df = conn.execute("""
            SELECT node_id, reach_id, x, y
            FROM nodes
            WHERE region = ?
        """, [self._region]).fetchdf()

        sword_x = nodes_df['x'].values
        sword_y = nodes_df['y'].values
        sword_nid = nodes_df['node_id'].values
        sword_nrid = nodes_df['reach_id'].values

        # Get reach IDs
        reach_ids = conn.execute("""
            SELECT reach_id FROM reaches WHERE region = ?
        """, [self._region]).fetchdf()['reach_id'].values

        # Compute level-2 basin code for each node (first 2 digits of node_id)
        sword_l2 = np.array([str(nid)[:2] for nid in sword_nid])
        unique_basins = np.unique(sword_l2)

        # Initialize flags
        node_tribs = np.zeros(len(sword_nid), dtype=int)
        rch_tribs = np.zeros(len(reach_ids), dtype=int)

        total_mhv_points = 0
        files_processed = 0

        # Process basin-by-basin (matching legacy algorithm)
        for basin_code in unique_basins:
            if basin_code not in basin_to_file:
                logger.warning(f"No MHV file for basin {basin_code}, skipping")
                continue

            gpkg_file = basin_to_file[basin_code]
            logger.info(f"Processing basin {basin_code}: {Path(gpkg_file).name}")

            # Get indices of nodes in this basin
            basin_mask = sword_l2 == basin_code
            basin_indices = np.where(basin_mask)[0]
            if len(basin_indices) == 0:
                continue

            swd_x = sword_x[basin_indices]
            swd_y = sword_y[basin_indices]
            swd_nid = sword_nid[basin_indices]
            swd_nrid = sword_nrid[basin_indices]

            try:
                mhv = gpd.read_file(gpkg_file)
            except Exception as e:
                logger.warning(f"Could not read {gpkg_file}: {e}")
                continue

            # Filter: sword_flag=0 AND strmorder>=3 (matching legacy exactly)
            subset = mhv[(mhv['sword_flag'] == 0) & (mhv['strmorder'] >= 3)]
            if len(subset) == 0:
                logger.debug(f"No unmatched MHV points in {Path(gpkg_file).name}")
                files_processed += 1
                continue

            mhv_x = subset['x'].values
            mhv_y = subset['y'].values
            total_mhv_points += len(subset)

            # Build KD-tree from MHV points
            mhv_pts = np.vstack((mhv_x, mhv_y)).T
            node_pts = np.vstack((swd_x, swd_y)).T
            kdt = sp.cKDTree(mhv_pts)

            # Query: find closest MHV point to each SWORD node in this basin
            # Note: k=10 matches legacy Add_Trib_Flag.py algorithm for consistency
            distances, _ = kdt.query(node_pts, k=10)
            distances = distances[:, 0]  # Use first (closest) neighbor

            # Flag nodes within threshold
            matches = distances <= distance_threshold
            matching_local_indices = np.where(matches)[0]

            if len(matching_local_indices) > 0:
                # Get the global node IDs that matched
                fg_nodes = np.unique(swd_nid[matching_local_indices])
                fg_rchs = np.unique(swd_nrid[matching_local_indices])

                # Find indices in global arrays and flag them
                node_match_mask = np.isin(sword_nid, fg_nodes)
                node_tribs[node_match_mask] = 1

                reach_match_mask = np.isin(reach_ids, fg_rchs)
                rch_tribs[reach_match_mask] = 1

            files_processed += 1

        nodes_flagged = int(np.sum(node_tribs))
        reaches_flagged = int(np.sum(rch_tribs))

        logger.info(
            f"Flagging {nodes_flagged} nodes and {reaches_flagged} reaches "
            f"as tributaries from {total_mhv_points} MHV points"
        )

        # Update database
        if self._provenance and self._enable_provenance:
            with self._provenance.operation(
                'RECALCULATE',
                'nodes',
                entity_ids=[],
                region=self._region,
                reason=reason or "Recalculate trib_flag from MHV data",
                details={
                    'mhv_data_dir': str(mhv_data_dir),
                    'distance_threshold': distance_threshold,
                    'nodes_flagged': nodes_flagged,
                    'reaches_flagged': reaches_flagged,
                },
            ):
                self._update_trib_flag_in_db(
                    sword_nid, node_tribs, reach_ids, rch_tribs
                )
        else:
            self._update_trib_flag_in_db(
                sword_nid, node_tribs, reach_ids, rch_tribs
            )

        return {
            'nodes_flagged': nodes_flagged,
            'reaches_flagged': reaches_flagged,
            'mhv_files_processed': files_processed,
            'total_mhv_points': total_mhv_points,
        }

    def _update_trib_flag_in_db(
        self,
        node_ids: 'np.ndarray',
        node_flags: 'np.ndarray',
        reach_ids: 'np.ndarray',
        reach_flags: 'np.ndarray',
    ) -> None:
        """Update trib_flag values in database for nodes and reaches."""
        import numpy as np

        conn = self._sword.db.conn

        # Update nodes
        flagged_node_ids = node_ids[node_flags == 1]
        unflagged_node_ids = node_ids[node_flags == 0]

        if len(flagged_node_ids) > 0:
            # Batch update for flagged nodes
            for batch_start in range(0, len(flagged_node_ids), 1000):
                batch = flagged_node_ids[batch_start:batch_start + 1000]
                # Convert numpy int64 to Python int for DuckDB
                batch_list = [int(x) for x in batch]
                placeholders = ','.join(['?'] * len(batch_list))
                conn.execute(f"""
                    UPDATE nodes SET trib_flag = 1
                    WHERE node_id IN ({placeholders}) AND region = ?
                """, batch_list + [self._region])

        if len(unflagged_node_ids) > 0:
            # Batch update for unflagged nodes
            for batch_start in range(0, len(unflagged_node_ids), 1000):
                batch = unflagged_node_ids[batch_start:batch_start + 1000]
                batch_list = [int(x) for x in batch]
                placeholders = ','.join(['?'] * len(batch_list))
                conn.execute(f"""
                    UPDATE nodes SET trib_flag = 0
                    WHERE node_id IN ({placeholders}) AND region = ?
                """, batch_list + [self._region])

        # Update reaches
        flagged_reach_ids = reach_ids[reach_flags == 1]
        unflagged_reach_ids = reach_ids[reach_flags == 0]

        if len(flagged_reach_ids) > 0:
            for batch_start in range(0, len(flagged_reach_ids), 1000):
                batch = flagged_reach_ids[batch_start:batch_start + 1000]
                batch_list = [int(x) for x in batch]
                placeholders = ','.join(['?'] * len(batch_list))
                conn.execute(f"""
                    UPDATE reaches SET trib_flag = 1
                    WHERE reach_id IN ({placeholders}) AND region = ?
                """, batch_list + [self._region])

        if len(unflagged_reach_ids) > 0:
            for batch_start in range(0, len(unflagged_reach_ids), 1000):
                batch = unflagged_reach_ids[batch_start:batch_start + 1000]
                batch_list = [int(x) for x in batch]
                placeholders = ','.join(['?'] * len(batch_list))
                conn.execute(f"""
                    UPDATE reaches SET trib_flag = 0
                    WHERE reach_id IN ({placeholders}) AND region = ?
                """, batch_list + [self._region])

        logger.info("Updated trib_flag in database")

    def recalculate_facc(
        self,
        merit_hydro_dir: str,
        facc_threshold: float = 0.01,
        k_neighbors: int = 20,
        dry_run: bool = False,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Recalculate flow accumulation (facc) from MERIT Hydro rasters.

        Improved algorithm:
        1. Extract facc from MERIT using KD-tree (median of k nearest river pixels)
        2. Validate using network topology (facc must increase downstream)
        3. Smooth within reaches for monotonic progression
        4. Sanity checks for unreasonable jumps

        Args:
            merit_hydro_dir: Path to MERIT Hydro data directory containing
                             region subfolders with upa rasters.
            facc_threshold: Minimum facc value to consider as river pixel.
                           Defaults to 0.01 (very permissive to catch small tribs).
            k_neighbors: Number of nearest MERIT points to use. Defaults to 20.
            dry_run: If True, return comparison without updating database.
            reason: Optional reason for the recalculation.

        Returns:
            Dict with extraction stats, validation results, and update counts.

        Raises:
            RuntimeError: If no database is loaded.
            FileNotFoundError: If MERIT Hydro data directory not found.
        """
        from pathlib import Path
        import numpy as np
        import rasterio
        from rasterio.errors import RasterioIOError
        from scipy import spatial as sp

        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        mh_path = Path(merit_hydro_dir)
        upa_path = mh_path / self._region / 'upa'
        if not upa_path.exists():
            raise FileNotFoundError(
                f"MERIT Hydro upa data not found for region {self._region}: {upa_path}"
            )

        logger.info(f"Recalculating facc from MERIT Hydro data at {upa_path}")

        # Get node coordinates and current facc
        conn = self._sword.db.conn
        nodes_df = conn.execute("""
            SELECT node_id, reach_id, x, y, facc, wse
            FROM nodes
            WHERE region = ?
            ORDER BY reach_id, node_id
        """, [self._region]).fetchdf()

        node_ids = nodes_df['node_id'].values
        reach_ids_per_node = nodes_df['reach_id'].values
        node_x = nodes_df['x'].values
        node_y = nodes_df['y'].values
        node_wse = nodes_df['wse'].values
        current_facc = nodes_df['facc'].values

        # Get reach topology for validation (downstream neighbors)
        topology_df = conn.execute("""
            SELECT reach_id, neighbor_reach_id
            FROM reach_topology
            WHERE region = ? AND direction = 'down'
        """, [self._region]).fetchdf()

        # Build tile index - find all upa tiles
        tile_files = {}
        for subdir in upa_path.iterdir():
            if subdir.is_dir() and subdir.name.startswith('upa_'):
                for tif in subdir.glob('*.tif'):
                    if not tif.name.startswith('._'):
                        tile_files[tif.stem.replace('_upa', '')] = str(tif)

        logger.info(f"Found {len(tile_files)} upa tiles")

        # =====================================================================
        # PHASE 1: Extract raw facc from MERIT Hydro
        # =====================================================================
        raw_facc = np.full(len(node_ids), np.nan, dtype=np.float64)
        tiles_used = set()

        # Group nodes by tile for efficient processing
        tile_to_indices = {}
        for i, (x, y) in enumerate(zip(node_x, node_y)):
            tile_name = self._get_merit_tile_name(x, y)
            if tile_name in tile_files:
                if tile_name not in tile_to_indices:
                    tile_to_indices[tile_name] = []
                tile_to_indices[tile_name].append(i)

        logger.info(f"Phase 1: Extracting from {len(tile_to_indices)} tiles")

        for tile_idx, (tile_name, indices) in enumerate(tile_to_indices.items()):
            if tile_idx % 20 == 0:
                logger.info(f"  Processing tile {tile_idx + 1}/{len(tile_to_indices)}")

            try:
                with rasterio.open(tile_files[tile_name]) as src:
                    data = src.read(1)
                    transform = src.transform

                    # Get coordinates of valid river pixels (facc > threshold)
                    valid_mask = data > facc_threshold
                    valid_rows, valid_cols = np.where(valid_mask)

                    if len(valid_rows) == 0:
                        continue

                    # Convert pixel coords to geographic coords
                    mh_x = transform.c + (valid_cols + 0.5) * transform.a
                    mh_y = transform.f + (valid_rows + 0.5) * transform.e
                    mh_facc = data[valid_rows, valid_cols]

                    # Build KD-tree of valid MERIT points
                    mh_pts = np.vstack((mh_x, mh_y)).T
                    kdt = sp.cKDTree(mh_pts)

                    # Query for each node in this tile
                    for i in indices:
                        x, y = node_x[i], node_y[i]
                        node_pt = np.array([[x, y]])

                        # Find k nearest valid MERIT points
                        k = min(k_neighbors, len(mh_pts))
                        dists, inds = kdt.query(node_pt, k=k)
                        dists = dists.flatten()
                        inds = inds.flatten()

                        # Get facc values of nearest points and take median
                        neighbor_facc = mh_facc[inds]
                        raw_facc[i] = np.median(neighbor_facc)

                    tiles_used.add(tile_name)

            except (RasterioIOError, IndexError) as e:
                logger.warning(f"Error reading tile {tile_name}: {e}")
                continue

        valid_extractions = int(np.sum(~np.isnan(raw_facc)))
        logger.info(f"Phase 1 complete: {valid_extractions}/{len(node_ids)} extracted")

        # =====================================================================
        # PHASE 2: Within-reach filtering (legacy algorithm improved)
        # =====================================================================
        logger.info("Phase 2: Within-reach filtering")
        filtered_facc = np.copy(raw_facc)
        unique_reaches = np.unique(reach_ids_per_node)

        for reach_id in unique_reaches:
            reach_mask = reach_ids_per_node == reach_id
            reach_facc = raw_facc[reach_mask]
            reach_wse = node_wse[reach_mask]

            if np.all(np.isnan(reach_facc)) or np.sum(~np.isnan(reach_facc)) < 2:
                continue

            # Apply improved filter: remove outliers, interpolate
            filtered_reach = self._filter_facc_reach(reach_facc, reach_wse)
            filtered_facc[reach_mask] = filtered_reach

        # =====================================================================
        # PHASE 3: Network topology validation
        # =====================================================================
        logger.info("Phase 3: Network topology validation")
        validated_facc = np.copy(filtered_facc)
        violations = []

        # Build reach-level facc (max of nodes in reach)
        reach_facc_new = {}
        for reach_id in unique_reaches:
            reach_mask = reach_ids_per_node == reach_id
            reach_vals = filtered_facc[reach_mask]
            if not np.all(np.isnan(reach_vals)):
                reach_facc_new[reach_id] = np.nanmax(reach_vals)

        # Check topology: facc should increase downstream
        for _, row in topology_df.iterrows():
            reach_id = row['reach_id']
            dn_id = row['neighbor_reach_id']

            if reach_id not in reach_facc_new or dn_id not in reach_facc_new:
                continue

            upstream_facc = reach_facc_new[reach_id]
            downstream_facc = reach_facc_new[dn_id]

            # Violation: upstream has MORE facc than downstream
            if upstream_facc > downstream_facc * 1.1:  # 10% tolerance
                violations.append({
                    'upstream_reach': int(reach_id),
                    'downstream_reach': int(dn_id),
                    'upstream_facc': float(upstream_facc),
                    'downstream_facc': float(downstream_facc),
                })

        logger.info(f"Phase 3 complete: {len(violations)} topology violations found")

        # =====================================================================
        # PHASE 4: Sanity checks
        # =====================================================================
        logger.info("Phase 4: Sanity checks")
        sanity_issues = {
            'negative_values': int(np.sum(validated_facc < 0)),
            'zero_values': int(np.sum(validated_facc == 0)),
            'topology_violations': len(violations),
        }

        # Check for unreasonable jumps within reaches
        jump_count = 0
        for reach_id in unique_reaches:
            reach_mask = reach_ids_per_node == reach_id
            reach_vals = validated_facc[reach_mask]
            if len(reach_vals) > 1 and not np.all(np.isnan(reach_vals)):
                diffs = np.abs(np.diff(reach_vals[~np.isnan(reach_vals)]))
                max_val = np.nanmax(reach_vals)
                if max_val > 0:
                    # Flag jumps > 50% of max value
                    big_jumps = np.sum(diffs > max_val * 0.5)
                    jump_count += big_jumps

        sanity_issues['large_jumps_within_reach'] = jump_count

        # =====================================================================
        # PHASE 5: Update database (unless dry_run)
        # =====================================================================
        result = {
            'total_nodes': len(node_ids),
            'valid_extractions': valid_extractions,
            'tiles_used': len(tiles_used),
            'topology_violations': len(violations),
            'sanity_issues': sanity_issues,
            'sample_violations': violations[:10] if violations else [],
        }

        # Compare with current values
        valid_both = ~np.isnan(validated_facc) & (current_facc > 0)
        if np.sum(valid_both) > 0:
            pct_diffs = np.abs(validated_facc[valid_both] - current_facc[valid_both])
            pct_diffs = pct_diffs / current_facc[valid_both] * 100
            result['comparison'] = {
                'nodes_compared': int(np.sum(valid_both)),
                'median_pct_diff': float(np.median(pct_diffs)),
                'nodes_within_10pct': int(np.sum(pct_diffs < 10)),
                'nodes_over_100pct': int(np.sum(pct_diffs > 100)),
            }

        if dry_run:
            logger.info("Dry run - no database updates")
            result['dry_run'] = True
            return result

        # Update nodes
        logger.info("Phase 5: Updating database")
        nodes_updated = 0
        for i, (nid, facc_val) in enumerate(zip(node_ids, validated_facc)):
            if not np.isnan(facc_val) and facc_val > 0:
                conn.execute("""
                    UPDATE nodes SET facc = ?
                    WHERE node_id = ? AND region = ?
                """, [float(facc_val), int(nid), self._region])
                nodes_updated += 1

        # Update reaches (max facc of nodes)
        reaches_updated = 0
        for reach_id in unique_reaches:
            reach_mask = reach_ids_per_node == reach_id
            reach_vals = validated_facc[reach_mask]
            if not np.all(np.isnan(reach_vals)):
                max_facc = np.nanmax(reach_vals)
                if max_facc > 0:
                    conn.execute("""
                        UPDATE reaches SET facc = ?
                        WHERE reach_id = ? AND region = ?
                    """, [float(max_facc), int(reach_id), self._region])
                    reaches_updated += 1

        result['nodes_updated'] = nodes_updated
        result['reaches_updated'] = reaches_updated
        logger.info(f"Updated {nodes_updated} nodes and {reaches_updated} reaches")

        return result

    def _get_merit_tile_name(self, lon: float, lat: float) -> str:
        """Get MERIT Hydro tile name for a coordinate."""
        import math

        # MERIT tiles are 5x5 degrees, named by lower-left corner
        lat_tile = int(math.floor(lat / 5) * 5)
        lon_tile = int(math.floor(lon / 5) * 5)

        # Format: n{lat}w{lon} or n{lat}e{lon}
        lat_str = f"n{abs(lat_tile):02d}" if lat_tile >= 0 else f"s{abs(lat_tile):02d}"
        lon_str = f"w{abs(lon_tile):03d}" if lon_tile < 0 else f"e{abs(lon_tile):03d}"

        return f"{lat_str}{lon_str}"

    def _filter_facc_reach(
        self,
        facc: 'np.ndarray',
        wse: 'np.ndarray',
    ) -> 'np.ndarray':
        """
        Filter facc values for a single reach.

        Ensures monotonic downstream increase by interpolating between
        min and max values based on elevation direction.
        """
        import numpy as np

        if len(facc) == 0 or np.all(np.isnan(facc)):
            return facc

        result = np.copy(facc)
        valid_mask = ~np.isnan(facc)

        if np.sum(valid_mask) < 2:
            return result

        # Get min/max of valid values
        facc_min = np.nanmin(facc)
        facc_max = np.nanmax(facc)

        # If range is too small, just return original values
        if facc_max - facc_min < facc_max * 0.01:
            return result

        # Interpolate between min and max
        interp_vals = np.linspace(facc_min, facc_max, len(facc))

        # Determine direction using elevation
        wse_valid = wse[~np.isnan(wse)]
        if len(wse_valid) >= 2:
            wse_beg = wse[0] if not np.isnan(wse[0]) else wse_valid[0]
            wse_end = wse[-1] if not np.isnan(wse[-1]) else wse_valid[-1]

            if wse_end == wse_beg:
                result[:] = facc_max
            elif wse_end > wse_beg:
                # End is higher = upstream, reverse interpolation
                result = interp_vals[::-1]
            else:
                # End is lower = downstream
                result = interp_vals
        else:
            result = interp_vals

        return result

    # =========================================================================
    # STATUS AND UTILITY METHODS
    # =========================================================================

    def status(self) -> Dict[str, Any]:
        """
        Get the current workflow status.

        Returns
        -------
        dict
            Status information including loaded state, counts, pending changes
        """
        status = {
            "is_loaded": self.is_loaded,
            "db_path": str(self._db_path) if self._db_path else None,
            "region": self._region,
            "has_pending_changes": self.has_pending_changes,
            "in_batch_mode": self._in_batch,
            "in_transaction": self._in_transaction,
            "provenance_enabled": self._enable_provenance,
        }

        if self.is_loaded:
            status["reach_count"] = len(self._sword.reaches)
            status["node_count"] = len(self._sword.nodes)

            if self._reactive:
                status["dirty_attributes"] = list(self._reactive._dirty_attrs)

            if self._provenance:
                status["session_id"] = self._provenance.session_id
                status["user_id"] = self._provenance.user_id

        return status

    def fix_topology_violations(
        self,
        min_ratio: float = 100.0,
        dry_run: bool = True,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Remove extreme topology violations where upstream facc >> downstream facc.

        These are cases where a large river is incorrectly linked as flowing INTO
        a small tributary, which is physically impossible. This method removes
        the erroneous downstream links from reach_topology.

        Args:
            min_ratio: Minimum facc ratio (upstream/downstream) to consider a violation.
                       Default 100 means upstream is 100x larger than downstream.
            dry_run: If True, only report what would be removed without making changes.
            reason: Optional reason for the modification.

        Returns:
            Dict with statistics about violations found and removed.

        Example:
            # Preview what would be removed
            result = workflow.fix_topology_violations(min_ratio=100, dry_run=True)
            print(f"Would remove {result['violations_found']} erroneous links")

            # Actually remove them
            result = workflow.fix_topology_violations(min_ratio=100, dry_run=False)
        """
        if not self.is_loaded:
            raise RuntimeError("No database loaded. Call load() first.")

        conn = self._sword.db.conn

        # Find extreme violations
        violations_df = conn.execute("""
            WITH reach_facc AS (
                SELECT
                    reach_id,
                    AVG(facc) as avg_facc
                FROM nodes
                WHERE facc IS NOT NULL AND region = ?
                GROUP BY reach_id
            ),
            violations AS (
                SELECT
                    t.reach_id as upstream_reach,
                    t.neighbor_reach_id as downstream_reach,
                    r1.avg_facc as upstream_facc,
                    r2.avg_facc as downstream_facc,
                    r1.avg_facc / r2.avg_facc as ratio
                FROM reach_topology t
                JOIN reach_facc r1 ON t.reach_id = r1.reach_id
                JOIN reach_facc r2 ON t.neighbor_reach_id = r2.reach_id
                WHERE t.direction = 'down'
                  AND t.region = ?
                  AND r2.avg_facc > 0
                  AND r1.avg_facc / r2.avg_facc >= ?
            )
            SELECT * FROM violations
            ORDER BY ratio DESC
        """, [self._region, self._region, min_ratio]).fetchdf()

        result = {
            'violations_found': len(violations_df),
            'min_ratio_threshold': min_ratio,
            'dry_run': dry_run,
        }

        if len(violations_df) == 0:
            logger.info(f"No topology violations found with ratio >= {min_ratio}")
            return result

        # Add sample violations to result
        samples = []
        for _, row in violations_df.head(10).iterrows():
            samples.append({
                'upstream_reach': int(row['upstream_reach']),
                'downstream_reach': int(row['downstream_reach']),
                'upstream_facc': float(row['upstream_facc']),
                'downstream_facc': float(row['downstream_facc']),
                'ratio': float(row['ratio']),
            })
        result['sample_violations'] = samples

        if dry_run:
            logger.info(
                f"Dry run: Would remove {len(violations_df)} erroneous topology links "
                f"(ratio >= {min_ratio})"
            )
            return result

        # Remove the erroneous links
        removed_count = 0
        for _, row in violations_df.iterrows():
            up_reach = int(row['upstream_reach'])
            dn_reach = int(row['downstream_reach'])

            # Delete the downstream link
            conn.execute("""
                DELETE FROM reach_topology
                WHERE reach_id = ?
                  AND neighbor_reach_id = ?
                  AND direction = 'down'
                  AND region = ?
            """, [up_reach, dn_reach, self._region])

            # Also delete the corresponding upstream link (reverse direction)
            conn.execute("""
                DELETE FROM reach_topology
                WHERE reach_id = ?
                  AND neighbor_reach_id = ?
                  AND direction = 'up'
                  AND region = ?
            """, [dn_reach, up_reach, self._region])

            removed_count += 1

        result['links_removed'] = removed_count
        logger.info(
            f"Removed {removed_count} erroneous topology links (ratio >= {min_ratio})"
        )

        return result

    # =========================================================================
    # IMAGERY METHODS
    # =========================================================================

    def get_imagery_for_reach(
        self,
        reach_id: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_cloud_cover: float = 20.0,
        use_otsu: bool = False,
    ) -> 'Any':
        """
        Get satellite imagery and water index for a reach.

        Parameters
        ----------
        reach_id : int
            SWORD reach ID
        start_date : str, optional
            Start date for imagery search (YYYY-MM-DD)
        end_date : str, optional
            End date for imagery search (YYYY-MM-DD)
        max_cloud_cover : float, optional
            Maximum cloud cover percentage (default: 20)
        use_otsu : bool, optional
            Use Otsu's method for automatic water threshold

        Returns
        -------
        ImageryResult
            Result containing NDWI, water mask, and statistics

        Example
        -------
        >>> result = workflow.get_imagery_for_reach(12345678901)
        >>> print(f"Water fraction: {result.stats['water_fraction']:.2%}")
        """
        if not self.is_loaded:
            raise ValueError("No SWORD database loaded. Call load() first.")

        return self.imagery.get_imagery_for_reach(
            reach_id=reach_id,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            use_otsu=use_otsu,
        )

    def get_imagery_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the imagery cache."""
        if self.imagery is None:
            return {"cache_enabled": False, "error": "No database loaded"}
        return self.imagery.get_cache_stats()

    def clear_imagery_cache(self) -> None:
        """Clear the imagery cache."""
        if self.imagery is not None:
            self.imagery.clear_cache()

    def __repr__(self) -> str:
        if self.is_loaded:
            return (
                f"SWORDWorkflow(db={self._db_path.name}, "
                f"region={self._region}, "
                f"pending={self.has_pending_changes})"
            )
        return "SWORDWorkflow(not loaded)"

    def __enter__(self) -> 'SWORDWorkflow':
        """Support using workflow as context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close database on context exit."""
        self.close()
        return False
