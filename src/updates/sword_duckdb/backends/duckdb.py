# -*- coding: utf-8 -*-
"""
DuckDB Backend Implementation
=============================

Wraps the existing DuckDB connection management to implement the
DatabaseBackend protocol.

DuckDB is optimized for:
- Single-user, local file access
- High-performance analytical queries
- In-process execution (no network latency)
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

import duckdb
import pandas as pd

from .base import (
    BackendType,
    BaseBackend,
    ConnectionType,
    IsolationLevel,
    TransactionContext,
)

logger = logging.getLogger(__name__)


class DuckDBBackend(BaseBackend):
    """
    DuckDB backend implementation.

    Provides single-connection access to a DuckDB file database.
    Handles spatial extension loading and proper resource management.

    Parameters
    ----------
    db_path : str or Path
        Path to the DuckDB database file. Use ':memory:' for in-memory database.
    read_only : bool, optional
        If True, opens database in read-only mode. Default is False.
    spatial : bool, optional
        If True, loads the DuckDB spatial extension. Default is True.

    Example
    -------
    >>> backend = DuckDBBackend('sword_v17c.duckdb')
    >>> conn = backend.connect()
    >>> df = backend.query("SELECT * FROM reaches LIMIT 10")
    >>> backend.close()
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        read_only: bool = False,
        spatial: bool = True,
    ):
        super().__init__()
        self.db_path = Path(db_path) if db_path != ':memory:' else db_path
        self.read_only = read_only
        self.spatial = spatial
        self._spatial_loaded = False

    @property
    def backend_type(self) -> BackendType:
        """Return the backend type."""
        return BackendType.DUCKDB

    @property
    def placeholder(self) -> str:
        """Return the SQL placeholder ('?' for DuckDB)."""
        return '?'

    def connect(self) -> duckdb.DuckDBPyConnection:
        """
        Get or create a database connection.

        Returns
        -------
        duckdb.DuckDBPyConnection
            Active database connection.
        """
        if self._connection is None:
            # Ensure parent directory exists for file-based databases
            if self.db_path != ':memory:':
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self._connection = duckdb.connect(
                str(self.db_path),
                read_only=self.read_only
            )

            # Load spatial extension if requested
            if self.spatial and not self._spatial_loaded:
                self._load_spatial()

        return self._connection

    def _load_spatial(self) -> None:
        """Load the DuckDB spatial extension."""
        try:
            self._connection.execute("INSTALL spatial;")
            self._connection.execute("LOAD spatial;")
            self._spatial_loaded = True
        except Exception as e:
            # Spatial extension might already be installed
            if "already installed" in str(e).lower():
                self._connection.execute("LOAD spatial;")
                self._spatial_loaded = True
            else:
                logger.warning(f"Could not load spatial extension: {e}")
                self._spatial_loaded = False

    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            self._spatial_loaded = False

    def query(
        self,
        sql: str,
        params: Optional[List[Any]] = None,
        connection: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to execute.
        params : list, optional
            Query parameters for parameterized queries.
        connection : optional
            Specific connection to use. If None, uses default.

        Returns
        -------
        pd.DataFrame
            Query results.
        """
        conn = connection or self.connect()
        if params:
            return conn.execute(sql, params).fetchdf()
        return conn.execute(sql).fetchdf()

    def execute(
        self,
        sql: str,
        params: Optional[List[Any]] = None,
        connection: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> duckdb.DuckDBPyResult:
        """
        Execute a SQL statement.

        Parameters
        ----------
        sql : str
            SQL statement to execute.
        params : list, optional
            Query parameters for parameterized queries.
        connection : optional
            Specific connection to use.

        Returns
        -------
        duckdb.DuckDBPyResult
            Query result object.
        """
        conn = connection or self.connect()
        if params:
            return conn.execute(sql, params)
        return conn.execute(sql)

    def executemany(
        self,
        sql: str,
        params_list: List[Tuple[Any, ...]],
        connection: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> None:
        """
        Execute a SQL statement with multiple parameter sets.

        DuckDB's executemany is efficient for batch operations.

        Parameters
        ----------
        sql : str
            SQL statement to execute.
        params_list : list of tuples
            List of parameter tuples.
        connection : optional
            Specific connection to use.
        """
        conn = connection or self.connect()
        conn.executemany(sql, params_list)

    @contextmanager
    def transaction(
        self,
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
        connection: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> Generator[TransactionContext, None, None]:
        """
        Context manager for database transactions.

        DuckDB uses auto-commit by default. This wraps operations in
        explicit BEGIN/COMMIT with proper rollback on error.

        Parameters
        ----------
        isolation_level : IsolationLevel
            Transaction isolation level (DuckDB uses SERIALIZABLE).
        connection : optional
            Specific connection to use.

        Yields
        ------
        TransactionContext
            Context object with transaction details.
        """
        conn = connection or self.connect()

        # Disable GC during transaction to avoid DuckDB issues
        gc_was_enabled = gc.isenabled()
        gc.disable()

        try:
            conn.execute("BEGIN TRANSACTION")

            ctx = TransactionContext(
                backend_type=BackendType.DUCKDB,
                isolation_level=isolation_level,
                connection=conn,
            )

            yield ctx

            conn.execute("COMMIT")

        except Exception:
            conn.execute("ROLLBACK")
            raise

        finally:
            if gc_was_enabled:
                gc.enable()

    def commit(self, connection: Optional[duckdb.DuckDBPyConnection] = None) -> None:
        """Commit the current transaction."""
        conn = connection or self._connection
        if conn:
            conn.execute("COMMIT")

    def rollback(self, connection: Optional[duckdb.DuckDBPyConnection] = None) -> None:
        """Rollback the current transaction."""
        conn = connection or self._connection
        if conn:
            conn.execute("ROLLBACK")

    def format_upsert(
        self,
        table: str,
        columns: List[str],
        key_columns: List[str],
    ) -> str:
        """
        Generate DuckDB UPSERT SQL using INSERT OR REPLACE.

        Parameters
        ----------
        table : str
            Table name.
        columns : list of str
            All columns to insert/update.
        key_columns : list of str
            Primary key columns (used for conflict detection).

        Returns
        -------
        str
            UPSERT SQL template with '?' placeholders.
        """
        placeholders = ', '.join(['?'] * len(columns))
        col_list = ', '.join(columns)

        # DuckDB uses INSERT OR REPLACE (requires primary key constraint)
        return f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"

    def format_array_literal(self, values: List[Any]) -> str:
        """
        Format a list as a DuckDB array literal.

        DuckDB uses [v1, v2, v3] syntax.

        Parameters
        ----------
        values : list
            Values to format as array.

        Returns
        -------
        str
            DuckDB array literal.
        """
        formatted = ', '.join(
            repr(v) if isinstance(v, str) else str(v)
            for v in values
        )
        return f"[{formatted}]"

    @property
    def spatial_available(self) -> bool:
        """Check if the spatial extension is available and loaded."""
        return self._spatial_loaded

    def get_regions(self) -> List[str]:
        """Get list of regions present in the database."""
        result = self.query("SELECT DISTINCT region FROM reaches ORDER BY region")
        return result['region'].tolist()

    def count_records(self, region: Optional[str] = None) -> dict:
        """
        Count records in each table.

        Parameters
        ----------
        region : str, optional
            If provided, count only records for this region.

        Returns
        -------
        dict
            Dictionary with table names as keys and record counts as values.
        """
        conn = self.connect()

        where_clause = ""
        params = []
        if region:
            where_clause = "WHERE region = ?"
            params = [region]

        counts = {}

        # Tables with region column
        for table in ['centerlines', 'nodes', 'reaches']:
            sql = f"SELECT COUNT(*) as cnt FROM {table} {where_clause}"
            result = conn.execute(sql, params).fetchone()
            counts[table] = result[0]

        # Derived tables
        if region:
            counts['reach_topology'] = conn.execute("""
                SELECT COUNT(*) FROM reach_topology t
                JOIN reaches r ON t.reach_id = r.reach_id
                WHERE r.region = ?
            """, [region]).fetchone()[0]
        else:
            counts['reach_topology'] = conn.execute(
                "SELECT COUNT(*) FROM reach_topology"
            ).fetchone()[0]

        return counts
