# -*- coding: utf-8 -*-
"""
SWORD Database Connection Manager
=================================

This module provides connection management for SWORD databases.
It supports both DuckDB (local file) and PostgreSQL (remote/concurrent) backends
through a unified interface.

Example Usage:
    from sword_db import SWORDDatabase

    # DuckDB (file path - default)
    db = SWORDDatabase('/path/to/sword.duckdb')
    conn = db.connect()
    result = conn.execute("SELECT * FROM reaches LIMIT 10").fetchdf()
    db.close()

    # PostgreSQL (connection URL)
    db = SWORDDatabase('postgresql://user:pass@localhost/sword')
    with db.transaction() as ctx:
        db.execute("UPDATE reaches SET facc = %s WHERE reach_id = %s", [100, 123])
    db.close()

    # Context manager usage
    with SWORDDatabase('/path/to/sword.duckdb') as db:
        result = db.query("SELECT * FROM reaches WHERE region = 'NA'")
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

import pandas as pd

from .backends import (
    BackendType,
    DatabaseBackend,
    DuckDBBackend,
    PostgresBackend,
    detect_backend_type,
    get_backend,
)
from .backends.base import IsolationLevel, TransactionContext
from .schema import create_schema, SCHEMA_VERSION

logger = logging.getLogger(__name__)


class SWORDDatabase:
    """
    Database connection manager for SWORD.

    Supports both DuckDB (local file) and PostgreSQL (remote/concurrent) backends
    through a unified interface. Auto-detects backend type from connection string.

    Parameters
    ----------
    db_path : str or Path
        Database connection string or file path.
        - File path (*.duckdb): Uses DuckDB backend
        - postgresql://...: Uses PostgreSQL backend
        - ':memory:': In-memory DuckDB
    read_only : bool, optional
        If True, opens database in read-only mode. Default is False.
        (Only applies to DuckDB backend)
    spatial : bool, optional
        If True, loads spatial extension (DuckDB: spatial, PG: PostGIS).
        Default is True.
    backend_type : BackendType, optional
        Force a specific backend type. If None, auto-detects from db_path.

    Attributes
    ----------
    db_path : Path or str
        Original database path/connection string.
    backend : DatabaseBackend
        The underlying backend instance.
    backend_type : BackendType
        Type of backend (DUCKDB or POSTGRES).

    Examples
    --------
    >>> # DuckDB (file path)
    >>> db = SWORDDatabase('sword_v17c.duckdb')
    >>> df = db.query("SELECT COUNT(*) FROM reaches")
    >>> db.close()

    >>> # PostgreSQL (connection URL)
    >>> db = SWORDDatabase('postgresql://user:pass@localhost/sword')
    >>> with db.acquire_region_lock('NA'):
    ...     df = db.query("SELECT * FROM reaches WHERE region = 'NA'")
    >>> db.close()

    >>> # Using context manager
    >>> with SWORDDatabase('sword_v18.duckdb') as db:
    ...     df = db.query("SELECT COUNT(*) FROM nodes WHERE region = 'NA'")
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        read_only: bool = False,
        spatial: bool = True,
        backend_type: Optional[BackendType] = None,
    ):
        self.db_path = db_path
        self._read_only = read_only
        self._spatial = spatial

        # Detect or use specified backend type
        if backend_type is None:
            backend_type = detect_backend_type(db_path)

        self._backend_type = backend_type

        # Create the appropriate backend
        if backend_type == BackendType.DUCKDB:
            self._backend = DuckDBBackend(
                db_path=db_path,
                read_only=read_only,
                spatial=spatial,
            )
        elif backend_type == BackendType.POSTGRES:
            self._backend = PostgresBackend(
                connection_string=str(db_path),
                enable_postgis=spatial,
            )
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

        # Track current connection for DuckDB compatibility
        self._conn = None

    @property
    def backend(self) -> DatabaseBackend:
        """Get the underlying backend instance."""
        return self._backend

    # Backward-compatible properties
    @property
    def read_only(self) -> bool:
        """Whether the connection is read-only (DuckDB only)."""
        return self._read_only

    @property
    def spatial(self) -> bool:
        """Whether spatial extension is requested."""
        return self._spatial

    @property
    def _spatial_loaded(self) -> bool:
        """Whether spatial extension is loaded (DuckDB only, for backward compat)."""
        if hasattr(self._backend, '_spatial_loaded'):
            return self._backend._spatial_loaded
        if hasattr(self._backend, '_postgis_enabled'):
            return self._backend._postgis_enabled
        return False

    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        return self._backend_type

    @property
    def placeholder(self) -> str:
        """Get the SQL placeholder for this backend ('?' or '%s')."""
        return self._backend.placeholder

    @property
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL backend."""
        return self._backend_type == BackendType.POSTGRES

    @property
    def is_duckdb(self) -> bool:
        """Check if using DuckDB backend."""
        return self._backend_type == BackendType.DUCKDB

    def connect(self):
        """
        Get or create a database connection.

        Returns
        -------
        connection
            Active database connection (type depends on backend).

        Notes
        -----
        For DuckDB: Returns duckdb.DuckDBPyConnection
        For PostgreSQL: Returns psycopg2.connection from pool
        """
        if self._conn is None:
            self._conn = self._backend.connect()
        return self._conn

    @property
    def conn(self):
        """Get the database connection (alias for connect())."""
        return self.connect()

    def close(self) -> None:
        """Close the database connection(s)."""
        self._backend.close()
        self._conn = None

    def __enter__(self) -> 'SWORDDatabase':
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def query(self, sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to execute.
        params : list, optional
            Query parameters for parameterized queries.

        Returns
        -------
        pd.DataFrame
            Query results.

        Notes
        -----
        SQL is automatically converted to use the correct placeholder
        for the backend ('?' for DuckDB, '%s' for PostgreSQL).

        Examples
        --------
        >>> db = SWORDDatabase('sword.duckdb')
        >>> df = db.query("SELECT * FROM reaches WHERE facc > ?", [10000])
        """
        # Convert placeholders if needed
        converted_sql = self._backend.convert_placeholders(sql)
        return self._backend.query(converted_sql, params)

    def execute(self, sql: str, params: Optional[List[Any]] = None) -> Any:
        """
        Execute a SQL statement.

        Parameters
        ----------
        sql : str
            SQL statement to execute.
        params : list, optional
            Query parameters for parameterized queries.

        Returns
        -------
        Any
            Backend-specific result object.
        """
        converted_sql = self._backend.convert_placeholders(sql)
        return self._backend.execute(converted_sql, params)

    def executemany(
        self,
        sql: str,
        params_list: List[Tuple[Any, ...]],
    ) -> None:
        """
        Execute a SQL statement with multiple parameter sets.

        Parameters
        ----------
        sql : str
            SQL statement to execute.
        params_list : list of tuples
            List of parameter tuples.
        """
        converted_sql = self._backend.convert_placeholders(sql)
        self._backend.executemany(converted_sql, params_list)

    @contextmanager
    def transaction(
        self,
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
    ) -> Generator[TransactionContext, None, None]:
        """
        Context manager for database transactions.

        Parameters
        ----------
        isolation_level : IsolationLevel
            Transaction isolation level.

        Yields
        ------
        TransactionContext
            Context object with transaction details.

        Example
        -------
        >>> with db.transaction() as ctx:
        ...     db.execute("UPDATE reaches SET facc = ? WHERE reach_id = ?", [100, 123])
        ...     # Automatically committed on success, rolled back on error
        """
        with self._backend.transaction(isolation_level) as ctx:
            yield ctx

    def commit(self) -> None:
        """Commit the current transaction."""
        self._backend.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self._backend.rollback()

    def init_schema(self) -> None:
        """
        Initialize the database schema.

        Creates all tables, indexes, and views if they don't exist.

        Raises
        ------
        Exception
            If schema creation fails.
        """
        conn = self.connect()
        create_schema(conn)

    def get_regions(self) -> List[str]:
        """
        Get list of regions present in the database.

        Returns
        -------
        list of str
            List of 2-letter region codes.
        """
        result = self.query("SELECT DISTINCT region FROM reaches ORDER BY region")
        return result['region'].tolist()

    def get_versions(self) -> pd.DataFrame:
        """
        Get version information from the database.

        Returns
        -------
        pd.DataFrame
            Version metadata including creation dates and notes.
        """
        return self.query("SELECT * FROM sword_versions ORDER BY version")

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
        placeholder = self.placeholder

        where_clause = ""
        params = []
        if region:
            where_clause = f"WHERE region = {placeholder}"
            params = [region]

        counts = {}

        # Tables with region column
        for table in ['centerlines', 'nodes', 'reaches']:
            sql = f"SELECT COUNT(*) as cnt FROM {table} {where_clause}"
            result = self.query(sql, params if params else None)
            counts[table] = result.iloc[0]['cnt']

        # Derived tables (no region column, need to join)
        if region:
            counts['reach_topology'] = self.query(f"""
                SELECT COUNT(*) as cnt FROM reach_topology t
                JOIN reaches r ON t.reach_id = r.reach_id
                WHERE r.region = {placeholder}
            """, [region]).iloc[0]['cnt']

            counts['reach_swot_orbits'] = self.query(f"""
                SELECT COUNT(*) as cnt FROM reach_swot_orbits o
                JOIN reaches r ON o.reach_id = r.reach_id
                WHERE r.region = {placeholder}
            """, [region]).iloc[0]['cnt']
        else:
            counts['reach_topology'] = self.query(
                "SELECT COUNT(*) as cnt FROM reach_topology"
            ).iloc[0]['cnt']

            counts['reach_swot_orbits'] = self.query(
                "SELECT COUNT(*) as cnt FROM reach_swot_orbits"
            ).iloc[0]['cnt']

        return counts

    def spatial_available(self) -> bool:
        """
        Check if spatial extension is available and loaded.

        Returns
        -------
        bool
            True if spatial queries are supported.
        """
        if hasattr(self._backend, 'spatial_available'):
            return self._backend.spatial_available
        if hasattr(self._backend, '_postgis_enabled'):
            return self._backend._postgis_enabled
        return False

    @property
    def schema_version(self) -> Optional[str]:
        """Get the schema version from the database."""
        try:
            result = self.query("""
                SELECT schema_version FROM sword_versions
                WHERE version = 'schema'
            """)
            if len(result) > 0:
                return result['schema_version'].iloc[0]
        except Exception:
            pass
        return None

    # =========================================================================
    # PostgreSQL-specific methods
    # =========================================================================

    @contextmanager
    def acquire_region_lock(
        self,
        region: str,
        timeout_ms: int = 30000,
    ) -> Generator[Any, None, None]:
        """
        Acquire an advisory lock for a region (PostgreSQL only).

        This provides exclusive access to a region, preventing concurrent
        edits to the same region by different users.

        Parameters
        ----------
        region : str
            Region code (e.g., 'NA', 'EU').
        timeout_ms : int
            Lock acquisition timeout in milliseconds. Default is 30s.

        Yields
        ------
        connection
            The connection holding the lock.

        Raises
        ------
        NotImplementedError
            If called on DuckDB backend (single-user, no locking needed).
        TimeoutError
            If lock cannot be acquired within timeout (PostgreSQL).

        Example
        -------
        >>> with db.acquire_region_lock('NA'):
        ...     # Exclusive access to NA region
        ...     db.execute("UPDATE reaches SET facc = 100 WHERE region = 'NA'")
        """
        if self._backend_type == BackendType.DUCKDB:
            # DuckDB is single-user, no locking needed
            yield self.connect()
        elif self._backend_type == BackendType.POSTGRES:
            with self._backend.acquire_region_lock(region, timeout_ms) as conn:
                yield conn
        else:
            raise NotImplementedError(f"Region locking not supported for {self._backend_type}")

    def has_region_lock(self, region: str) -> bool:
        """
        Check if we hold the lock for a region (PostgreSQL only).

        Parameters
        ----------
        region : str
            Region code.

        Returns
        -------
        bool
            True if we hold the lock, False otherwise.
            Always returns True for DuckDB (single-user).
        """
        if self._backend_type == BackendType.DUCKDB:
            return True  # Single-user, always "locked"
        elif self._backend_type == BackendType.POSTGRES:
            return self._backend.has_region_lock(region)
        return False

    def format_upsert(
        self,
        table: str,
        columns: List[str],
        key_columns: List[str],
    ) -> str:
        """
        Generate backend-specific UPSERT SQL.

        Parameters
        ----------
        table : str
            Table name.
        columns : list of str
            All columns to insert/update.
        key_columns : list of str
            Primary key columns for conflict detection.

        Returns
        -------
        str
            UPSERT SQL template with appropriate placeholders.
        """
        return self._backend.format_upsert(table, columns, key_columns)

    def format_array_literal(self, values: List[Any]) -> str:
        """
        Format a list as a SQL array literal.

        Parameters
        ----------
        values : list
            Values to format as array.

        Returns
        -------
        str
            Backend-specific array literal.
        """
        return self._backend.format_array_literal(values)


def create_database(
    db_path: Union[str, Path],
    overwrite: bool = False,
) -> SWORDDatabase:
    """
    Create a new SWORD database with initialized schema.

    Parameters
    ----------
    db_path : str or Path
        Path where the database file will be created.
    overwrite : bool, optional
        If True and database exists, it will be overwritten.
        Default is False.

    Returns
    -------
    SWORDDatabase
        Database instance with schema initialized.

    Raises
    ------
    FileExistsError
        If database exists and overwrite is False.

    Examples
    --------
    >>> db = create_database('/data/duckdb/sword_v18.duckdb')
    >>> db.count_records()
    {'centerlines': 0, 'nodes': 0, 'reaches': 0, ...}
    """
    db_path = Path(db_path)

    if db_path.exists():
        if overwrite:
            db_path.unlink()
        else:
            raise FileExistsError(
                f"Database already exists at {db_path}. "
                "Set overwrite=True to replace it."
            )

    db = SWORDDatabase(db_path)
    db.init_schema()

    return db
