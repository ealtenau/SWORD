# -*- coding: utf-8 -*-
"""
SWORD DuckDB Connection Manager
===============================

This module provides connection management for the SWORD DuckDB database.
It handles connection pooling, spatial extension loading, and context management.

Example Usage:
    from sword_db import SWORDDatabase

    # Basic usage
    db = SWORDDatabase('/path/to/sword.duckdb')
    conn = db.connect()
    result = conn.execute("SELECT * FROM reaches LIMIT 10").fetchdf()
    db.close()

    # Context manager usage
    with SWORDDatabase('/path/to/sword.duckdb') as db:
        result = db.query("SELECT * FROM reaches WHERE region = 'NA'")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union, Any

import duckdb
import pandas as pd

from .schema import create_schema, SCHEMA_VERSION

logger = logging.getLogger(__name__)


class SWORDDatabase:
    """
    DuckDB connection manager for SWORD database.

    Provides connection pooling, spatial extension loading, and
    convenience methods for common database operations.

    Parameters
    ----------
    db_path : str or Path
        Path to the DuckDB database file. Use ':memory:' for in-memory database.
    read_only : bool, optional
        If True, opens database in read-only mode. Default is False.
    spatial : bool, optional
        If True, loads the DuckDB spatial extension. Default is True.

    Attributes
    ----------
    db_path : Path
        Path to the database file.
    read_only : bool
        Whether the connection is read-only.
    spatial : bool
        Whether the spatial extension is loaded.

    Examples
    --------
    >>> db = SWORDDatabase('sword_v18.duckdb')
    >>> conn = db.connect()
    >>> df = conn.execute("SELECT * FROM reaches LIMIT 5").fetchdf()
    >>> db.close()

    >>> # Using context manager
    >>> with SWORDDatabase('sword_v18.duckdb') as db:
    ...     df = db.query("SELECT COUNT(*) FROM nodes WHERE region = 'NA'")
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        read_only: bool = False,
        spatial: bool = True
    ):
        self.db_path = Path(db_path) if db_path != ':memory:' else db_path
        self.read_only = read_only
        self.spatial = spatial
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._spatial_loaded = False

    def connect(self) -> duckdb.DuckDBPyConnection:
        """
        Get or create a database connection.

        Returns
        -------
        duckdb.DuckDBPyConnection
            Active database connection.

        Notes
        -----
        The connection is created lazily and reused for subsequent calls.
        The spatial extension is loaded automatically if `spatial=True`.
        """
        if self._conn is None:
            # Ensure parent directory exists for file-based databases
            if self.db_path != ':memory:':
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self._conn = duckdb.connect(
                str(self.db_path),
                read_only=self.read_only
            )

            # Load spatial extension if requested
            if self.spatial and not self._spatial_loaded:
                self._load_spatial()

        return self._conn

    def _load_spatial(self) -> None:
        """Load the DuckDB spatial extension."""
        try:
            self._conn.execute("INSTALL spatial;")
            self._conn.execute("LOAD spatial;")
            self._spatial_loaded = True
        except Exception as e:
            # Spatial extension might already be installed
            if "already installed" in str(e).lower():
                self._conn.execute("LOAD spatial;")
                self._spatial_loaded = True
            else:
                logger.warning(f"Could not load spatial extension: {e}")
                self._spatial_loaded = False

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get the database connection (alias for connect())."""
        return self.connect()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._spatial_loaded = False

    def __enter__(self) -> 'SWORDDatabase':
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def query(self, sql: str, params: list = None) -> pd.DataFrame:
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

        Examples
        --------
        >>> db = SWORDDatabase('sword.duckdb')
        >>> df = db.query("SELECT * FROM reaches WHERE facc > ?", [10000])
        """
        conn = self.connect()
        if params:
            return conn.execute(sql, params).fetchdf()
        return conn.execute(sql).fetchdf()

    def execute(self, sql: str, params: list = None) -> Any:
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
        duckdb.DuckDBPyResult
            Query result object.
        """
        conn = self.connect()
        if params:
            return conn.execute(sql, params)
        return conn.execute(sql)

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

    def get_regions(self) -> list[str]:
        """
        Get list of regions present in the database.

        Returns
        -------
        list[str]
            List of 2-letter region codes.
        """
        result = self.query("""
            SELECT DISTINCT region FROM reaches ORDER BY region
        """)
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

    def count_records(self, region: str = None) -> dict:
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

        # Derived tables (no region column, need to join)
        if region:
            counts['reach_topology'] = conn.execute("""
                SELECT COUNT(*) FROM reach_topology t
                JOIN reaches r ON t.reach_id = r.reach_id
                WHERE r.region = ?
            """, [region]).fetchone()[0]

            counts['reach_swot_orbits'] = conn.execute("""
                SELECT COUNT(*) FROM reach_swot_orbits o
                JOIN reaches r ON o.reach_id = r.reach_id
                WHERE r.region = ?
            """, [region]).fetchone()[0]
        else:
            counts['reach_topology'] = conn.execute(
                "SELECT COUNT(*) FROM reach_topology"
            ).fetchone()[0]

            counts['reach_swot_orbits'] = conn.execute(
                "SELECT COUNT(*) FROM reach_swot_orbits"
            ).fetchone()[0]

        return counts

    def spatial_available(self) -> bool:
        """
        Check if the spatial extension is available and loaded.

        Returns
        -------
        bool
            True if spatial queries are supported.
        """
        return self._spatial_loaded

    @property
    def schema_version(self) -> str:
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


def create_database(
    db_path: Union[str, Path],
    overwrite: bool = False
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
