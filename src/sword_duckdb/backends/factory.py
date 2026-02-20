# -*- coding: utf-8 -*-
"""
Backend Factory
===============

Provides factory functions to create the appropriate database backend
based on connection string or explicit type selection.

Usage:
    from sword_duckdb.backends import get_backend

    # Auto-detect from path/URL
    backend = get_backend('data/sword.duckdb')  # DuckDB
    backend = get_backend('postgresql://user:pass@host/db')  # PostgreSQL

    # Explicit type
    backend = get_backend('mydb.duckdb', backend_type=BackendType.DUCKDB)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from .base import BackendType, DatabaseBackend
from .duckdb import DuckDBBackend
from .postgres import PostgresBackend


def detect_backend_type(connection: Union[str, Path]) -> BackendType:
    """
    Detect the backend type from a connection string or path.

    Parameters
    ----------
    connection : str or Path
        Database connection string or file path.

    Returns
    -------
    BackendType
        Detected backend type.

    Examples
    --------
    >>> detect_backend_type('data/sword.duckdb')
    BackendType.DUCKDB

    >>> detect_backend_type('postgresql://user:pass@localhost/sword')
    BackendType.POSTGRES

    >>> detect_backend_type(':memory:')
    BackendType.DUCKDB
    """
    conn_str = str(connection)

    # Check for PostgreSQL URL patterns
    if conn_str.startswith(('postgresql://', 'postgres://', 'psql://')):
        return BackendType.POSTGRES

    # Check for environment variable override
    env_backend = os.environ.get('SWORD_PRIMARY_BACKEND', '').lower()
    if env_backend == 'postgres':
        # If SWORD_POSTGRES_URL is set, use PostgreSQL
        if os.environ.get('SWORD_POSTGRES_URL'):
            return BackendType.POSTGRES

    # Default to DuckDB for file paths
    return BackendType.DUCKDB


def get_backend(
    connection: Union[str, Path],
    backend_type: Optional[BackendType] = None,
    **kwargs,
) -> DatabaseBackend:
    """
    Create a database backend based on connection string or explicit type.

    This is the main factory function for creating backends. It auto-detects
    the backend type from the connection string, or uses an explicit type.

    Parameters
    ----------
    connection : str or Path
        Database connection string or file path.
        - File path: Creates DuckDBBackend
        - postgresql://...: Creates PostgresBackend
    backend_type : BackendType, optional
        Explicit backend type. If None, auto-detects.
    **kwargs
        Additional arguments passed to the backend constructor.

    Returns
    -------
    DatabaseBackend
        Configured backend instance.

    Examples
    --------
    >>> # Auto-detect DuckDB from file path
    >>> backend = get_backend('data/duckdb/sword_v17c.duckdb')
    >>> type(backend)
    <class 'DuckDBBackend'>

    >>> # Auto-detect PostgreSQL from URL
    >>> backend = get_backend('postgresql://user:pass@localhost/sword')
    >>> type(backend)
    <class 'PostgresBackend'>

    >>> # DuckDB with options
    >>> backend = get_backend('sword.duckdb', read_only=True, spatial=False)

    >>> # PostgreSQL with pool options
    >>> backend = get_backend(
    ...     'postgresql://localhost/sword',
    ...     min_connections=2,
    ...     max_connections=20,
    ... )
    """
    if backend_type is None:
        backend_type = detect_backend_type(connection)

    if backend_type == BackendType.DUCKDB:
        # Extract DuckDB-specific kwargs
        duckdb_kwargs = {
            'db_path': connection,
            'read_only': kwargs.pop('read_only', False),
            'spatial': kwargs.pop('spatial', True),
        }
        return DuckDBBackend(**duckdb_kwargs)

    elif backend_type == BackendType.POSTGRES:
        # Handle PostgreSQL connection string
        conn_str = str(connection)

        # Support environment variable for connection string
        if not conn_str.startswith(('postgresql://', 'postgres://', 'psql://')):
            conn_str = os.environ.get('SWORD_POSTGRES_URL', conn_str)

        # Extract PostgreSQL-specific kwargs
        pg_kwargs = {
            'connection_string': conn_str,
            'min_connections': kwargs.pop('min_connections', 1),
            'max_connections': kwargs.pop('max_connections', 10),
            'enable_postgis': kwargs.pop('enable_postgis', True),
        }
        return PostgresBackend(**pg_kwargs)

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def get_backend_from_env() -> Optional[DatabaseBackend]:
    """
    Create a backend from environment variables.

    Checks the following environment variables:
    - SWORD_PRIMARY_BACKEND: 'duckdb' or 'postgres'
    - SWORD_DUCKDB_PATH: Path to DuckDB file (for duckdb backend)
    - SWORD_POSTGRES_URL: PostgreSQL connection URL (for postgres backend)

    Returns
    -------
    DatabaseBackend or None
        Configured backend, or None if environment not configured.

    Example
    -------
    >>> import os
    >>> os.environ['SWORD_PRIMARY_BACKEND'] = 'postgres'
    >>> os.environ['SWORD_POSTGRES_URL'] = 'postgresql://localhost/sword'
    >>> backend = get_backend_from_env()
    >>> type(backend)
    <class 'PostgresBackend'>
    """
    backend_type_str = os.environ.get('SWORD_PRIMARY_BACKEND', '').lower()

    if backend_type_str == 'postgres':
        pg_url = os.environ.get('SWORD_POSTGRES_URL')
        if not pg_url:
            raise ValueError(
                "SWORD_PRIMARY_BACKEND=postgres but SWORD_POSTGRES_URL not set"
            )
        return get_backend(pg_url, BackendType.POSTGRES)

    elif backend_type_str == 'duckdb':
        db_path = os.environ.get('SWORD_DUCKDB_PATH')
        if not db_path:
            raise ValueError(
                "SWORD_PRIMARY_BACKEND=duckdb but SWORD_DUCKDB_PATH not set"
            )
        return get_backend(db_path, BackendType.DUCKDB)

    # No explicit configuration
    return None
