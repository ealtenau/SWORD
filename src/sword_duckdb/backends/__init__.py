# -*- coding: utf-8 -*-
"""
SWORD Database Backends
=======================

This module provides a backend abstraction layer that allows SWORD to work
with either DuckDB (local file) or PostgreSQL (remote/concurrent access).

The backend system enables:
- DuckDB for local, single-user, high-performance operations
- PostgreSQL for concurrent multi-user access with proper locking

Example Usage:
    from sword_duckdb.backends import get_backend, BackendType

    # DuckDB (file path)
    backend = get_backend('data/duckdb/sword_v17c.duckdb')

    # PostgreSQL (connection string)
    backend = get_backend('postgresql://user:pass@localhost/sword')

    # Use the backend
    with backend.connect() as conn:
        result = backend.query(conn, "SELECT * FROM reaches LIMIT 10")
"""

from .base import (
    DatabaseBackend,
    BackendType,
    TransactionContext,
    ConnectionType,
)
from .duckdb import DuckDBBackend
from .postgres import PostgresBackend
from .factory import get_backend, detect_backend_type

__all__ = [
    # Protocol and base classes
    "DatabaseBackend",
    "BackendType",
    "TransactionContext",
    "ConnectionType",
    # Implementations
    "DuckDBBackend",
    "PostgresBackend",
    # Factory
    "get_backend",
    "detect_backend_type",
]
