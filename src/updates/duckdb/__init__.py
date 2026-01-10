# -*- coding: utf-8 -*-
"""
SWORD DuckDB Backend
====================

This module provides DuckDB-based storage for the SWOT River Database (SWORD).
It offers SQL query capabilities, spatial queries, and improved performance
while maintaining compatibility with existing workflows.

Modules:
    sword_db: Connection management and database operations
    schema: Table definitions and schema creation
    migrations: NetCDF to DuckDB migration utilities
    queries: Common SQL query patterns (TODO)
    export: GeoParquet, Shapefile, and GeoPackage export functions (TODO)
"""

from .sword_db import SWORDDatabase, create_database
from .schema import create_schema, get_schema_sql, SCHEMA_VERSION
from .migrations import migrate_region, migrate_all_regions, validate_migration

__all__ = [
    # Connection management
    'SWORDDatabase',
    'create_database',
    # Schema
    'create_schema',
    'get_schema_sql',
    'SCHEMA_VERSION',
    # Migration
    'migrate_region',
    'migrate_all_regions',
    'validate_migration',
]
