# -*- coding: utf-8 -*-
"""
SWORD DuckDB Backend
====================

This module provides DuckDB-based storage for the SWOT River Database (SWORD).
It offers SQL query capabilities, spatial queries, and improved performance
while maintaining compatibility with existing workflows.

Modules:
    sword_class: DuckDB-backed SWORD class (drop-in replacement for original)
    sword_db: Connection management and database operations
    schema: Table definitions and schema creation
    migrations: NetCDF to DuckDB migration utilities
    views: View wrapper classes for numpy-array-style access
    reactive: Automatic recalculation of derived attributes
    queries: Common SQL query patterns (TODO)
    export: GeoParquet, Shapefile, and GeoPackage export functions (TODO)

Example Usage:
    from sword_duckdb import SWORD

    # Load from DuckDB (same interface as original SWORD class)
    sword = SWORD('data/duckdb/sword_v17b.duckdb', 'NA')

    # Access data
    print(sword.reaches.wse[:5])
    print(sword.nodes.facc[:5])

    # Reactive updates (automatic recalculation)
    from sword_duckdb import SWORDReactive, mark_geometry_changed
    reactive = SWORDReactive(sword)
    mark_geometry_changed(reactive, reach_ids=[12345678901])
    reactive.recalculate()
"""

from .sword_db import SWORDDatabase, create_database
from .schema import create_schema, get_schema_sql, SCHEMA_VERSION
from .migrations import migrate_region, migrate_all_regions, validate_migration, build_all_geometry
from .sword_class import SWORD
from .views import CenterlinesView, NodesView, ReachesView, WritableArray
from .reactive import (
    SWORDReactive,
    DependencyGraph,
    ChangeType,
    mark_geometry_changed,
    mark_topology_changed,
    full_recalculate,
)

__all__ = [
    # Main SWORD class (drop-in replacement)
    'SWORD',
    # View classes
    'CenterlinesView',
    'NodesView',
    'ReachesView',
    'WritableArray',
    # Reactive updates
    'SWORDReactive',
    'DependencyGraph',
    'ChangeType',
    'mark_geometry_changed',
    'mark_topology_changed',
    'full_recalculate',
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
    'build_all_geometry',
]
