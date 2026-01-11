# SWORD DuckDB Migration Status

## Project Overview
Migrating SWORD (SWOT River Database) from NetCDF4 to DuckDB for SQL query capabilities, better performance, and simpler tooling.

## Current State: Phase 2 COMPLETE

### Completed Phases

**Phase 1: Schema & Infrastructure** ✅
- Created `src/updates/sword_duckdb/` package (renamed from `duckdb` to avoid namespace conflict)
- Files: `__init__.py`, `schema.py`, `sword_db.py`, `migrations.py`
- Schema v1.1.0 with composite primary keys

**Phase 2: Data Migration** ✅
- All 6 regions migrated and validated
- Database: `data/duckdb/sword_v17b.duckdb` (9.9 GB)

| Region | Centerlines | Nodes | Reaches |
|--------|-------------|-------|---------|
| NA | 10,349,756 | 1,705,705 | 38,696 |
| SA | 12,861,259 | 2,135,054 | 42,159 |
| EU | 7,048,975 | 1,174,236 | 31,103 |
| AF | 6,725,494 | 1,116,775 | 21,441 |
| OC | 4,125,498 | 686,595 | 15,089 |
| AS | 25,820,450 | 4,294,089 | 100,185 |
| **Total** | **66,931,432** | **11,112,454** | **248,673** |

### Key Technical Decisions

1. **Composite Primary Keys** - cl_id and node_id are NOT globally unique, only unique within region:
   - centerlines: (cl_id, region)
   - nodes: (node_id, region)
   - reaches: (reach_id, region) - reach_id IS globally unique but kept consistent

2. **Package Name** - `sword_duckdb` not `duckdb` to avoid conflict with duckdb library

3. **Geometry Deferred** - `build_geometry=False` by default, use `build_all_geometry()` after migration

4. **Batch Size** - 25,000 records (reduced from 100K to avoid memory crashes)

### Git Status
- Branch: `gearon_dev3`
- Latest commit: `30d1424` - "Rename duckdb package to sword_duckdb and add composite key support"
- GitHub Issue: #1 (https://github.com/ealtenau/SWORD/issues/1)

## Next Steps: Phase 3 - SWORD Class Refactoring

### Goals
1. Create DuckDB-backed version of SWORD class (`src/updates/sword.py`)
2. Replace NumPy array operations with SQL queries
3. Maintain backward compatibility with existing pipelines
4. Add export functions (GeoParquet, Shapefile, GeoPackage)

### Key Files to Modify/Create
- `src/updates/sword.py` - Main SWORD class (~1500 lines, uses NumPy arrays)
- `src/updates/sword_utils.py` - I/O utilities with read_nc(), write_nc()
- `src/updates/sword_duckdb/queries.py` - Common SQL patterns (TODO)
- `src/updates/sword_duckdb/export.py` - Export functions (TODO)

### Architecture Notes
Current SWORD class uses:
- `centerlines`, `nodes`, `reaches` objects with NumPy arrays
- Direct attribute access like `sword.reaches.wse`, `sword.nodes.facc`
- Multi-dimensional arrays: `reach_id[4,N]`, `node_id[4,N]` for neighbors

DuckDB approach should:
- Provide SQL query interface
- Return pandas DataFrames or allow attribute-style access
- Support spatial queries via DuckDB Spatial extension
- Export to Shapefile/GeoPackage for downstream users

### Usage Example (Target API)
```python
from sword_duckdb import SWORDDatabase

db = SWORDDatabase('data/duckdb/sword_v17b.duckdb')

# SQL queries
df = db.query("SELECT * FROM reaches WHERE facc > 10000 AND region = 'NA'")

# Spatial queries
df = db.query("""
    SELECT * FROM nodes
    WHERE ST_Within(geom, ST_MakeEnvelope(-100, 30, -90, 40))
""")

# Export
db.export_shapefile('reaches', 'output/na_reaches.shp', region='NA')
```

## Files Reference

```
src/updates/
├── sword_duckdb/           # NEW - DuckDB backend
│   ├── __init__.py         # Exports: SWORDDatabase, migrate_region, etc.
│   ├── schema.py           # Table definitions (v1.1.0)
│   ├── sword_db.py         # Connection manager
│   ├── migrations.py       # NetCDF → DuckDB migration
│   ├── queries.py          # TODO: Common SQL patterns
│   └── export.py           # TODO: Export functions
├── sword.py                # Original SWORD class (to refactor)
└── sword_utils.py          # I/O utilities
```

## Testing the Migration

```python
import sys
sys.path.insert(0, './src/updates')
from sword_duckdb import SWORDDatabase

db = SWORDDatabase('data/duckdb/sword_v17b.duckdb', read_only=True)
print(db.get_regions())  # ['AF', 'AS', 'EU', 'NA', 'OC', 'SA']
print(db.count_records())  # Shows all table counts

# Example query
df = db.query("SELECT region, COUNT(*) as cnt FROM reaches GROUP BY region")
print(df)
```
