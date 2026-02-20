---
name: export-v17c
description: Use when exporting SWORD v17c data to NetCDF, GeoPackage, Parquet, PostgreSQL, or DuckDB formats for release, sharing, or backup
---

# Export v17c

Export SWORD v17c data from DuckDB to one or more output formats.

## Usage

```
/export-v17c NA
/export-v17c all --formats netcdf,gpkg
/export-v17c EU --formats parquet,postgres --output-dir /path/to/output
```

## Arguments

| Arg | Values | Default |
|-----|--------|---------|
| region | NA, SA, EU, AF, AS, OC, all | required |
| --formats | netcdf, gpkg, parquet, postgres, duckdb | all five |
| --output-dir | path | output/exports/ |

Validate region is one of the six codes or "all". Normalize to uppercase.

## Bug: Do NOT Use workflow.export()

`workflow._do_export()` passes `self._sword.db` (a DuckDB connection) where export functions expect a `SWORD` instance. This causes AttributeError on `.reaches`, `.nodes`, etc. **Call export functions directly or use raw SQL.**

## Execution Steps

### 1. Pre-export Lint (ERROR checks only)

```bash
python -m src.sword_duckdb.lint.cli \
  --db data/duckdb/sword_v17c.duckdb \
  --region {REGION} --fail-on-error
```

If errors found, show them and ask user: continue or abort?

### 2. Load Data

```python
from sword_duckdb import SWORDWorkflow

workflow = SWORDWorkflow(user_id="export")
sword = workflow.load('data/duckdb/sword_v17c.duckdb', region)
# sword instance is workflow._sword
```

### 3. Export Per Format

#### GeoPackage

```python
from sword_duckdb.export import export_to_geopackage

export_to_geopackage(
    sword=workflow._sword,          # SWORD instance, NOT .db
    output_path=f"{out_dir}/sword_{region}_v17c.gpkg",
    tables=['reaches', 'nodes'],
)
```

**Limitation:** Only exports ~10 columns (reach_id, dist_out, facc, wse, width, slope, river_name, stream_order, main_side, end_rch). Full column export is issue #90.

#### GeoParquet

```python
from sword_duckdb.export import export_to_geoparquet

# One call per table (reaches, nodes, centerlines)
for table in ['reaches', 'nodes', 'centerlines']:
    export_to_geoparquet(
        sword=workflow._sword,      # SWORD instance, NOT .db
        output_path=f"{out_dir}/sword_{region}_v17c_{table}.parquet",
        table=table,
        compression='snappy',
    )
```

**Same column limitation as GeoPackage.**

#### NetCDF

```python
# save_nc is on the SWORD class, NOT in export module
workflow._sword.save_nc(f"{out_dir}/sword_{region}_v17c.nc")
```

**Limitation:** Only exports v17b-era variables. v17c-specific columns (hydro_dist_out, is_mainstem_edge, etc.) are NOT included. Full v17c NetCDF is issue #89.

#### PostgreSQL

```python
import os
from sword_duckdb.export import export_to_postgres

pg_url = os.environ.get('SWORD_POSTGRES_URL')
if not pg_url:
    # Ask user for connection string or check .env
    raise RuntimeError("Set SWORD_POSTGRES_URL in .env")

export_to_postgres(
    sword=workflow._sword,          # SWORD instance, NOT .db
    connection_string=pg_url,
    tables=['reaches', 'nodes'],
    prefix=f"{region.lower()}_",
    drop_existing=True,             # ask user first
    include_topology=True,
)
```

**Note:** Requires `psycopg2` installed. Uses connection string from `SWORD_POSTGRES_URL` env var.

#### DuckDB Copy

No existing function. Use raw SQL:

```python
import duckdb

src = duckdb.connect('data/duckdb/sword_v17c.duckdb', read_only=True)
out_path = f"{out_dir}/sword_{region}_v17c.duckdb"
src.execute(f"ATTACH '{out_path}' AS out")

for table in ['reaches', 'nodes', 'centerlines']:
    src.execute(f"""
        CREATE TABLE out.{table} AS
        SELECT * FROM {table} WHERE region = '{region}'
    """)

# Topology has no region column — filter by reach_id
src.execute(f"""
    CREATE TABLE out.reach_topology AS
    SELECT rt.* FROM reach_topology rt
    JOIN reaches r ON rt.reach_id = r.reach_id
    WHERE r.region = '{region}'
""")

src.execute("DETACH out")
src.close()
```

### 4. Multi-Region ("all")

Loop sequentially over `['NA', 'SA', 'EU', 'AF', 'AS', 'OC']`. Each region requires a fresh `workflow.load()`. Close previous workflow before loading next to avoid DuckDB lock contention.

### 5. Verify Outputs

For each exported file:

| Check | How |
|-------|-----|
| File exists | `Path(f).exists()` |
| Row count | Read back and compare to source `SELECT COUNT(*) FROM {table} WHERE region='{R}'` |
| CRS (gpkg/parquet) | `geopandas.read_file(f).crs == 'EPSG:4326'` |
| NetCDF structure | Open with netCDF4, verify groups (centerlines, nodes, reaches) exist |
| DuckDB tables | `duckdb.connect(f).execute("SHOW TABLES").fetchall()` |

### 6. Report Summary

Print a table:

```
| Format   | File                              | Size   | Rows    | Status |
|----------|-----------------------------------|--------|---------|--------|
| gpkg     | sword_NA_v17c.gpkg                | 45 MB  | 38,696  | OK     |
| parquet  | sword_NA_v17c_reaches.parquet     | 12 MB  | 38,696  | OK     |
| netcdf   | sword_NA_v17c.nc                  | 890 MB | 38,696  | OK     |
| postgres | postgresql://...                  | —      | 38,696  | OK     |
| duckdb   | sword_NA_v17c.duckdb              | 200 MB | 38,696  | OK     |
```

Include any warnings (lint failures, missing columns, skipped formats).

## Known Limitations

- GeoPackage/Parquet export only ~10 columns (issue #90)
- NetCDF missing v17c-specific columns (issue #89)
- `workflow.export()` is broken — always call export functions directly
- DuckDB copy doesn't include provenance tables (sword_operations, sword_value_snapshots)
