# export-v17c

Export SWORD v17c data to various formats.

## Usage
```
/export-v17c NA
/export-v17c all --formats netcdf,gpkg
/export-v17c EU --formats parquet --output-dir /path/to/output
```

## Arguments
- `region`: Region code (NA, SA, EU, AF, AS, OC) or "all"
- `--formats`: Comma-separated list (netcdf, gpkg, parquet, duckdb). Default: all
- `--output-dir`: Output directory. Default: output/exports/

## Instructions

When the user invokes this skill:

1. **Parse arguments**
   - Validate region code
   - Parse format list (default: all formats)
   - Validate output directory exists or create it

2. **Load data**
   ```python
   from updates.sword_duckdb import SWORDWorkflow

   workflow = SWORDWorkflow(user_id="export-skill")
   sword = workflow.load('data/duckdb/sword_v17c.duckdb', region)
   ```

3. **Run pre-export validation**
   - Quick lint check (ERROR severity only)
   - Warn if there are unresolved errors
   - Ask user to confirm if errors exist

4. **Export each format**

   ### NetCDF (Primary public release)
   - Must match v17b structure exactly
   - Verify variable names, dimensions, attributes
   - Output: `{region}_reaches.nc`, `{region}_nodes.nc`, `{region}_centerlines.nc`

   ### GeoPackage (GIS users)
   - CRS: EPSG:4326
   - Layers: reaches, nodes, centerlines
   - Output: `sword_v17c_{region}.gpkg`

   ### Parquet (Cloud/analytics)
   - GeoParquet spec compliance
   - Geometry as WKB
   - Output: `sword_v17c_{region}_reaches.parquet`, etc.

   ### DuckDB (Dev only)
   - Direct copy with region filter
   - Output: `sword_v17c_{region}.duckdb`

5. **Verify exports**
   - Round-trip test: read back and compare row counts
   - CRS verification for spatial formats
   - Schema verification for NetCDF

6. **Report summary**
   - Files created with sizes
   - Any warnings/issues
   - Checksum for each file (optional)

## Format Details

| Format | Extension | Use Case | CRS |
|--------|-----------|----------|-----|
| NetCDF | .nc | Official release | EPSG:4326 |
| GeoPackage | .gpkg | GIS analysis | EPSG:4326 |
| Parquet | .parquet | Cloud analytics | WKB geometry |
| DuckDB | .duckdb | Development | N/A |

## Notes

- NetCDF export must be compatible with v17b readers
- GeoPackage layer names: sword_reaches, sword_nodes, sword_centerlines
- Parquet includes geo metadata per GeoParquet spec
