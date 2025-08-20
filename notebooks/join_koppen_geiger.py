#!/usr/bin/env python3
"""
Join Köppen–Geiger climate classes from a raster TIF to a point GeoJSON.

Usage:
  python notebooks/join_koppen_geiger.py --geojson data/analysis/weekly_variability_20734.geojson \
      --tif "/Users/jakegearon/Downloads/Koppen Geiger TIF/1991_2020/koppen_geiger_0p00833333.tif" \
      --out data/analysis/weekly_variability_20734_kg.geojson

Notes:
- Adds integer `kg_class` (raster value). Mapping to textual classes can be joined later.
- Reprojects points to TIF CRS before sampling.
"""

import argparse
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from pyproj import Transformer


def sample_raster_at_points(geojson_path: Path, tif_path: Path, out_path: Path) -> None:
    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        raise ValueError("Input GeoJSON has no features")

    # Ensure geometry present
    if 'geometry' not in gdf.columns:
        raise ValueError("Input GeoJSON must have point geometry")

    # Open raster and build transformer to its CRS
    with rasterio.open(tif_path) as src:
        tif_crs = src.crs
        # Build coordinate arrays in raster CRS
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        transformer = Transformer.from_crs(gdf.crs, tif_crs, always_xy=True)
        xs = gdf.geometry.x.values
        ys = gdf.geometry.y.values
        xr, yr = transformer.transform(xs, ys)

        # Sample values using a warped VRT to avoid manual reprojection issues
        with WarpedVRT(src, crs=tif_crs, resampling=Resampling.nearest) as vrt:
            # Build iterable of (x,y) pairs in raster CRS
            coords = list(zip(xr, yr))
            values = list(vrt.sample(coords))  # returns array per band
            vals = [v[0] if v.size else np.nan for v in values]

    gdf['kg_class'] = np.array(vals)
    # Cast to Int64 (nullable integer) if possible
    try:
        gdf['kg_class'] = gdf['kg_class'].astype('Int64')
    except Exception:
        pass

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver='GeoJSON')
    print(f"Wrote: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geojson', required=True, help='Input point GeoJSON path')
    ap.add_argument('--tif', required=True, help='Köppen–Geiger TIF path')
    ap.add_argument('--out', required=True, help='Output GeoJSON with kg_class appended')
    args = ap.parse_args()
    sample_raster_at_points(Path(args.geojson), Path(args.tif), Path(args.out))


if __name__ == '__main__':
    main()
