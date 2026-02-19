# -*- coding: utf-8 -*-
"""
MERIT Guided Search for Facc Correction
========================================

Searches MERIT Hydro upa (flow accumulation) values near reach geometries
to find values matching model estimates.

Key insight: SWORD and MERIT Hydro are not aligned (positional offset + different
D8 routing decisions). Direct sampling would give wrong values. Instead, we:
1. Estimate what facc *should* be from regression
2. Search MERIT for values within same order of magnitude
3. Use best match or fall back to topology-based estimate

MERIT Hydro structure:
    {base_path}/{region}/upa/{tile_group}/{tile}_upa.tif

Example:
    /Volumes/SWORD_DATA/data/MERIT_Hydro/NA/upa/upa_n30w090/n35w085_upa.tif
"""

from typing import Optional, List, Tuple, Dict, Any, Union
from pathlib import Path
import numpy as np
from osgeo import gdal
from shapely.geometry import LineString, Polygon, box
import logging

logger = logging.getLogger(__name__)


class MeritGuidedSearch:
    """
    Search MERIT upa values matching model estimates.

    Parameters
    ----------
    merit_base_path : str
        Path to MERIT Hydro base directory containing region folders.
        Example: /Volumes/SWORD_DATA/data/MERIT_Hydro

    Attributes
    ----------
    base_path : Path
        Base path to MERIT Hydro data.
    tile_cache : dict
        Cache of loaded tile data to avoid re-reading.
    """

    # Tile size in degrees (MERIT uses 5x5 degree tiles)
    TILE_SIZE = 5.0

    # Region mapping for MERIT directory structure
    REGION_MAP = {
        'NA': 'NA',
        'SA': 'SA',
        'EU': 'EU',
        'AF': 'AF',
        'AS': 'AS',
        'OC': 'OC',
    }

    def __init__(self, merit_base_path: str):
        self.base_path = Path(merit_base_path)
        self.tile_cache: Dict[str, Dict[str, Any]] = {}
        self._validate_path()

    def _validate_path(self):
        """Validate MERIT Hydro directory exists."""
        if not self.base_path.exists():
            raise FileNotFoundError(f"MERIT Hydro path not found: {self.base_path}")

        # Check for at least one region
        regions_found = [
            r for r in self.REGION_MAP.values()
            if (self.base_path / r / 'upa').exists()
        ]
        if not regions_found:
            raise ValueError(f"No upa directories found under {self.base_path}")
        logger.info(f"MERIT Hydro initialized with regions: {regions_found}")

    def _get_tile_name(self, lon: float, lat: float) -> str:
        """
        Get MERIT tile name for a coordinate.

        MERIT tiles are named like 'n35w085' where:
        - n/s = north/south of equator
        - 35 = latitude of SW corner
        - w/e = west/east of prime meridian
        - 085 = longitude of SW corner (absolute value)

        Tiles are 5x5 degrees.
        """
        # Round down to tile corner
        lat_tile = int(np.floor(lat / self.TILE_SIZE) * self.TILE_SIZE)
        lon_tile = int(np.floor(lon / self.TILE_SIZE) * self.TILE_SIZE)

        lat_prefix = 'n' if lat_tile >= 0 else 's'
        lon_prefix = 'e' if lon_tile >= 0 else 'w'

        tile_name = f"{lat_prefix}{abs(lat_tile):02d}{lon_prefix}{abs(lon_tile):03d}"
        return tile_name

    def _get_tile_group(self, lon: float, lat: float) -> str:
        """
        Get MERIT tile group directory name.

        Tile groups are 30x30 degree blocks named like 'upa_n30w090'.
        """
        # Round down to 30-degree block
        lat_group = int(np.floor(lat / 30) * 30)
        lon_group = int(np.floor(lon / 30) * 30)

        lat_prefix = 'n' if lat_group >= 0 else 's'
        lon_prefix = 'e' if lon_group >= 0 else 'w'

        return f"upa_{lat_prefix}{abs(lat_group):02d}{lon_prefix}{abs(lon_group):03d}"

    def _get_tile_path(self, region: str, lon: float, lat: float,
                       layer: str = 'upa') -> Optional[Path]:
        """Get path to MERIT tile for a coordinate.

        Parameters
        ----------
        layer : str
            'upa' for flow accumulation, 'dir' for D8 flow direction.
        """
        merit_region = self.REGION_MAP.get(region.upper())
        if not merit_region:
            logger.warning(f"Unknown region: {region}")
            return None

        tile_group = self._get_tile_group(lon, lat).replace('upa_', f'{layer}_')
        tile_name = self._get_tile_name(lon, lat)
        tile_path = self.base_path / merit_region / layer / tile_group / f"{tile_name}_{layer}.tif"

        if tile_path.exists():
            return tile_path
        return None

    def _load_tile(self, tile_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load a MERIT upa tile, with caching.

        Returns dict with 'data', 'transform', 'xmin', 'ymax', etc.
        """
        cache_key = str(tile_path)
        if cache_key in self.tile_cache:
            return self.tile_cache[cache_key]

        try:
            ds = gdal.Open(str(tile_path))
            if ds is None:
                logger.warning(f"Failed to open tile: {tile_path}")
                return None

            transform = ds.GetGeoTransform()
            data = ds.GetRasterBand(1).ReadAsArray()

            tile_info = {
                'data': data,
                'xmin': transform[0],
                'xres': transform[1],
                'ymax': transform[3],
                'yres': transform[5],  # negative
                'width': ds.RasterXSize,
                'height': ds.RasterYSize,
            }

            ds = None  # Close dataset
            self.tile_cache[cache_key] = tile_info
            return tile_info

        except Exception as e:
            logger.error(f"Error loading tile {tile_path}: {e}")
            return None

    def _sample_point(self, tile_info: Dict, lon: float, lat: float) -> Optional[float]:
        """Sample a single point from loaded tile data."""
        col = int((lon - tile_info['xmin']) / tile_info['xres'])
        row = int((lat - tile_info['ymax']) / tile_info['yres'])

        if 0 <= col < tile_info['width'] and 0 <= row < tile_info['height']:
            value = tile_info['data'][row, col]
            # MERIT uses -9999 or similar for nodata
            if value > 0:
                return float(value)
        return None

    def _sample_in_polygon(
        self,
        polygon: Polygon,
        region: str,
        sample_resolution: float = 0.001  # ~100m in degrees
    ) -> List[float]:
        """
        Sample all MERIT upa values within a polygon.

        Parameters
        ----------
        polygon : Polygon
            Search area polygon (in WGS84 degrees).
        region : str
            SWORD region code.
        sample_resolution : float
            Grid resolution for sampling (degrees).

        Returns
        -------
        list of float
            All valid upa values found within the polygon.
        """
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = bounds

        # Find all tiles that overlap this polygon
        tiles_needed = set()
        for lon in np.arange(minx, maxx + self.TILE_SIZE, self.TILE_SIZE):
            for lat in np.arange(miny, maxy + self.TILE_SIZE, self.TILE_SIZE):
                tile_path = self._get_tile_path(region, lon, lat)
                if tile_path:
                    tiles_needed.add(tile_path)

        if not tiles_needed:
            logger.debug(f"No tiles found for region {region} in bounds {bounds}")
            return []

        values = []

        # Load each tile and sample within polygon
        for tile_path in tiles_needed:
            tile_info = self._load_tile(tile_path)
            if tile_info is None:
                continue

            # Compute tile bounds
            tile_minx = tile_info['xmin']
            tile_maxx = tile_minx + tile_info['width'] * tile_info['xres']
            tile_maxy = tile_info['ymax']
            tile_miny = tile_maxy + tile_info['height'] * tile_info['yres']

            # Clip sampling bounds to tile and polygon
            sample_minx = max(minx, tile_minx)
            sample_maxx = min(maxx, tile_maxx)
            sample_miny = max(miny, tile_miny)
            sample_maxy = min(maxy, tile_maxy)

            # Sample on grid
            for lon in np.arange(sample_minx, sample_maxx, sample_resolution):
                for lat in np.arange(sample_miny, sample_maxy, sample_resolution):
                    # Check if point is in polygon (for non-rectangular buffers)
                    from shapely.geometry import Point
                    if polygon.contains(Point(lon, lat)):
                        val = self._sample_point(tile_info, lon, lat)
                        if val is not None:
                            values.append(val)

        return values

    def search_near_reach(
        self,
        geom: Union[LineString, Any],
        region: str,
        facc_expected: float,
        buffer_m: Optional[float] = None,
        width: Optional[float] = None,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Search MERIT for upa values near reach matching expected facc.

        Parameters
        ----------
        geom : LineString or similar
            Reach geometry (LINESTRING in WGS84).
        region : str
            SWORD region code.
        facc_expected : float
            Expected facc value from regression estimate.
        buffer_m : float, optional
            Search buffer in meters. If None, computed from width.
        width : float, optional
            Reach width in meters (used to compute buffer if not provided).

        Returns
        -------
        tuple of (best_match, metadata)
            best_match : float or None
                Best matching MERIT upa value, or None if no good match.
            metadata : dict
                Search statistics including searched, matched counts.
        """
        metadata = {
            'searched': 0,
            'matched': 0,
            'candidates': [],
            'buffer_m': None,
            'facc_expected': facc_expected,
        }

        # Compute buffer
        if buffer_m is None:
            buffer_m = min(2 * (width or 100), 2000)
        metadata['buffer_m'] = buffer_m

        # Convert meters to approximate degrees
        # At equator, 1 degree ≈ 111km. Use centroid latitude for better estimate.
        try:
            centroid = geom.centroid
            lat = centroid.y
        except Exception:
            lat = 0

        meters_per_degree = 111320 * np.cos(np.radians(lat))
        buffer_deg = buffer_m / meters_per_degree

        # Buffer the geometry
        try:
            buffered = geom.buffer(buffer_deg)
        except Exception as e:
            logger.warning(f"Failed to buffer geometry: {e}")
            return None, metadata

        # Sample all MERIT values in buffer
        merit_values = self._sample_in_polygon(buffered, region)
        metadata['searched'] = len(merit_values)

        if not merit_values:
            return None, metadata

        # Filter to same order of magnitude as expected
        if facc_expected > 0:
            order_low = facc_expected / 10
            order_high = facc_expected * 10
            candidates = [v for v in merit_values if order_low <= v <= order_high]
        else:
            candidates = merit_values

        metadata['matched'] = len(candidates)
        metadata['candidates'] = candidates[:100]  # Limit stored candidates

        if not candidates:
            return None, metadata

        # Return closest to estimate
        best = min(candidates, key=lambda x: abs(x - facc_expected))
        return best, metadata

    def search_near_reach_topo(
        self,
        geom: Union[LineString, Any],
        region: str,
        topo_expected: float,
        buffer_m: Optional[float] = None,
        width: Optional[float] = None,
        order_of_magnitude: float = 10.0,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Search MERIT for upa values near reach using topology-consistent selection.

        Unlike search_near_reach (which picks closest to a regression estimate),
        this picks the candidate closest to the topology expectation:
        ``sum(corrected_upstream) + median_lateral``.

        Parameters
        ----------
        geom : LineString or similar
            Reach geometry (LINESTRING in WGS84).
        region : str
            SWORD region code.
        topo_expected : float
            Topology-based expected facc (sum corrected upstream + median lateral).
        buffer_m : float, optional
            Search buffer in meters. If None, computed from width.
        width : float, optional
            Reach width in meters (used to compute buffer if not provided).
        order_of_magnitude : float
            Candidate filter: keep values within this factor of topo_expected.

        Returns
        -------
        tuple of (best_match, metadata)
            best_match : float or None
                Best matching MERIT upa value, or None if no good match.
            metadata : dict
                Search statistics.
        """
        metadata = {
            'searched': 0,
            'matched': 0,
            'candidates': [],
            'buffer_m': None,
            'topo_expected': topo_expected,
            'selection': 'topo_consistent',
        }

        if buffer_m is None:
            buffer_m = min(3 * (width or 100), 3000)
        metadata['buffer_m'] = buffer_m

        try:
            centroid = geom.centroid
            lat = centroid.y
        except Exception:
            lat = 0

        meters_per_degree = 111320 * np.cos(np.radians(lat))
        if meters_per_degree <= 0:
            meters_per_degree = 111320
        buffer_deg = buffer_m / meters_per_degree

        try:
            buffered = geom.buffer(buffer_deg)
        except Exception as e:
            logger.warning(f"Failed to buffer geometry: {e}")
            return None, metadata

        merit_values = self._sample_in_polygon(buffered, region)
        metadata['searched'] = len(merit_values)

        if not merit_values:
            return None, metadata

        # Filter to within order_of_magnitude of topo_expected
        if topo_expected > 0:
            order_low = topo_expected / order_of_magnitude
            order_high = topo_expected * order_of_magnitude
            candidates = [v for v in merit_values if order_low <= v <= order_high]
        else:
            candidates = merit_values

        metadata['matched'] = len(candidates)
        metadata['candidates'] = candidates[:100]

        if not candidates:
            return None, metadata

        # Pick candidate closest to topology expectation
        best = min(candidates, key=lambda x: abs(x - topo_expected))
        return best, metadata

    # D8 direction encoding: value -> (row_offset, col_offset)
    # 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
    D8_OFFSETS = {
        1: (0, 1),      # East
        2: (1, 1),      # SE
        4: (1, 0),      # South
        8: (1, -1),     # SW
        16: (0, -1),    # West
        32: (-1, -1),   # NW
        64: (-1, 0),    # North
        128: (-1, 1),   # NE
    }

    def walk_d8_downstream(
        self,
        lon: float,
        lat: float,
        region: str,
        target_min: float,
        max_steps: int = 150,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Walk downstream along MERIT D8 flow direction from a point.

        Starts at the MERIT pixel nearest to (lon, lat), then follows the D8
        flow direction for up to max_steps. At each cell, checks UPA. Returns
        the first UPA value >= target_min, or the maximum found if none qualifies.

        This "snaps" to MERIT's actual thalweg, solving the structural mismatch
        where SWORD's junction is offset from MERIT's confluence.

        Parameters
        ----------
        lon, lat : float
            Starting point in WGS84.
        region : str
            SWORD region code.
        target_min : float
            Minimum UPA needed (corrected upstream value).
        max_steps : int
            Maximum D8 cells to walk (default 150 ≈ 13.5km at 90m).

        Returns
        -------
        (best_value, metadata)
        """
        metadata = {
            'start_lon': lon,
            'start_lat': lat,
            'steps_walked': 0,
            'values_seen': [],
            'found_above_target': False,
            'target_min': target_min,
        }

        # Load UPA and DIR tiles at starting point
        upa_path = self._get_tile_path(region, lon, lat, layer='upa')
        dir_path = self._get_tile_path(region, lon, lat, layer='dir')
        if upa_path is None or dir_path is None:
            return None, metadata

        upa_info = self._load_tile(upa_path)
        dir_info = self._load_tile(dir_path)
        if upa_info is None or dir_info is None:
            return None, metadata

        # Starting pixel
        col = int((lon - upa_info['xmin']) / upa_info['xres'])
        row = int((lat - upa_info['ymax']) / upa_info['yres'])

        best_above = None
        best_below = None
        visited = set()

        for step in range(max_steps):
            if not (0 <= col < upa_info['width'] and 0 <= row < upa_info['height']):
                break
            if (row, col) in visited:
                break  # Cycle or flat
            visited.add((row, col))

            upa_val = float(upa_info['data'][row, col])
            if upa_val > 0:
                metadata['values_seen'].append(round(upa_val, 2))

                if upa_val >= target_min:
                    if best_above is None or upa_val < best_above:
                        best_above = upa_val  # Min value above target
                    metadata['found_above_target'] = True
                else:
                    if best_below is None or upa_val > best_below:
                        best_below = upa_val  # Max value below target

            # Follow D8 direction
            d8_val = int(dir_info['data'][row, col])
            if d8_val not in self.D8_OFFSETS:
                break  # Nodata, ocean, or flat
            dr, dc = self.D8_OFFSETS[d8_val]
            row += dr
            col += dc

        metadata['steps_walked'] = len(visited)

        if best_above is not None:
            return best_above, metadata
        elif best_below is not None:
            return best_below, metadata
        return None, metadata

    def search_batch(
        self,
        reaches_df,
        region_col: str = 'region',
        geom_col: str = 'geometry',
        facc_expected_col: str = 'facc_expected',
        width_col: str = 'width',
    ) -> List[Tuple[Optional[float], Dict[str, Any]]]:
        """
        Search MERIT for multiple reaches.

        Parameters
        ----------
        reaches_df : DataFrame
            DataFrame with reach geometries and expected facc values.
        region_col : str
            Column name for region.
        geom_col : str
            Column name for geometry.
        facc_expected_col : str
            Column name for expected facc.
        width_col : str
            Column name for width.

        Returns
        -------
        list of (best_match, metadata) tuples
            One result per row in reaches_df.
        """
        results = []

        for idx, row in reaches_df.iterrows():
            geom = row.get(geom_col)
            region = row.get(region_col)
            facc_expected = row.get(facc_expected_col, 0)
            width = row.get(width_col, 100)

            if geom is None or region is None:
                results.append((None, {'error': 'missing geometry or region'}))
                continue

            result = self.search_near_reach(
                geom=geom,
                region=region,
                facc_expected=facc_expected,
                width=width,
            )
            results.append(result)

        return results

    def clear_cache(self):
        """Clear tile cache to free memory."""
        self.tile_cache.clear()


def create_merit_search(merit_path: Optional[str] = None) -> Optional[MeritGuidedSearch]:
    """
    Factory function to create MeritGuidedSearch if path is valid.

    Parameters
    ----------
    merit_path : str, optional
        Path to MERIT Hydro base directory.

    Returns
    -------
    MeritGuidedSearch or None
        Search object if path is valid, None otherwise.
    """
    if merit_path is None:
        return None

    try:
        return MeritGuidedSearch(merit_path)
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not initialize MERIT search: {e}")
        return None
