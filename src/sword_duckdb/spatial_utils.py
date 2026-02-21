"""
Spatial utility functions for SWORD.

Ported from src/_legacy/updates/geo_utils.py.
"""

from __future__ import annotations

import math
from collections import deque

import duckdb
import numpy as np
import utm as _utm
from pyproj import Proj


def meters_to_degrees(meters: float, latitude: float) -> float:
    """Convert meters to decimal degrees at a given latitude.

    Args:
        meters: Distance in meters.
        latitude: Latitude in decimal degrees (WGS84).

    Returns:
        Equivalent distance in decimal degrees of longitude.
    """
    cos_lat = math.cos(math.radians(latitude))
    if cos_lat == 0:
        return float("inf")
    return meters / (111320.0 * cos_lat)


def reproject_utm(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Project lat/lon arrays to UTM using the most common zone.

    Args:
        latitudes: Array of latitudes (WGS84 EPSG:4326).
        longitudes: Array of longitudes (WGS84 EPSG:4326).

    Returns:
        (easting, northing, zone_number, zone_letter)
    """
    zone_nums = []
    zone_lets = []
    for lat, lon in zip(latitudes, longitudes):
        _, _, zn, zl = _utm.from_latlon(float(lat), float(lon))
        zone_nums.append(zn)
        zone_lets.append(zl)

    zone_nums_arr = np.array(zone_nums)
    values, counts = np.unique(zone_nums_arr, return_counts=True)
    dominant_zone = int(values[np.argmax(counts)])
    dominant_letter = zone_lets[np.where(zone_nums_arr == dominant_zone)[0][0]]

    south = "+south " if np.mean(latitudes) < 0 else ""
    proj = Proj(
        f"+proj=utm {south}+zone={dominant_zone} +ellps=WGS84 +datum=WGS84 +units=m"
    )
    easting, northing = proj(
        np.asarray(longitudes, dtype=float),
        np.asarray(latitudes, dtype=float),
    )
    return (
        np.asarray(easting),
        np.asarray(northing),
        dominant_zone,
        dominant_letter,
    )


def get_all_upstream(
    con: duckdb.DuckDBPyConnection,
    reach_id: int,
    region: str | None = None,
) -> set[int]:
    """BFS to collect all upstream reach IDs from reach_topology.

    Args:
        con: DuckDB connection.
        reach_id: Starting reach ID.
        region: Optional region filter.

    Returns:
        Set of all upstream reach IDs (excluding start).
    """
    return _bfs_topology(con, reach_id, direction="up", region=region)


def get_all_downstream(
    con: duckdb.DuckDBPyConnection,
    reach_id: int,
    region: str | None = None,
) -> set[int]:
    """BFS to collect all downstream reach IDs from reach_topology.

    Args:
        con: DuckDB connection.
        reach_id: Starting reach ID.
        region: Optional region filter.

    Returns:
        Set of all downstream reach IDs (excluding start).
    """
    return _bfs_topology(con, reach_id, direction="down", region=region)


def _bfs_topology(
    con: duckdb.DuckDBPyConnection,
    start_id: int,
    direction: str,
    region: str | None = None,
) -> set[int]:
    """Internal BFS over reach_topology table."""
    region_clause = f"AND region = '{region}'" if region else ""
    visited: set[int] = set()
    queue: deque[int] = deque([start_id])

    while queue:
        current = queue.popleft()
        rows = con.execute(
            f"""
            SELECT neighbor_reach_id
            FROM reach_topology
            WHERE reach_id = ? AND direction = ? {region_clause}
              AND neighbor_reach_id != 0
            """,
            [current, direction],
        ).fetchall()

        for (neighbor_id,) in rows:
            if neighbor_id not in visited and neighbor_id != start_id:
                visited.add(neighbor_id)
                queue.append(neighbor_id)

    return visited
