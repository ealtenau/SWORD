#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create minimal SWORD test database fixture.

Generates a small (~100 reaches, <10MB) database for fast unit testing.
The database includes realistic synthetic data with proper topology,
network structure, and all required tables populated.

Usage:
    python create_test_db.py                    # Create at default location
    python create_test_db.py /path/to/output.duckdb  # Create at custom path
    SWORD_REGEN_TEST_DB=true pytest ...         # Regenerate during tests

Data Structure:
    - 100 reaches with 11-digit IDs (11000000000-11000000099)
    - ~500 nodes with 14-digit IDs (5 per reach)
    - ~2000 centerlines (20 per reach)
    - Sparse topology (10 headwaters, 1 outlet, branching network)
    - SWOT orbits (8 per reach)
    - Ice flags (366 days × 100 reaches)
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
main_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(main_dir))

import duckdb
from src.updates.sword_duckdb.schema import create_schema


# ==============================================================================
# Configuration
# ==============================================================================

# Network structure
NUM_REACHES = 100
NODES_PER_REACH = 5
CENTERLINES_PER_REACH = 20
ORBITS_PER_REACH = 8

# ID patterns (matching SWORD conventions)
# Reach IDs: 11-digit, first 2 digits encode region (11=NA)
BASE_REACH_ID = 11000000000

# Node IDs: 14-digit, derived from reach ID
# Pattern: 140 + reach_id(11) = 14 digits

# Centerline IDs: sequential starting from 1


# ==============================================================================
# Data Generators
# ==============================================================================


def generate_reach_data(num_reaches: int, region: str, version: str) -> list:
    """Generate synthetic reach data."""
    reaches = []

    # Create a simple network topology:
    # - 10 headwaters (indices 0-9)
    # - 1 outlet (index 99)
    # - Linear chain with tributaries

    for i in range(num_reaches):
        reach_id = BASE_REACH_ID + i

        # Position (spread across NA region)
        x = -100.0 + (i % 10) * 2.0  # Longitude
        y = 40.0 + (i // 10) * 1.0  # Latitude

        # Bounding box
        x_min = x - 0.1
        x_max = x + 0.1
        y_min = y - 0.05
        y_max = y + 0.05

        # Reach metrics (realistic values)
        reach_length = 5000.0 + np.random.uniform(-1000, 1000)
        n_nodes = NODES_PER_REACH
        wse = 100.0 + (100 - i) * 0.5  # Decreasing downstream
        wse_var = 0.1 + np.random.uniform(0, 0.2)
        width = 50.0 + np.random.uniform(0, 100)
        width_var = 5.0 + np.random.uniform(0, 10)
        slope = 0.0001 + np.random.uniform(0, 0.0005)
        max_width = width * 1.5
        facc = 1000.0 + i * 100  # Increasing downstream
        dist_out = (100 - i) * 5000.0  # Decreasing downstream

        # Network attributes
        stream_order = max(1, 5 - (i // 20))  # Higher order near outlet
        path_freq = 10**stream_order
        path_order = i + 1
        path_segs = (i // 10) + 1
        network = 1  # All in same network

        # End reach classification
        if i < 10:
            end_reach = 1  # Headwater
        elif i == 99:
            end_reach = 2  # Outlet
        elif i % 10 == 0:
            end_reach = 3  # Junction
        else:
            end_reach = 0  # Main

        # Topology counts (set later based on actual topology)
        n_rch_up = 0
        n_rch_down = 0

        reaches.append(
            {
                "reach_id": reach_id,
                "region": region,
                "x": x,
                "y": y,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "geom": None,
                "cl_id_min": i * CENTERLINES_PER_REACH + 1,
                "cl_id_max": (i + 1) * CENTERLINES_PER_REACH,
                "reach_length": reach_length,
                "n_nodes": n_nodes,
                "wse": wse,
                "wse_var": wse_var,
                "width": width,
                "width_var": width_var,
                "slope": slope,
                "max_width": max_width,
                "facc": facc,
                "dist_out": dist_out,
                "lakeflag": 0,
                "obstr_type": 0,
                "grod_id": None,
                "hfalls_id": None,
                "n_chan_max": 1,
                "n_chan_mod": 1,
                "n_rch_up": n_rch_up,
                "n_rch_down": n_rch_down,
                "swot_obs": 8,
                "iceflag": 0,
                "low_slope_flag": 0,
                "river_name": f"Test River {i}",
                "edit_flag": None,
                "trib_flag": 0,
                "path_freq": path_freq,
                "path_order": path_order,
                "path_segs": path_segs,
                "stream_order": stream_order,
                "main_side": 0,
                "end_reach": end_reach,
                "network": network,
                "add_flag": 0,
                "version": version,
            }
        )

    return reaches


def generate_node_data(reaches: list, region: str, version: str) -> list:
    """Generate synthetic node data for each reach."""
    nodes = []
    node_counter = 0

    for reach in reaches:
        reach_id = reach["reach_id"]
        base_x = reach["x"]
        base_y = reach["y"]

        for j in range(NODES_PER_REACH):
            # 14-digit node ID: unique sequential ID padded to 14 digits
            # Format: 14 + reach_index(3) + node_seq(2) + padding
            # Example: 14000000000001, 14000000000002, etc.
            node_id = 14000000000000 + node_counter

            # Position along reach
            x = base_x + (j - NODES_PER_REACH // 2) * 0.01
            y = base_y + (j - NODES_PER_REACH // 2) * 0.005

            # Centerline range
            cl_id_min = reach["cl_id_min"] + j * (
                CENTERLINES_PER_REACH // NODES_PER_REACH
            )
            cl_id_max = cl_id_min + (CENTERLINES_PER_REACH // NODES_PER_REACH) - 1

            # Node metrics (derived from reach)
            node_length = reach["reach_length"] / NODES_PER_REACH
            wse = reach["wse"] + (NODES_PER_REACH // 2 - j) * 0.1
            width = reach["width"] * (0.9 + np.random.uniform(0, 0.2))

            nodes.append(
                {
                    "node_id": node_id,
                    "region": region,
                    "x": x,
                    "y": y,
                    "geom": None,
                    "cl_id_min": cl_id_min,
                    "cl_id_max": cl_id_max,
                    "reach_id": reach_id,
                    "node_length": node_length,
                    "wse": wse,
                    "wse_var": reach["wse_var"],
                    "width": width,
                    "width_var": reach["width_var"],
                    "max_width": reach["max_width"],
                    "facc": reach["facc"],
                    "dist_out": reach["dist_out"]
                    + (NODES_PER_REACH - j - 1) * node_length,
                    "lakeflag": reach["lakeflag"],
                    "obstr_type": reach["obstr_type"],
                    "grod_id": None,
                    "hfalls_id": None,
                    "n_chan_max": 1,
                    "n_chan_mod": 1,
                    "wth_coef": 1.0,
                    "ext_dist_coef": 1.0,
                    "meander_length": None,
                    "sinuosity": 1.1 + np.random.uniform(0, 0.3),
                    "river_name": reach["river_name"],
                    "manual_add": 0,
                    "edit_flag": None,
                    "trib_flag": 0,
                    "path_freq": reach["path_freq"],
                    "path_order": reach["path_order"],
                    "path_segs": reach["path_segs"],
                    "stream_order": reach["stream_order"],
                    "main_side": 0,
                    "end_reach": 0,
                    "network": reach["network"],
                    "add_flag": 0,
                    "version": version,
                }
            )

            node_counter += 1

    return nodes


def generate_centerline_data(
    reaches: list, nodes: list, region: str, version: str
) -> list:
    """Generate synthetic centerline data."""
    centerlines = []
    cl_id = 1

    for reach in reaches:
        reach_id = reach["reach_id"]
        base_x = reach["x"]
        base_y = reach["y"]

        # Get nodes for this reach
        reach_nodes = [n for n in nodes if n["reach_id"] == reach_id]

        for j in range(CENTERLINES_PER_REACH):
            # Position along reach centerline
            x = base_x + (j - CENTERLINES_PER_REACH // 2) * 0.002
            y = base_y + (j - CENTERLINES_PER_REACH // 2) * 0.001

            # Assign to closest node
            node_idx = min(
                j * NODES_PER_REACH // CENTERLINES_PER_REACH, NODES_PER_REACH - 1
            )
            node_id = reach_nodes[node_idx]["node_id"]

            centerlines.append(
                {
                    "cl_id": cl_id,
                    "region": region,
                    "x": x,
                    "y": y,
                    "geom": None,
                    "reach_id": reach_id,
                    "node_id": node_id,
                    "version": version,
                }
            )

            cl_id += 1

    return centerlines


def generate_topology(reaches: list, region: str) -> list:
    """
    Generate network topology (upstream/downstream relationships).

    Creates a simple network structure:
    - Indices 0-9: Headwaters (no upstream)
    - Index 99: Outlet (no downstream)
    - Others: Linear chain with tributaries joining
    """
    topology = []

    # Build adjacency: reach i flows to reach i+1 (except outlet)
    for i, reach in enumerate(reaches):
        reach_id = reach["reach_id"]

        # Downstream connections
        if i < 99:  # Not outlet
            downstream_id = BASE_REACH_ID + i + 1
            topology.append(
                {
                    "reach_id": reach_id,
                    "region": region,
                    "direction": "down",
                    "neighbor_rank": 0,
                    "neighbor_reach_id": downstream_id,
                }
            )

        # Upstream connections (inverse of downstream)
        if i > 0:  # Not first reach
            upstream_id = BASE_REACH_ID + i - 1
            topology.append(
                {
                    "reach_id": reach_id,
                    "region": region,
                    "direction": "up",
                    "neighbor_rank": 0,
                    "neighbor_reach_id": upstream_id,
                }
            )

    return topology


def generate_swot_orbits(reaches: list, region: str) -> list:
    """Generate SWOT orbit data for each reach."""
    orbits = []

    for reach in reaches:
        reach_id = reach["reach_id"]

        for rank in range(ORBITS_PER_REACH):
            # Generate orbit IDs (realistic pattern)
            orbit_id = 100000000 + reach_id % 1000 + rank * 1000

            orbits.append(
                {
                    "reach_id": reach_id,
                    "region": region,
                    "orbit_rank": rank,
                    "orbit_id": orbit_id,
                }
            )

    return orbits


def generate_ice_flags(reaches: list) -> list:
    """Generate daily ice flags for each reach (366 days)."""
    ice_flags = []

    for reach in reaches:
        reach_id = reach["reach_id"]

        # Simple seasonal pattern: ice from day 1-60 and 305-366
        for day in range(1, 367):
            if day <= 60 or day >= 305:
                iceflag = 1  # Ice
            else:
                iceflag = 0  # No ice

            ice_flags.append(
                {
                    "reach_id": reach_id,
                    "julian_day": day,
                    "iceflag": iceflag,
                }
            )

    return ice_flags


# ==============================================================================
# Geometry Helpers
# ==============================================================================


def _populate_geometries(conn):
    """Build actual geometries from x/y coordinates using DuckDB spatial.

    Reaches: LineString from their nodes' x/y positions.
    Nodes: Point from x/y.
    Centerlines: Point from x/y.
    """
    try:
        conn.execute("LOAD spatial")
    except Exception:
        try:
            conn.execute("INSTALL spatial; LOAD spatial")
        except Exception:
            print("    Spatial extension unavailable, skipping geometry population")
            return

    # Nodes → Point
    conn.execute("""
        UPDATE nodes SET geom = ST_Point(x, y)
        WHERE x IS NOT NULL AND y IS NOT NULL
    """)

    # Centerlines → Point
    conn.execute("""
        UPDATE centerlines SET geom = ST_Point(x, y)
        WHERE x IS NOT NULL AND y IS NOT NULL
    """)

    # Reaches → LineString built from their nodes (ordered by node_id)
    conn.execute("""
        UPDATE reaches SET geom = sub.line
        FROM (
            SELECT n.reach_id, n.region,
                   ST_MakeLine(LIST(ST_Point(n.x, n.y) ORDER BY n.node_id)) AS line
            FROM nodes n
            GROUP BY n.reach_id, n.region
            HAVING COUNT(*) >= 2
        ) sub
        WHERE reaches.reach_id = sub.reach_id
          AND reaches.region = sub.region
    """)


def _inject_geometry_test_data(conn, region):
    """Add deliberate bad data to exercise geometry checks G013-G021.

    Uses reach indices 90-97 (IDs 11000000090-11000000097) which are
    'normal' reaches in the middle of the network — safe to modify.
    """
    base = 11000000000

    # G013: width > length (reach 90) — set width = 10000, length stays ~5000
    conn.execute(f"""
        UPDATE reaches SET width = 10000.0
        WHERE reach_id = {base + 90} AND region = '{region}'
    """)

    # G015: node far from reach (move first node of reach 91 far away)
    # Shift node x by +2.0 degrees (~200 km) so it's far from its parent reach
    conn.execute(f"""
        UPDATE nodes SET x = x + 2.0,
            geom = ST_Point(x + 2.0, y)
        WHERE reach_id = {base + 91} AND region = '{region}'
        AND node_id = (
            SELECT MIN(node_id) FROM nodes
            WHERE reach_id = {base + 91} AND region = '{region}'
        )
    """)

    # G014: duplicate geometry — make reach 92 and 93 share identical geometry
    conn.execute(f"""
        UPDATE reaches SET geom = (
            SELECT geom FROM reaches
            WHERE reach_id = {base + 92} AND region = '{region}'
        )
        WHERE reach_id = {base + 93} AND region = '{region}'
    """)

    # G016: node spacing outlier — make one node of reach 94 have 5x avg length
    conn.execute(f"""
        UPDATE nodes SET node_length = node_length * 5.0
        WHERE reach_id = {base + 94} AND region = '{region}'
        AND node_id = (
            SELECT MIN(node_id) FROM nodes
            WHERE reach_id = {base + 94} AND region = '{region}'
        )
    """)

    # G018: dist_out gap mismatch — set reach 95 dist_out very high
    conn.execute(f"""
        UPDATE reaches SET dist_out = 999999.0
        WHERE reach_id = {base + 95} AND region = '{region}'
    """)


# ==============================================================================
# Database Creation
# ==============================================================================


def create_minimal_test_db(db_path: str, region: str = "NA", version: str = "v17b"):
    """
    Create a minimal SWORD test database.

    Parameters
    ----------
    db_path : str
        Path to the output DuckDB file.
    region : str
        Region code (default: 'NA').
    version : str
        SWORD version (default: 'v17b').
    """
    print(f"Creating minimal test database: {db_path}")

    # Remove existing file
    if os.path.exists(db_path):
        os.remove(db_path)

    # Create connection
    conn = duckdb.connect(db_path)

    # Try to load spatial extension (optional)
    try:
        conn.execute("INSTALL spatial; LOAD spatial;")
    except Exception:
        print("  Note: Spatial extension not available, skipping geometry")

    # Create schema
    print("  Creating schema...")
    create_schema(conn)

    # Generate data
    print("  Generating reaches...")
    reaches = generate_reach_data(NUM_REACHES, region, version)

    print("  Generating nodes...")
    nodes = generate_node_data(reaches, region, version)

    print("  Generating centerlines...")
    centerlines = generate_centerline_data(reaches, nodes, region, version)

    print("  Generating topology...")
    topology = generate_topology(reaches, region)

    print("  Generating SWOT orbits...")
    orbits = generate_swot_orbits(reaches, region)

    print("  Generating ice flags...")
    ice_flags = generate_ice_flags(reaches)

    # Insert data
    print("  Inserting reaches...")
    for r in reaches:
        conn.execute(
            """
            INSERT INTO reaches (
                reach_id, region, x, y, x_min, x_max, y_min, y_max, geom,
                cl_id_min, cl_id_max, reach_length, n_nodes, wse, wse_var,
                width, width_var, slope, max_width, facc, dist_out, lakeflag,
                obstr_type, grod_id, hfalls_id, n_chan_max, n_chan_mod,
                n_rch_up, n_rch_down, swot_obs, iceflag, low_slope_flag,
                river_name, edit_flag, trib_flag, path_freq, path_order,
                path_segs, stream_order, main_side, end_reach, network,
                add_flag, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                     ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                     ?, ?, ?, ?, ?, ?)
        """,
            list(r.values()),
        )

    print("  Inserting nodes...")
    for n in nodes:
        conn.execute(
            """
            INSERT INTO nodes (
                node_id, region, x, y, geom, cl_id_min, cl_id_max, reach_id,
                node_length, wse, wse_var, width, width_var, max_width, facc,
                dist_out, lakeflag, obstr_type, grod_id, hfalls_id, n_chan_max,
                n_chan_mod, wth_coef, ext_dist_coef, meander_length, sinuosity,
                river_name, manual_add, edit_flag, trib_flag, path_freq,
                path_order, path_segs, stream_order, main_side, end_reach,
                network, add_flag, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                     ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            list(n.values()),
        )

    print("  Inserting centerlines...")
    for c in centerlines:
        conn.execute(
            """
            INSERT INTO centerlines (cl_id, region, x, y, geom, reach_id, node_id, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            list(c.values()),
        )

    print("  Inserting topology...")
    for t in topology:
        conn.execute(
            """
            INSERT INTO reach_topology (reach_id, region, direction, neighbor_rank, neighbor_reach_id)
            VALUES (?, ?, ?, ?, ?)
        """,
            list(t.values()),
        )

    print("  Inserting SWOT orbits...")
    for o in orbits:
        conn.execute(
            """
            INSERT INTO reach_swot_orbits (reach_id, region, orbit_rank, orbit_id)
            VALUES (?, ?, ?, ?)
        """,
            list(o.values()),
        )

    print("  Inserting ice flags...")
    # Batch insert for ice flags (36,600 rows)
    conn.executemany(
        """
        INSERT INTO reach_ice_flags (reach_id, julian_day, iceflag)
        VALUES (?, ?, ?)
    """,
        [(f["reach_id"], f["julian_day"], f["iceflag"]) for f in ice_flags],
    )

    # Populate geometries from x/y coordinates
    print("  Populating geometries...")
    _populate_geometries(conn)

    # Inject bad data for geometry check tests
    print("  Injecting geometry test data...")
    _inject_geometry_test_data(conn, region)

    # Update topology counts on reaches
    print("  Updating topology counts...")
    conn.execute("""
        UPDATE reaches SET n_rch_up = (
            SELECT COUNT(*) FROM reach_topology t
            WHERE t.reach_id = reaches.reach_id
              AND t.region = reaches.region
              AND t.direction = 'up'
        )
    """)
    conn.execute("""
        UPDATE reaches SET n_rch_down = (
            SELECT COUNT(*) FROM reach_topology t
            WHERE t.reach_id = reaches.reach_id
              AND t.region = reaches.region
              AND t.direction = 'down'
        )
    """)

    # Record version
    conn.execute(
        """
        INSERT INTO sword_versions (version, schema_version, notes)
        VALUES (?, '1.2.0', 'Minimal test database fixture')
    """,
        [version],
    )

    conn.commit()
    conn.close()

    # Report size
    file_size = os.path.getsize(db_path) / (1024 * 1024)
    print(f"  Done! Database size: {file_size:.2f} MB")
    print(f"  Reaches: {NUM_REACHES}")
    print(f"  Nodes: {NUM_REACHES * NODES_PER_REACH}")
    print(f"  Centerlines: {NUM_REACHES * CENTERLINES_PER_REACH}")
    print(f"  Topology entries: {len(topology)}")
    print(f"  SWOT orbits: {NUM_REACHES * ORBITS_PER_REACH}")
    print(f"  Ice flags: {NUM_REACHES * 366}")


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    # Default output path
    default_path = Path(__file__).parent / "sword_test_minimal.duckdb"

    # Allow custom path via argument
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = str(default_path)

    create_minimal_test_db(output_path)
