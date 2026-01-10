# -*- coding: utf-8 -*-
"""
SWORD DuckDB Schema Definitions
===============================

This module defines the DuckDB schema for the SWOT River Database (SWORD).
It normalizes the multi-dimensional NetCDF arrays into proper relational tables.

Tables:
    centerlines - Dense geospatial points forming river centerlines
    centerline_neighbors - Normalized neighbor relationships (from [4,N] array)
    nodes - Points at ~200m intervals with hydrologic attributes
    reaches - River segments between major junctions
    reach_topology - Normalized upstream/downstream relationships (from [4,N] arrays)
    reach_swot_orbits - SWOT orbit data (normalized from [75,N] array)
    reach_ice_flags - Daily ice flags (normalized from [366,N] array)
    reach_discharge_params - Discharge algorithm parameters
    sword_versions - Version tracking metadata
"""

# Schema version for migration tracking
SCHEMA_VERSION = "1.0.0"

# Core table definitions
CENTERLINES_TABLE = """
CREATE TABLE IF NOT EXISTS centerlines (
    -- Primary key
    cl_id BIGINT PRIMARY KEY,

    -- Coordinates
    x DOUBLE NOT NULL,
    y DOUBLE NOT NULL,

    -- Geometry (populated after insert via ST_Point)
    geom GEOMETRY,

    -- Primary associations (row 0 of the [4,N] arrays)
    reach_id BIGINT NOT NULL,
    node_id BIGINT NOT NULL,

    -- Metadata
    region VARCHAR(2) NOT NULL,
    version VARCHAR(10) NOT NULL
);
"""

CENTERLINE_NEIGHBORS_TABLE = """
CREATE TABLE IF NOT EXISTS centerline_neighbors (
    -- Composite primary key
    cl_id BIGINT NOT NULL,
    neighbor_rank TINYINT NOT NULL,  -- 1, 2, or 3 (0 is in main table)

    -- Neighbor IDs (from rows 1-3 of the [4,N] arrays)
    reach_id BIGINT,
    node_id BIGINT,

    PRIMARY KEY (cl_id, neighbor_rank)
);
"""

NODES_TABLE = """
CREATE TABLE IF NOT EXISTS nodes (
    -- Primary key
    node_id BIGINT PRIMARY KEY,

    -- Coordinates
    x DOUBLE NOT NULL,
    y DOUBLE NOT NULL,

    -- Geometry (populated after insert via ST_Point)
    geom GEOMETRY,

    -- Centerline range (from cl_ids[2,N])
    cl_id_min BIGINT,
    cl_id_max BIGINT,

    -- Parent reach
    reach_id BIGINT NOT NULL,

    -- Core measurements
    node_length DOUBLE,          -- len
    wse DOUBLE,                  -- water surface elevation (m)
    wse_var DOUBLE,              -- wse variance (m^2)
    width DOUBLE,                -- wth (m)
    width_var DOUBLE,            -- wth_var (m^2)
    max_width DOUBLE,            -- max_wth (m)

    -- Flow and hydrology
    facc DOUBLE,                 -- flow accumulation (km^2)
    dist_out DOUBLE,             -- distance from outlet (m)
    lakeflag INTEGER,            -- 0=river, 1=lake, 2=canal, 3=tidal

    -- Obstructions
    obstr_type INTEGER,          -- grod: 0=none, 1=dam, 2=lock, 3=low-perm, 4=waterfall
    grod_id BIGINT,              -- GROD database ID
    hfalls_id BIGINT,            -- HydroFALLS database ID

    -- Channel info
    n_chan_max INTEGER,          -- nchan_max
    n_chan_mod INTEGER,          -- nchan_mod

    -- SWOT search parameters
    wth_coef DOUBLE,             -- width coefficient for search window
    ext_dist_coef DOUBLE,        -- max search window coefficient

    -- Morphology
    meander_length DOUBLE,       -- meand_len (m)
    sinuosity DOUBLE,            -- sinuosity ratio

    -- Metadata
    river_name VARCHAR,          -- semicolon-separated if multiple
    manual_add INTEGER,          -- 0=not manual, 1=manual

    -- Quality flags
    edit_flag VARCHAR,           -- comma-separated update codes
    trib_flag INTEGER,           -- 0=no tributary, 1=tributary

    -- Network analysis
    path_freq BIGINT,            -- traversal count
    path_order BIGINT,           -- 1=longest to N=shortest
    path_segs BIGINT,            -- segment ID between junctions
    stream_order INTEGER,        -- strm_order (log scale of path_freq)
    main_side INTEGER,           -- 0=main, 1=side, 2=secondary outlet
    end_reach INTEGER,           -- end_rch: 0=main, 1=headwater, 2=outlet, 3=junction
    network INTEGER,             -- connected network ID

    -- Addition flag (optional, may not exist in older versions)
    add_flag INTEGER,            -- 0=not added, 1=added from MERIT Hydro

    -- Metadata
    region VARCHAR(2) NOT NULL,
    version VARCHAR(10) NOT NULL
);
"""

REACHES_TABLE = """
CREATE TABLE IF NOT EXISTS reaches (
    -- Primary key
    reach_id BIGINT PRIMARY KEY,

    -- Centroid coordinates
    x DOUBLE,
    y DOUBLE,

    -- Bounding box
    x_min DOUBLE,
    x_max DOUBLE,
    y_min DOUBLE,
    y_max DOUBLE,

    -- Geometry (LINESTRING, populated separately from centerlines)
    geom GEOMETRY,

    -- Centerline range (from cl_ids[2,N])
    cl_id_min BIGINT,
    cl_id_max BIGINT,

    -- Core measurements
    reach_length DOUBLE,         -- len (m)
    n_nodes INTEGER,             -- rch_n_nodes
    wse DOUBLE,                  -- water surface elevation (m)
    wse_var DOUBLE,              -- wse variance (m^2)
    width DOUBLE,                -- wth (m)
    width_var DOUBLE,            -- wth_var (m^2)
    slope DOUBLE,                -- slope (m/km)
    max_width DOUBLE,            -- max_wth (m)

    -- Flow and hydrology
    facc DOUBLE,                 -- flow accumulation (km^2)
    dist_out DOUBLE,             -- distance from outlet (m)
    lakeflag INTEGER,            -- 0=river, 1=lake, 2=canal, 3=tidal

    -- Obstructions
    obstr_type INTEGER,          -- grod
    grod_id BIGINT,              -- GROD database ID
    hfalls_id BIGINT,            -- HydroFALLS database ID

    -- Channel info
    n_chan_max INTEGER,          -- nchan_max
    n_chan_mod INTEGER,          -- nchan_mod

    -- Topology counts
    n_rch_up INTEGER,            -- number of upstream neighbors
    n_rch_down INTEGER,          -- number of downstream neighbors

    -- SWOT observations
    swot_obs INTEGER,            -- max_obs: max SWOT passes in 21-day cycle

    -- Flags
    iceflag INTEGER,             -- ice/seasonal flag
    low_slope_flag INTEGER,      -- low_slope: 1=too low for discharge estimation

    -- Metadata
    river_name VARCHAR,          -- semicolon-separated if multiple

    -- Quality flags
    edit_flag VARCHAR,           -- comma-separated update codes
    trib_flag INTEGER,           -- 0=no tributary, 1=tributary

    -- Network analysis
    path_freq BIGINT,            -- traversal count
    path_order BIGINT,           -- 1=longest to N=shortest
    path_segs BIGINT,            -- segment ID between junctions
    stream_order INTEGER,        -- strm_order
    main_side INTEGER,           -- 0=main, 1=side, 2=secondary outlet
    end_reach INTEGER,           -- end_rch: 0=main, 1=headwater, 2=outlet, 3=junction
    network INTEGER,             -- connected network ID

    -- Addition flag (optional)
    add_flag INTEGER,            -- 0=not added, 1=added from MERIT Hydro

    -- Metadata
    region VARCHAR(2) NOT NULL,
    version VARCHAR(10) NOT NULL
);
"""

REACH_TOPOLOGY_TABLE = """
CREATE TABLE IF NOT EXISTS reach_topology (
    -- Composite primary key
    reach_id BIGINT NOT NULL,
    direction VARCHAR(4) NOT NULL,  -- 'up' or 'down'
    neighbor_rank TINYINT NOT NULL,  -- 0-3

    -- Neighbor reach ID
    neighbor_reach_id BIGINT NOT NULL,

    PRIMARY KEY (reach_id, direction, neighbor_rank)
);
"""

REACH_SWOT_ORBITS_TABLE = """
CREATE TABLE IF NOT EXISTS reach_swot_orbits (
    -- Composite primary key
    reach_id BIGINT NOT NULL,
    orbit_rank TINYINT NOT NULL,  -- 0-74

    -- SWOT orbit pass_tile ID
    orbit_id BIGINT NOT NULL,

    PRIMARY KEY (reach_id, orbit_rank)
);
"""

REACH_ICE_FLAGS_TABLE = """
CREATE TABLE IF NOT EXISTS reach_ice_flags (
    -- Composite primary key
    reach_id BIGINT NOT NULL,
    julian_day SMALLINT NOT NULL,  -- 1-366

    -- Ice flag value
    iceflag INTEGER NOT NULL,

    PRIMARY KEY (reach_id, julian_day)
);
"""

SWORD_VERSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sword_versions (
    version VARCHAR(10) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    production_date TIMESTAMP,
    schema_version VARCHAR(10),
    notes VARCHAR
);
"""

# Index definitions
INDEXES = [
    # Spatial indexes (for DuckDB Spatial extension)
    "CREATE INDEX IF NOT EXISTS idx_centerlines_geom ON centerlines USING RTREE (geom);",
    "CREATE INDEX IF NOT EXISTS idx_nodes_geom ON nodes USING RTREE (geom);",
    "CREATE INDEX IF NOT EXISTS idx_reaches_geom ON reaches USING RTREE (geom);",

    # Foreign key lookups (most common query patterns)
    "CREATE INDEX IF NOT EXISTS idx_centerlines_reach ON centerlines(reach_id);",
    "CREATE INDEX IF NOT EXISTS idx_centerlines_node ON centerlines(node_id);",
    "CREATE INDEX IF NOT EXISTS idx_nodes_reach ON nodes(reach_id);",

    # Regional partitioning queries
    "CREATE INDEX IF NOT EXISTS idx_centerlines_region ON centerlines(region);",
    "CREATE INDEX IF NOT EXISTS idx_nodes_region ON nodes(region);",
    "CREATE INDEX IF NOT EXISTS idx_reaches_region ON reaches(region);",

    # Topology traversal
    "CREATE INDEX IF NOT EXISTS idx_topology_reach ON reach_topology(reach_id);",
    "CREATE INDEX IF NOT EXISTS idx_topology_neighbor ON reach_topology(neighbor_reach_id);",
    "CREATE INDEX IF NOT EXISTS idx_topology_direction ON reach_topology(reach_id, direction);",

    # Common analytical queries
    "CREATE INDEX IF NOT EXISTS idx_reaches_dist_out ON reaches(dist_out);",
    "CREATE INDEX IF NOT EXISTS idx_reaches_facc ON reaches(facc);",
    "CREATE INDEX IF NOT EXISTS idx_reaches_stream_order ON reaches(stream_order);",
    "CREATE INDEX IF NOT EXISTS idx_reaches_network ON reaches(network);",
    "CREATE INDEX IF NOT EXISTS idx_nodes_stream_order ON nodes(stream_order);",

    # SWOT orbits lookup
    "CREATE INDEX IF NOT EXISTS idx_swot_orbits_reach ON reach_swot_orbits(reach_id);",
]

# Regional views for backward compatibility
REGIONAL_VIEWS = """
-- North America
CREATE OR REPLACE VIEW na_centerlines AS SELECT * FROM centerlines WHERE region = 'NA';
CREATE OR REPLACE VIEW na_nodes AS SELECT * FROM nodes WHERE region = 'NA';
CREATE OR REPLACE VIEW na_reaches AS SELECT * FROM reaches WHERE region = 'NA';

-- South America
CREATE OR REPLACE VIEW sa_centerlines AS SELECT * FROM centerlines WHERE region = 'SA';
CREATE OR REPLACE VIEW sa_nodes AS SELECT * FROM nodes WHERE region = 'SA';
CREATE OR REPLACE VIEW sa_reaches AS SELECT * FROM reaches WHERE region = 'SA';

-- Europe
CREATE OR REPLACE VIEW eu_centerlines AS SELECT * FROM centerlines WHERE region = 'EU';
CREATE OR REPLACE VIEW eu_nodes AS SELECT * FROM nodes WHERE region = 'EU';
CREATE OR REPLACE VIEW eu_reaches AS SELECT * FROM reaches WHERE region = 'EU';

-- Africa
CREATE OR REPLACE VIEW af_centerlines AS SELECT * FROM centerlines WHERE region = 'AF';
CREATE OR REPLACE VIEW af_nodes AS SELECT * FROM nodes WHERE region = 'AF';
CREATE OR REPLACE VIEW af_reaches AS SELECT * FROM reaches WHERE region = 'AF';

-- Oceania
CREATE OR REPLACE VIEW oc_centerlines AS SELECT * FROM centerlines WHERE region = 'OC';
CREATE OR REPLACE VIEW oc_nodes AS SELECT * FROM nodes WHERE region = 'OC';
CREATE OR REPLACE VIEW oc_reaches AS SELECT * FROM reaches WHERE region = 'OC';

-- Asia
CREATE OR REPLACE VIEW as_centerlines AS SELECT * FROM centerlines WHERE region = 'AS';
CREATE OR REPLACE VIEW as_nodes AS SELECT * FROM nodes WHERE region = 'AS';
CREATE OR REPLACE VIEW as_reaches AS SELECT * FROM reaches WHERE region = 'AS';
"""


def get_schema_sql() -> str:
    """
    Returns the complete schema SQL as a single string.

    Returns
    -------
    str
        Complete SQL for creating all tables, indexes, and views.
    """
    tables = [
        CENTERLINES_TABLE,
        CENTERLINE_NEIGHBORS_TABLE,
        NODES_TABLE,
        REACHES_TABLE,
        REACH_TOPOLOGY_TABLE,
        REACH_SWOT_ORBITS_TABLE,
        REACH_ICE_FLAGS_TABLE,
        SWORD_VERSIONS_TABLE,
    ]

    schema_parts = []
    schema_parts.append("-- SWORD DuckDB Schema")
    schema_parts.append(f"-- Schema Version: {SCHEMA_VERSION}")
    schema_parts.append("")

    # Tables
    schema_parts.append("-- Core Tables")
    schema_parts.extend(tables)

    # Indexes
    schema_parts.append("\n-- Indexes")
    schema_parts.extend(INDEXES)

    # Views
    schema_parts.append("\n-- Regional Views")
    schema_parts.append(REGIONAL_VIEWS)

    return "\n".join(schema_parts)


def create_schema(conn) -> None:
    """
    Creates all SWORD tables, indexes, and views in the database.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.

    Raises
    ------
    Exception
        If schema creation fails.
    """
    # Create tables
    tables = [
        CENTERLINES_TABLE,
        CENTERLINE_NEIGHBORS_TABLE,
        NODES_TABLE,
        REACHES_TABLE,
        REACH_TOPOLOGY_TABLE,
        REACH_SWOT_ORBITS_TABLE,
        REACH_ICE_FLAGS_TABLE,
        SWORD_VERSIONS_TABLE,
    ]

    for table_sql in tables:
        conn.execute(table_sql)

    # Create indexes (may fail if spatial extension not loaded)
    for index_sql in INDEXES:
        try:
            conn.execute(index_sql)
        except Exception as e:
            # Skip spatial indexes if spatial extension not available
            if "RTREE" in index_sql and "spatial" in str(e).lower():
                continue
            raise

    # Create regional views
    for view_sql in REGIONAL_VIEWS.strip().split(";"):
        view_sql = view_sql.strip()
        if view_sql:
            conn.execute(view_sql)

    # Record schema version
    conn.execute("""
        INSERT OR REPLACE INTO sword_versions (version, schema_version, notes)
        VALUES ('schema', ?, 'Initial schema creation')
    """, [SCHEMA_VERSION])


def drop_schema(conn) -> None:
    """
    Drops all SWORD tables from the database.

    WARNING: This will delete all data!

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    """
    # Drop views first
    views = [
        'na_centerlines', 'na_nodes', 'na_reaches',
        'sa_centerlines', 'sa_nodes', 'sa_reaches',
        'eu_centerlines', 'eu_nodes', 'eu_reaches',
        'af_centerlines', 'af_nodes', 'af_reaches',
        'oc_centerlines', 'oc_nodes', 'oc_reaches',
        'as_centerlines', 'as_nodes', 'as_reaches',
    ]

    for view in views:
        conn.execute(f"DROP VIEW IF EXISTS {view};")

    # Drop tables in reverse dependency order
    tables = [
        'reach_ice_flags',
        'reach_swot_orbits',
        'reach_topology',
        'centerline_neighbors',
        'nodes',
        'centerlines',
        'reaches',
        'sword_versions',
    ]

    for table in tables:
        conn.execute(f"DROP TABLE IF EXISTS {table};")
