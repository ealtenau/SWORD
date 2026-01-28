#!/usr/bin/env python3
"""
SWORD Topology & FACC Reviewer
==============================
Streamlit UI to review and fix topology/facc issues in SWORD database.

Run with: streamlit run topology_reviewer.py

Issue Types:
- Topology Ratio: upstream facc >> downstream facc
- Monotonicity: facc > downstream facc (should decrease)
- Headwaters: n_rch_up=0 but high facc
- Suspect: Already flagged as facc_quality='suspect'
- Fix History: View and export all logged fixes
- Lake Sandwich: River reaches between lake reaches (C001)
"""

import streamlit as st
import duckdb
import pandas as pd
import pydeck as pdk
import folium
from streamlit_folium import st_folium
from datetime import datetime
import json
import os
from pathlib import Path

# Import lint checks
from src.updates.sword_duckdb.lint.checks.classification import check_lake_sandwich, check_lakeflag_type_consistency
from src.updates.sword_duckdb.lint.checks.topology import check_facc_monotonicity, check_orphan_reaches

# =============================================================================
# LOCAL PERSISTENCE (JSON file backup for session fixes)
# =============================================================================
FIXES_DIR = Path("output/lint_fixes")
FIXES_DIR.mkdir(parents=True, exist_ok=True)


def get_session_file(region: str, check_id: str = "all") -> Path:
    """Get the session fixes file path for a region and check type."""
    return FIXES_DIR / f"lint_session_{region}_{check_id}.json"


def load_session_fixes(region: str, check_id: str = "all") -> dict:
    """Load fixes from local JSON file."""
    session_file = get_session_file(region, check_id)
    if session_file.exists():
        with open(session_file, 'r') as f:
            return json.load(f)
    return {"fixes": [], "skips": [], "pending": []}


def save_session_fixes(region: str, fixes: list, skips: list, pending: list, check_id: str = "all"):
    """Save fixes to local JSON file."""
    session_file = get_session_file(region, check_id)
    data = {
        "region": region,
        "check_id": check_id,
        "last_updated": datetime.now().isoformat(),
        "fixes": fixes,
        "skips": skips,
        "pending": pending
    }
    with open(session_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def append_fix_to_session(region: str, fix_record: dict, check_id: str = "all"):
    """Append a single fix to the session file."""
    session = load_session_fixes(region, check_id)
    session["fixes"].append(fix_record)
    save_session_fixes(region, session["fixes"], session["skips"], session.get("pending", []), check_id)


def append_skip_to_session(region: str, skip_record: dict, check_id: str = "all"):
    """Append a single skip to the session file."""
    session = load_session_fixes(region, check_id)
    session["skips"].append(skip_record)
    save_session_fixes(region, session["fixes"], session["skips"], session.get("pending", []), check_id)


def export_session_csv(region: str, check_id: str = "all") -> str:
    """Export session fixes to CSV format."""
    session = load_session_fixes(region, check_id)
    all_records = []
    for fix in session["fixes"]:
        fix["action"] = "fix"
        all_records.append(fix)
    for skip in session["skips"]:
        skip["action"] = "skip"
        all_records.append(skip)
    if all_records:
        df = pd.DataFrame(all_records)
        return df.to_csv(index=False)
    return ""

# Page config
st.set_page_config(
    page_title="SWORD Reviewer",
    page_icon="🌊",
    layout="wide"
)

# Database connection
@st.cache_resource
def get_connection():
    conn = duckdb.connect('data/duckdb/sword_v17c.duckdb')
    try:
        conn.execute("INSTALL spatial")
        conn.execute("LOAD spatial")
    except:
        pass
    # Ensure fix log table exists
    conn.execute('''
        CREATE TABLE IF NOT EXISTS facc_fix_log (
            fix_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reach_id BIGINT,
            region VARCHAR,
            fix_type VARCHAR,
            old_facc DOUBLE,
            new_facc DOUBLE,
            old_edit_flag VARCHAR,
            new_edit_flag VARCHAR,
            notes VARCHAR,
            source VARCHAR DEFAULT 'manual'
        )
    ''')
    # Ensure lint fix log table exists (for C001 lake sandwich fixes)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS lint_fix_log (
            fix_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            check_id VARCHAR,
            reach_id BIGINT,
            region VARCHAR,
            action VARCHAR,
            column_changed VARCHAR,
            old_value VARCHAR,
            new_value VARCHAR,
            notes VARCHAR,
            undone BOOLEAN DEFAULT FALSE
        )
    ''')
    return conn

conn = get_connection()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def log_fix(conn, reach_id, region, fix_type, old_facc, new_facc, notes=""):
    """Log a fix to the facc_fix_log table."""
    # Get current edit_flag
    old_flag = conn.execute(
        "SELECT edit_flag FROM reaches WHERE reach_id = ? AND region = ?",
        [reach_id, region]
    ).fetchone()
    old_flag = old_flag[0] if old_flag else None

    # Get next fix_id
    max_id = conn.execute("SELECT COALESCE(MAX(fix_id), 0) FROM facc_fix_log").fetchone()[0]
    new_id = max_id + 1

    conn.execute("""
        INSERT INTO facc_fix_log (fix_id, reach_id, region, fix_type, old_facc, new_facc, old_edit_flag, notes, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'manual')
    """, [new_id, reach_id, region, fix_type, old_facc, new_facc, old_flag, notes])


def apply_facc_fix(conn, reach_id, region, new_facc, fix_type, notes=""):
    """Apply a facc fix to both reaches and nodes tables, with logging."""
    # Get old facc
    old_facc = conn.execute(
        "SELECT facc FROM reaches WHERE reach_id = ? AND region = ?",
        [reach_id, region]
    ).fetchone()
    old_facc = old_facc[0] if old_facc else 0

    # Log the fix
    log_fix(conn, reach_id, region, fix_type, old_facc, new_facc, notes)

    # Update reaches table
    conn.execute("""
        UPDATE reaches SET facc = ?, facc_quality = 'manual_fix',
        edit_flag = CASE
            WHEN edit_flag IS NULL OR edit_flag = 'NaN' THEN ?
            ELSE edit_flag || ',' || ?
        END
        WHERE reach_id = ? AND region = ?
    """, [new_facc, fix_type, fix_type, reach_id, region])

    # Update nodes table
    conn.execute("""
        UPDATE nodes SET facc = ?
        WHERE reach_id = ? AND region = ?
    """, [new_facc, reach_id, region])

    # COMMIT to ensure persistence
    conn.commit()

    return old_facc


@st.cache_data(ttl=300)
def get_reach_geometry(_conn, reach_id):
    """Get nodes for a reach ordered to form a line."""
    nodes = _conn.execute("""
        SELECT x, y FROM nodes WHERE reach_id = ? ORDER BY dist_out DESC
    """, [reach_id]).fetchdf()
    if len(nodes) == 0:
        return None
    return nodes[['x', 'y']].values.tolist()


@st.cache_data(ttl=300)
def get_reach_info(_conn, reach_id, region):
    """Get detailed reach info."""
    return _conn.execute("""
        SELECT reach_id, facc, width, river_name, lakeflag, n_rch_up, n_rch_down,
               facc_quality, edit_flag, x, y
        FROM reaches WHERE reach_id = ? AND region = ?
    """, [reach_id, region]).fetchdf()


@st.cache_data(ttl=60)
def get_upstream_chain(_conn, reach_id, region, max_hops=5):
    """Get upstream reaches for context."""
    chain = []
    current_id = reach_id
    for hop in range(max_hops):
        info = _conn.execute("""
            SELECT r.reach_id, r.facc, r.width, r.river_name,
                   t.neighbor_reach_id as up_id
            FROM reaches r
            LEFT JOIN reach_topology t ON r.reach_id = t.reach_id AND r.region = t.region AND t.direction = 'up'
            WHERE r.reach_id = ? AND r.region = ?
            ORDER BY t.neighbor_rank LIMIT 1
        """, [current_id, region]).fetchone()
        if not info:
            break
        chain.append({
            'hop': hop, 'reach_id': info[0], 'facc': info[1],
            'width': info[2], 'river_name': info[3]
        })
        if info[4] is None or info[4] <= 0:
            break
        current_id = int(info[4])
    return pd.DataFrame(chain)


@st.cache_data(ttl=60)
def get_downstream_chain(_conn, reach_id, region, max_hops=5):
    """Get downstream reaches for context."""
    chain = []
    current_id = reach_id
    for hop in range(max_hops):
        info = _conn.execute("""
            SELECT r.reach_id, r.facc, r.width, r.river_name,
                   t.neighbor_reach_id as dn_id
            FROM reaches r
            LEFT JOIN reach_topology t ON r.reach_id = t.reach_id AND r.region = t.region AND t.direction = 'down'
            WHERE r.reach_id = ? AND r.region = ?
            ORDER BY t.neighbor_rank LIMIT 1
        """, [current_id, region]).fetchone()
        if not info:
            break
        chain.append({
            'hop': hop, 'reach_id': info[0], 'facc': info[1],
            'width': info[2], 'river_name': info[3]
        })
        if info[4] is None or info[4] <= 0:
            break
        current_id = int(info[4])
    return pd.DataFrame(chain)


# =============================================================================
# LAKE SANDWICH (C001) HELPER FUNCTIONS
# =============================================================================

def get_neighbors(conn, reach_ids, region, hops=2):
    """Get upstream + downstream neighbor IDs (configurable hops deep)."""
    if not reach_ids:
        return set()

    all_neighbors = set()
    current_ids = set(reach_ids)

    for _ in range(hops):
        if not current_ids:
            break
        placeholders = ','.join(['?' for _ in current_ids])
        result = conn.execute(f"""
            SELECT DISTINCT neighbor_reach_id
            FROM reach_topology
            WHERE reach_id IN ({placeholders}) AND region = ?
        """, list(current_ids) + [region]).fetchall()

        new_ids = {row[0] for row in result if row[0]}
        all_neighbors.update(new_ids)
        current_ids = new_ids - set(reach_ids) - all_neighbors

    return all_neighbors


def run_c001_check(conn, region, reach_ids=None):
    """Run C001 lake sandwich check on full region or specific reaches + neighbors."""
    result = check_lake_sandwich(conn, region)

    if reach_ids is not None and len(reach_ids) > 0:
        # Filter to specific reaches + their neighbors
        neighbor_ids = get_neighbors(conn, reach_ids, region, hops=2)
        all_ids = set(reach_ids) | neighbor_ids
        if len(result.details) > 0:
            result.details = result.details[result.details['reach_id'].isin(all_ids)]

    return result


def get_neighbor_lakeflags(conn, reach_id, region):
    """Get lakeflag values for upstream and downstream neighbors."""
    up_result = conn.execute("""
        SELECT r.lakeflag
        FROM reach_topology t
        JOIN reaches r ON t.neighbor_reach_id = r.reach_id AND t.region = r.region
        WHERE t.reach_id = ? AND t.region = ? AND t.direction = 'up'
    """, [reach_id, region]).fetchall()

    dn_result = conn.execute("""
        SELECT r.lakeflag
        FROM reach_topology t
        JOIN reaches r ON t.neighbor_reach_id = r.reach_id AND t.region = r.region
        WHERE t.reach_id = ? AND t.region = ? AND t.direction = 'down'
    """, [reach_id, region]).fetchall()

    up_flags = [row[0] for row in up_result]
    dn_flags = [row[0] for row in dn_result]

    return up_flags, dn_flags


def get_reach_slope(conn, reach_id, region):
    """Get slope for a reach if available."""
    result = conn.execute("""
        SELECT slope FROM reaches WHERE reach_id = ? AND region = ?
    """, [reach_id, region]).fetchone()
    return result[0] if result and result[0] is not None else None


def get_reach_facc(conn, reach_id, region):
    """Get facc for a reach."""
    result = conn.execute("""
        SELECT facc FROM reaches WHERE reach_id = ? AND region = ?
    """, [reach_id, region]).fetchone()
    return result[0] if result and result[0] is not None else 0


def apply_lakeflag_fix(conn, reach_id, region, new_lakeflag):
    """Apply a lakeflag fix to reaches table, with logging and local backup."""
    # Get old lakeflag
    old_lakeflag = conn.execute(
        "SELECT lakeflag FROM reaches WHERE reach_id = ? AND region = ?",
        [reach_id, region]
    ).fetchone()
    old_lakeflag = old_lakeflag[0] if old_lakeflag else None

    # Get next fix_id
    max_id = conn.execute("SELECT COALESCE(MAX(fix_id), 0) FROM lint_fix_log").fetchone()[0]
    new_id = max_id + 1

    timestamp = datetime.now().isoformat()

    # Log the fix to database
    conn.execute("""
        INSERT INTO lint_fix_log (fix_id, check_id, reach_id, region, action, column_changed, old_value, new_value, notes)
        VALUES (?, 'C001', ?, ?, 'fix', 'lakeflag', ?, ?, '')
    """, [new_id, reach_id, region, str(old_lakeflag), str(new_lakeflag)])

    # Update reaches table
    conn.execute("""
        UPDATE reaches SET lakeflag = ?
        WHERE reach_id = ? AND region = ?
    """, [new_lakeflag, reach_id, region])

    # COMMIT to ensure persistence
    conn.commit()

    # Also save to local JSON file as backup
    fix_record = {
        "fix_id": new_id,
        "timestamp": timestamp,
        "check_id": "C001",
        "reach_id": int(reach_id),
        "region": region,
        "column_changed": "lakeflag",
        "old_value": old_lakeflag,
        "new_value": new_lakeflag,
        "undone": False
    }
    append_fix_to_session(region, fix_record, check_id="C001")

    return old_lakeflag


def log_skip(conn, reach_id, region, check_id, notes):
    """Log a skip action (false positive) with required explanation and local backup."""
    max_id = conn.execute("SELECT COALESCE(MAX(fix_id), 0) FROM lint_fix_log").fetchone()[0]
    new_id = max_id + 1

    timestamp = datetime.now().isoformat()

    conn.execute("""
        INSERT INTO lint_fix_log (fix_id, check_id, reach_id, region, action, column_changed, old_value, new_value, notes)
        VALUES (?, ?, ?, ?, 'skip', NULL, NULL, NULL, ?)
    """, [new_id, check_id, reach_id, region, notes])

    # COMMIT to ensure persistence
    conn.commit()

    # Also save to local JSON file as backup
    skip_record = {
        "fix_id": new_id,
        "timestamp": timestamp,
        "check_id": check_id,
        "reach_id": int(reach_id),
        "region": region,
        "notes": notes,
        "undone": False
    }
    append_skip_to_session(region, skip_record, check_id=check_id)


def undo_last_fix(conn, region):
    """Undo the most recent lakeflag fix for the region."""
    # Get most recent non-undone fix
    last = conn.execute("""
        SELECT fix_id, reach_id, old_value
        FROM lint_fix_log
        WHERE region = ? AND action = 'fix' AND NOT undone
        ORDER BY timestamp DESC LIMIT 1
    """, [region]).fetchone()

    if not last:
        return None

    fix_id, reach_id, old_value = last

    # Restore old value
    if old_value is not None:
        conn.execute("""
            UPDATE reaches SET lakeflag = ?
            WHERE reach_id = ? AND region = ?
        """, [int(old_value), reach_id, region])

    # Mark fix as undone
    conn.execute("""
        UPDATE lint_fix_log SET undone = TRUE WHERE fix_id = ?
    """, [fix_id])

    # Log undo action
    max_id = conn.execute("SELECT COALESCE(MAX(fix_id), 0) FROM lint_fix_log").fetchone()[0]
    conn.execute("""
        INSERT INTO lint_fix_log (fix_id, check_id, reach_id, region, action, column_changed, old_value, new_value, notes)
        VALUES (?, 'C001', ?, ?, 'undo', 'lakeflag', NULL, ?, ?)
    """, [max_id + 1, reach_id, region, old_value, f'Undo of fix_id={fix_id}'])

    # COMMIT to ensure persistence
    conn.commit()

    # Update local JSON - mark the fix as undone
    session = load_session_fixes(region, check_id="C001")
    for fix in session["fixes"]:
        if fix.get("fix_id") == fix_id:
            fix["undone"] = True
            break
    save_session_fixes(region, session["fixes"], session["skips"], session.get("pending", []), check_id="C001")

    return reach_id


def get_nearby_reaches(conn, center_lon, center_lat, radius_deg, region, exclude_ids=None):
    """Get all reaches within a bounding box (for showing unconnected reaches)."""
    exclude_ids = exclude_ids or []
    exclude_str = ','.join([str(int(r)) for r in exclude_ids]) if exclude_ids else '0'

    query = f"""
        SELECT reach_id, x, y
        FROM reaches
        WHERE region = ?
          AND x BETWEEN ? AND ?
          AND y BETWEEN ? AND ?
          AND reach_id NOT IN ({exclude_str})
        LIMIT 200
    """
    return conn.execute(query, [
        region,
        center_lon - radius_deg, center_lon + radius_deg,
        center_lat - radius_deg, center_lat + radius_deg
    ]).fetchdf()


def render_reach_map_satellite(reach_id, region, conn, hops=None):
    """Render a map centered on a reach with Esri satellite basemap using folium."""
    geom = get_reach_geometry(conn, reach_id)
    if not geom:
        st.warning("No geometry available")
        return

    # Use sidebar settings
    if hops is None:
        hops = st.session_state.get('map_hops', 5)
    show_all = st.session_state.get('show_all_reaches', True)

    # Get upstream and downstream geometries
    up_chain = get_upstream_chain(conn, reach_id, region, hops)
    dn_chain = get_downstream_chain(conn, reach_id, region, hops)

    all_coords = list(geom)
    connected_ids = {reach_id}

    # Collect upstream geometries
    up_geoms = []
    for i, row in up_chain.iterrows():
        if row['reach_id'] == reach_id:
            continue
        connected_ids.add(row['reach_id'])
        up_geom = get_reach_geometry(conn, int(row['reach_id']))
        if up_geom:
            up_geoms.append((up_geom, i, row['reach_id']))
            all_coords.extend(up_geom)

    # Collect downstream geometries
    dn_geoms = []
    for i, row in dn_chain.iterrows():
        if row['reach_id'] == reach_id:
            continue
        connected_ids.add(row['reach_id'])
        dn_geom = get_reach_geometry(conn, int(row['reach_id']))
        if dn_geom:
            dn_geoms.append((dn_geom, i, row['reach_id']))
            all_coords.extend(dn_geom)

    # Calculate bounds and center
    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2
    extent = max(max(lons) - min(lons), max(lats) - min(lats))

    # Expand extent to show more area
    view_radius = max(extent * 0.75, 0.02)  # At least 0.02 degrees

    zoom = 15 if extent < 0.02 else 14 if extent < 0.05 else 12 if extent < 0.1 else 10

    # Create folium map with Esri satellite tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=None  # Start with no tiles
    )

    # Add Esri World Imagery (free satellite tiles) - default
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)

    # Add CartoDB Dark for contrast option
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='CartoDB',
        name='Dark',
        overlay=False,
        control=True
    ).add_to(m)

    # Add layer control to toggle between basemaps
    folium.LayerControl().add_to(m)

    # Get and display ALL nearby reaches (unconnected ones in gray)
    nearby_unconnected = []
    if show_all:
        nearby = get_nearby_reaches(conn, center_lon, center_lat, view_radius, region, list(connected_ids))
        for _, row in nearby.iterrows():
            nearby_geom = get_reach_geometry(conn, int(row['reach_id']))
            if nearby_geom:
                nearby_unconnected.append((nearby_geom, row['reach_id']))

        # Draw unconnected reaches first (gray, thin) so they're behind
        for nearby_geom, rid in nearby_unconnected:
            coords = [[c[1], c[0]] for c in nearby_geom]
            folium.PolyLine(
                coords,
                color='#ffffff',
                weight=2,
                opacity=0.6,
                tooltip=f"Unconnected: {rid}"
            ).add_to(m)

    # Add upstream reaches (orange, fading based on distance)
    for up_geom, i, rid in up_geoms:
        # Convert [lon, lat] to [lat, lon] for folium
        coords = [[c[1], c[0]] for c in up_geom]
        # Scale opacity based on total hops
        opacity = max(0.2, 1.0 - (i / max(hops, 1)) * 0.8)
        folium.PolyLine(
            coords,
            color='orange',
            weight=3,
            opacity=opacity,
            tooltip=f"Upstream {i+1}: {rid}"
        ).add_to(m)

    # Add downstream reaches (blue, fading based on distance)
    for dn_geom, i, rid in dn_geoms:
        coords = [[c[1], c[0]] for c in dn_geom]
        opacity = max(0.2, 1.0 - (i / max(hops, 1)) * 0.8)
        folium.PolyLine(
            coords,
            color='#0066ff',
            weight=3,
            opacity=opacity,
            tooltip=f"Downstream {i+1}: {rid}"
        ).add_to(m)

    # Add main reach (red, thicker)
    main_coords = [[c[1], c[0]] for c in geom]
    folium.PolyLine(
        main_coords,
        color='red',
        weight=6,
        opacity=1.0,
        tooltip=f"Selected: {reach_id}"
    ).add_to(m)

    # Fit bounds (expand slightly to show context)
    padding = view_radius * 0.2
    m.fit_bounds([
        [min(lats) - padding, min(lons) - padding],
        [max(lats) + padding, max(lons) + padding]
    ])

    # Render in streamlit
    st_folium(m, width=None, height=500, returned_objects=[])
    if show_all and nearby_unconnected:
        st.caption(f"🔴 Selected | 🟠 Upstream ({len(up_geoms)}) | 🔵 Downstream ({len(dn_geoms)}) | ⚪ Unconnected ({len(nearby_unconnected)})")
    else:
        st.caption(f"🔴 Selected | 🟠 Upstream ({len(up_geoms)}) | 🔵 Downstream ({len(dn_geoms)})")


def get_lint_fix_history(conn, region=None, limit=100):
    """Get lint fix history from log table."""
    if region:
        return conn.execute("""
            SELECT * FROM lint_fix_log WHERE region = ? ORDER BY timestamp DESC LIMIT ?
        """, [region, limit]).fetchdf()
    else:
        return conn.execute("""
            SELECT * FROM lint_fix_log ORDER BY timestamp DESC LIMIT ?
        """, [limit]).fetchdf()


# =============================================================================
# ISSUE QUERIES
# =============================================================================

@st.cache_data(ttl=60)
def get_monotonicity_issues(_conn, region, limit=500):
    """Get reaches where facc > downstream facc."""
    return _conn.execute("""
        SELECT r.reach_id, r.facc, r.width, r.river_name, r.x, r.y, r.facc_quality,
               rd.reach_id as dn_reach_id, rd.facc as dn_facc, rd.width as dn_width,
               r.facc - rd.facc as diff
        FROM reaches r
        JOIN reach_topology t ON r.reach_id = t.reach_id AND r.region = t.region
        JOIN reaches rd ON t.neighbor_reach_id = rd.reach_id AND rd.region = ?
        WHERE r.region = ? AND t.direction = 'down'
        AND r.facc > rd.facc * 1.01
        AND (r.facc_quality IS NULL OR r.facc_quality NOT IN ('traced', 'manual_fix'))
        ORDER BY r.facc - rd.facc DESC
        LIMIT ?
    """, [region, region, limit]).fetchdf()


@st.cache_data(ttl=60)
def get_headwater_issues(_conn, region, min_facc=1000, limit=500):
    """Get headwaters with high facc."""
    return _conn.execute("""
        SELECT reach_id, facc, width, river_name, x, y, lakeflag, facc_quality,
               facc / NULLIF(width, 0) as ratio
        FROM reaches
        WHERE region = ? AND n_rch_up = 0 AND facc > ?
        AND (facc_quality IS NULL OR facc_quality NOT IN ('traced', 'manual_fix'))
        ORDER BY facc DESC
        LIMIT ?
    """, [region, min_facc, limit]).fetchdf()


@st.cache_data(ttl=60)
def get_suspect_reaches(_conn, region, limit=500):
    """Get reaches flagged as suspect."""
    return _conn.execute("""
        SELECT reach_id, facc, width, river_name, x, y, facc_quality, edit_flag,
               facc / NULLIF(width, 0) as ratio
        FROM reaches
        WHERE region = ? AND facc_quality = 'suspect'
        ORDER BY facc DESC
        LIMIT ?
    """, [region, limit]).fetchdf()


@st.cache_data(ttl=60)
def get_ratio_violations(_conn, region, min_ratio=10, limit=500):
    """Get topology ratio violations (original query)."""
    return _conn.execute("""
        SELECT
            r1.reach_id as upstream_reach, r1.facc as upstream_facc, r1.width as up_width,
            r1.river_name as upstream_name, r1.x as up_x, r1.y as up_y,
            r2.reach_id as downstream_reach, r2.facc as downstream_facc, r2.width as dn_width,
            r2.river_name as downstream_name, r2.x as dn_x, r2.y as dn_y,
            r1.facc / NULLIF(r2.facc, 0) as ratio,
            t.topology_suspect as is_flagged,
            COALESCE(t.topology_approved, FALSE) as is_approved
        FROM reach_topology t
        JOIN reaches r1 ON t.reach_id = r1.reach_id AND t.region = r1.region
        JOIN reaches r2 ON t.neighbor_reach_id = r2.reach_id AND t.region = r2.region
        WHERE t.direction = 'down' AND t.region = ?
        AND r2.facc > 0 AND r1.facc / r2.facc >= ?
        AND (t.topology_approved = FALSE OR t.topology_approved IS NULL)
        ORDER BY r1.facc / r2.facc DESC
        LIMIT ?
    """, [region, min_ratio, limit]).fetchdf()


def get_fix_history(_conn, region=None, limit=100):
    """Get fix history from log table."""
    if region:
        return _conn.execute("""
            SELECT * FROM facc_fix_log WHERE region = ? ORDER BY timestamp DESC LIMIT ?
        """, [region, limit]).fetchdf()
    else:
        return _conn.execute("""
            SELECT * FROM facc_fix_log ORDER BY timestamp DESC LIMIT ?
        """, [limit]).fetchdf()


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_reach_map(reach_id, region, title="Reach Location"):
    """Render a map centered on a reach with upstream/downstream context."""
    geom = get_reach_geometry(conn, reach_id)
    if not geom:
        st.warning("No geometry available")
        return

    # Get upstream and downstream geometries
    up_chain = get_upstream_chain(conn, reach_id, region, 3)
    dn_chain = get_downstream_chain(conn, reach_id, region, 3)

    layers = []
    all_coords = list(geom)

    # Main reach (RED)
    layers.append(pdk.Layer(
        "PathLayer",
        data=[{"path": geom, "color": [255, 0, 0]}],
        get_path="path", get_color="color",
        width_scale=30, width_min_pixels=5,
    ))

    # Upstream context (ORANGE gradient)
    for i, row in up_chain.iterrows():
        if row['reach_id'] == reach_id:
            continue
        up_geom = get_reach_geometry(conn, int(row['reach_id']))
        if up_geom:
            alpha = 255 - (i * 50)
            layers.append(pdk.Layer(
                "PathLayer",
                data=[{"path": up_geom, "color": [255, 165, 0, alpha], "name": f"Up {i}: facc={row['facc']:,.0f}"}],
                get_path="path", get_color="color",
                width_scale=20, width_min_pixels=3, pickable=True,
            ))
            all_coords.extend(up_geom)

    # Downstream context (BLUE gradient)
    for i, row in dn_chain.iterrows():
        if row['reach_id'] == reach_id:
            continue
        dn_geom = get_reach_geometry(conn, int(row['reach_id']))
        if dn_geom:
            alpha = 255 - (i * 50)
            layers.append(pdk.Layer(
                "PathLayer",
                data=[{"path": dn_geom, "color": [0, 100, 255, alpha], "name": f"Dn {i}: facc={row['facc']:,.0f}"}],
                get_path="path", get_color="color",
                width_scale=20, width_min_pixels=3, pickable=True,
            ))
            all_coords.extend(dn_geom)

    # Calculate view
    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    center_lon = (min(lons) + max(lons)) / 2
    center_lat = (min(lats) + max(lats)) / 2
    extent = max(max(lons) - min(lons), max(lats) - min(lats))
    zoom = 12 if extent < 0.1 else 10 if extent < 0.5 else 8 if extent < 1 else 7

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom),
        tooltip={"text": "{name}"}
    ))
    st.caption("🔴 Selected | 🟠 Upstream | 🔵 Downstream")


def render_fix_panel(reach_id, region, current_facc, issue_type):
    """Render fix options for a reach."""
    st.markdown("### 🔧 Fix Options")

    # Get context
    up_chain = get_upstream_chain(conn, reach_id, region, 5)
    dn_chain = get_downstream_chain(conn, reach_id, region, 3)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Upstream chain:**")
        if len(up_chain) > 0:
            st.dataframe(up_chain[['hop', 'reach_id', 'facc', 'width']], height=150)
        else:
            st.info("No upstream")

    with col2:
        st.markdown("**Downstream chain:**")
        if len(dn_chain) > 0:
            st.dataframe(dn_chain[['hop', 'reach_id', 'facc', 'width']], height=150)
        else:
            st.info("No downstream")

    st.divider()

    # Use issue_type in keys to avoid conflicts across tabs
    key_prefix = f"{issue_type}_{reach_id}"

    # Fix options
    notes = st.text_input("Notes (optional)", key=f"notes_{key_prefix}")

    # Option 1: Set to upstream value
    if len(up_chain) > 1:
        up_facc = up_chain.iloc[1]['facc'] if len(up_chain) > 1 else 0
        if up_facc > 0 and up_facc < current_facc * 0.5:
            st.markdown(f"**Option 1:** Use upstream facc = **{up_facc:,.0f}**")
            if st.button(f"Set to {up_facc:,.0f}", key=f"fix_up_{key_prefix}"):
                old = apply_facc_fix(conn, reach_id, region, up_facc, issue_type, notes)
                st.success(f"Fixed! {old:,.0f} → {up_facc:,.0f}")
                st.cache_data.clear()
                st.rerun()

    # Option 2: Set to downstream value
    if len(dn_chain) > 1:
        dn_facc = dn_chain.iloc[1]['facc'] if len(dn_chain) > 1 else 0
        if dn_facc > 0:
            st.markdown(f"**Option 2:** Match downstream facc = **{dn_facc:,.0f}**")
            if st.button(f"Set to {dn_facc:,.0f}", key=f"fix_dn_{key_prefix}"):
                old = apply_facc_fix(conn, reach_id, region, dn_facc, issue_type, notes)
                st.success(f"Fixed! {old:,.0f} → {dn_facc:,.0f}")
                st.cache_data.clear()
                st.rerun()

    # Option 3: Custom value
    custom_facc = st.number_input("Custom facc value", min_value=0.0, value=float(current_facc), key=f"custom_{key_prefix}")
    if st.button("Apply custom value", key=f"fix_custom_{key_prefix}"):
        old = apply_facc_fix(conn, reach_id, region, custom_facc, f"{issue_type}_custom", notes)
        st.success(f"Fixed! {old:,.0f} → {custom_facc:,.0f}")
        st.cache_data.clear()
        st.rerun()

    # Option 4: Flag as unfixable
    if st.button("🚩 Flag as unfixable", key=f"flag_{key_prefix}"):
        conn.execute("""
            UPDATE reaches SET facc_quality = 'unfixable'
            WHERE reach_id = ? AND region = ?
        """, [reach_id, region])
        log_fix(conn, reach_id, region, "flagged_unfixable", current_facc, current_facc, notes)
        st.warning("Flagged as unfixable")
        st.cache_data.clear()
        st.rerun()


# =============================================================================
# MAIN APP
# =============================================================================

st.title("🌊 SWORD Topology & FACC Reviewer")

# Sidebar
st.sidebar.header("Settings")
region = st.sidebar.selectbox("Region", ["NA", "SA", "EU", "AF", "AS", "OC"], index=0)

# Network display settings
st.sidebar.subheader("Map Settings")
st.session_state.map_hops = st.sidebar.slider(
    "Network depth (reaches)",
    min_value=1,
    max_value=15,
    value=st.session_state.get('map_hops', 5),
    help="How many reaches to show upstream/downstream"
)
st.session_state.show_all_reaches = st.sidebar.checkbox(
    "Show ALL reaches in area",
    value=st.session_state.get('show_all_reaches', True),
    help="Show unconnected reaches (white) to spot missing topology"
)

# Tabs for different issue types
# NOTE: Tabs 1-2 (Ratio Violations, Monotonicity) hidden until facc strategy decided
# To restore, uncomment those tabs and their implementations below
tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🏔️ Headwaters",
    "⚠️ Suspect",
    "📜 Fix History",
    "🥪 Lake Sandwich",
    "🏷️ Lakeflag/Type",
    "📈 FACC Mono",
    "🏝️ Orphans"
])

# =============================================================================
# TAB 1: Ratio Violations (HIDDEN - uncomment when facc strategy decided)
# =============================================================================
# with tab1:
#     st.header("Topology Ratio Violations")
#     st.caption("Reaches where upstream facc >> downstream facc")
#
#     min_ratio = st.slider("Minimum ratio", 2, 100, 10, key="ratio_slider")
#     violations = get_ratio_violations(conn, region, min_ratio)
#
#     st.metric("Total", len(violations))
#
#     if len(violations) > 0:
#         selected_idx = st.selectbox(
#             "Select violation",
#             range(len(violations)),
#             format_func=lambda i: f"#{i+1}: {violations.iloc[i]['upstream_name']} ({violations.iloc[i]['ratio']:.0f}x)",
#             key="ratio_select"
#         )
#
#         v = violations.iloc[selected_idx]
#
#         col1, col2 = st.columns([2, 1])
#
#         with col1:
#             render_reach_map(int(v['upstream_reach']), region, "Upstream Reach")
#
#         with col2:
#             st.markdown(f"**Upstream:** `{v['upstream_reach']}`")
#             st.markdown(f"facc: **{v['upstream_facc']:,.0f}** | width: {v['up_width']:.0f}m")
#             st.markdown(f"**Downstream:** `{v['downstream_reach']}`")
#             st.markdown(f"facc: **{v['downstream_facc']:,.0f}** | width: {v['dn_width']:.0f}m")
#             st.markdown(f"**Ratio: {v['ratio']:.0f}x**")
#
#             st.divider()
#             render_fix_panel(int(v['upstream_reach']), region, v['upstream_facc'], "ratio_violation")

# =============================================================================
# TAB 2: Monotonicity Violations (HIDDEN - uncomment when facc strategy decided)
# =============================================================================
# with tab2:
#     st.header("Monotonicity Violations")
#     st.caption("Reaches where facc > downstream facc (should decrease downstream)")
#
#     mono_issues = get_monotonicity_issues(conn, region)
#
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Total", len(mono_issues))
#     col2.metric("Severe (>100k diff)", len(mono_issues[mono_issues['diff'] > 100000]))
#     col3.metric("Minor (<1k diff)", len(mono_issues[mono_issues['diff'] < 1000]))
#
#     if len(mono_issues) > 0:
#         selected_idx = st.selectbox(
#             "Select issue",
#             range(len(mono_issues)),
#             format_func=lambda i: f"#{i+1}: {mono_issues.iloc[i]['reach_id']} (diff={mono_issues.iloc[i]['diff']:,.0f})",
#             key="mono_select"
#         )
#
#         m = mono_issues.iloc[selected_idx]
#
#         col1, col2 = st.columns([2, 1])
#
#         with col1:
#             render_reach_map(int(m['reach_id']), region)
#
#         with col2:
#             st.markdown(f"**Reach:** `{m['reach_id']}`")
#             st.markdown(f"**facc:** {m['facc']:,.0f} km²")
#             st.markdown(f"**width:** {m['width']:.0f}m")
#             st.markdown(f"**River:** {m['river_name']}")
#             st.divider()
#             st.markdown(f"**Downstream:** `{m['dn_reach_id']}`")
#             st.markdown(f"**dn_facc:** {m['dn_facc']:,.0f} km²")
#             st.markdown(f"**Difference:** {m['diff']:,.0f} km²")
#
#             st.divider()
#             render_fix_panel(int(m['reach_id']), region, m['facc'], "monotonicity")

# =============================================================================
# TAB 3: Headwater Issues
# =============================================================================
with tab3:
    st.header("🏔️ Suspicious Headwaters")

    # Initialize session state
    if 'hw_pending' not in st.session_state:
        st.session_state.hw_pending = []

    min_facc = st.slider("Minimum facc threshold", 1000, 100000, 5000, key="hw_slider")
    hw_issues = get_headwater_issues(conn, region, min_facc)

    if len(hw_issues) == 0:
        st.success(f"✅ No suspicious headwaters (facc > {min_facc:,} km²)")
    else:
        total = len(hw_issues)
        done = len(st.session_state.hw_pending)
        remaining = len([r for r in hw_issues['reach_id'].tolist() if r not in st.session_state.hw_pending])

        # Progress
        col1, col2, col3 = st.columns(3)
        col1.metric("Remaining", remaining)
        col2.metric("Reviewed", done)
        col3.metric("Total", total)
        st.progress(done / total if total > 0 else 0)

        if remaining == 0:
            st.success("🎉 All suspicious headwaters reviewed!")
            if st.button("🔄 Reset reviews", key="reset_hw"):
                st.session_state.hw_pending = []
                st.rerun()
        else:
            available = [r for r in hw_issues['reach_id'].tolist() if r not in st.session_state.hw_pending]
            selected = available[0]
            h = hw_issues[hw_issues['reach_id'] == selected].iloc[0]

            # ===== PROBLEM BOX =====
            st.markdown("---")
            st.subheader(f"🔍 Issue #{done + 1}: Reach `{selected}`")

            # Clear problem statement
            st.error(f"""
            **PROBLEM:** This reach has **NO upstream neighbors** but has **HIGH facc ({h['facc']:,.0f} km²)**

            True headwaters should have LOW facc (small catchment). High facc suggests:
            - Lake outlet (facc inherited from lake)
            - Missing upstream topology
            - Wrong flow direction
            """)

            col1, col2 = st.columns([2, 1])

            with col1:
                render_reach_map_satellite(int(selected), region, conn)

            with col2:
                st.markdown("### 📊 Details")
                st.markdown(f"**FACC:** {h['facc']:,.0f} km²")
                st.markdown(f"**Width:** {h['width']:.0f}m")
                st.markdown(f"**River:** {h['river_name'] or 'Unnamed'}")
                laketype = {0: 'River', 1: 'Lake', 2: 'Canal', 3: 'Tidal'}.get(h['lakeflag'], '?')

                # Hint based on type
                if h['lakeflag'] == 1:
                    st.success(f"**Type:** {laketype} → Likely valid lake outlet")
                else:
                    st.warning(f"**Type:** {laketype} → Check if topology is correct")

                st.markdown("---")
                st.markdown("### 🎯 Classification")

                # Action buttons
                if st.button("✅ Lake Outlet (valid)", key=f"hw_lake_{selected}", type="primary", use_container_width=True):
                    log_skip(conn, selected, region, 'HW', 'Lake outlet - valid high facc')
                    st.session_state.hw_pending.append(selected)
                    st.rerun()

                if st.button("🔗 Missing Upstream", key=f"hw_missing_{selected}", use_container_width=True):
                    log_skip(conn, selected, region, 'HW', 'Missing upstream topology')
                    st.session_state.hw_pending.append(selected)
                    st.rerun()

                if st.button("↩️ Wrong Flow Direction", key=f"hw_flow_{selected}", use_container_width=True):
                    log_skip(conn, selected, region, 'HW', 'Wrong flow direction')
                    st.session_state.hw_pending.append(selected)
                    st.rerun()

                if st.button("✓ Valid Headwater", key=f"hw_valid_{selected}", use_container_width=True):
                    log_skip(conn, selected, region, 'HW', 'Valid headwater')
                    st.session_state.hw_pending.append(selected)
                    st.rerun()

# =============================================================================
# TAB 4: Suspect Reaches
# =============================================================================
with tab4:
    st.header("⚠️ Suspect Reaches")

    # Initialize session state
    if 'suspect_pending' not in st.session_state:
        st.session_state.suspect_pending = []

    suspect = get_suspect_reaches(conn, region)

    if len(suspect) == 0:
        st.success("✅ No suspect reaches in this region")
    else:
        total = len(suspect)
        done = len(st.session_state.suspect_pending)
        remaining = len([r for r in suspect['reach_id'].tolist() if r not in st.session_state.suspect_pending])

        # Progress
        col1, col2, col3 = st.columns(3)
        col1.metric("Remaining", remaining)
        col2.metric("Reviewed", done)
        col3.metric("Total", total)
        st.progress(done / total if total > 0 else 0)

        if remaining == 0:
            st.success("🎉 All suspect reaches reviewed!")
            if st.button("🔄 Reset reviews", key="reset_suspect"):
                st.session_state.suspect_pending = []
                st.rerun()
        else:
            available = [r for r in suspect['reach_id'].tolist() if r not in st.session_state.suspect_pending]
            selected = available[0]
            s = suspect[suspect['reach_id'] == selected].iloc[0]

            # ===== PROBLEM BOX =====
            st.markdown("---")
            st.subheader(f"🔍 Issue #{done + 1}: Reach `{selected}`")

            st.warning(f"""
            **PROBLEM:** This reach was flagged as `facc_quality='suspect'`

            Automated methods couldn't determine the correct facc value. Manual review needed.
            - **Current FACC:** {s['facc']:,.0f} km²
            - **Width:** {s['width']:.0f}m
            """)

            col1, col2 = st.columns([2, 1])

            with col1:
                render_reach_map_satellite(int(selected), region, conn)

            with col2:
                st.markdown(f"**River:** {s['river_name'] or 'Unnamed'}")
                st.markdown(f"**Edit flag:** {s['edit_flag']}")

                st.markdown("---")
                st.markdown("### 🎯 Classification")

                if st.button("✅ FACC looks correct", key=f"suspect_ok_{selected}", type="primary", use_container_width=True):
                    log_skip(conn, selected, region, 'SUSPECT', 'FACC value looks correct')
                    st.session_state.suspect_pending.append(selected)
                    st.rerun()

                if st.button("❌ FACC is wrong", key=f"suspect_wrong_{selected}", use_container_width=True):
                    log_skip(conn, selected, region, 'SUSPECT', 'FACC value is incorrect')
                    st.session_state.suspect_pending.append(selected)
                    st.rerun()

                if st.button("🔍 Needs more investigation", key=f"suspect_inv_{selected}", use_container_width=True):
                    log_skip(conn, selected, region, 'SUSPECT', 'Needs more investigation')
                    st.session_state.suspect_pending.append(selected)
                    st.rerun()

# =============================================================================
# TAB 5: Fix History
# =============================================================================
with tab5:
    st.header("Fix History")
    st.caption("Log of all manual fixes for analysis")

    show_all_regions = st.checkbox("Show all regions", value=False)

    history = get_fix_history(conn, None if show_all_regions else region, 500)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Fixes", len(history))
    if len(history) > 0:
        col2.metric("Fix Types", history['fix_type'].nunique())
        col3.metric("Avg Reduction", f"{(history['old_facc'] - history['new_facc']).mean():,.0f}")

    if len(history) > 0:
        # Summary by fix type
        st.subheader("By Fix Type")
        summary = history.groupby('fix_type').agg({
            'reach_id': 'count',
            'old_facc': 'mean',
            'new_facc': 'mean'
        }).rename(columns={'reach_id': 'count', 'old_facc': 'avg_old', 'new_facc': 'avg_new'})
        summary['avg_reduction'] = summary['avg_old'] - summary['avg_new']
        st.dataframe(summary)

        # Full table
        st.subheader("All Fixes")
        st.dataframe(history, height=400)

        # Export
        st.subheader("Export")
        csv = history.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"facc_fixes_{region}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
    else:
        st.info("No fixes logged yet. Make some fixes in the other tabs!")

# =============================================================================
# TAB 6: Lake Sandwich (C001)
# =============================================================================
with tab6:
    st.header("🥪 Lake Sandwich Fixer")

    # Initialize session state
    if 'pending_fixes' not in st.session_state:
        st.session_state.pending_fixes = []
    if 'last_fix' not in st.session_state:
        st.session_state.last_fix = None
    if 'c001_results' not in st.session_state:
        st.session_state.c001_results = None

    # Auto-run C001 check on first load or when region changes
    if st.session_state.c001_results is None or st.session_state.get('c001_region') != region:
        with st.spinner("Running lake sandwich check..."):
            st.session_state.c001_results = run_c001_check(conn, region)
            st.session_state.c001_region = region
            session = load_session_fixes(region, "C001")
            st.session_state.pending_fixes = session.get("pending", [])

    # Progress bar at top
    result = st.session_state.c001_results
    if result:
        issues = result.details
        total = len(issues)
        done = len(st.session_state.pending_fixes)
        remaining = len([r for r in issues['reach_id'].tolist() if r not in st.session_state.pending_fixes]) if total > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Remaining", remaining)
        col2.metric("Reviewed", done)
        col3.metric("Total", total)
        if st.session_state.last_fix:
            col4.button(f"↩️ Undo", key="undo_c001", on_click=lambda: None)  # Placeholder

        if total > 0:
            st.progress(done / total if total > 0 else 0, text=f"{done}/{total} reviewed")

        if remaining == 0 and total > 0:
            st.success("🎉 All lake sandwiches reviewed!")
            if st.button("🔄 Re-run check to verify", key="rerun_c001"):
                st.session_state.c001_results = None
                st.session_state.pending_fixes = []
                st.rerun()
        elif total == 0:
            st.success("✅ No lake sandwich issues in this region!")
        else:
            # Get next issue
            available = [r for r in issues['reach_id'].tolist() if r not in st.session_state.pending_fixes]
            selected = available[0]  # Auto-select first available
            issue = issues[issues['reach_id'] == selected].iloc[0]

            # Get decision data
            up_flags, dn_flags = get_neighbor_lakeflags(conn, selected, region)
            slope = get_reach_slope(conn, selected, region)
            width = issue['width'] if pd.notna(issue['width']) else 0

            # ===== PROBLEM BOX =====
            st.markdown("---")
            st.subheader(f"🔍 Issue #{done + 1}: Reach `{selected}`")

            # Visual sandwich diagram
            up_label = "🔵 LAKE" if up_flags and 1 in up_flags else "〰️ River" if up_flags else "?"
            dn_label = "🔵 LAKE" if dn_flags and 1 in dn_flags else "〰️ River" if dn_flags else "?"
            st.markdown(f"""
            ```
            Upstream:   {up_label}
                          ↓
            THIS REACH: 〰️ River (lakeflag=0)  ← Is this actually a lake?
                          ↓
            Downstream: {dn_label}
            ```
            """)

            # ===== DECISION HELPER =====
            col1, col2 = st.columns([2, 1])

            with col1:
                render_reach_map_satellite(int(selected), region, conn)

            with col2:
                st.markdown("### 📊 Key Indicators")

                # Width indicator
                if width > 500:
                    st.success(f"**Width:** {width:.0f}m → Wide (likely lake)")
                elif width > 200:
                    st.warning(f"**Width:** {width:.0f}m → Medium")
                elif width > 0:
                    st.error(f"**Width:** {width:.0f}m → Narrow (likely river)")
                else:
                    st.info("**Width:** N/A")

                # Slope indicator
                if slope is not None:
                    if slope < 0.0001:
                        st.success(f"**Slope:** {slope:.6f} → Flat (likely lake)")
                    elif slope < 0.001:
                        st.warning(f"**Slope:** {slope:.6f} → Low slope")
                    else:
                        st.error(f"**Slope:** {slope:.6f} → Steep (likely river)")
                else:
                    st.info("**Slope:** N/A")

                # Neighbor indicator
                both_lakes = (up_flags and 1 in up_flags) and (dn_flags and 1 in dn_flags)
                if both_lakes:
                    st.success("**Neighbors:** Both lakes → Likely lake")
                else:
                    st.warning("**Neighbors:** Mixed types")

                st.markdown(f"**River name:** {issue['river_name'] or 'Unnamed'}")
                st.markdown(f"**Length:** {issue['reach_length']:.0f}m" if pd.notna(issue['reach_length']) else "")

            # ===== BIG DECISION BUTTONS =====
            st.markdown("---")
            st.markdown("### 🎯 Your Decision")

            btn_col1, btn_col2 = st.columns(2)

            with btn_col1:
                st.markdown("**It's a LAKE** (convert lakeflag to 1)")
                if st.button("✅ YES, IT'S A LAKE", key=f"fix_{selected}", type="primary", use_container_width=True):
                    apply_lakeflag_fix(conn, selected, region, 1)
                    st.session_state.pending_fixes.append(selected)
                    st.session_state.last_fix = selected
                    st.cache_data.clear()
                    st.rerun()

            with btn_col2:
                st.markdown("**Keep as RIVER** (no change)")
                skip_reason = st.selectbox(
                    "Why is it a river?",
                    ["Flowing water visible", "Narrow channel", "Dam/weir", "Canal", "Tidal", "Other"],
                    key=f"skip_reason_{selected}",
                    label_visibility="collapsed"
                )
                if st.button("❌ NO, IT'S A RIVER", key=f"skip_{selected}", use_container_width=True):
                    log_skip(conn, selected, region, 'C001', skip_reason)
                    st.session_state.pending_fixes.append(selected)
                    st.cache_data.clear()
                    st.rerun()

            # Undo option (small, at bottom)
            if st.session_state.last_fix:
                if st.button(f"↩️ Undo last ({st.session_state.last_fix})", key="undo_last"):
                    undone_id = undo_last_fix(conn, region)
                    if undone_id:
                        st.session_state.pending_fixes = [p for p in st.session_state.pending_fixes if p != undone_id]
                        st.session_state.last_fix = None
                        st.cache_data.clear()
                        st.rerun()

    # Refresh button at bottom
    st.markdown("---")
    if st.button("🔄 Refresh Check", key="refresh_c001"):
        st.session_state.c001_results = None
        st.session_state.pending_fixes = []
        st.rerun()

# =============================================================================
# TAB 7: Lakeflag/Type Consistency (C004)
# =============================================================================
with tab7:
    st.header("🏷️ Lakeflag/Type Mismatch")

    # Initialize session state for C004
    if 'c004_results' not in st.session_state:
        st.session_state.c004_results = None
    if 'c004_pending' not in st.session_state:
        st.session_state.c004_pending = []

    # Auto-run check
    if st.session_state.c004_results is None or st.session_state.get('c004_region') != region:
        with st.spinner("Running C004 check..."):
            st.session_state.c004_results = check_lakeflag_type_consistency(conn, region)
            st.session_state.c004_region = region

    result = st.session_state.c004_results

    # Check if type column exists
    if result and result.total_checked == 0:
        st.warning("⚠️ **Type column not in database** - this check requires adding the 'type' column first.")
        st.info("Once the type column is added, this tab will show reaches where lakeflag and type don't match.")
    elif result is None or len(result.details) == 0:
        st.success("✅ No lakeflag/type mismatches in this region!")
    else:
        issues = result.details
        total = len(issues)
        done = len(st.session_state.c004_pending)
        remaining = len([r for r in issues['reach_id'].tolist() if r not in st.session_state.c004_pending])

        # Progress
        col1, col2, col3 = st.columns(3)
        col1.metric("Remaining", remaining)
        col2.metric("Reviewed", done)
        col3.metric("Total", total)
        st.progress(done / total if total > 0 else 0)

        if remaining == 0:
            st.success("🎉 All mismatches reviewed!")
        else:
            available = [r for r in issues['reach_id'].tolist() if r not in st.session_state.c004_pending]
            selected = available[0]
            issue = issues[issues['reach_id'] == selected].iloc[0]

            lakeflag_map = {0: 'River', 1: 'Lake', 2: 'Canal', 3: 'Tidal'}
            type_map = {1: 'river', 2: 'lake', 3: 'tidal_river', 4: 'artificial', 5: 'unassigned', 6: 'unreliable'}
            lf = issue['lakeflag']
            tp = issue['type']

            # Problem box
            st.markdown("---")
            st.subheader(f"🔍 Issue #{done + 1}: Reach `{selected}`")
            st.error(f"""
            **Mismatch detected:**
            - **Lakeflag:** {lf} ({lakeflag_map.get(lf, '?')})
            - **Type:** {tp} ({type_map.get(tp, '?')})
            - **Issue:** {issue['issue_type']}
            """)

            col1, col2 = st.columns([2, 1])
            with col1:
                render_reach_map_satellite(int(selected), region, conn)

            with col2:
                st.markdown(f"**River:** {issue['river_name'] or 'Unnamed'}")

                st.markdown("---")
                st.markdown("### 🎯 Fix the mismatch")

                # Determine fix options based on issue type
                if 'lake' in issue['issue_type']:
                    if st.button("✅ Set type=2 (lake)", key=f"c004_fix_{selected}", type="primary", use_container_width=True):
                        conn.execute("UPDATE reaches SET type = 2 WHERE reach_id = ? AND region = ?", [selected, region])
                        conn.commit()
                        log_skip(conn, selected, region, 'C004', f'Fixed: type→2')
                        st.session_state.c004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()
                elif 'river' in issue['issue_type']:
                    if st.button("✅ Set type=1 (river)", key=f"c004_fix_{selected}", type="primary", use_container_width=True):
                        conn.execute("UPDATE reaches SET type = 1 WHERE reach_id = ? AND region = ?", [selected, region])
                        conn.commit()
                        log_skip(conn, selected, region, 'C004', f'Fixed: type→1')
                        st.session_state.c004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()
                elif 'canal' in issue['issue_type']:
                    if st.button("✅ Set type=4 (artificial)", key=f"c004_fix_{selected}", type="primary", use_container_width=True):
                        conn.execute("UPDATE reaches SET type = 4 WHERE reach_id = ? AND region = ?", [selected, region])
                        conn.commit()
                        log_skip(conn, selected, region, 'C004', f'Fixed: type→4')
                        st.session_state.c004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()
                elif 'tidal' in issue['issue_type']:
                    if st.button("✅ Set type=3 (tidal)", key=f"c004_fix_{selected}", type="primary", use_container_width=True):
                        conn.execute("UPDATE reaches SET type = 3 WHERE reach_id = ? AND region = ?", [selected, region])
                        conn.commit()
                        log_skip(conn, selected, region, 'C004', f'Fixed: type→3')
                        st.session_state.c004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()

                if st.button("⏭️ Skip (correct as-is)", key=f"c004_skip_{selected}", use_container_width=True):
                    log_skip(conn, selected, region, 'C004', 'Skipped: correct as-is')
                    st.session_state.c004_pending.append(selected)
                    st.cache_data.clear()
                    st.rerun()

# =============================================================================
# TAB 8: FACC Monotonicity (T003)
# =============================================================================
with tab8:
    st.header("📈 FACC Monotonicity Issues")

    # Initialize session state for T003
    if 't003_results' not in st.session_state:
        st.session_state.t003_results = None
    if 't003_pending' not in st.session_state:
        st.session_state.t003_pending = []

    # Auto-run check
    if st.session_state.t003_results is None or st.session_state.get('t003_region') != region:
        with st.spinner("Running T003 check..."):
            st.session_state.t003_results = check_facc_monotonicity(conn, region)
            st.session_state.t003_region = region

    result = st.session_state.t003_results
    if result is None or len(result.details) == 0:
        st.success("✅ No FACC monotonicity issues in this region!")
    else:
        issues = result.details
        total = len(issues)
        done = len(st.session_state.t003_pending)
        remaining = len([r for r in issues['reach_id'].tolist() if r not in st.session_state.t003_pending])

        # Progress
        col1, col2, col3 = st.columns(3)
        col1.metric("Remaining", remaining)
        col2.metric("Reviewed", done)
        col3.metric("Total", total)
        st.progress(done / total if total > 0 else 0)

        # Important note about D8
        st.info("💡 **Most T003 issues are D8 routing artifacts** (distributaries, deltas, braided rivers). These are NOT errors - just mark as 'D8 artifact'.")

        if remaining == 0:
            st.success("🎉 All FACC issues reviewed!")
        else:
            available = [r for r in issues['reach_id'].tolist() if r not in st.session_state.t003_pending]
            selected = available[0]
            issue = issues[issues['reach_id'] == selected].iloc[0]

            # Problem box
            st.markdown("---")
            st.subheader(f"🔍 Issue #{done + 1}: Reach `{selected}`")

            # Visual showing the problem
            st.warning(f"""
            **FACC decreases downstream** (should increase):
            - **This reach facc:** {issue['facc_up']:,.0f} km²
            - **Downstream facc:** {issue['facc_down']:,.0f} km²
            - **Decrease:** {issue['facc_decrease']:,.0f} km²
            """)

            col1, col2 = st.columns([2, 1])
            with col1:
                render_reach_map_satellite(int(selected), region, conn)

            with col2:
                st.markdown(f"**River:** {issue['river_name'] or 'Unnamed'}")

                # Decision helper
                decrease_pct = (issue['facc_decrease'] / issue['facc_up'] * 100) if issue['facc_up'] > 0 else 0
                if decrease_pct > 50:
                    st.error(f"**{decrease_pct:.0f}% decrease** → Likely distributary/delta")
                elif decrease_pct > 20:
                    st.warning(f"**{decrease_pct:.0f}% decrease** → Could be bifurcation")
                else:
                    st.info(f"**{decrease_pct:.0f}% decrease** → Minor, likely D8 artifact")

                st.markdown("---")
                st.markdown("### 🎯 Classification")

                # Main decision: D8 artifact or needs review
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("✅ D8 Artifact", key=f"t003_d8_{selected}", type="primary", use_container_width=True):
                        log_skip(conn, selected, region, 'T003', 'D8 routing artifact')
                        st.session_state.t003_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()

                with btn_col2:
                    if st.button("🔍 Needs Review", key=f"t003_review_{selected}", use_container_width=True):
                        log_skip(conn, selected, region, 'T003', 'Needs manual review')
                        st.session_state.t003_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()

                # Other options
                other_reason = st.selectbox(
                    "Or classify as:",
                    ["", "Delta/bifurcation", "Braided river", "Lake outlet", "Other"],
                    key=f"t003_other_{selected}"
                )
                if other_reason and st.button("Mark with this reason", key=f"t003_other_btn_{selected}"):
                    log_skip(conn, selected, region, 'T003', other_reason)
                    st.session_state.t003_pending.append(selected)
                    st.cache_data.clear()
                    st.rerun()

# =============================================================================
# TAB 9: Orphan Reaches (T004)
# =============================================================================
with tab9:
    st.header("🏝️ Orphan Reaches")

    # Initialize session state for T004
    if 't004_results' not in st.session_state:
        st.session_state.t004_results = None
    if 't004_pending' not in st.session_state:
        st.session_state.t004_pending = []

    # Auto-run check
    if st.session_state.t004_results is None or st.session_state.get('t004_region') != region:
        with st.spinner("Running T004 check..."):
            st.session_state.t004_results = check_orphan_reaches(conn, region)
            st.session_state.t004_region = region

    result = st.session_state.t004_results
    if result is None or len(result.details) == 0:
        st.success("✅ No orphan reaches in this region! Topology is fully connected.")
    else:
        issues = result.details
        total = len(issues)
        done = len(st.session_state.t004_pending)
        remaining = len([r for r in issues['reach_id'].tolist() if r not in st.session_state.t004_pending])

        # Progress
        col1, col2, col3 = st.columns(3)
        col1.metric("Remaining", remaining)
        col2.metric("Reviewed", done)
        col3.metric("Total Orphans", total)
        st.progress(done / total if total > 0 else 0)

        st.info("💡 **Orphans** are reaches with no upstream AND no downstream neighbors. Most are valid isolated water bodies (ponds, small lakes).")

        if remaining == 0:
            st.success("🎉 All orphans reviewed!")
        else:
            available = [r for r in issues['reach_id'].tolist() if r not in st.session_state.t004_pending]
            selected = available[0]
            issue = issues[issues['reach_id'] == selected].iloc[0]

            # Problem box
            st.markdown("---")
            st.subheader(f"🔍 Orphan #{done + 1}: Reach `{selected}`")

            width = issue['width'] if pd.notna(issue['width']) else 0
            length = issue['reach_length'] if pd.notna(issue['reach_length']) else 0

            st.warning(f"""
            **Disconnected reach** (no neighbors):
            - **Length:** {length:.0f}m
            - **Width:** {width:.0f}m
            - **Network ID:** {issue['network']}
            """)

            col1, col2 = st.columns([2, 1])
            with col1:
                render_reach_map_satellite(int(selected), region, conn)

            with col2:
                st.markdown(f"**River:** {issue['river_name'] or 'Unnamed'}")

                # Size-based hint
                if width > 100 or length > 1000:
                    st.warning("Large feature → May need topology connection")
                else:
                    st.success("Small feature → Likely valid isolated pond")

                st.markdown("---")
                st.markdown("### 🎯 Classification")

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("✅ Valid Orphan", key=f"t004_valid_{selected}", type="primary", use_container_width=True):
                        log_skip(conn, selected, region, 'T004', 'Valid orphan - isolated water body')
                        st.session_state.t004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()

                with btn_col2:
                    if st.button("🔗 Needs Connection", key=f"t004_connect_{selected}", use_container_width=True):
                        log_skip(conn, selected, region, 'T004', 'Needs topology connection')
                        st.session_state.t004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()

# Footer
st.divider()
st.caption(f"SWORD Reviewer | Region: {region} | DB: sword_v17c.duckdb")
