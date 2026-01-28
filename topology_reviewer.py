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


def render_reach_map_satellite(reach_id, region, conn):
    """Render a map centered on a reach with Esri satellite basemap using folium."""
    geom = get_reach_geometry(conn, reach_id)
    if not geom:
        st.warning("No geometry available")
        return

    # Get upstream and downstream geometries
    up_chain = get_upstream_chain(conn, reach_id, region, 3)
    dn_chain = get_downstream_chain(conn, reach_id, region, 3)

    all_coords = list(geom)

    # Collect upstream geometries
    up_geoms = []
    for i, row in up_chain.iterrows():
        if row['reach_id'] == reach_id:
            continue
        up_geom = get_reach_geometry(conn, int(row['reach_id']))
        if up_geom:
            up_geoms.append((up_geom, i))
            all_coords.extend(up_geom)

    # Collect downstream geometries
    dn_geoms = []
    for i, row in dn_chain.iterrows():
        if row['reach_id'] == reach_id:
            continue
        dn_geom = get_reach_geometry(conn, int(row['reach_id']))
        if dn_geom:
            dn_geoms.append((dn_geom, i))
            all_coords.extend(dn_geom)

    # Calculate bounds and center
    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2
    extent = max(max(lons) - min(lons), max(lats) - min(lats))
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

    # Add upstream reaches (orange, fading)
    for up_geom, i in up_geoms:
        # Convert [lon, lat] to [lat, lon] for folium
        coords = [[c[1], c[0]] for c in up_geom]
        opacity = 1.0 - (i * 0.2)
        folium.PolyLine(
            coords,
            color='orange',
            weight=4,
            opacity=max(0.3, opacity),
            tooltip=f"Upstream {i}"
        ).add_to(m)

    # Add downstream reaches (blue, fading)
    for dn_geom, i in dn_geoms:
        coords = [[c[1], c[0]] for c in dn_geom]
        opacity = 1.0 - (i * 0.2)
        folium.PolyLine(
            coords,
            color='#0066ff',
            weight=4,
            opacity=max(0.3, opacity),
            tooltip=f"Downstream {i}"
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

    # Fit bounds
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    # Render in streamlit
    st_folium(m, width=None, height=400, returned_objects=[])
    st.caption("🔴 Selected | 🟠 Upstream | 🔵 Downstream")


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
    st.header("Headwater Issues")
    st.caption("Headwater reaches (no upstream) with suspiciously high facc")

    min_facc = st.slider("Minimum facc", 1000, 100000, 5000, key="hw_slider")
    hw_issues = get_headwater_issues(conn, region, min_facc)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", len(hw_issues))
    col2.metric("Rivers", len(hw_issues[hw_issues['lakeflag'] == 0]))
    col3.metric("Lakes", len(hw_issues[hw_issues['lakeflag'] == 1]))

    if len(hw_issues) > 0:
        selected_idx = st.selectbox(
            "Select issue",
            range(len(hw_issues)),
            format_func=lambda i: f"#{i+1}: {hw_issues.iloc[i]['reach_id']} (facc={hw_issues.iloc[i]['facc']:,.0f})",
            key="hw_select"
        )

        h = hw_issues.iloc[selected_idx]

        col1, col2 = st.columns([2, 1])

        with col1:
            render_reach_map(int(h['reach_id']), region)

        with col2:
            st.markdown(f"**Reach:** `{h['reach_id']}`")
            st.markdown(f"**facc:** {h['facc']:,.0f} km²")
            st.markdown(f"**width:** {h['width']:.0f}m")
            st.markdown(f"**ratio:** {h['ratio']:,.0f}")
            st.markdown(f"**River:** {h['river_name']}")
            laketype = {0: 'River', 1: 'Lake', 2: 'Canal', 3: 'Tidal'}.get(h['lakeflag'], 'Unknown')
            st.markdown(f"**Type:** {laketype}")

            st.divider()
            render_fix_panel(int(h['reach_id']), region, h['facc'], "headwater")

# =============================================================================
# TAB 4: Suspect Reaches
# =============================================================================
with tab4:
    st.header("Suspect Reaches")
    st.caption("Reaches flagged as facc_quality='suspect' (unfixable by automated methods)")

    suspect = get_suspect_reaches(conn, region)

    st.metric("Total Suspect", len(suspect))

    if len(suspect) > 0:
        selected_idx = st.selectbox(
            "Select reach",
            range(len(suspect)),
            format_func=lambda i: f"#{i+1}: {suspect.iloc[i]['reach_id']} (facc={suspect.iloc[i]['facc']:,.0f})",
            key="suspect_select"
        )

        s = suspect.iloc[selected_idx]

        col1, col2 = st.columns([2, 1])

        with col1:
            render_reach_map(int(s['reach_id']), region)

        with col2:
            st.markdown(f"**Reach:** `{s['reach_id']}`")
            st.markdown(f"**facc:** {s['facc']:,.0f} km²")
            st.markdown(f"**width:** {s['width']:.0f}m")
            st.markdown(f"**ratio:** {s['ratio']:,.0f}")
            st.markdown(f"**River:** {s['river_name']}")
            st.markdown(f"**edit_flag:** {s['edit_flag']}")

            st.divider()
            render_fix_panel(int(s['reach_id']), region, s['facc'], "suspect")

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
    st.header("C001: Lake Sandwich Fixer")
    st.caption("River reaches (lakeflag=0) sandwiched between lake reaches (lakeflag=1)")

    # Help expander with tooltips
    with st.expander("ℹ️ How to identify lake sandwiches"):
        st.markdown("""
        **What to look for:**
        - **Width**: Lakes typically >500m wide. Narrow reaches are likely rivers.
        - **Slope**: Lakes have slope ~0. Rivers have slope >0.001 m/km.
        - **Neighbors**: If both upstream and downstream are lakes, this is probably a lake too.
        - **Satellite**: Check the map - is it wide open water or a flowing channel?

        **Actions:**
        - **Convert to Lake**: Sets lakeflag=1 (use when it's clearly part of a lake)
        - **Skip**: Mark as false positive with required explanation (stays lakeflag=0)
        """)

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
            # Load pending from local session
            session = load_session_fixes(region, "C001")
            st.session_state.pending_fixes = session.get("pending", [])

    # Manual refresh button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Refresh C001 Check", key="run_c001"):
            with st.spinner("Running lake sandwich check..."):
                st.session_state.c001_results = run_c001_check(conn, region)
                st.session_state.pending_fixes = []
            st.rerun()

    # Display results
    if st.session_state.c001_results is None:
        st.info("Loading lake sandwich issues...")
    else:
        result = st.session_state.c001_results
        issues = result.details

        # Metrics row
        col1, col2, col3 = st.columns(3)
        col1.metric("Lake Sandwiches Found", len(issues))
        col2.metric("Pending Fixes", len(st.session_state.pending_fixes))
        col3.metric("Total River Reaches", result.total_checked)

        if len(issues) == 0:
            st.success("No lake sandwich issues found!")
        else:
            # Issue selector
            issue_options = issues['reach_id'].tolist()
            # Filter out already-fixed reaches
            available_options = [r for r in issue_options if r not in st.session_state.pending_fixes]

            if len(available_options) == 0:
                st.success("All issues addressed! Click 'Re-validate Batch' to verify fixes.")
            else:
                selected = st.selectbox(
                    "Select issue to review",
                    available_options,
                    format_func=lambda r: f"{r} - {issues[issues['reach_id']==r].iloc[0]['river_name'] or 'Unnamed'}",
                    key="c001_select"
                )

                issue = issues[issues['reach_id'] == selected].iloc[0]

                # Main content: Map and info side by side
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("Location")
                    render_reach_map_satellite(int(selected), region, conn)

                with col2:
                    st.subheader("Reach Info")

                    # Get additional info
                    up_flags, dn_flags = get_neighbor_lakeflags(conn, selected, region)
                    slope = get_reach_slope(conn, selected, region)
                    facc = get_reach_facc(conn, selected, region)

                    # Format lakeflag display
                    def lakeflag_str(flags):
                        if not flags:
                            return "None"
                        labels = {0: 'River', 1: 'Lake', 2: 'Canal', 3: 'Tidal'}
                        return ', '.join([f"{labels.get(f, '?')}({f})" for f in flags])

                    # Display 6 key attributes
                    st.markdown(f"**Reach ID:** `{selected}`")
                    st.markdown(f"**Width:** {issue['width']:.0f}m" if pd.notna(issue['width']) else "**Width:** N/A")
                    st.markdown(f"**Length:** {issue['reach_length']:.0f}m" if pd.notna(issue['reach_length']) else "**Length:** N/A")
                    st.markdown(f"**Upstream lakeflags:** {lakeflag_str(up_flags)}")
                    st.markdown(f"**Downstream lakeflags:** {lakeflag_str(dn_flags)}")
                    st.markdown(f"**FACC:** {facc:,.0f} km²" if facc else "**FACC:** N/A")
                    st.markdown(f"**Slope:** {slope:.6f} m/km" if slope is not None else "**Slope:** N/A")

                    st.divider()

                    # Action buttons
                    st.subheader("Actions")

                    if st.button("🔵 Convert to Lake (lakeflag=1)", key=f"fix_{selected}", type="primary"):
                        old_val = apply_lakeflag_fix(conn, selected, region, 1)
                        st.session_state.pending_fixes.append(selected)
                        st.session_state.last_fix = selected
                        st.success(f"Fixed! lakeflag: {old_val} → 1")
                        st.cache_data.clear()
                        st.rerun()

                    # Skip with note (false positives)
                    with st.expander("⏭️ Skip (not a lake sandwich)"):
                        # Preset skip reasons
                        skip_reasons = [
                            "Select reason...",
                            "River - clearly flowing water",
                            "Canal - artificial channel",
                            "Tidal - tidal influence",
                            "Narrow channel between lakes",
                            "Dam/weir structure",
                            "Imagery shows river",
                            "Other (specify below)"
                        ]
                        skip_choice = st.selectbox(
                            "Reason",
                            skip_reasons,
                            key=f"skip_reason_{selected}"
                        )
                        # Custom reason if "Other" selected
                        custom_reason = ""
                        if skip_choice == "Other (specify below)":
                            custom_reason = st.text_input(
                                "Specify reason",
                                key=f"skip_custom_{selected}"
                            )

                        # Determine final skip note
                        skip_note = ""
                        if skip_choice and skip_choice != "Select reason...":
                            if skip_choice == "Other (specify below)":
                                skip_note = custom_reason
                            else:
                                skip_note = skip_choice

                        if st.button("Skip this issue", key=f"skip_{selected}", disabled=not skip_note):
                            log_skip(conn, selected, region, 'C001', skip_note)
                            st.session_state.pending_fixes.append(selected)  # Remove from list
                            st.info(f"Skipped: {skip_note}")
                            st.cache_data.clear()
                            st.rerun()

                    # Undo last fix
                    if st.session_state.last_fix:
                        st.divider()
                        if st.button(f"↩️ Undo last fix ({st.session_state.last_fix})", key="undo_last"):
                            undone_id = undo_last_fix(conn, region)
                            if undone_id:
                                st.session_state.pending_fixes = [p for p in st.session_state.pending_fixes if p != undone_id]
                                st.session_state.last_fix = None
                                st.warning(f"Undone fix for reach {undone_id}")
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.error("No fix to undo")

        # Batch validation section
        st.divider()
        st.subheader("Batch Validation")

        pending = st.session_state.pending_fixes
        st.write(f"**Pending fixes/skips:** {len(pending)}")

        if len(pending) > 0:
            st.write(f"Reaches: {pending[:10]}{'...' if len(pending) > 10 else ''}")

        if st.button("🔄 Re-validate Batch", key="revalidate", disabled=len(pending) == 0, type="primary"):
            with st.spinner("Re-validating..."):
                before_count = len(issues) if st.session_state.c001_results else 0

                # Run C001 on fixed reaches + neighbors
                new_results = run_c001_check(conn, region, reach_ids=pending)
                full_results = run_c001_check(conn, region)  # Also get full count

                # Analyze results
                still_failing = [r for r in pending if r in new_results.details['reach_id'].values]
                new_total = len(full_results.details)
                # New issues = issues that weren't in original set AND weren't in pending
                original_ids = set(issues['reach_id'].values) if len(issues) > 0 else set()
                new_issues = [r for r in full_results.details['reach_id'].values
                              if r not in original_ids and r not in pending]

            # Display results
            col1, col2, col3 = st.columns(3)
            fixed_count = len(pending) - len(still_failing)
            col1.metric("✅ Fixed", fixed_count)
            col2.metric("❌ Still Failing", len(still_failing), delta_color="inverse")
            col3.metric("⚠️ New Issues", len(new_issues), delta_color="inverse")

            if len(still_failing) == 0 and len(new_issues) == 0:
                st.success("All fixes validated successfully!")
            elif len(still_failing) > 0:
                st.warning(f"Still failing: {still_failing}")
            if len(new_issues) > 0:
                st.warning(f"New issues created: {new_issues[:5]}{'...' if len(new_issues) > 5 else ''}")

            # Clear pending and update results
            st.session_state.pending_fixes = []
            st.session_state.c001_results = full_results
            st.session_state.last_fix = None

        # Local session file (persistent across refreshes)
        st.divider()
        st.subheader("📁 Local Session File")
        st.caption("Fixes are saved locally and persist across refreshes")

        session_file = get_session_file(region, "C001")
        session_data = load_session_fixes(region, "C001")

        col1, col2, col3 = st.columns(3)
        col1.metric("Fixes Saved", len(session_data.get("fixes", [])))
        col2.metric("Skips Saved", len(session_data.get("skips", [])))
        if session_data.get("last_updated"):
            col3.write(f"**Last update:** {session_data['last_updated'][:19]}")

        # Show session file path
        st.code(str(session_file), language=None)

        # Export local session as CSV
        col1, col2 = st.columns(2)
        with col1:
            local_csv = export_session_csv(region, "C001")
            if local_csv:
                st.download_button(
                    "📥 Export Local Session (CSV)",
                    local_csv,
                    f"c001_session_{region}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    key="export_local_csv"
                )
            else:
                st.info("No local fixes to export")

        with col2:
            # Export as JSON
            if session_data.get("fixes") or session_data.get("skips"):
                json_str = json.dumps(session_data, indent=2, default=str)
                st.download_button(
                    "📥 Export Local Session (JSON)",
                    json_str,
                    f"c001_session_{region}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    key="export_local_json"
                )

        # Clear session button
        if st.button("🗑️ Clear Local Session", key="clear_session"):
            save_session_fixes(region, [], [], [], check_id="C001")
            st.success("Local session cleared!")
            st.rerun()

        # Fix history for C001 (from database)
        st.divider()
        st.subheader("C001 Fix History (Database)")
        lint_history = get_lint_fix_history(conn, region, 50)
        if len(lint_history) > 0:
            st.dataframe(lint_history[['timestamp', 'reach_id', 'action', 'old_value', 'new_value', 'notes', 'undone']], height=200)

            # Export
            csv = lint_history.to_csv(index=False)
            st.download_button(
                "📥 Download DB Fix History",
                csv,
                f"c001_fixes_db_{region}_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                key="export_db_csv"
            )
        else:
            st.info("No C001 fixes in database yet")

# =============================================================================
# TAB 7: Lakeflag/Type Consistency (C004)
# =============================================================================
with tab7:
    st.header("C004: Lakeflag/Type Consistency")
    st.caption("Reaches where lakeflag and type fields don't match expected mappings")

    with st.expander("ℹ️ Expected mappings"):
        st.markdown("""
        | Lakeflag | Expected Type |
        |----------|---------------|
        | 0 (river) | 1 (river) or 3 (tidal_river) |
        | 1 (lake) | 2 (lake) |
        | 2 (canal) | 4 (artificial) |
        | 3 (tidal) | 3 (tidal_river) |

        **Actions:**
        - **Fix lakeflag**: Change lakeflag to match type
        - **Fix type**: Change type to match lakeflag
        - **Skip**: Mark as reviewed (no change needed)
        """)

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

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Refresh C004", key="refresh_c004"):
            with st.spinner("Running C004 check..."):
                st.session_state.c004_results = check_lakeflag_type_consistency(conn, region)
            st.rerun()

    result = st.session_state.c004_results
    if result is None or len(result.details) == 0:
        if result and result.total_checked == 0:
            st.warning("⚠️ Type column not present in database - cannot check consistency. This check requires adding the 'type' column first.")
        else:
            st.success("No lakeflag/type mismatches found!")
    else:
        issues = result.details

        col1, col2, col3 = st.columns(3)
        col1.metric("Mismatches Found", len(issues))
        col2.metric("Total Checked", result.total_checked)
        col3.metric("Issue %", f"{result.issue_pct:.2f}%")

        # Group by issue type
        if 'issue_type' in issues.columns:
            st.subheader("By Issue Type")
            issue_counts = issues['issue_type'].value_counts()
            st.dataframe(issue_counts)

        # Issue selector
        available = [r for r in issues['reach_id'].tolist() if r not in st.session_state.c004_pending]

        if len(available) == 0:
            st.success("All issues reviewed!")
        else:
            selected = st.selectbox(
                "Select issue",
                available,
                format_func=lambda r: f"{r} - {issues[issues['reach_id']==r].iloc[0]['issue_type']}",
                key="c004_select"
            )

            issue = issues[issues['reach_id'] == selected].iloc[0]

            col1, col2 = st.columns([2, 1])

            with col1:
                render_reach_map_satellite(int(selected), region, conn)

            with col2:
                st.subheader("Issue Details")
                st.markdown(f"**Reach ID:** `{selected}`")

                lakeflag_map = {0: 'River', 1: 'Lake', 2: 'Canal', 3: 'Tidal'}
                type_map = {1: 'river', 2: 'lake', 3: 'tidal_river', 4: 'artificial', 5: 'unassigned', 6: 'unreliable'}

                lf = issue['lakeflag']
                tp = issue['type']
                st.markdown(f"**Lakeflag:** {lf} ({lakeflag_map.get(lf, '?')})")
                st.markdown(f"**Type:** {tp} ({type_map.get(tp, '?')})")
                st.markdown(f"**Issue:** {issue['issue_type']}")
                st.markdown(f"**River:** {issue['river_name'] or 'Unnamed'}")

                st.divider()
                st.subheader("Actions")

                # Fix options based on issue type
                if issue['issue_type'] == 'lake_type_mismatch':
                    # lakeflag=1 (lake) but type != 2
                    if st.button("Set type=2 (lake)", key=f"c004_fix_type_{selected}", type="primary"):
                        conn.execute("UPDATE reaches SET type = 2 WHERE reach_id = ? AND region = ?", [selected, region])
                        conn.commit()
                        log_skip(conn, selected, region, 'C004', f'Fixed: set type=2 (was {tp})')
                        st.session_state.c004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()
                    if st.button("Set lakeflag=0 (river)", key=f"c004_fix_lf_{selected}"):
                        conn.execute("UPDATE reaches SET lakeflag = 0 WHERE reach_id = ? AND region = ?", [selected, region])
                        conn.commit()
                        log_skip(conn, selected, region, 'C004', f'Fixed: set lakeflag=0 (was {lf})')
                        st.session_state.c004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()

                elif issue['issue_type'] == 'river_type_mismatch':
                    # lakeflag=0 (river) but type not in (1, 3)
                    if st.button("Set type=1 (river)", key=f"c004_fix_type_{selected}", type="primary"):
                        conn.execute("UPDATE reaches SET type = 1 WHERE reach_id = ? AND region = ?", [selected, region])
                        conn.commit()
                        log_skip(conn, selected, region, 'C004', f'Fixed: set type=1 (was {tp})')
                        st.session_state.c004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()
                    if st.button("Set lakeflag=1 (lake)", key=f"c004_fix_lf_{selected}"):
                        conn.execute("UPDATE reaches SET lakeflag = 1 WHERE reach_id = ? AND region = ?", [selected, region])
                        conn.commit()
                        log_skip(conn, selected, region, 'C004', f'Fixed: set lakeflag=1 (was {lf})')
                        st.session_state.c004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()

                elif issue['issue_type'] == 'canal_type_mismatch':
                    if st.button("Set type=4 (artificial)", key=f"c004_fix_type_{selected}", type="primary"):
                        conn.execute("UPDATE reaches SET type = 4 WHERE reach_id = ? AND region = ?", [selected, region])
                        conn.commit()
                        log_skip(conn, selected, region, 'C004', f'Fixed: set type=4 (was {tp})')
                        st.session_state.c004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()

                elif issue['issue_type'] == 'tidal_type_mismatch':
                    if st.button("Set type=3 (tidal_river)", key=f"c004_fix_type_{selected}", type="primary"):
                        conn.execute("UPDATE reaches SET type = 3 WHERE reach_id = ? AND region = ?", [selected, region])
                        conn.commit()
                        log_skip(conn, selected, region, 'C004', f'Fixed: set type=3 (was {tp})')
                        st.session_state.c004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()

                # Skip option
                with st.expander("⏭️ Skip (leave as-is)"):
                    skip_reasons = [
                        "Select reason...",
                        "Correct as-is",
                        "Needs manual review",
                        "Data quality issue",
                        "Other (specify below)"
                    ]
                    skip_choice = st.selectbox("Reason", skip_reasons, key=f"c004_skip_reason_{selected}")
                    custom_reason = ""
                    if skip_choice == "Other (specify below)":
                        custom_reason = st.text_input("Specify", key=f"c004_skip_custom_{selected}")

                    skip_note = custom_reason if skip_choice == "Other (specify below)" else skip_choice
                    if st.button("Skip", key=f"c004_skip_{selected}", disabled=skip_choice == "Select reason..."):
                        log_skip(conn, selected, region, 'C004', f'Skipped: {skip_note}')
                        st.session_state.c004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()

# =============================================================================
# TAB 8: FACC Monotonicity (T003)
# =============================================================================
with tab8:
    st.header("T003: FACC Monotonicity")
    st.caption("Reaches where flow accumulation decreases downstream (should increase)")

    with st.expander("ℹ️ Understanding FACC issues"):
        st.markdown("""
        **Why facc decreases downstream:**
        - **D8 routing artifact**: D8 picks ONE downstream cell, so distributaries get wrong values
        - **Delta/bifurcation**: Flow splits but D8 only tracks one branch
        - **Braided rivers**: Multiple channels, D8 picks one
        - **Data error**: Actual mistake in source data

        **Most T003 issues are D8 artifacts and can't be "fixed" without changing the routing algorithm.**

        **Actions:**
        - **Skip (D8 artifact)**: Mark as known limitation
        - **Fix facc**: Only if you're sure it's a real error
        """)

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

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Refresh T003", key="refresh_t003"):
            with st.spinner("Running T003 check..."):
                st.session_state.t003_results = check_facc_monotonicity(conn, region)
            st.rerun()

    result = st.session_state.t003_results
    if result is None or len(result.details) == 0:
        st.success("No FACC monotonicity issues found!")
    else:
        issues = result.details

        col1, col2, col3 = st.columns(3)
        col1.metric("Issues Found", len(issues))
        col2.metric("Total Checked", result.total_checked)
        col3.metric("Issue %", f"{result.issue_pct:.2f}%")

        # Issue selector
        available = [r for r in issues['reach_id'].tolist() if r not in st.session_state.t003_pending]

        if len(available) == 0:
            st.success("All issues reviewed!")
        else:
            selected = st.selectbox(
                "Select issue",
                available,
                format_func=lambda r: f"{r} - decrease: {issues[issues['reach_id']==r].iloc[0]['facc_decrease']:,.0f} km²",
                key="t003_select"
            )

            issue = issues[issues['reach_id'] == selected].iloc[0]

            col1, col2 = st.columns([2, 1])

            with col1:
                render_reach_map_satellite(int(selected), region, conn)

            with col2:
                st.subheader("Issue Details")
                st.markdown(f"**Reach ID:** `{selected}`")
                st.markdown(f"**Upstream facc:** {issue['facc_up']:,.0f} km²")
                st.markdown(f"**Downstream facc:** {issue['facc_down']:,.0f} km²")
                st.markdown(f"**Decrease:** {issue['facc_decrease']:,.0f} km²")
                st.markdown(f"**River:** {issue['river_name'] or 'Unnamed'}")

                st.divider()
                st.subheader("Actions")

                # Skip options (most common for T003)
                skip_reasons = [
                    "Select reason...",
                    "D8 distributary artifact",
                    "Delta/bifurcation - expected",
                    "Braided river section",
                    "Anastomosing channel",
                    "Lake outlet artifact",
                    "Real error - needs fix",
                    "Other (specify below)"
                ]
                skip_choice = st.selectbox("Classification", skip_reasons, key=f"t003_reason_{selected}")

                custom_reason = ""
                if skip_choice == "Other (specify below)":
                    custom_reason = st.text_input("Specify", key=f"t003_custom_{selected}")

                skip_note = custom_reason if skip_choice == "Other (specify below)" else skip_choice

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ Mark Reviewed", key=f"t003_skip_{selected}",
                                 disabled=skip_choice == "Select reason...", type="primary"):
                        log_skip(conn, selected, region, 'T003', skip_note)
                        st.session_state.t003_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()

                # Fix option (rare)
                with st.expander("🔧 Fix facc (use sparingly)"):
                    st.warning("Only fix if you're certain this is a real data error, not a D8 artifact")
                    new_facc = st.number_input("New facc value", min_value=0.0,
                                               value=float(issue['facc_down']), key=f"t003_facc_{selected}")
                    if st.button("Apply fix", key=f"t003_fix_{selected}"):
                        apply_facc_fix(conn, selected, region, new_facc, "T003_manual", skip_note)
                        st.session_state.t003_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()

# =============================================================================
# TAB 9: Orphan Reaches (T004)
# =============================================================================
with tab9:
    st.header("T004: Orphan Reaches")
    st.caption("Reaches with no upstream AND no downstream neighbors (disconnected)")

    with st.expander("ℹ️ Understanding orphan reaches"):
        st.markdown("""
        **Why reaches become orphans:**
        - **Small isolated water body**: Pond, small lake not connected to network
        - **Data processing artifact**: Topology wasn't built correctly
        - **Island in lake**: Water body that should be connected
        - **Coastal feature**: Tidal inlet or lagoon

        **Actions:**
        - **Skip (valid orphan)**: Small pond, isolated feature - correct as-is
        - **Flag for topology fix**: Should be connected but isn't
        """)

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

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Refresh T004", key="refresh_t004"):
            with st.spinner("Running T004 check..."):
                st.session_state.t004_results = check_orphan_reaches(conn, region)
            st.rerun()

    result = st.session_state.t004_results
    if result is None or len(result.details) == 0:
        st.success("No orphan reaches found!")
    else:
        issues = result.details

        col1, col2, col3 = st.columns(3)
        col1.metric("Orphans Found", len(issues))
        col2.metric("Total Reaches", result.total_checked)
        col3.metric("Orphan %", f"{result.issue_pct:.2f}%")

        # Summary stats
        if 'width' in issues.columns:
            col1, col2 = st.columns(2)
            col1.metric("Avg Width", f"{issues['width'].mean():.0f}m")
            col2.metric("Avg Length", f"{issues['reach_length'].mean():.0f}m")

        # Issue selector
        available = [r for r in issues['reach_id'].tolist() if r not in st.session_state.t004_pending]

        if len(available) == 0:
            st.success("All orphans reviewed!")
        else:
            selected = st.selectbox(
                "Select orphan",
                available,
                format_func=lambda r: f"{r} - {issues[issues['reach_id']==r].iloc[0]['river_name'] or 'Unnamed'} ({issues[issues['reach_id']==r].iloc[0]['reach_length']:.0f}m)",
                key="t004_select"
            )

            issue = issues[issues['reach_id'] == selected].iloc[0]

            col1, col2 = st.columns([2, 1])

            with col1:
                render_reach_map_satellite(int(selected), region, conn)

            with col2:
                st.subheader("Reach Details")
                st.markdown(f"**Reach ID:** `{selected}`")
                st.markdown(f"**River:** {issue['river_name'] or 'Unnamed'}")
                st.markdown(f"**Length:** {issue['reach_length']:.0f}m")
                st.markdown(f"**Width:** {issue['width']:.0f}m" if pd.notna(issue['width']) else "**Width:** N/A")
                st.markdown(f"**Network ID:** {issue['network']}")
                st.markdown(f"**n_rch_up:** {issue['n_rch_up']} | **n_rch_down:** {issue['n_rch_down']}")

                st.divider()
                st.subheader("Classification")

                skip_reasons = [
                    "Select reason...",
                    "Valid orphan - small pond",
                    "Valid orphan - isolated lake",
                    "Valid orphan - coastal feature",
                    "Needs topology connection",
                    "Should be deleted",
                    "Other (specify below)"
                ]
                skip_choice = st.selectbox("Classification", skip_reasons, key=f"t004_reason_{selected}")

                custom_reason = ""
                if skip_choice == "Other (specify below)":
                    custom_reason = st.text_input("Specify", key=f"t004_custom_{selected}")

                skip_note = custom_reason if skip_choice == "Other (specify below)" else skip_choice

                if st.button("✅ Mark Reviewed", key=f"t004_mark_{selected}",
                             disabled=skip_choice == "Select reason...", type="primary"):
                    log_skip(conn, selected, region, 'T004', skip_note)
                    st.session_state.t004_pending.append(selected)
                    st.cache_data.clear()
                    st.rerun()

# Footer
st.divider()
st.caption(f"SWORD Reviewer | Region: {region} | DB: sword_v17c.duckdb")
