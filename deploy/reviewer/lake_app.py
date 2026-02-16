#!/usr/bin/env python3
"""
SWORD Lake QA Reviewer (Cloud Run deploy)
==========================================
Streamlit UI for QA review of lake-related issues only.
Tabs: Lakeflag/Type (C004), Lake Sandwich (C001), Fix History.

Run with: streamlit run lake_app.py

Shares the same DuckDB file and lint_fix_log table as topology_reviewer.py.
Cannot run both simultaneously (DuckDB single-writer lock).
"""

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import duckdb
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from lint.checks.classification import check_lake_sandwich

# =============================================================================
# LOCAL PERSISTENCE (JSON file backup for session fixes)
# =============================================================================
FIXES_DIR = Path(os.environ.get("FIXES_DIR", "output/lint_fixes"))
FIXES_DIR.mkdir(parents=True, exist_ok=True)


def get_session_file(region: str, check_id: str = "all") -> Path:
    return FIXES_DIR / f"lint_session_{region}_{check_id}.json"


def load_session_fixes(region: str, check_id: str = "all") -> dict:
    session_file = get_session_file(region, check_id)
    if session_file.exists():
        with open(session_file) as f:
            return json.load(f)
    return {"fixes": [], "skips": [], "pending": []}


def save_session_fixes(region, fixes, skips, pending, check_id="all"):
    session_file = get_session_file(region, check_id)
    data = {
        "region": region,
        "check_id": check_id,
        "last_updated": datetime.now().isoformat(),
        "fixes": fixes,
        "skips": skips,
        "pending": pending,
    }
    with open(session_file, "w") as f:
        json.dump(data, f, indent=2, default=str)


def append_fix_to_session(region, fix_record, check_id="all"):
    session = load_session_fixes(region, check_id)
    session["fixes"].append(fix_record)
    save_session_fixes(
        region, session["fixes"], session["skips"], session.get("pending", []), check_id
    )


def append_skip_to_session(region, skip_record, check_id="all"):
    session = load_session_fixes(region, check_id)
    session["skips"].append(skip_record)
    save_session_fixes(
        region, session["fixes"], session["skips"], session.get("pending", []), check_id
    )


# =============================================================================
# PAGE CONFIG + DB CONNECTION
# =============================================================================
st.set_page_config(page_title="SWORD Lake Reviewer", page_icon="🏷️", layout="wide")


@st.cache_resource
def get_connection():
    conn = duckdb.connect(
        os.environ.get("SWORD_DB_PATH", "data/duckdb/sword_v17c.duckdb")
    )
    try:
        conn.execute("INSTALL spatial")
        conn.execute("LOAD spatial")
    except Exception:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lint_fix_log (
            fix_id INTEGER, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            check_id VARCHAR, reach_id BIGINT, region VARCHAR, action VARCHAR,
            column_changed VARCHAR, old_value VARCHAR, new_value VARCHAR,
            notes VARCHAR, undone BOOLEAN DEFAULT FALSE
        )
    """)
    return conn


conn = get_connection()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
@st.cache_data(ttl=300)
def get_reach_geometry(_conn, reach_id):
    nodes = _conn.execute(
        "SELECT x, y FROM nodes WHERE reach_id = ? ORDER BY dist_out DESC", [reach_id]
    ).fetchdf()
    if len(nodes) == 0:
        return None
    return nodes[["x", "y"]].values.tolist()


@st.cache_data(ttl=60)
def get_upstream_chain(_conn, reach_id, region, max_hops=5):
    chain = []
    current_id = reach_id
    for hop in range(max_hops):
        info = _conn.execute(
            """
            SELECT r.reach_id, r.facc, r.width, r.river_name, t.neighbor_reach_id as up_id
            FROM reaches r
            LEFT JOIN reach_topology t ON r.reach_id = t.reach_id AND r.region = t.region AND t.direction = 'up'
            WHERE r.reach_id = ? AND r.region = ?
            ORDER BY t.neighbor_rank LIMIT 1
        """,
            [current_id, region],
        ).fetchone()
        if not info:
            break
        chain.append(
            {
                "hop": hop,
                "reach_id": info[0],
                "facc": info[1],
                "width": info[2],
                "river_name": info[3],
            }
        )
        if info[4] is None or info[4] <= 0:
            break
        current_id = int(info[4])
    return pd.DataFrame(chain)


@st.cache_data(ttl=60)
def get_downstream_chain(_conn, reach_id, region, max_hops=5):
    chain = []
    current_id = reach_id
    for hop in range(max_hops):
        info = _conn.execute(
            """
            SELECT r.reach_id, r.facc, r.width, r.river_name, t.neighbor_reach_id as dn_id
            FROM reaches r
            LEFT JOIN reach_topology t ON r.reach_id = t.reach_id AND r.region = t.region AND t.direction = 'down'
            WHERE r.reach_id = ? AND r.region = ?
            ORDER BY t.neighbor_rank LIMIT 1
        """,
            [current_id, region],
        ).fetchone()
        if not info:
            break
        chain.append(
            {
                "hop": hop,
                "reach_id": info[0],
                "facc": info[1],
                "width": info[2],
                "river_name": info[3],
            }
        )
        if info[4] is None or info[4] <= 0:
            break
        current_id = int(info[4])
    return pd.DataFrame(chain)


def get_neighbors(conn, reach_ids, region, hops=2):
    if not reach_ids:
        return set()
    all_neighbors = set()
    current_ids = set(reach_ids)
    for _ in range(hops):
        if not current_ids:
            break
        placeholders = ",".join(["?" for _ in current_ids])
        result = conn.execute(
            f"""
            SELECT DISTINCT neighbor_reach_id FROM reach_topology
            WHERE reach_id IN ({placeholders}) AND region = ?
        """,
            list(current_ids) + [region],
        ).fetchall()
        new_ids = {row[0] for row in result if row[0]}
        all_neighbors.update(new_ids)
        current_ids = new_ids - set(reach_ids) - all_neighbors
    return all_neighbors


def run_c001_check(conn, region, reach_ids=None):
    result = check_lake_sandwich(conn, region)
    if reach_ids is not None and len(reach_ids) > 0:
        neighbor_ids = get_neighbors(conn, reach_ids, region, hops=2)
        all_ids = set(reach_ids) | neighbor_ids
        if len(result.details) > 0:
            result.details = result.details[result.details["reach_id"].isin(all_ids)]
    return result


def get_neighbor_lakeflags(conn, reach_id, region):
    up_result = conn.execute(
        """
        SELECT r.lakeflag FROM reach_topology t
        JOIN reaches r ON t.neighbor_reach_id = r.reach_id AND t.region = r.region
        WHERE t.reach_id = ? AND t.region = ? AND t.direction = 'up'
    """,
        [reach_id, region],
    ).fetchall()
    dn_result = conn.execute(
        """
        SELECT r.lakeflag FROM reach_topology t
        JOIN reaches r ON t.neighbor_reach_id = r.reach_id AND t.region = r.region
        WHERE t.reach_id = ? AND t.region = ? AND t.direction = 'down'
    """,
        [reach_id, region],
    ).fetchall()
    return [row[0] for row in up_result], [row[0] for row in dn_result]


def get_reach_slope(conn, reach_id, region):
    result = conn.execute(
        "SELECT slope FROM reaches WHERE reach_id = ? AND region = ?",
        [reach_id, region],
    ).fetchone()
    return result[0] if result and result[0] is not None else None


def apply_lakeflag_fix(conn, reach_id, region, new_lakeflag):
    old = conn.execute(
        "SELECT lakeflag FROM reaches WHERE reach_id = ? AND region = ?",
        [reach_id, region],
    ).fetchone()
    old_lakeflag = old[0] if old else None
    max_id = conn.execute(
        "SELECT COALESCE(MAX(fix_id), 0) FROM lint_fix_log"
    ).fetchone()[0]
    new_id = max_id + 1
    timestamp = datetime.now().isoformat()
    conn.execute(
        """
        INSERT INTO lint_fix_log (fix_id, check_id, reach_id, region, action, column_changed, old_value, new_value, notes)
        VALUES (?, 'C001', ?, ?, 'fix', 'lakeflag', ?, ?, '')
    """,
        [new_id, reach_id, region, str(old_lakeflag), str(new_lakeflag)],
    )
    conn.execute(
        "UPDATE reaches SET lakeflag = ? WHERE reach_id = ? AND region = ?",
        [new_lakeflag, reach_id, region],
    )
    conn.commit()
    append_fix_to_session(
        region,
        {
            "fix_id": new_id,
            "timestamp": timestamp,
            "check_id": "C001",
            "reach_id": int(reach_id),
            "region": region,
            "column_changed": "lakeflag",
            "old_value": old_lakeflag,
            "new_value": new_lakeflag,
            "undone": False,
        },
        check_id="C001",
    )
    return old_lakeflag


def log_skip(conn, reach_id, region, check_id, notes):
    max_id = conn.execute(
        "SELECT COALESCE(MAX(fix_id), 0) FROM lint_fix_log"
    ).fetchone()[0]
    new_id = max_id + 1
    timestamp = datetime.now().isoformat()
    conn.execute(
        """
        INSERT INTO lint_fix_log (fix_id, check_id, reach_id, region, action, column_changed, old_value, new_value, notes)
        VALUES (?, ?, ?, ?, 'skip', NULL, NULL, NULL, ?)
    """,
        [new_id, check_id, reach_id, region, notes],
    )
    conn.commit()
    append_skip_to_session(
        region,
        {
            "fix_id": new_id,
            "timestamp": timestamp,
            "check_id": check_id,
            "reach_id": int(reach_id),
            "region": region,
            "notes": notes,
            "undone": False,
        },
        check_id=check_id,
    )


def undo_last_fix(conn, region):
    last = conn.execute(
        """
        SELECT fix_id, reach_id, old_value FROM lint_fix_log
        WHERE region = ? AND action = 'fix' AND NOT undone ORDER BY timestamp DESC LIMIT 1
    """,
        [region],
    ).fetchone()
    if not last:
        return None
    fix_id, reach_id, old_value = last
    if old_value is not None:
        conn.execute(
            "UPDATE reaches SET lakeflag = ? WHERE reach_id = ? AND region = ?",
            [int(old_value), reach_id, region],
        )
    conn.execute("UPDATE lint_fix_log SET undone = TRUE WHERE fix_id = ?", [fix_id])
    max_id = conn.execute(
        "SELECT COALESCE(MAX(fix_id), 0) FROM lint_fix_log"
    ).fetchone()[0]
    conn.execute(
        """
        INSERT INTO lint_fix_log (fix_id, check_id, reach_id, region, action, column_changed, old_value, new_value, notes)
        VALUES (?, 'C001', ?, ?, 'undo', 'lakeflag', NULL, ?, ?)
    """,
        [max_id + 1, reach_id, region, old_value, f"Undo of fix_id={fix_id}"],
    )
    conn.commit()
    session = load_session_fixes(region, check_id="C001")
    for fix in session["fixes"]:
        if fix.get("fix_id") == fix_id:
            fix["undone"] = True
            break
    save_session_fixes(
        region,
        session["fixes"],
        session["skips"],
        session.get("pending", []),
        check_id="C001",
    )
    return reach_id


def get_nearby_reaches(
    conn,
    center_lon,
    center_lat,
    radius_deg,
    region,
    exclude_ids=None,
    include_all=False,
    max_reaches=2000,
):
    exclude_ids = exclude_ids or []
    exclude_str = ",".join([str(int(r)) for r in exclude_ids]) if exclude_ids else "0"
    if include_all:
        query = f"""SELECT reach_id, x, y, lakeflag, facc, width, n_rch_up, n_rch_down FROM reaches
            WHERE region = ? AND x BETWEEN ? AND ? AND y BETWEEN ? AND ? LIMIT {max_reaches}"""
    else:
        query = f"""SELECT reach_id, x, y, lakeflag, facc, width, n_rch_up, n_rch_down FROM reaches
            WHERE region = ? AND x BETWEEN ? AND ? AND y BETWEEN ? AND ? AND reach_id NOT IN ({exclude_str}) LIMIT {max_reaches}"""
    return conn.execute(
        query,
        [
            region,
            center_lon - radius_deg,
            center_lon + radius_deg,
            center_lat - radius_deg,
            center_lat + radius_deg,
        ],
    ).fetchdf()


def add_flow_arrows(m, coords, color, num_arrows=2, size=0.003):
    """Add vector arrows along a polyline to show flow direction."""
    if len(coords) < 2:
        return
    for i in range(num_arrows):
        frac = (i + 1) / (num_arrows + 1)
        idx = int(frac * (len(coords) - 1))
        idx = max(0, min(idx, len(coords) - 2))
        p1, p2 = coords[idx], coords[idx + 1]
        dx = p2[1] - p1[1]
        dy = p2[0] - p1[0]
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            continue
        dx /= length
        dy /= length
        cx, cy = (p1[1] + p2[1]) / 2, (p1[0] + p2[0]) / 2
        tail_lon, tail_lat = cx - dx * size, cy - dy * size
        tip_lon, tip_lat = cx + dx * size, cy + dy * size
        head_size = size * 0.6
        angle1 = math.atan2(dy, dx) + math.radians(150)
        angle2 = math.atan2(dy, dx) - math.radians(150)
        v1_lon = tip_lon + math.cos(angle1) * head_size
        v1_lat = tip_lat + math.sin(angle1) * head_size
        v2_lon = tip_lon + math.cos(angle2) * head_size
        v2_lat = tip_lat + math.sin(angle2) * head_size
        for c, w in [("black", 5), (color, 3)]:
            folium.PolyLine(
                [[tail_lat, tail_lon], [tip_lat, tip_lon]], color=c, weight=w, opacity=1
            ).add_to(m)
            folium.PolyLine(
                [[v1_lat, v1_lon], [tip_lat, tip_lon], [v2_lat, v2_lon]],
                color=c,
                weight=w,
                opacity=1,
            ).add_to(m)


def render_reach_map_satellite(reach_id, region, conn, hops=None, color_by_type=False):
    """Render a map centered on a reach with Esri satellite basemap using folium."""
    geom = get_reach_geometry(conn, reach_id)
    if not geom:
        st.warning("No geometry available")
        return
    if hops is None:
        hops = st.session_state.get("map_hops", 25)
    show_all = st.session_state.get("show_all_reaches", True)
    up_chain = get_upstream_chain(conn, reach_id, region, hops)
    dn_chain = get_downstream_chain(conn, reach_id, region, hops)
    all_coords = list(geom)
    connected_ids = {reach_id}
    up_geoms = []
    for i, row in up_chain.iterrows():
        if row["reach_id"] == reach_id:
            continue
        connected_ids.add(row["reach_id"])
        ug = get_reach_geometry(conn, int(row["reach_id"]))
        if ug:
            up_geoms.append((ug, i, row["reach_id"]))
            all_coords.extend(ug)
    dn_geoms = []
    for i, row in dn_chain.iterrows():
        if row["reach_id"] == reach_id:
            continue
        connected_ids.add(row["reach_id"])
        dg = get_reach_geometry(conn, int(row["reach_id"]))
        if dg:
            dn_geoms.append((dg, i, row["reach_id"]))
            all_coords.extend(dg)
    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2
    extent = max(max(lons) - min(lons), max(lats) - min(lats))
    view_radius = max(extent * 0.75, 0.02)
    zoom = 15 if extent < 0.02 else 14 if extent < 0.05 else 12 if extent < 0.1 else 10
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True,
        show=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
        name="Dark",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)
    folium.LayerControl().add_to(m)
    nearby_unconnected = []
    if show_all:
        search_radius = st.session_state.get("map_radius", 4.0)
        max_r = st.session_state.get("max_reaches", 15000)
        nearby = get_nearby_reaches(
            conn,
            center_lon,
            center_lat,
            search_radius,
            region,
            list(connected_ids),
            include_all=False,
            max_reaches=max_r,
        )
        type_colors = {0: "#ffffff", 1: "#00ffff", 2: "#ffff00", 3: "#ff00ff"}
        type_names = {0: "River", 1: "Lake", 2: "Canal", 3: "Tidal"}
        highlight_id = st.session_state.get("highlight_reach") or st.session_state.get(
            "clicked_reach"
        )
        for _, row in nearby.iterrows():
            ng = get_reach_geometry(conn, int(row["reach_id"]))
            if ng:
                rid = row["reach_id"]
                lf = int(row["lakeflag"]) if pd.notna(row["lakeflag"]) else 0
                facc = row["facc"] if pd.notna(row["facc"]) else 0
                w = row["width"] if pd.notna(row["width"]) else 0
                nearby_unconnected.append((ng, rid, lf))
                coords = [[c[1], c[0]] for c in ng]
                tn = type_names.get(lf, "?")
                clr = type_colors.get(lf, "#ffffff")
                if highlight_id and int(rid) == int(highlight_id):
                    folium.PolyLine(
                        coords,
                        color="#00ff00",
                        weight=6,
                        opacity=1.0,
                        tooltip=f"SELECTED: {rid} ({tn}, facc={facc:,.0f}, w={w:.0f}m)",
                    ).add_to(m)
                    add_flow_arrows(m, coords, "#00ff00", num_arrows=2)
                else:
                    folium.PolyLine(
                        coords,
                        color=clr,
                        weight=3,
                        opacity=0.9,
                        tooltip=f"{tn}: {rid} (facc={facc:,.0f}, w={w:.0f}m)",
                    ).add_to(m)
                    add_flow_arrows(m, coords, clr, num_arrows=1)
    connected_lakeflags = {}
    if color_by_type:
        all_connected = [rid for _, _, rid in up_geoms] + [
            rid for _, _, rid in dn_geoms
        ]
        if all_connected:
            ph = ",".join(["?"] * len(all_connected))
            lf_rows = conn.execute(
                f"SELECT reach_id, lakeflag FROM reaches WHERE reach_id IN ({ph}) AND region = ?",
                all_connected + [region],
            ).fetchall()
            connected_lakeflags = {
                r[0]: int(r[1]) if r[1] is not None else 0 for r in lf_rows
            }
    tc = {0: "#ffffff", 1: "#00ffff", 2: "#ffff00", 3: "#ff00ff"}
    tn_map = {0: "River", 1: "Lake", 2: "Canal", 3: "Tidal"}
    for ug, i, rid in up_geoms:
        coords = [[c[1], c[0]] for c in ug]
        opacity = max(0.4, 1.0 - (i / max(hops, 1)) * 0.6)
        if color_by_type:
            lf = connected_lakeflags.get(rid, 0)
            folium.PolyLine(
                coords,
                color=tc.get(lf, "#ffffff"),
                weight=4,
                opacity=opacity,
                tooltip=f"Up {i + 1}: {rid} ({tn_map.get(lf, '?')})",
            ).add_to(m)
        else:
            folium.PolyLine(
                coords,
                color="orange",
                weight=4,
                opacity=opacity,
                tooltip=f"Upstream {i + 1}: {rid}",
            ).add_to(m)
        add_flow_arrows(
            m,
            coords,
            tc.get(connected_lakeflags.get(rid, 0), "orange")
            if color_by_type
            else "orange",
            num_arrows=2,
        )
    for dg, i, rid in dn_geoms:
        coords = [[c[1], c[0]] for c in dg]
        opacity = max(0.4, 1.0 - (i / max(hops, 1)) * 0.6)
        if color_by_type:
            lf = connected_lakeflags.get(rid, 0)
            folium.PolyLine(
                coords,
                color=tc.get(lf, "#ffffff"),
                weight=4,
                opacity=opacity,
                tooltip=f"Down {i + 1}: {rid} ({tn_map.get(lf, '?')})",
            ).add_to(m)
        else:
            folium.PolyLine(
                coords,
                color="#0066ff",
                weight=4,
                opacity=opacity,
                tooltip=f"Downstream {i + 1}: {rid}",
            ).add_to(m)
        add_flow_arrows(
            m,
            coords,
            tc.get(connected_lakeflags.get(rid, 0), "#0066ff")
            if color_by_type
            else "#0066ff",
            num_arrows=2,
        )
    main_coords = [[c[1], c[0]] for c in geom]
    folium.PolyLine(main_coords, color="black", weight=12, opacity=1.0).add_to(m)
    folium.PolyLine(
        main_coords,
        color="#FFFF00",
        weight=8,
        opacity=1.0,
        tooltip=f"SELECTED: {reach_id}",
    ).add_to(m)
    add_flow_arrows(m, main_coords, "#FFFF00", num_arrows=2)
    if len(main_coords) > 0:
        center = main_coords[len(main_coords) // 2]
        folium.CircleMarker(
            center,
            radius=15,
            color="black",
            fill=True,
            fill_color="#FFFF00",
            fill_opacity=1.0,
            weight=3,
            tooltip=f"SELECTED: {reach_id}",
        ).add_to(m)
    padding = view_radius * 0.2
    m.fit_bounds(
        [
            [min(lats) - padding, min(lons) - padding],
            [max(lats) + padding, max(lons) + padding],
        ]
    )
    st.session_state.last_network_ids = list(connected_ids)
    st.session_state.nearby_reach_ids = (
        [int(rid) for _, rid, _ in nearby_unconnected] if show_all else []
    )
    map_data = st_folium(
        m, width=None, height=500, returned_objects=["last_object_clicked_tooltip"]
    )
    if map_data and map_data.get("last_object_clicked_tooltip"):
        tooltip = map_data["last_object_clicked_tooltip"]
        match = re.search(
            r"(?:River|Lake|Canal|Tidal|Unconnected|SELECTED):\s*(\d+)", tooltip
        )
        if match:
            clicked_id = int(match.group(1))
            if clicked_id in st.session_state.get("nearby_reach_ids", []):
                st.session_state.clicked_reach = clicked_id
    if show_all and nearby_unconnected:
        type_counts = {}
        for _, rid, lf in nearby_unconnected:
            type_counts[lf] = type_counts.get(lf, 0) + 1
        legend = f"SELECTED | Up ({len(up_geoms)}) | Down ({len(dn_geoms)})"
        legend += f" | River ({type_counts.get(0, 0)}) | Lake ({type_counts.get(1, 0)})"
        legend += (
            f" | Canal ({type_counts.get(2, 0)}) | Tidal ({type_counts.get(3, 0)})"
        )
        st.caption(legend)
    else:
        st.caption(
            f"SELECTED | Upstream ({len(up_geoms)}) | Downstream ({len(dn_geoms)})"
        )


# =============================================================================
# MAIN APP
# =============================================================================
st.title("SWORD Lake QA Reviewer")
st.session_state.map_hops = 30
st.session_state.map_radius = 4.0
st.session_state.max_reaches = 15000
st.session_state.show_all_reaches = True

st.sidebar.header("Settings")
region = st.sidebar.selectbox("Region", ["NA", "SA", "EU", "AF", "AS", "OC"], index=0)
st.sidebar.subheader("Saved to Database")
try:
    saved_summary = conn.execute(
        """
        SELECT check_id, COUNT(DISTINCT reach_id) as cnt FROM lint_fix_log
        WHERE region = ? AND NOT undone AND check_id IN ('C001', 'C004') GROUP BY check_id
    """,
        [region],
    ).fetchdf()
    if len(saved_summary) == 0:
        st.sidebar.info("No saved reviews yet")
    else:
        counts = dict(zip(saved_summary["check_id"], saved_summary["cnt"]))
        c001 = int(counts.get("C001", 0))
        c004 = int(counts.get("C004", 0))
        st.sidebar.success(f"**{c001 + c004} reviews**")
        if c001:
            st.sidebar.write(f"  Lake Sandwich: {c001}")
        if c004:
            st.sidebar.write(f"  Type Mismatch: {c004}")
except Exception as e:
    st.sidebar.warning(f"Could not load summary: {e}")

tab_c004, tab_c001, tab_history = st.tabs(
    ["Lakeflag/Type", "Lake Sandwich", "Fix History"]
)

with tab_c004:
    st.header("Lakeflag/Type Mismatch")
    if "c004_issues" not in st.session_state:
        st.session_state.c004_issues = None
    if "c004_pending" not in st.session_state:
        st.session_state.c004_pending = []
    reviewed_c004 = (
        conn.execute(
            """
        SELECT DISTINCT reach_id FROM lint_fix_log
        WHERE region = ? AND check_id = 'C004' AND NOT undone
    """,
            [region],
        )
        .fetchdf()["reach_id"]
        .tolist()
    )
    all_reviewed_c004 = set(reviewed_c004) | set(st.session_state.c004_pending)
    if (
        st.session_state.c004_issues is None
        or st.session_state.get("c004_region") != region
    ):
        with st.spinner("Finding lakeflag/type mismatches..."):
            try:
                c004_df = conn.execute(
                    """
                    SELECT reach_id, region, river_name, x, y, lakeflag, type,
                        CASE
                            WHEN lakeflag = 1 AND type = 1 THEN 'lake_labeled_as_river_type'
                            WHEN lakeflag = 0 AND type = 3 THEN 'river_labeled_as_lake_type'
                            WHEN lakeflag = 2 AND type NOT IN (1, 4, 5, 6) THEN 'canal_type_mismatch'
                            WHEN lakeflag = 3 AND type NOT IN (5, 6) THEN 'tidal_type_mismatch'
                            ELSE 'other_mismatch'
                        END as issue_type
                    FROM reaches
                    WHERE region = ? AND lakeflag IS NOT NULL AND type IS NOT NULL
                        AND NOT (
                            (lakeflag = 0 AND type IN (1, 4))
                            OR (lakeflag = 1 AND type IN (3, 4))
                            OR (lakeflag = 2 AND type IN (1, 4))
                            OR type IN (5, 6)
                        )
                    ORDER BY reach_id
                """,
                    [region],
                ).fetchdf()
                st.session_state.c004_issues = c004_df
            except Exception:
                st.session_state.c004_issues = pd.DataFrame()
            st.session_state.c004_region = region
    issues = st.session_state.c004_issues
    if issues is None or len(issues) == 0:
        st.success("No lakeflag/type mismatches in this region!")
    else:
        total = len(issues)
        remaining = len(
            [r for r in issues["reach_id"].tolist() if r not in all_reviewed_c004]
        )
        done = len(all_reviewed_c004)
        col1, col2, col3 = st.columns(3)
        col1.metric("Remaining", remaining)
        col2.metric("Reviewed", done)
        col3.metric("Total", total)
        st.progress(done / total if total > 0 else 0)
        if remaining == 0:
            st.success("All mismatches reviewed!")
        else:
            available = [
                r for r in issues["reach_id"].tolist() if r not in all_reviewed_c004
            ]
            selected = available[0]
            issue = issues[issues["reach_id"] == selected].iloc[0]
            lakeflag_map = {0: "River", 1: "Lake", 2: "Canal", 3: "Tidal"}
            type_map = {
                1: "river",
                2: "lake",
                3: "tidal_river",
                4: "artificial",
                5: "unassigned",
                6: "unreliable",
            }
            lf = issue["lakeflag"]
            tp = issue["type"]
            st.markdown("---")
            st.subheader(f"Issue #{done + 1}: Reach `{selected}`")
            st.error(f"""
            **Mismatch detected:**
            - **Lakeflag:** {lf} ({lakeflag_map.get(lf, "?")})
            - **Type:** {tp} ({type_map.get(tp, "?")})
            - **Issue:** {issue["issue_type"]}
            """)
            col1, col2 = st.columns([2, 1])
            with col1:
                render_reach_map_satellite(
                    int(selected), region, conn, color_by_type=True
                )
            with col2:
                st.markdown(f"**River:** {issue['river_name'] or 'Unnamed'}")
                st.markdown("---")
                st.markdown("### Fix the mismatch")
                if "lake" in issue["issue_type"]:
                    if st.button(
                        "Set type=2 (lake)",
                        key=f"c004_fix_{selected}",
                        type="primary",
                        use_container_width=True,
                    ):
                        conn.execute(
                            "UPDATE reaches SET type = 2 WHERE reach_id = ? AND region = ?",
                            [selected, region],
                        )
                        conn.commit()
                        log_skip(conn, selected, region, "C004", "Fixed: type->2")
                        st.session_state.c004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()
                elif "river" in issue["issue_type"]:
                    if st.button(
                        "Set type=1 (river)",
                        key=f"c004_fix_{selected}",
                        type="primary",
                        use_container_width=True,
                    ):
                        conn.execute(
                            "UPDATE reaches SET type = 1 WHERE reach_id = ? AND region = ?",
                            [selected, region],
                        )
                        conn.commit()
                        log_skip(conn, selected, region, "C004", "Fixed: type->1")
                        st.session_state.c004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()
                elif "canal" in issue["issue_type"]:
                    if st.button(
                        "Set type=4 (artificial)",
                        key=f"c004_fix_{selected}",
                        type="primary",
                        use_container_width=True,
                    ):
                        conn.execute(
                            "UPDATE reaches SET type = 4 WHERE reach_id = ? AND region = ?",
                            [selected, region],
                        )
                        conn.commit()
                        log_skip(conn, selected, region, "C004", "Fixed: type->4")
                        st.session_state.c004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()
                elif "tidal" in issue["issue_type"]:
                    if st.button(
                        "Set lakeflag=0 (river)",
                        key=f"c004_fix_lf_{selected}",
                        type="primary",
                        use_container_width=True,
                    ):
                        conn.execute(
                            "UPDATE reaches SET lakeflag = 0 WHERE reach_id = ? AND region = ?",
                            [selected, region],
                        )
                        conn.commit()
                        log_skip(
                            conn,
                            selected,
                            region,
                            "C004",
                            "Fixed: lakeflag->0 (was tidal)",
                        )
                        st.session_state.c004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()
                    if st.button(
                        "Set type=1 (river)",
                        key=f"c004_fix_tp_{selected}",
                        use_container_width=True,
                    ):
                        conn.execute(
                            "UPDATE reaches SET type = 1 WHERE reach_id = ? AND region = ?",
                            [selected, region],
                        )
                        conn.commit()
                        log_skip(
                            conn, selected, region, "C004", "Fixed: type->1 (was tidal)"
                        )
                        st.session_state.c004_pending.append(selected)
                        st.cache_data.clear()
                        st.rerun()
                if st.button(
                    "Skip (correct as-is)",
                    key=f"c004_skip_{selected}",
                    use_container_width=True,
                ):
                    log_skip(conn, selected, region, "C004", "Skipped: correct as-is")
                    st.session_state.c004_pending.append(selected)
                    st.cache_data.clear()
                    st.rerun()
with tab_c001:
    st.header("Lake Sandwich Fixer")
    if "pending_fixes" not in st.session_state:
        st.session_state.pending_fixes = []
    if "last_fix" not in st.session_state:
        st.session_state.last_fix = None
    if "c001_results" not in st.session_state:
        st.session_state.c001_results = None
    if (
        st.session_state.c001_results is None
        or st.session_state.get("c001_region") != region
    ):
        with st.spinner("Running lake sandwich check..."):
            st.session_state.c001_results = run_c001_check(conn, region)
            st.session_state.c001_region = region
            session = load_session_fixes(region, "C001")
            st.session_state.pending_fixes = session.get("pending", [])
    result = st.session_state.c001_results
    if result:
        issues = result.details
        total = len(issues)
        done = len(st.session_state.pending_fixes)
        remaining = (
            len(
                [
                    r
                    for r in issues["reach_id"].tolist()
                    if r not in st.session_state.pending_fixes
                ]
            )
            if total > 0
            else 0
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Remaining", remaining)
        col2.metric("Reviewed", done)
        col3.metric("Total", total)
        if total > 0:
            st.progress(
                done / total if total > 0 else 0, text=f"{done}/{total} reviewed"
            )
        if remaining == 0 and total > 0:
            st.success("All lake sandwiches reviewed!")
            if st.button("Re-run check to verify", key="rerun_c001"):
                st.session_state.c001_results = None
                st.session_state.pending_fixes = []
                st.rerun()
        elif total == 0:
            st.success("No lake sandwich issues in this region!")
        else:
            available = [
                r
                for r in issues["reach_id"].tolist()
                if r not in st.session_state.pending_fixes
            ]
            selected = available[0]
            issue = issues[issues["reach_id"] == selected].iloc[0]
            up_flags, dn_flags = get_neighbor_lakeflags(conn, selected, region)
            slope = get_reach_slope(conn, selected, region)
            width = issue["width"] if pd.notna(issue["width"]) else 0
            st.markdown("---")
            st.subheader(f"Issue #{done + 1}: Reach `{selected}`")
            up_label = (
                "LAKE" if up_flags and 1 in up_flags else "River" if up_flags else "?"
            )
            dn_label = (
                "LAKE" if dn_flags and 1 in dn_flags else "River" if dn_flags else "?"
            )
            st.markdown(f"""
            ```
            Upstream:   {up_label}
                          |
            THIS REACH: River (lakeflag=0)  <-- Is this actually a lake?
                          |
            Downstream: {dn_label}
            ```
            """)
            col1, col2 = st.columns([2, 1])
            with col1:
                render_reach_map_satellite(int(selected), region, conn)
            with col2:
                st.markdown("### Key Indicators")
                if width > 500:
                    st.success(f"**Width:** {width:.0f}m -> Wide (likely lake)")
                elif width > 200:
                    st.warning(f"**Width:** {width:.0f}m -> Medium")
                elif width > 0:
                    st.error(f"**Width:** {width:.0f}m -> Narrow (likely river)")
                else:
                    st.info("**Width:** N/A")
                if slope is not None:
                    if slope < 0.0001:
                        st.success(f"**Slope:** {slope:.6f} -> Flat (likely lake)")
                    elif slope < 0.001:
                        st.warning(f"**Slope:** {slope:.6f} -> Low slope")
                    else:
                        st.error(f"**Slope:** {slope:.6f} -> Steep (likely river)")
                else:
                    st.info("**Slope:** N/A")
                both_lakes = (up_flags and 1 in up_flags) and (
                    dn_flags and 1 in dn_flags
                )
                if both_lakes:
                    st.success("**Neighbors:** Both lakes -> Likely lake")
                else:
                    st.warning("**Neighbors:** Mixed types")
                st.markdown(f"**River name:** {issue['river_name'] or 'Unnamed'}")
                st.markdown(
                    f"**Length:** {issue['reach_length']:.0f}m"
                    if pd.notna(issue["reach_length"])
                    else ""
                )
            st.markdown("---")
            st.markdown("### Your Decision")
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.markdown("**It's a LAKE** (convert lakeflag to 1)")
                if st.button(
                    "YES, IT'S A LAKE",
                    key=f"fix_{selected}",
                    type="primary",
                    use_container_width=True,
                ):
                    apply_lakeflag_fix(conn, selected, region, 1)
                    st.session_state.pending_fixes.append(selected)
                    st.session_state.last_fix = selected
                    st.cache_data.clear()
                    st.rerun()
            with btn_col2:
                st.markdown("**Keep as RIVER** (no change)")
                skip_reason = st.selectbox(
                    "Why is it a river?",
                    [
                        "Flowing water visible",
                        "Narrow channel",
                        "Dam/weir",
                        "Canal",
                        "Tidal",
                        "Other",
                    ],
                    key=f"skip_reason_{selected}",
                    label_visibility="collapsed",
                )
                if st.button(
                    "NO, IT'S A RIVER", key=f"skip_{selected}", use_container_width=True
                ):
                    log_skip(conn, selected, region, "C001", skip_reason)
                    st.session_state.pending_fixes.append(selected)
                    st.cache_data.clear()
                    st.rerun()
            if st.session_state.last_fix:
                if st.button(
                    f"Undo last ({st.session_state.last_fix})", key="undo_last"
                ):
                    undone_id = undo_last_fix(conn, region)
                    if undone_id:
                        st.session_state.pending_fixes = [
                            p for p in st.session_state.pending_fixes if p != undone_id
                        ]
                        st.session_state.last_fix = None
                        st.cache_data.clear()
                        st.rerun()
    st.markdown("---")
    if st.button("Refresh Check", key="refresh_c001"):
        st.session_state.c001_results = None
        st.session_state.pending_fixes = []
        st.rerun()
with tab_history:
    st.header("Fix History (Lake Reviews)")
    st.caption("Log of all C001 and C004 review actions")
    show_all_regions = st.checkbox("Show all regions", value=False)
    if show_all_regions:
        history = conn.execute("""
            SELECT * FROM lint_fix_log WHERE check_id IN ('C001', 'C004') ORDER BY timestamp DESC LIMIT 500
        """).fetchdf()
    else:
        history = conn.execute(
            """
            SELECT * FROM lint_fix_log WHERE region = ? AND check_id IN ('C001', 'C004') ORDER BY timestamp DESC LIMIT 500
        """,
            [region],
        ).fetchdf()
    col1, col2 = st.columns(2)
    col1.metric("Total Actions", len(history))
    if len(history) > 0:
        action_counts = history["action"].value_counts().to_dict()
        fixes = action_counts.get("fix", 0)
        skips = action_counts.get("skip", 0)
        col2.metric("Fixes / Skips", f"{fixes} / {skips}")
    if len(history) > 0:
        st.subheader("By Check")
        summary = (
            history.groupby(["check_id", "action"])
            .agg(count=("reach_id", "count"))
            .reset_index()
        )
        st.dataframe(summary)
        st.subheader("All Actions")
        st.dataframe(history, height=400)
        st.subheader("Export")
        csv = history.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            f"lake_fixes_{region}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
        )
    else:
        st.info(
            "No lake review actions logged yet. Make some reviews in the other tabs!"
        )

st.divider()
st.caption(f"SWORD Lake QA Reviewer | Region: {region} | DB: sword_v17c.duckdb")
