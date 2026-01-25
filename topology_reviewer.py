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
"""

import streamlit as st
import duckdb
import pandas as pd
import pydeck as pdk
from datetime import datetime

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

    # Fix options
    notes = st.text_input("Notes (optional)", key=f"notes_{reach_id}")

    # Option 1: Set to upstream value
    if len(up_chain) > 1:
        up_facc = up_chain.iloc[1]['facc'] if len(up_chain) > 1 else 0
        if up_facc > 0 and up_facc < current_facc * 0.5:
            st.markdown(f"**Option 1:** Use upstream facc = **{up_facc:,.0f}**")
            if st.button(f"Set to {up_facc:,.0f}", key=f"fix_up_{reach_id}"):
                old = apply_facc_fix(conn, reach_id, region, up_facc, issue_type, notes)
                st.success(f"Fixed! {old:,.0f} → {up_facc:,.0f}")
                st.cache_data.clear()
                st.rerun()

    # Option 2: Set to downstream value
    if len(dn_chain) > 1:
        dn_facc = dn_chain.iloc[1]['facc'] if len(dn_chain) > 1 else 0
        if dn_facc > 0:
            st.markdown(f"**Option 2:** Match downstream facc = **{dn_facc:,.0f}**")
            if st.button(f"Set to {dn_facc:,.0f}", key=f"fix_dn_{reach_id}"):
                old = apply_facc_fix(conn, reach_id, region, dn_facc, issue_type, notes)
                st.success(f"Fixed! {old:,.0f} → {dn_facc:,.0f}")
                st.cache_data.clear()
                st.rerun()

    # Option 3: Custom value
    custom_facc = st.number_input("Custom facc value", min_value=0.0, value=float(current_facc), key=f"custom_{reach_id}")
    if st.button("Apply custom value", key=f"fix_custom_{reach_id}"):
        old = apply_facc_fix(conn, reach_id, region, custom_facc, f"{issue_type}_custom", notes)
        st.success(f"Fixed! {old:,.0f} → {custom_facc:,.0f}")
        st.cache_data.clear()
        st.rerun()

    # Option 4: Flag as unfixable
    if st.button("🚩 Flag as unfixable", key=f"flag_{reach_id}"):
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Ratio Violations",
    "📉 Monotonicity",
    "🏔️ Headwaters",
    "⚠️ Suspect",
    "📜 Fix History"
])

# =============================================================================
# TAB 1: Ratio Violations (original functionality)
# =============================================================================
with tab1:
    st.header("Topology Ratio Violations")
    st.caption("Reaches where upstream facc >> downstream facc")

    min_ratio = st.slider("Minimum ratio", 2, 100, 10, key="ratio_slider")
    violations = get_ratio_violations(conn, region, min_ratio)

    st.metric("Total", len(violations))

    if len(violations) > 0:
        selected_idx = st.selectbox(
            "Select violation",
            range(len(violations)),
            format_func=lambda i: f"#{i+1}: {violations.iloc[i]['upstream_name']} ({violations.iloc[i]['ratio']:.0f}x)",
            key="ratio_select"
        )

        v = violations.iloc[selected_idx]

        col1, col2 = st.columns([2, 1])

        with col1:
            render_reach_map(int(v['upstream_reach']), region, "Upstream Reach")

        with col2:
            st.markdown(f"**Upstream:** `{v['upstream_reach']}`")
            st.markdown(f"facc: **{v['upstream_facc']:,.0f}** | width: {v['up_width']:.0f}m")
            st.markdown(f"**Downstream:** `{v['downstream_reach']}`")
            st.markdown(f"facc: **{v['downstream_facc']:,.0f}** | width: {v['dn_width']:.0f}m")
            st.markdown(f"**Ratio: {v['ratio']:.0f}x**")

            st.divider()
            render_fix_panel(int(v['upstream_reach']), region, v['upstream_facc'], "ratio_violation")

# =============================================================================
# TAB 2: Monotonicity Violations
# =============================================================================
with tab2:
    st.header("Monotonicity Violations")
    st.caption("Reaches where facc > downstream facc (should decrease downstream)")

    mono_issues = get_monotonicity_issues(conn, region)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", len(mono_issues))
    col2.metric("Severe (>100k diff)", len(mono_issues[mono_issues['diff'] > 100000]))
    col3.metric("Minor (<1k diff)", len(mono_issues[mono_issues['diff'] < 1000]))

    if len(mono_issues) > 0:
        selected_idx = st.selectbox(
            "Select issue",
            range(len(mono_issues)),
            format_func=lambda i: f"#{i+1}: {mono_issues.iloc[i]['reach_id']} (diff={mono_issues.iloc[i]['diff']:,.0f})",
            key="mono_select"
        )

        m = mono_issues.iloc[selected_idx]

        col1, col2 = st.columns([2, 1])

        with col1:
            render_reach_map(int(m['reach_id']), region)

        with col2:
            st.markdown(f"**Reach:** `{m['reach_id']}`")
            st.markdown(f"**facc:** {m['facc']:,.0f} km²")
            st.markdown(f"**width:** {m['width']:.0f}m")
            st.markdown(f"**River:** {m['river_name']}")
            st.divider()
            st.markdown(f"**Downstream:** `{m['dn_reach_id']}`")
            st.markdown(f"**dn_facc:** {m['dn_facc']:,.0f} km²")
            st.markdown(f"**Difference:** {m['diff']:,.0f} km²")

            st.divider()
            render_fix_panel(int(m['reach_id']), region, m['facc'], "monotonicity")

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

# Footer
st.divider()
st.caption(f"SWORD Reviewer | Region: {region} | DB: sword_v17c.duckdb")
