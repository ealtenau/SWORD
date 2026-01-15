#!/usr/bin/env python3
"""
Topology Violation Reviewer
===========================
Simple Streamlit UI to review and flag topology violations in SWORD database.

Run with: streamlit run topology_reviewer.py
"""

import streamlit as st
import duckdb
import pandas as pd
import pydeck as pdk
import json

# Page config
st.set_page_config(
    page_title="SWORD Topology Reviewer",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 SWORD Topology Violation Reviewer")

# Database connection
@st.cache_resource
def get_connection():
    conn = duckdb.connect('data/duckdb/sword_v17b.duckdb')
    # Load spatial extension for RTREE index support
    try:
        conn.execute("INSTALL spatial")
        conn.execute("LOAD spatial")
    except:
        pass
    return conn

conn = get_connection()

# Sidebar filters
st.sidebar.header("Filters")

min_ratio = st.sidebar.slider(
    "Minimum facc ratio",
    min_value=2,
    max_value=1000,
    value=10,
    help="Show violations where upstream/downstream facc ratio exceeds this"
)

show_only_unflagged = st.sidebar.checkbox(
    "Show only unflagged violations",
    value=False
)

hide_approved = st.sidebar.checkbox(
    "Hide approved (valid) violations",
    value=True,
    help="Hide violations marked as valid distributaries"
)

region = st.sidebar.selectbox(
    "Region",
    ["NA", "SA", "EU", "AF", "AS", "OC"],
    index=0
)

# Get violations
@st.cache_data(ttl=60)
def get_violations(_conn, region, min_ratio, show_unflagged, hide_approved):
    flag_filter = "AND t.topology_suspect = FALSE" if show_unflagged else ""
    approved_filter = "AND (t.topology_approved = FALSE OR t.topology_approved IS NULL)" if hide_approved else ""

    query = f"""
        WITH reach_facc AS (
            SELECT reach_id, AVG(facc) as avg_facc
            FROM nodes WHERE facc IS NOT NULL AND region = ?
            GROUP BY reach_id
        ),
        reach_coords AS (
            SELECT reach_id, AVG(x) as lon, AVG(y) as lat
            FROM nodes WHERE region = ?
            GROUP BY reach_id
        )
        SELECT
            t.reach_id as upstream_reach,
            t.neighbor_reach_id as downstream_reach,
            t.topology_suspect as is_flagged,
            COALESCE(t.topology_approved, FALSE) as is_approved,
            r1.river_name as upstream_name,
            r2.river_name as downstream_name,
            rf1.avg_facc as upstream_facc,
            rf2.avg_facc as downstream_facc,
            rf1.avg_facc / rf2.avg_facc as ratio,
            c1.lon as up_lon, c1.lat as up_lat,
            c2.lon as dn_lon, c2.lat as dn_lat
        FROM reach_topology t
        JOIN reach_facc rf1 ON t.reach_id = rf1.reach_id
        JOIN reach_facc rf2 ON t.neighbor_reach_id = rf2.reach_id
        JOIN reaches r1 ON t.reach_id = r1.reach_id
        JOIN reaches r2 ON t.neighbor_reach_id = r2.reach_id
        JOIN reach_coords c1 ON t.reach_id = c1.reach_id
        JOIN reach_coords c2 ON t.neighbor_reach_id = c2.reach_id
        WHERE t.direction = 'down'
          AND t.region = ?
          AND rf2.avg_facc > 0
          AND rf1.avg_facc / rf2.avg_facc >= ?
          {flag_filter}
          {approved_filter}
        ORDER BY rf1.avg_facc / rf2.avg_facc DESC
        LIMIT 500
    """
    return _conn.execute(query, [region, region, region, min_ratio]).fetchdf()

# Get reach geometry (nodes as line)
@st.cache_data(ttl=300)
def get_reach_geometry(_conn, reach_id):
    """Get nodes for a reach ordered to form a line."""
    nodes = _conn.execute("""
        SELECT x, y
        FROM nodes
        WHERE reach_id = ?
        ORDER BY dist_out DESC
    """, [reach_id]).fetchdf()

    if len(nodes) == 0:
        return None

    # Return as list of [lon, lat] coordinates
    return nodes[['x', 'y']].values.tolist()

# Get nearby reaches for context (with facc)
@st.cache_data(ttl=300)
def get_nearby_reaches(_conn, center_lon, center_lat, region, radius_deg=0.5, limit=100):
    """Get reaches within radius of center point, including facc values."""
    reaches = _conn.execute("""
        WITH reach_stats AS (
            SELECT reach_id, AVG(x) as cx, AVG(y) as cy, AVG(facc) as avg_facc
            FROM nodes
            WHERE region = ?
            GROUP BY reach_id
        )
        SELECT rs.reach_id, rs.avg_facc, r.river_name
        FROM reach_stats rs
        LEFT JOIN reaches r ON rs.reach_id = r.reach_id
        WHERE ABS(rs.cx - ?) < ? AND ABS(rs.cy - ?) < ?
        LIMIT ?
    """, [region, center_lon, radius_deg, center_lat, radius_deg, limit]).fetchdf()

    # Return as dict with reach_id -> {facc, name}
    return {
        row['reach_id']: {
            'facc': row['avg_facc'],
            'name': row['river_name'] or 'Unknown'
        }
        for _, row in reaches.iterrows()
    }

@st.cache_data(ttl=300)
def get_multiple_reach_geometries(_conn, reach_ids):
    """Get geometries for multiple reaches."""
    if not reach_ids:
        return {}

    geometries = {}
    for rid in reach_ids:
        nodes = _conn.execute("""
            SELECT x, y
            FROM nodes
            WHERE reach_id = ?
            ORDER BY dist_out DESC
        """, [rid]).fetchdf()

        if len(nodes) > 1:
            geometries[rid] = nodes[['x', 'y']].values.tolist()

    return geometries

violations = get_violations(conn, region, min_ratio, show_only_unflagged, hide_approved)

# Summary stats
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Violations", len(violations))
with col2:
    flagged = violations['is_flagged'].sum() if len(violations) > 0 else 0
    st.metric("Flagged (problem)", flagged)
with col3:
    approved = violations['is_approved'].sum() if len(violations) > 0 else 0
    st.metric("Approved (valid)", approved)
with col4:
    extreme = len(violations[violations['ratio'] > 100]) if len(violations) > 0 else 0
    st.metric("Extreme (>100x)", extreme)
with col5:
    avg_ratio = violations['ratio'].mean() if len(violations) > 0 else 0
    st.metric("Avg Ratio", f"{avg_ratio:.0f}x")

st.divider()

if len(violations) == 0:
    st.info("No violations found with current filters.")
else:
    # Selection first
    st.subheader("Select Violation to Review")

    selected_idx = st.selectbox(
        "Choose a violation",
        range(len(violations)),
        format_func=lambda i: f"#{i+1}: {violations.iloc[i]['upstream_name']} → {violations.iloc[i]['downstream_name']} ({violations.iloc[i]['ratio']:.0f}x)"
    )

    v = violations.iloc[selected_idx]

    # Get geometries for selected reaches
    up_geom = get_reach_geometry(conn, int(v['upstream_reach']))
    dn_geom = get_reach_geometry(conn, int(v['downstream_reach']))

    # Two columns: map and details
    map_col, detail_col = st.columns([2, 1])

    with detail_col:
        st.subheader("📋 Details")

        # Upstream reach info
        st.markdown("### 🔴 Upstream Reach")
        st.markdown(f"**Reach ID:** `{v['upstream_reach']}`")
        st.markdown(f"**River:** {v['upstream_name']}")
        st.markdown(f"**facc:** {v['upstream_facc']:,.0f}")
        if up_geom:
            st.markdown(f"**Nodes:** {len(up_geom)}")

        st.markdown("### ⬇️")

        # Downstream reach info
        st.markdown("### 🔵 Downstream Reach")
        st.markdown(f"**Reach ID:** `{v['downstream_reach']}`")
        st.markdown(f"**River:** {v['downstream_name']}")
        st.markdown(f"**facc:** {v['downstream_facc']:,.0f}")
        if dn_geom:
            st.markdown(f"**Nodes:** {len(dn_geom)}")

        st.markdown(f"### Ratio: {v['ratio']:,.0f}x")

        st.divider()

        # Approval status (valid distributary)
        if v['is_approved']:
            st.success("✅ Approved as valid (e.g., distributary)")
            if st.button("❌ Remove approval", key="unapprove"):
                conn.execute("""
                    UPDATE reach_topology
                    SET topology_approved = FALSE
                    WHERE reach_id = ? AND neighbor_reach_id = ? AND direction = 'down' AND region = ?
                """, [int(v['upstream_reach']), int(v['downstream_reach']), region])
                st.success("Approval removed!")
                st.cache_data.clear()
                st.rerun()
        else:
            if st.button("✅ Approve as valid (distributary OK)", key="approve"):
                conn.execute("""
                    UPDATE reach_topology
                    SET topology_approved = TRUE
                    WHERE reach_id = ? AND neighbor_reach_id = ? AND direction = 'down' AND region = ?
                """, [int(v['upstream_reach']), int(v['downstream_reach']), region])
                st.success("Approved as valid!")
                st.cache_data.clear()
                st.rerun()

        st.divider()

        # Flag status and actions
        if v['is_flagged']:
            st.warning("⚠️ Flagged as suspect (problem)")
            if st.button("🔄 Unflag", key="unflag"):
                conn.execute("""
                    UPDATE reach_topology
                    SET topology_suspect = FALSE
                    WHERE reach_id = ? AND neighbor_reach_id = ? AND direction = 'down' AND region = ?
                """, [int(v['upstream_reach']), int(v['downstream_reach']), region])
                st.success("Unflagged!")
                st.cache_data.clear()
                st.rerun()
        else:
            if st.button("🚩 Flag as suspect (problem)", key="flag"):
                conn.execute("""
                    UPDATE reach_topology
                    SET topology_suspect = TRUE
                    WHERE reach_id = ? AND neighbor_reach_id = ? AND direction = 'down' AND region = ?
                """, [int(v['upstream_reach']), int(v['downstream_reach']), region])
                st.success("Flagged as problem!")
                st.cache_data.clear()
                st.rerun()

        st.divider()

        # FIX: Propagate facc
        st.markdown("### 🔧 Fix facc")

        # Option 1: Propagate downstream (upstream + downstream → downstream)
        new_dn_facc = v['upstream_facc'] + v['downstream_facc']
        st.markdown(f"**Option 1:** Set downstream = {v['upstream_facc']:,.0f} + {v['downstream_facc']:,.0f} = **{new_dn_facc:,.0f}**")
        if st.button("⬇️ Fix downstream (propagate)", key="fix_downstream"):
            conn.execute("""
                UPDATE nodes SET facc = ?
                WHERE reach_id = ? AND region = ?
            """, [new_dn_facc, int(v['downstream_reach']), region])
            st.success(f"Updated downstream reach {v['downstream_reach']} facc to {new_dn_facc:,.0f}")
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")

        # Option 2: Set upstream to match downstream (for erroneous upstream)
        st.markdown(f"**Option 2:** Set upstream = downstream = **{v['downstream_facc']:,.0f}**")
        if st.button("⬆️ Fix upstream (match downstream)", key="fix_upstream"):
            conn.execute("""
                UPDATE nodes SET facc = ?
                WHERE reach_id = ? AND region = ?
            """, [v['downstream_facc'], int(v['upstream_reach']), region])
            st.success(f"Updated upstream reach {v['upstream_reach']} facc to {v['downstream_facc']:,.0f}")
            st.cache_data.clear()
            st.rerun()

        # Navigation
        st.divider()
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if selected_idx > 0:
                if st.button("⬅️ Previous"):
                    st.session_state['selected'] = selected_idx - 1
                    st.rerun()
        with nav_col2:
            if selected_idx < len(violations) - 1:
                if st.button("Next ➡️"):
                    st.session_state['selected'] = selected_idx + 1
                    st.rerun()

    with map_col:
        st.subheader("🗺️ Reach Geometry")

        # Sidebar control for context radius
        context_radius = st.slider("Context radius (degrees)", 0.1, 2.0, 0.5, 0.1)

        layers = []
        all_coords = []

        # Get center point for context
        center_lon = (v['up_lon'] + v['dn_lon']) / 2
        center_lat = (v['up_lat'] + v['dn_lat']) / 2

        # Get nearby reaches for context (now returns dict with facc info)
        nearby_data = get_nearby_reaches(conn, center_lon, center_lat, region, context_radius, 200)

        # Exclude the main two reaches from context
        context_ids = [rid for rid in nearby_data.keys() if rid not in [v['upstream_reach'], v['downstream_reach']]]

        # Get geometries for context reaches
        context_geoms = get_multiple_reach_geometries(conn, context_ids)

        # Add context reaches (GRAY) with facc in tooltip
        if context_geoms:
            context_path_data = []
            for rid, geom in context_geoms.items():
                info = nearby_data.get(rid, {'facc': 0, 'name': 'Unknown'})
                facc_val = info['facc'] if info['facc'] else 0
                context_path_data.append({
                    "path": geom,
                    "name": f"{info['name']} (facc={facc_val:,.0f})",
                    "color": [128, 128, 128, 150],  # Gray, semi-transparent
                })
                all_coords.extend(geom)

            layers.append(pdk.Layer(
                "PathLayer",
                data=context_path_data,
                get_path="path",
                get_color="color",
                width_scale=15,
                width_min_pixels=2,
                pickable=True,
            ))

        # Upstream reach (RED - the big one that shouldn't flow into small)
        if up_geom and len(up_geom) > 1:
            up_path_data = [{
                "path": up_geom,
                "name": f"🔴 Upstream: {v['upstream_name']} (facc={v['upstream_facc']:,.0f})",
                "color": [255, 0, 0],  # Red
            }]
            layers.append(pdk.Layer(
                "PathLayer",
                data=up_path_data,
                get_path="path",
                get_color="color",
                width_scale=25,
                width_min_pixels=4,
                pickable=True,
            ))
            all_coords.extend(up_geom)

        # Downstream reach (BLUE - the small one)
        if dn_geom and len(dn_geom) > 1:
            dn_path_data = [{
                "path": dn_geom,
                "name": f"🔵 Downstream: {v['downstream_name']} (facc={v['downstream_facc']:,.0f})",
                "color": [0, 100, 255],  # Blue
            }]
            layers.append(pdk.Layer(
                "PathLayer",
                data=dn_path_data,
                get_path="path",
                get_color="color",
                width_scale=25,
                width_min_pixels=4,
                pickable=True,
            ))
            all_coords.extend(dn_geom)

        # Connection point (where they supposedly connect)
        if up_geom and dn_geom:
            # Downstream end of upstream reach
            up_end = up_geom[-1]
            # Upstream end of downstream reach
            dn_start = dn_geom[0]

            connection_data = pd.DataFrame([
                {"lon": up_end[0], "lat": up_end[1], "label": "Upstream end", "color": [255, 0, 0]},
                {"lon": dn_start[0], "lat": dn_start[1], "label": "Downstream start", "color": [0, 100, 255]},
            ])

            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=connection_data,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius=200,
                pickable=True,
            ))

        if all_coords:
            # Calculate bounds
            lons = [c[0] for c in all_coords]
            lats = [c[1] for c in all_coords]
            center_lon = (min(lons) + max(lons)) / 2
            center_lat = (min(lats) + max(lats)) / 2

            # Calculate zoom based on extent
            lon_range = max(lons) - min(lons)
            lat_range = max(lats) - min(lats)
            extent = max(lon_range, lat_range)

            if extent > 1:
                zoom = 7
            elif extent > 0.5:
                zoom = 8
            elif extent > 0.1:
                zoom = 10
            else:
                zoom = 12

            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=zoom,
                pitch=0,
            )

            st.pydeck_chart(pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                tooltip={"text": "{name}"}
            ))

            st.caption("🔴 Upstream reach (large facc) | 🔵 Downstream reach (small facc)")
        else:
            st.warning("No geometry available for selected reaches")

    # Table view
    st.divider()
    with st.expander("📊 All Violations Table"):
        display_df = violations[[
            'upstream_reach', 'upstream_name', 'upstream_facc',
            'downstream_reach', 'downstream_name', 'downstream_facc',
            'ratio', 'is_approved', 'is_flagged'
        ]].copy()
        display_df.columns = [
            'Up Reach', 'Up River', 'Up facc',
            'Dn Reach', 'Dn River', 'Dn facc',
            'Ratio', 'Approved', 'Flagged'
        ]
        display_df['Up facc'] = display_df['Up facc'].apply(lambda x: f"{x:,.0f}")
        display_df['Dn facc'] = display_df['Dn facc'].apply(lambda x: f"{x:,.0f}")
        display_df['Ratio'] = display_df['Ratio'].apply(lambda x: f"{x:,.0f}x")

        st.dataframe(display_df, use_container_width=True, height=400)

    # Bulk actions
    with st.expander("⚡ Bulk Actions"):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚩 Flag ALL displayed violations"):
                for _, row in violations.iterrows():
                    if not row['is_flagged']:
                        conn.execute("""
                            UPDATE reach_topology
                            SET topology_suspect = TRUE
                            WHERE reach_id = ? AND neighbor_reach_id = ? AND direction = 'down' AND region = ?
                        """, [int(row['upstream_reach']), int(row['downstream_reach']), region])
                st.success(f"Flagged {len(violations[~violations['is_flagged']])} violations!")
                st.cache_data.clear()
                st.rerun()

        with col2:
            if st.button("✅ Unflag ALL displayed violations"):
                for _, row in violations.iterrows():
                    if row['is_flagged']:
                        conn.execute("""
                            UPDATE reach_topology
                            SET topology_suspect = FALSE
                            WHERE reach_id = ? AND neighbor_reach_id = ? AND direction = 'down' AND region = ?
                        """, [int(row['upstream_reach']), int(row['downstream_reach']), region])
                st.success(f"Unflagged {len(violations[violations['is_flagged']])} violations!")
                st.cache_data.clear()
                st.rerun()

# Footer
st.divider()
st.caption("SWORD Topology Reviewer | Data from sword_v17b.duckdb")
