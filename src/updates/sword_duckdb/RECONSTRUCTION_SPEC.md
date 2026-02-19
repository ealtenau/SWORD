# SWORD Reconstruction Specification

This document provides comprehensive documentation of all algorithms used to compute SWORD attributes, derived from analysis of the original construction code. This serves as the specification for the reconstruction system.

## Table of Contents
1. [Centerline Processing](#1-centerline-processing)
2. [Node Creation](#2-node-creation)
3. [Reach Definition](#3-reach-definition)
4. [Topology Operations](#4-topology-operations)
5. [Attribute Computation](#5-attribute-computation)
6. [Post-Processing Operations](#6-post-processing-operations)
7. [External Data Integration](#7-external-data-integration)

---

## 1. Centerline Processing

### 1.1 GRWL Smoothing
**Source:** `GRWL_Updates_v04.py:smooth_grwl()` (lines 146-192)

**Purpose:** Remove "stair-step" effects from 30m Landsat-derived centerlines.

**Algorithm:**
```python
# For each segment with >10 points:
# 1. Perform spatial query to find 5 closest points to each point
pts = np.vstack((grwl.x[seg], grwl.y[seg])).T
kdt = sp.cKDTree(pts)
pt_dist, pt_ind = kdt.query(pts, k=5)

# 2. Average the 5 closest points (moving window smoothing)
new_vals_x = np.mean(grwl.x[seg][pt_ind[:]], axis=1)
new_vals_y = np.mean(grwl.y[seg][pt_ind[:]], axis=1)

# 3. Preserve endpoints (don't smooth)
keep = [0, 1, len(seg)-2, len(seg)-1]
new_vals_x[keep] = original_x[keep]
new_vals_y[keep] = original_y[keep]
```

### 1.2 Segment ID Correction
**Source:** `GRWL_Updates_v04.py:update_segID()` (lines 532-584)

**Purpose:** Fix non-unique segment IDs in original GRWL.

**Algorithm:**
```python
# For each segment:
# 1. Calculate distance between consecutive points
dist = np.sqrt((x[seg][0]-x[seg])**2 + (y[seg][0]-y[seg])**2)
dist_diff = np.diff(dist)

# 2. If distance > 100m between points, split segment
seg_divs = list(np.where(abs(dist_diff) > 100)[0]+1)

# 3. Renumber split segments with unique IDs
```

### 1.3 Lake Flag Remapping
**Source:** `GRWL_Updates_v04.py:read_grwl()` (lines 260-269)

**Mapping:**
| Original Value | New Value | Meaning |
|----------------|-----------|---------|
| 255, 250 | 0 | River |
| 180, 181, 163 | 1 | Lake/Reservoir |
| 126, 125 | 3 | Tidally influenced |
| 86 | 2 | Canal |

---

## 2. Node Creation

### 2.1 Node Spacing (~200m intervals)
**Source:** `Reach_Definition_Tools_v11.py:node_reaches()` (lines 4318-4430)

**Algorithm:**
```python
# 1. For each reach, calculate number of divisions
node_len = 200  # meters (parameter)
divs = np.round(reach_length / node_len)

# 2. Calculate division spacing
divs_dist = reach_length / divs

# 3. Find break points at regular intervals along flow distance
for d in range(int(divs)):
    break_dist = d * divs_dist
    # Find centerline point closest to break_dist
```

### 2.2 Node Ordering by Flow Accumulation
**Source:** `Reach_Definition_Tools_v11.py:node_reaches()`

**Algorithm:**
```python
# 1. Assign temporary node IDs (1, 2, 3...)
# 2. Determine flow direction using median facc at endpoints
first_node_facc = np.median(facc[first_node_points])
last_node_facc = np.median(facc[last_node_points])

# 3. If first < last (upstream to downstream): reverse numbering
if first_node_facc < last_node_facc:
    node_ids = node_ids[::-1]

# Node numbers increase going UPSTREAM within a reach
```

### 2.3 Node ID Format
**Format:** `CBBBBBRRRRNNNT` (11-digit ID)
- C = Continent code (1 digit)
- B = Pfafstetter basin code (6 digits)
- R = Reach number (4 digits)
- N = Node number (3 digits)
- T = Type (1 digit)

---

## 3. Reach Definition

### 3.1 Reach Cutting (Max 10km)
**Source:** `Reach_Definition_Tools_v11.py:cut_reaches()` (lines 876-961)

**Algorithm:**
```python
MAX_REACH_LENGTH = 10000  # meters

# For reaches exceeding max distance:
divs = np.around(reach_length / MAX_REACH_LENGTH)
divs_dist = reach_length / divs

# Find break points at regular distance intervals
for d in range(int(divs)):
    break_dist = d * divs_dist
    # Split reach at break_dist
    # Renumber segments sequentially
```

### 3.2 Reach Topology Ordering
**Source:** `Reach_Definition_Tools_v11.py:reach_topology()` (lines 4217-4314)

**Algorithm:**
```python
# 1. Calculate reach metrics:
reach_facc = np.max(facc[reach_points])  # Maximum flow accumulation
reach_wse = np.median(elv[reach_points])  # Median elevation

# 2. Separate rivers from deltas:
rivers = type_flag < 5
deltas = type_flag == 5

# 3. Order reaches within basin by connectivity
# Reaches numbered 1,2,3... following flow direction
# Deltas get separate numbering
```

### 3.3 Reach ID Format
**Format:** `CBBBBBRRRRRT` (10-digit ID)
- C = Continent code
- B = Pfafstetter basin code (6 digits)
- R = Reach number (4 digits)
- T = Type (1 digit)

---

## 4. Topology Operations

### 4.1 Neighbor Detection
**Source:** `Reach_Definition_Tools_v11.py:find_neighbors()` (lines 965-1079)

**Algorithm:**
```python
# Variable search radii based on reach length:
if reach_length < 100:
    radius = 100  # meters
    k = 4
elif 100 <= reach_length < 200:
    radius = 100
    k = 10
else:
    radius = 200
    k = 10

# cKDTree spatial query
kdt = sp.cKDTree(endpoint_coords)
distances, indices = kdt.query(query_points, k=k, distance_upper_bound=radius)

# Identify upstream/downstream based on flow accumulation
```

### 4.2 Ghost Reaches (Headwater/Outlet Markers)
**Source:** `Reach_Definition_Tools_v11.py:ghost_reaches()` (lines 5845-5951)

**Algorithm:**
```python
# 1. Identify ghost nodes:
# Spatial query: find 10 nearest neighbors within 500m
kdt = sp.cKDTree(all_points)
distances, indices = kdt.query(endpoints, k=10, distance_upper_bound=500)

# 2. Threshold: 2nd nearest neighbor >= 60m indicates isolated endpoint
isolated = distances[:, 1] >= 60

# 3. Filter: if 4th nearest < 100m, remove from ghost list
# (too many neighbors = not really isolated)

# 4. Ghost reaches get type='6' (vs type='1' for rivers)
ghost_type = 6
```

### 4.3 Tributary Junction Detection
**Source:** `GRWL_Updates_v04.py:find_tributary_junctions()` (lines 717-795)

**Algorithm:**
```python
# For each edited segment endpoint:
# 1. Spatial query for nearest GRWL points
if len(seg) < 3:
    distance_upper_bound = 45.0
    k = 4
elif 3 <= len(seg) <= 6:
    distance_upper_bound = 100.0
    k = 10
else:
    distance_upper_bound = 200.0
    k = 10

kdt = sp.cKDTree(grwl_pts)
pt_dist, pt_ind = kdt.query(endpoint_pts, k=k, distance_upper_bound=distance_upper_bound)

# 2. If endpoint falls in MIDDLE of a neighboring segment, mark as tributary
if min_ind > ep_min + 5 and max_ind < ep_max - 5:
    tribs[index] = 1  # tributary junction
```

---

## 5. Attribute Computation

### 5.1 Water Surface Elevation (WSE)
**Source:** `Reach_Definition_Tools_v11.py:basin_node_attributes()`

```python
# Node level: median of centerline elevations
node_wse = np.median(elevation[node_centerline_points])

# Reach level: median of node elevations
reach_wse = np.median(node_wse[reach_nodes])
```

### 5.2 Width
**Source:** `Reach_Definition_Tools_v11.py:basin_node_attributes()`

```python
# Node level: median of centerline widths (from GRWL)
node_width = np.median(grwl_width[node_centerline_points])
node_width_var = np.var(grwl_width[node_centerline_points])

# Reach level: median of node widths
reach_width = np.median(node_width[reach_nodes])
reach_width_var = np.var(node_width[reach_nodes])
```

### 5.3 Flow Accumulation (FACC)
**Source:** `Reach_Definition_Tools_v11.py`, `Merge_Tools_v06.py:filter_facc()`

```python
# Raw facc from MERIT Hydro, then filtered:

# 1. Calculate median and std per segment
median_facc = np.median(facc[segment])
std_facc = np.std(facc[segment])

# 2. Remove outliers
valid = facc[(facc <= median + std) & (facc >= median - std)]

# 3. Special case for high variability:
if std > median:
    valid = facc[facc >= median - (median/2)]

# 4. Linear interpolation from min to max
interp_facc = np.linspace(valid.min(), valid.max(), len(segment))

# 5. Ensure downstream increase (flip if elevation increases downstream)

# Node/Reach: maximum facc
node_facc = np.max(facc[node_points])
reach_facc = np.max(facc[reach_points])
```

### 5.4 Slope
**Source:** `Reach_Definition_Tools_v11.py:reach_attributes()`

```python
# Linear regression of elevation vs distance
# slope = elevation_change / distance (units: m/km)

dist_km = node_distances / 1000
elevation = node_wse

# Linear least squares fit
A = np.vstack([dist_km, np.ones(len(dist_km))]).T
slope, intercept = np.linalg.lstsq(A, elevation, rcond=None)[0]

reach_slope = abs(slope)  # m/km
```

### 5.5 Number of Channels (n_chan)
**Source:** `Reach_Definition_Tools_v11.py:basin_node_attributes()`

```python
# n_chan_max: Maximum number of channels
node_n_chan_max = np.max(grwl_nchan[node_points])
reach_n_chan_max = np.max(node_n_chan_max[reach_nodes])

# n_chan_mod: Mode (most frequent) number of channels
from scipy.stats import mode
node_n_chan_mod = mode(grwl_nchan[node_points])[0][0]
reach_n_chan_mod = mode(node_n_chan_mod[reach_nodes])[0][0]
```

### 5.6 Lake Flag
**Source:** `Merge_Tools_v06.py:format_data()`

```python
# Mode of lakeflag values (most frequent)
from scipy.stats import mode
node_lakeflag = mode(grwl_lakeflag[node_points])[0][0]
reach_lakeflag = mode(node_lakeflag[reach_nodes])[0][0]
```

### 5.7 Distance to Outlet (dist_out)
**Source:** `Reach_Definition_Tools_v11.py` (path accumulation)

```python
# BFS graph traversal from outlets, accumulating reach lengths

# 1. Find all outlets (n_rch_down = 0)
outlets = [r for r in reaches if n_rch_down[r] == 0]

# 2. BFS from each outlet upstream
for outlet in outlets:
    queue = [(outlet, 0)]  # (reach_id, cumulative_distance)

    while queue:
        reach_id, dist = queue.pop(0)
        dist_out[reach_id] = dist

        # Add upstream reaches
        for upstream_id in upstream_neighbors[reach_id]:
            new_dist = dist + reach_length[reach_id]
            queue.append((upstream_id, new_dist))
```

### 5.8 Stream Order
**Source:** `stream_order.py`

```python
# Logarithmic transformation of path frequency
stream_order = round(np.log(path_freq)) + 1
```

### 5.9 Sinuosity
**Source:** `SWORD-Sinuosity/Code/SinuosityMinAreaVarMinReach.m`

**Algorithm:**
```python
# Leopold and Wolman (1960): sinuosity = arc_length / (meander_wavelength / 2)
# Meander wavelength ≈ 10 * river width (Leopold & Wolman 1960)
# Soar and Thorne (2001): wavelength = 10.23 * bankfull_width

lambda_meander = 10 * np.median(width)
evaluation_dist = max(lambda_meander / 2, 150)  # minimum 150m

# 1. Project to UTM and smooth centerline
X = smooth(utm_x)  # 5-point moving average
Y = smooth(utm_y)

# 2. Calculate cumulative distance along centerline
D = np.cumsum(np.sqrt(np.diff(X)**2 + np.diff(Y)**2))

# 3. Find inflection points (changes in curvature direction)
# Cross product of consecutive vectors
Dx = np.diff(X)
Dy = np.diff(Y)
cross_product = Dx[:-1] * Dy[1:] - Dx[1:] * Dy[:-1]
inflection_points = np.where(np.diff(np.sign(cross_product)) != 0)[0]

# 4. For each bend (between inflection points):
for i in range(len(inflection_points) - 1):
    start, end = inflection_points[i], inflection_points[i+1]
    arc_length = D[end] - D[start]
    straight_line = np.sqrt((X[end]-X[start])**2 + (Y[end]-Y[start])**2)
    sinuosity[start:end] = arc_length / straight_line
    wavelength[start:end] = straight_line * 2

# 5. Aggregate to node level
node_sinuosity = np.nanmean(sinuosity[node_points])
node_wavelength = np.nanmean(wavelength[node_points])
```

### 5.10 Extreme Distance Coefficient (ext_dist_coef)
**Source:** `Identify_Lake_Nodes.py`

**Purpose:** Adjust SWOT pixel search distance near lakes.

**Algorithm:**
```python
# Default coefficient
ext_dist_coef = 20

# For nodes near lakes:
# 1. Find mask pixels intersecting lakes
lake_pixels = gpd.sjoin(lakes, pixels, how="inner", op='intersects')

# 2. Find 10 closest nodes to each lake pixel
kdt = sp.cKDTree(node_coords)
dist, index = kdt.query(lake_pixel_coords, k=10)

# 3. Assign coefficients based on number of channels
# Single channel near lake: coefficient = 1
# Multi-channel near lake: coefficient = 2
ext_dist_coef[single_channel_nodes] = 1
ext_dist_coef[multi_channel_nodes] = 2
```

---

## 6. Post-Processing Operations

### 6.1 Tributary Flag (trib_flag)
**Source:** `Add_Trib_Flag.py`

**Algorithm:**
```python
# cKDTree proximity search
# Find nodes within 0.003 degrees (~333m at equator)

kdt = sp.cKDTree(all_node_coords)
distances, indices = kdt.query(query_coords, k=10)

# Node is a tributary junction if:
# 1. Multiple reaches converge within threshold distance
# 2. Upstream reach count > 1

trib_flag = 1  # tributary junction
```

### 6.2 Short Reach Aggregation
**Source:** `GRWL_Updates_v04.py:edit_short_segments()` (lines 839-969)

**Algorithm:**
```python
# For segments with < 100 points:
if len(segment) < 100 and not is_lake:
    # Find neighboring segments
    kdt = sp.cKDTree(all_points)

    # Assign to appropriate neighbor based on:
    # 1. Number of neighboring segments at each endpoint
    # 2. Width similarity (prefer similar width)

    if len(ep1_segs) == 1 and len(ep2_segs) == 1:
        # Both ends have single neighbor - assign to wider one
        if ep1_width > ep2_width:
            new_seg = ep1_segs
        else:
            new_seg = ep2_segs
```

### 6.3 Coastal Flag
**Source:** `Merge_Tools_v06.py:format_data()`

```python
# For each segment:
lake_percentage = len(lakeflag >= 3) / len(segment)
delta_percentage = len(in_delta) / len(segment)

# If either > 25%: mark as coastal
if lake_percentage > 0.25 or delta_percentage > 0.25:
    coastal_flag = 1
```

---

## 7. External Data Integration

### 7.1 Dam/Obstruction Attachment (GROD, GRanD)
**Source:** `Merge_Tools_v06.py:add_dams()`

```python
# GRanD dams: 1000m threshold
# GROD obstructions: 2000m threshold

kdt = sp.cKDTree(obstruction_coords)
distances, indices = kdt.query(node_coords, k=1)

# Assign obstruction if within threshold
for i, (dist, idx) in enumerate(zip(distances, indices)):
    if dist <= threshold:
        obstr_type[i] = obstruction_type[idx]
        grod_id[i] = obstruction_id[idx]

# obstr_type values > 4 are reset to 0 (invalid)
obstr_type[obstr_type > 4] = 0
```

### 7.2 HydroFalls Filtering
**Source:** `hydrofalls_filtering.py`

```python
# Remove HydroFalls points within 500m of GROD obstructions
# (to avoid double-counting)

kdt = sp.cKDTree(hydrofalls_coords)
grod_dist, grod_idx = kdt.query(grod_coords, k=1)

# Remove HydroFalls within 500m of any GROD point
close_pts = np.unique(grod_idx[grod_dist <= 500])
hydrofalls_filtered = np.delete(hydrofalls, close_pts)
```

### 7.3 Basin Code Assignment (HydroBASINS)
**Source:** `Merge_Tools_v06.py:add_basins()`

```python
# Spatial join with HydroBASINS polygons
points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat))
basins_gdf = gpd.read_file(hydrobasins_path)

joined = gpd.sjoin(points_gdf, basins_gdf, how='left', op='within')
basin_code = joined['PFAF_ID']

# Fill missing basin codes with mode from neighbors
missing = np.where(basin_code == 0)[0]
for idx in missing:
    neighbors = kdt.query(coords[idx], k=25)
    basin_code[idx] = mode(basin_code[neighbors])
```

### 7.4 Lake Database Attachment
**Source:** `Merge_Tools_v06.py:add_lakedb()`

```python
# Spatial join with prior lake database
joined = gpd.sjoin(points_gdf, lakes_gdf, how='left', op='within')
lake_id = joined['lake_id']
```

### 7.5 Delta Extent Assignment
**Source:** `Merge_Tools_v06.py:add_deltas()`

```python
# Spatial join with global delta extents
joined = gpd.sjoin(points_gdf, deltas_gdf, how='left', op='within')
in_delta = ~joined['delta_id'].isna()

# Coastal flag if > 25% in delta
```

### 7.6 SWOT Track Assignment
**Source:** `Merge_Tools_v06.py:add_tracks()`

```python
# Spatial intersection with SWOT orbit polygons
joined = gpd.sjoin(points_gdf, swot_tracks_gdf, how='left', op='intersects')

# Count observations per 21-day cycle
swot_obs = joined.groupby('point_id')['orbit_id'].count()
orbit_array = joined['orbit_id'].values
```

---

## Edit Flag Values

| Flag | Meaning |
|------|---------|
| 1 | Reach type change |
| 2 | Node order change |
| 3 | Reach neighbor change |
| 41 | Flow accumulation update |
| 42 | Elevation update |
| 43 | Width update |
| 44 | Slope update |
| 45 | River name update |
| 5 | Reach ID change |
| 6 | Reach boundary change |
| 7 | Reach/node addition |

Multiple updates separated by comma (e.g., "41,2").

---

## Type Flag Values

| Flag | Meaning |
|------|---------|
| 1 | River |
| 2 | Lake/Reservoir |
| 3 | Lake on river |
| 4 | Dam |
| 5 | No topology (deltas) |
| 6 | Ghost reach (headwater/outlet) |

---

## Implementation Status

| Algorithm | Implemented | Notes |
|-----------|-------------|-------|
| WSE (median) | Yes | `_reconstruct_reach_wse` |
| Width (median) | Yes | `_reconstruct_reach_width` |
| Width variance | Yes | `_reconstruct_reach_width_var` |
| WSE variance | Yes | `_reconstruct_reach_wse_var` |
| FACC (max) | Yes | `_reconstruct_reach_facc` |
| Slope (regression) | Yes | `_reconstruct_reach_slope` |
| dist_out (BFS) | Yes | `_reconstruct_reach_dist_out` |
| Stream order | Yes | `_reconstruct_reach_stream_order` |
| n_nodes (count) | Yes | `_reconstruct_reach_n_nodes` |
| Geometry (x,y bounds) | Yes | Multiple methods |
| Lakeflag (mode) | Yes | `_reconstruct_reach_lakeflag` |
| n_chan_max/mod | Yes | `_reconstruct_reach_n_chan_max/mod` |
| Network ID | Yes | `_reconstruct_reach_network` |
| End reach type | Yes | `_reconstruct_reach_end_reach` |
| Path frequency | Yes | `_reconstruct_reach_path_freq` |
| Sinuosity | Pending | Needs MATLAB port |
| ext_dist_coef | Pending | Needs lake polygons |
| Trib flag | Pending | Needs cKDTree implementation |
| FACC filtering | Pending | Needs raw facc values |
