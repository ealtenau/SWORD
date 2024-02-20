from __future__ import division
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
from pyproj import Proj
import time
import geopy.distance
import pandas as pd
import argparse
import re
import os 
import geopandas as gp
from shapely.geometry import Point
import matplotlib.pyplot as plt

fn = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v15/netcdf/na_sword_v15.nc'
# fn = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/Custom_CalVal_Rchs/na_822810_calval_v15.nc'
sword = nc.Dataset(fn, 'r+')

rch = 82281000041
cl_x = sword.groups['centerlines'].variables['x'][:]
cl_y = sword.groups['centerlines'].variables['y'][:]
cl_ids = sword.groups['centerlines'].variables['cl_id'][:]
cl_rch_ids = sword.groups['centerlines'].variables['reach_id'][:]
cl_node_ids = sword.groups['centerlines'].variables['node_id'][:]

vals = np.where(cl_rch_ids[0,:] == rch)[0]
# cl_ids[vals]
# cl_node_ids[0,vals]

# cl_ids[vals[208]]
# cl_node_ids[0,vals[208]]

# cl_ids[vals[209]]
# cl_node_ids[0,vals[209]]


new_vals = np.zeros(len(vals))
new_vals[0:52] = cl_ids[vals[209:261]][::-1]
new_vals[52:261] = cl_ids[vals[0:209]]

cl_ids[vals] = new_vals
sword.groups['centerlines'].variables['cl_id'][vals] = new_vals

sword.close()

plt.scatter(cl_x[vals], cl_y[vals], c=cl_ids[vals])
plt.show()

plt.scatter(cl_x[vals], cl_y[vals], c=new_vals)
plt.show()


region = 'OC'
version = 'v15b'
sword_dir = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'
sword = nc.Dataset(sword_dir+region.lower()+'_sword_'+version+'.nc', 'r+')

node_lon = sword.groups['nodes'].variables['x'][:]
node_lat = sword.groups['nodes'].variables['y'][:]
node_extd = sword.groups['nodes'].variables['ext_dist_coef'][:]

df = pd.DataFrame(np.array([node_lon, node_lat, node_extd]).T)
df.to_csv('/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/Temp/sa_ext_dist_coef.csv')

sword.close()


#####################################
region = 'NA'
fn_sword_v16 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf/'+region.lower()+'_sword_v16.nc'
rch_changes_fn = '/Users/ealteanau/Documents/SWORD_Dev/update_requests/v16/calval_reaches/'+region+'_rch_id_changes.csv'

sword = nc.Dataset(fn_sword_v16)
new_rchs = pd.read_csv(rch_changes_fn)
unq_rch = np.array(new_rchs['new_rch_id'])
old_rch = np.array(new_rchs['old_rch_id'])

all_rchs = sword.groups['reaches'].variables['reach_id'][:] 
check = sword.groups['reaches'].variables['reach_id'][:].shape
for ind in list(range(len(old_rch))):
    rch = np.where(all_rchs == old_rch[ind])[0]
    all_rchs[rch] = unq_rch[ind]

print('DONE', check, len(np.unique(all_rchs)))



region = 'NA'
sword = nc.Dataset('/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf/'+region.lower()+'_sword_v16.nc')
cl_rch = np.array(sword.groups['centerlines'].variables['reach_id'][:])
cl_id = np.array(sword.groups['centerlines'].variables['cl_id'][:])
# nodes = np.array(sword.groups['nodes'].variables['node_id'][:])
# reaches = np.array(sword.groups['reaches'].variables['reach_id'][:])
len(np.unique(cl_id))
cl_id.shape

from collections import Counter
d = Counter(cl_id)
new_list = list([item for item in d if d[item]>1])
print(new_list)

dup = np.where(cl_id == 28294498)[0]
cl_rch[0,dup]

# rch = np.where(cl_rch == 82269100133)[0]
# cl_rch[rch]
# np.where(sword.groups['reaches'].variables['reach_id'][:] == 82269100133)[0]
# sword.groups['reaches'].variables['reach_id'][32045]
# sword.groups['reaches'].variables['reach_length'][32045]




calval_fn = '/Users/ealteanau/Documents/SWORD_Dev/update_requests/v16/type_updates_v16.csv'
rch_changes_fn = '/Users/ealteanau/Documents/SWORD_Dev/update_requests/v16/NA_rch_id_changes.csv'

calval_rchs = pd.read_csv(calval_fn)
new_rchs = pd.read_csv(rch_changes_fn)
old_rch = np.array(new_rchs['old_rch_id'])

all_rchs = np.unique(calval_rchs['reach_id']) 
for ind in list(range(len(old_rch))):
    rch = np.where(all_rchs == old_rch[ind])[0]
    if len(rch) > 0:
        print(old_rch[ind])
print('DONE')



region = 'OC'
fn_sword1 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v15/netcdf/'+region.lower()+'_sword_v15.nc'
fn_sword2 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v15b/netcdf/'+region.lower()+'_sword_v15b.nc'

sword = nc.Dataset(fn_sword1)
sword2 = nc.Dataset(fn_sword2)

var1 = sword.groups['reaches'].variables['reach_id'][:]
var2 = sword2.groups['reaches'].variables['reach_id'][:]
print(len(var1), len(var2))
print(np.unique(var1-var2))


# plt.scatter(sword.groups['reaches'].variables['x'][:], sword.groups['reaches'].variables['y'][:], c=np.log(var1), s=2)
# plt.show()



region = 'AS'
fn = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/shp/AS/'+region.lower()+'_sword_reaches_hb45_v16.shp'
shp = gp.read_file(fn)

coords = shp.geometry.apply(lambda geom: list(geom.coords))
test = coords[3719]

x = np.array([t[0] for t in test])
y = np.array([t[1] for t in test])
plt.plot(x,y)
plt.show()



region = 'NA'
fn = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf/'+region.lower()+'_sword_v16.nc'
sword = nc.Dataset(fn)
reach_id = sword.groups['centerlines'].variables['reach_id'][0,:]
index = sword.groups['centerlines'].variables['cl_id'][:]
x = sword.groups['centerlines'].variables['x'][:]
y = sword.groups['centerlines'].variables['y'][:]

rch1 = np.where(reach_id == 74222900443)[0]
rch2 = np.where(reach_id == 74222900411)[0]
rch3 = np.where(reach_id == 74222900433)[0]

plt.scatter(x[rch1], y[rch1], c=index[rch1])
plt.scatter(x[rch2], y[rch2], c=index[rch2])
plt.scatter(x[rch3], y[rch3], c=index[rch3])
plt.show()


plt.scatter(x[rch1], y[rch1], c='red')
plt.scatter(x[rch2], y[rch2], c='blue')
plt.scatter(x[rch3], y[rch3], c='green')
plt.show()

# converting meters to decimal degree distance.
latitude = np.array([0,10,20,30,40,50,60,70,75,80,85])
meters = 45
for lat in enumerate(latitude):
    print(np.round(meters/(111.32 * 1000 * math.cos(lat[1] * (math.pi / 180))),5))


river_min_dist = 10000
subcls.rch_id2, subcls.rch_len2,\
    subcls.type2 = aggregate_rivers(subcls, river_min_dist)




fn = '/Users/ealteanau/Documents/SWORD_Dev/inputs/GRIT/gis_files/GRIT_hb31.gpkg'
grit = gp.read_file(fn)






plt.scatter(x_coords, y_coords, c='blue')
plt.scatter(ngh_x, ngh_y, c='red')
plt.scatter(ngh_x[0], ngh_y[0], c='gold')
plt.show()


coords_1 = (y_pt, x_pt)

#end 1 would use coords_2, end 2 would use coords_3
ngh_x = cl_x[np.where(reach_id[0,:] == reach_id[0,end1_pt])]
ngh_y = cl_y[np.where(reach_id[0,:] == reach_id[0,end1_pt])]
d=[]
for c in list(range(len(ngh_x))):
    temp_coords = (ngh_y[c], ngh_x[c])
    d.append(geopy.distance.geodesic(coords_2, temp_coords).m)
append_x = ngh_x[np.where(d == np.min(d))]
append_y = ngh_y[np.where(d == np.min(d))]
x_coords = np.insert(x_coords, 0, append_x, axis=0)
y_coords = np.insert(y_coords, 0, append_y, axis=0)

fn='/Users/ealteanau/Documents/SWORD_Dev/swot_data/temp_data/tiles/SWOT_L2_HR_PIXC_501_014_215L_20230425T052849_20230425T052900_PIB0_01.nc'
data=nc.Dataset(fn)
data.geospatial_lon_min
data.geospatial_lon_max
data.geospatial_lat_min
data.geospatial_lat_max
data.close()

region = 'OC'
fn17 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf_beta/'+region.lower()+'_sword_v17.nc'
fn16 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf/'+region.lower()+'_sword_v16.nc'

sword17 = nc.Dataset(fn17)
sword16 = nc.Dataset(fn16)

print(len(sword17.groups['centerlines'].variables['cl_id'][:]), len(np.unique(sword17.groups['centerlines'].variables['cl_id'][:])))
print(len(sword16.groups['centerlines'].variables['cl_id'][:]), len(np.unique(sword16.groups['centerlines'].variables['cl_id'][:])))
print(len(np.unique(sword17.groups['centerlines'].variables['reach_id'][0,:])), len(sword17.groups['reaches'].variables['reach_id'][:]))
print(len(np.unique(sword16.groups['centerlines'].variables['reach_id'][0,:])), len(sword16.groups['reaches'].variables['reach_id'][:]))
print(len(np.unique(sword17.groups['centerlines'].variables['node_id'][0,:])), len(sword17.groups['nodes'].variables['node_id'][:]))
print(len(np.unique(sword16.groups['centerlines'].variables['node_id'][0,:])), len(sword16.groups['nodes'].variables['node_id'][:]))

sword17.close()
sword16.close()



x = [23.45, 24.56, 25.55, 25.55, 45.67, 89.07]
y = [10.23, 11.45, 13.33, 13.33, 22.21, 67.53]

ar = np.array([x,y]).T
all_ind = np.array(list(range(ar.shape[0])))
vals, keep_ind, count = np.unique(ar, return_counts=True, axis = 0, return_index=True)
rmv = list(set(all_ind).difference(keep_ind))




fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/netcdf/na_sword_v17_dup_pts_rmv.nc'


pkl_fn = '/Users/ealtenau/Desktop/lr_icesat2_binned_sword16_subset74.pkl'
import pickle
with open(pkl_fn, 'rb') as f:
    data = pickle.load(f)


pkl_rchs = np.unique(data.reach_id)


shp_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16_original/shp/NA/na_sword_reaches_hb74_v16.shp'
sword = gp.read_file(shp_fn)

sword_rchs = np.unique(sword.reach_id)

len(sword_rchs)
len(pkl_rchs)


import numpy as np
import rasterio
from rasterio.plot import show

r = rasterio.open('/Users/ealtenau/Documents/SWORD_Dev/swot_data/Ganges/v2.0/raster/495_097F_wse_clip.tif')

sig0  = rasterio.open('/Users/ealtenau/Documents/SWORD_Dev/swot_data/Ganges/v2.0/raster/495_097F_sig0_clip.tif').read(1)
wse = rasterio.open('/Users/ealtenau/Documents/SWORD_Dev/swot_data/Ganges/v2.0/raster/495_097F_wse_clip.tif').read(1)
xtrack  = rasterio.open('/Users/ealtenau/Documents/SWORD_Dev/swot_data/Ganges/v2.0/raster/495_097F_xtrack_clip.tif').read(1)


# Create a copy of first raster
wse_masked = wse.copy()
sig0_masked = sig0.copy()
xtrack_masked = xtrack.copy()

# Set a pixel value to 0 as an example, which will signify NoData
# (top right pixel)


# Mask out any NoData (0) values
wse_masked = np.ma.masked_array(wse_masked, mask = (wse_masked == 9.96921e+36))
sig0_masked = np.ma.masked_array(sig0_masked, mask = (sig0_masked == 9.96921e+36))
xtrack_masked = np.ma.masked_array(xtrack_masked, mask = (xtrack_masked == 9.96921e+36))

wse_0 = wse_masked.copy()
wse_0 = ma.masked_where(sig0_masked < 15, wse_0)
wse_0 = ma.masked_where(xtrack_masked <= 10000, wse_0)

show(wse_masked, cmap='terrain')
show(sig0_masked, cmap='plasma')
show(xtrack_masked, cmap='viridis')

show(wse_0, cmap='viridis')


out_meta = r.meta.copy()

out_meta.update({'nodata': 9.96921e+36,
                 'height' : r.shape[0],
                 'width' : r.shape[1],
                 'transform' : r.transform})



outLoc = '/Users/ealtenau/Documents/SWORD_Dev/swot_data/Ganges/v2.0/raster/495_097F_filt_clip.tif'
with rasterio.open(outLoc,'w',**out_meta) as dst:
    dst.write(wse_0,1)

newClip = rasterio.open(outLoc)
clipIM = newClip.read(1,masked=True)
show(clipIM, cmap='viridis')

#######################################################################################################

import numpy as np
import rasterio
from rasterio.plot import show
import netCDF4 as nc

lr = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/swot_data/Ganges/v2.0/LR/SWOT_L2_LR_SSH_Expert_009_495_20240121T191011_20240121T200045_PIC0_01.nc')

lat = lr.variables['latitude'][:].flatten()
lon = lr.variables['longitude'][:].flatten()
vals = lr.variables['depth_or_elevation'][:].flatten()

lr_points = [(lon[i], lat[i]) for i in range(len(lon))]
lr_pts = np.array(lr_points)

xmin = 88.5
xmax = 89.5
ymin = 21.5
ymax = 22.5
ll = np.array([xmin, ymin])  # lower-left
ur = np.array([xmax, ymax])  # upper-right
        
lr_idx = np.all(np.logical_and(ll <= lr_pts, lr_pts <= ur), axis=1)
lr_lon_crop = lon[lr_idx]
lr_lat_crop = lat[lr_idx]
lr_vals_crop = vals[lr_idx]


plt.scatter(lr_lon_crop,lr_lat_crop,c=lr_vals_crop,cmap='terrain')
plt.show()

nodes = gp.GeoDataFrame([
    lr_lon_crop,
    lr_lat_crop,
    lr_vals_crop,
]).T

#rename columns.
nodes.rename(
    columns={
        0:"x",
        1:"y",
        2:"elv",
        },inplace=True)

nodes = nodes.apply(pd.to_numeric, errors='ignore') # nodes.dtypes
geom = gp.GeoSeries(map(Point, zip(lr_lon_crop, lr_lat_crop)))
nodes['geometry'] = geom
nodes = gp.GeoDataFrame(nodes)
nodes.set_geometry(col='geometry')
nodes = nodes.set_crs(4326, allow_override=True)
outdir = '/Users/ealtenau/Documents/SWORD_Dev/swot_data/Ganges/v2.0/LR/lr_elv.gpkg'
nodes.to_file(outdir, driver='GPKG', layer='points')

###################
import xarray as xr
import rioxarray as rio

#Open the NetCDF
#Download the sample from https://www.unidata.ucar.edu/software/netcdf/examples/sresa1b_ncar_ccsm3-example.nc
ncfile = nc.Dataset('/Users/ealtenau/Documents/SWORD_Dev/swot_data/Ganges/v2.0/LR/SWOT_L2_LR_SSH_Expert_009_495_20240121T191011_20240121T200045_PIC0_01.nc')

lon = np.array(ncfile.variables['longitude'][:]).flatten()
lat = np.array(ncfile.variables['latitude'][:]).flatten()
elv = np.array(ncfile.variables['depth_or_elevation'][:]).flatten()


ds = xr.Dataset(
    {"elv": (("x", "y"), elv.T)},
    coords={
        "x": list(lon[0,:]),
        "y": list(lat[:,0]),
    },
)


ds = xr.Dataset(
    {"foo": (("x", "y"), np.random.rand(4, 5))},
    coords={
        "x": [10, 20, 30, 40],
        "y": pd.date_range("2000-01-01", periods=5),
        "z": ("x", list("abcd")),
    },
)



start = time.time()
dist = calc_path_dist(lon,lat)
end = time.time()
print(end-start)

start = time.time()
gdf = gp.GeoDataFrame(geometry=gp.points_from_xy(lon, lat),crs="EPSG:4326").to_crs("EPSG:3857") 
diff = gdf.distance(gdf.shift(1)); diff[0] = 0
dist = np.cumsum(test)
end = time.time()
print(end-start)


#Check for the CRS
ds.rio.crs

#(Optional) If your CRS is not discovered, you should be able to add it like so:
ds.rio.set_crs("epsg:4326")



ds.rio.to_raster('/Users/ealtenau/Documents/SWORD_Dev/swot_data/Ganges/v2.0/LR/lr_elv.tif')







import numpy as np
from scipy import interpolate

A = rch_paths_dist[pts]
A[np.where(A == 0)] = np.nan

inds = np.arange(A.shape[0])
good = np.where(np.isfinite(A))
f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
B = np.where(np.isfinite(A),A,f(inds))
