import numpy as np
import netCDF4 as nc
import geopandas as gp
import geopy.distance
from shapely.geometry import LineString, Point
from geopandas import GeoSeries
import pandas as pd
import time
import argparse
import os

start_all = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("local_processing", help="'True' for local machine, 'False' for server", type = str)
args = parser.parse_args()

region = args.region
version = args.version

if args.local_processing == 'True':
    outdir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'
else:
    outdir = '/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/outputs/Reaches_Nodes/'

outpath = outdir+version+'/'
fn1 = outpath+'netcdf/'+region.lower()+'_sword_'+version+'.nc'
fn2 = outpath+'reach_geometry/'+region.lower()+'_sword_'+version+'_connectivity.nc'
shp_dir = outpath[0:-27]+'gis_files/continent_buffer_0.1deg.gpkg'
# gpkg_dir = outpath+'gpkg/'+region.lower()+'_sword_reaches_'+version+'.gpkg'
# gpkg_node_dir = outpath+'gpkg/'+region.lower()+'_sword_nodes_'+version+'.gpkg'

#read in netcdf data.
data1 = nc.Dataset(fn1)
data2 = nc.Dataset(fn2, 'r+')
buffer = gp.read_file(shp_dir)
# gpkg = gp.read_file(gpkg_dir)
# gpkg_node = gp.read_file(gpkg_node_dir)

#pull centerline level data. 
node_id = data1.groups['centerlines'].variables['node_id'][0,:]
reach_id = data2.groups['centerlines'].variables['reach_id'][:]
cl_id = data2.groups['centerlines'].variables['cl_id'][:]
cl_x = data2.groups['centerlines'].variables['x'][:]
cl_y = data2.groups['centerlines'].variables['y'][:]
common = data2.groups['centerlines'].variables['common'][:]
end_rch = np.zeros(len(common))

rch_binary = np.copy(reach_id)
rch_binary[np.where(rch_binary > 0)] = 1
row_sums_all = np.sum(rch_binary, axis = 0)
junctions = np.where(row_sums_all > 2)[0]
end_rch[junctions] = 3

print('Finding End Reaches')
unq_rch = np.unique(reach_id[0,:])
for ind in list(range(len(unq_rch))):
    pts = np.where(reach_id[0,:] == unq_rch[ind])[0]
    
    rch_sub = reach_id[:,pts]
    rch_sub[np.where(rch_sub > 0)] = 1
    row_sums = np.sum(rch_sub, axis = 0)
    
    ends = np.where(row_sums > 1)[0]
    if len(ends) <= 1:
        if len(ends) == 0:
            mx = np.where(cl_id[pts] == np.min(cl_id[pts]))[0]
            end_rch[pts[mx]] = 1
        
        if len(ends) == 1:
            mn = np.where(cl_id[pts] == np.min(cl_id[pts]))[0]
            mx = np.where(cl_id[pts] == np.max(cl_id[pts]))[0]
            if ends == mn:
                end_rch[pts[mx]] = 1
            elif ends == mx:
                end_rch[pts[mn]] = 1
            else:
                end_rch[pts[mx]] = 1

#use the continent buffer to identify outlet points. 
# Finding where delta shapefiles intersect the GRWL shapefile.
points = pd.DataFrame(np.array([cl_id,cl_x, cl_y]).T)
geom = gp.GeoSeries(map(Point, zip(cl_x, cl_y)))
points['geometry'] = geom
points = gp.GeoDataFrame(points)
points.set_geometry(col='geometry')
points = points.set_crs(4326, allow_override=True)

poly = buffer
intersect = gp.sjoin(poly, points, how="inner")
intersect = pd.DataFrame(intersect)
intersect = intersect.drop_duplicates(subset='index_right', keep='first')

# Identifying the overlaps.
ids = np.array(intersect.index_right)
overlap = np.where(end_rch[ids] == 1)[0]
end_rch[ids[overlap]] = 2

headwaters = np.unique(reach_id[0,np.where(end_rch == 1)])        
outlets = np.unique(reach_id[0,np.where(end_rch == 2)])        
juncts = np.unique(reach_id[0,np.where(end_rch == 3)])

node_headwaters = np.unique(node_id[np.where(end_rch == 1)])        
node_outlets = np.unique(node_id[np.where(end_rch == 2)])        
node_juncts = np.unique(node_id[np.where(end_rch == 3)])

#update gpkg.
# print('Updating GPKG Files')
# gpkg_rchs = np.array(gpkg['reach_id'])
# ind_head = np.in1d(gpkg_rchs, headwaters)
# ind_out = np.in1d(gpkg_rchs, outlets)
# ind_jun = np.in1d(gpkg_rchs, juncts)
# end_reaches = np.zeros(len(gpkg_rchs))
# end_reaches[ind_jun] = 3
# end_reaches[ind_head] = 1
# end_reaches[ind_out] = 2
# gpkg['end_reach'] = end_reaches
# gpkg.to_file(gpkg_dir, driver='GPKG', layer='reaches')

# gpkg_nds = np.array(gpkg_node['node_id'])
# node_ind_head = np.in1d(gpkg_nds, node_headwaters)
# node_ind_out = np.in1d(gpkg_nds, node_outlets)
# node_ind_jun = np.in1d(gpkg_nds, node_juncts)
# node_end_reaches = np.zeros(len(gpkg_nds))
# node_end_reaches[node_ind_jun] = 3
# node_end_reaches[node_ind_head] = 1
# node_end_reaches[node_ind_out] = 2
# gpkg_node['end_reach'] = node_end_reaches
# gpkg_node.to_file(gpkg_node_dir, driver='GPKG', layer='nodes')

#re-saving shp files. 
# print('Updating SHPs')
# rch_level2 = np.array([int(str(r)[0:2]) for r in np.array(gpkg['reach_id'])])
# node_level2 = np.array([int(str(r)[0:2]) for r in np.array(gpkg_node['node_id'])])
# unq_basins = np.unique(rch_level2)
# for b in list(range(len(unq_basins))):
#     rch_outshp = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/shp/'+region+'/'+region.lower()+\
#         '_sword_reaches_hb'+str(unq_basins[b])+'_'+version+'.shp'
#     node_outshp = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/shp/'+region+'/'+region.lower()+\
#         '_sword_nodes_hb'+str(unq_basins[b])+'_'+version+'.shp'
#     rch_basin = np.where(rch_level2 == unq_basins[b])[0]
#     node_basin = np.where(node_level2 == unq_basins[b])[0]
#     #subset continental gpkg file. 
#     rch_subset = gpkg.iloc[rch_basin]
#     rch_subset.to_file(rch_outshp)
#     node_subset = gpkg_node.iloc[node_basin]
#     node_subset.to_file(node_outshp)
#     del(rch_subset); del(node_subset)

#write outlets and headwaters to point files.
print('Saving Headwaters and Outlets')
headwaters = np.where(end_rch == 1)[0]
hw = pd.DataFrame(np.array([cl_x[headwaters], cl_y[headwaters], reach_id[0,headwaters]]).T)
hw.rename(
    columns={
        0:"x",
        1:"y",
        2:"reach_id",
        },inplace=True)

hw_geom = gp.GeoSeries(map(Point, zip(cl_x[headwaters], cl_y[headwaters])))
hw['geometry'] = hw_geom
hw = gp.GeoDataFrame(hw)
hw.set_geometry(col='geometry')
hw = hw.set_crs(4326, allow_override=True)
outgpkg='/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/network_testing/'+region.lower()+'_headwaters_sword_'+version+'.gpkg'
hw.to_file(outgpkg, driver='GPKG', layer='headwaters')

outlets = np.where(end_rch == 2)[0]
ol = pd.DataFrame(np.array([cl_x[outlets], cl_y[outlets], reach_id[0,outlets]]).T)
ol.rename(
    columns={
        0:"x",
        1:"y",
        2:"reach_id",
        },inplace=True)

ol_geom = gp.GeoSeries(map(Point, zip(cl_x[outlets], cl_y[outlets])))
ol['geometry'] = ol_geom
ol = gp.GeoDataFrame(ol)
ol.set_geometry(col='geometry')
ol = ol.set_crs(4326, allow_override=True)
outgpkg='/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/network_testing/'+region.lower()+'_outlets_sword_'+version+'.gpkg'
ol.to_file(outgpkg, driver='GPKG', layer='outlets')

#update connectivity netcdf.
print('Updating NetCDF')
if 'end_reach' in data2.groups['centerlines'].variables:
    data2.groups['centerlines'].variables['end_reach'][:] = end_rch
    data2.groups['centerlines'].variables['node_id'][:] = node_id
    data2.close()
else:
    # create variables. 
    data2.groups['centerlines'].createVariable('end_reach', 'i4', ('num_points',))
    data2.groups['centerlines'].createVariable('node_id', 'i8', ('num_points',))
    # populate variables. 
    data2.groups['centerlines'].variables['end_reach'][:] = end_rch
    data2.groups['centerlines'].variables['node_id'][:] = node_id
    data2.close()

end_all = time.time()
print('Finished in: '+str(np.round((end_all-start_all)/60,2))+' mins')