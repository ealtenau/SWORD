from __future__ import division
import os
if os.path.exists('/Users/ealteanau/Documents/SWORD_Dev/src/SWORD/post_swot_launch_updates/mhv_sword/'):
    os.chdir('/Users/ealteanau/Documents/SWORD_Dev/src/SWORD/post_swot_launch_updates/mhv_sword/')
else:
    os.chdir('/afs/cas.unc.edu/users/e/a/ealtenau/SWORD/post_swot_launch_updates/mhv_sword/')
import mhv_reach_def_tools as rdt
# import Write_Database_Files as wf
import time
import numpy as np
import geopandas as gp
from shapely.geometry import Point
import pandas as pd
import argparse

###############################################################################
##################### Defining Reach and Node Locations #######################
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
parser.add_argument("local_processing", help="'True' for local machine, 'False' for server", type = str)
args = parser.parse_args()

start_all = time.time()
region = args.region

# Input file(s).
if args.local_processing == 'True':
    main_dir = '/Users/ealteanau/Documents/SWORD_Dev/inputs/'
else:
    main_dir = '/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/inputs/'
nc_file = main_dir+'MHV_SWORD/'+region+'_mhv_sword.nc'

# Output files.
# nc_outpath = main_dir+'Reaches_Nodes/netcdf/'+region.lower()+'_sword_'+version+'.nc'
# swot_outpath = main_dir+'SWOT_Coverage/'+region.lower()+'_swot_obs_'+version+'.nc'

# Reading in data.
data = rdt.read_merge_netcdf(nc_file)

# Making sure flow accumulation minimum isn't zero.
data.facc[np.where(data.facc == 0)[0]] = 0.001
# Cutting basins to 6 digits.
data.basins = np.array([int(str(ind)[0:6]) for ind in data.basins])

# Creating empty objects to fill with attributes.
centerlines = rdt.Object()
reaches = rdt.Object()
nodes = rdt.Object()

# Loop through each level 2 basin. Subsetting per level 2 basin speeds up the script.
level2_basins = np.array([int(str(ind)[0:2]) for ind in data.basins])
uniq_level2 = np.unique(level2_basins)
uniq_level2 = np.delete(uniq_level2, 0)
cnt = 0
start_id = 0
for ind in list(range(len(uniq_level2))):

    print('STARTING BASIN: ' + str(uniq_level2[ind]))

    start = time.time()

    # Define objects to assign attributes.
    subcls = rdt.Object()
    subreaches = rdt.Object()
    subnodes = rdt.Object()

    # Subset data.
    level2 = np.where(level2_basins == uniq_level2[ind])[0]
    subcls.id = data.id[level2]
    subcls.lon = data.lon[level2]
    subcls.lat = data.lat[level2]
    subcls.seg = data.seg[level2]
    subcls.ind = data.ind[level2]
    subcls.dist = data.segDist[level2]
    subcls.wth = data.wth[level2]
    subcls.elv = data.elv[level2]
    subcls.facc = data.facc[level2]
    subcls.lake = data.lake[level2]
    subcls.delta = data.delta[level2]
    subcls.nchan = data.nchan[level2]
    subcls.grand = data.grand[level2]
    subcls.grod = data.grod[level2]
    subcls.grod_fid = data.grod_fid[level2]
    subcls.hfalls_fid = data.hfalls_fid[level2]
    subcls.basins = data.basins[level2]
    subcls.manual = data.manual[level2]
    subcls.num_obs = data.num_obs[level2]
    subcls.orbits = data.orbits[level2]
    subcls.lake_id = data.lake_id[level2]
    subcls.strorder = data.strorder[level2]
    subcls.eps = data.eps[level2]
    subcls.lon[np.where(subcls.lon < -180)] = -180.0
    subcls.lon[np.where(subcls.lon > 180)] = 180.0
    subcls.x = np.copy(subcls.lon)
    subcls.y = np.copy(subcls.lat)

    #########################################################################
    #REACH DEFINITION

    print('Defining Intial Reaches')
    # Find all reach boundaries.
    dam_radius = 225
    subcls.lake_bnds, subcls.lake_dist, subcls.lake_num,\
     subcls.dam_bnds, subcls.dam_dist, subcls.dam_id, subcls.swot_bnds,\
     subcls.swot_dist, subcls.swot_id, subcls.all_bnds, \
     subcls.type0 = rdt.find_all_bounds(subcls, dam_radius)

    # Define inital reaches.
    subcls.rch_id0, subcls.rch_len0 = rdt.number_reaches(subcls)

    print('Cutting Long Reaches')
    subcls.rch_id1, subcls.rch_len1 = rdt.cut_reaches(subcls.rch_id0,
                                                      subcls.rch_len0, subcls.dist,
                                                      subcls.ind, 20000)

    # Make sure there is only one "type" value per reach.
    subcls.type1 = np.zeros(len(subcls.rch_id1))
    uniq_rch = np.unique(subcls.rch_id1)
    for idx in list(range(len(uniq_rch))):
        rch = np.where(subcls.rch_id1 == uniq_rch[idx])[0]
        if len(rch) <=2:
            subcls.type1[rch] = np.min(subcls.type0[rch])
        else:
            subcls.type1[rch] = max(set(list(subcls.type0[rch])),
                                    key=list(subcls.type0[rch]).count)

    print('Aggregating River Reaches')
    # Aggregating river reach types.
    river_min_dist = 10000
    subcls.rch_id2, subcls.rch_len2,\
     subcls.type2 = rdt.aggregate_rivers(subcls, river_min_dist)
    # Updating reach indexes.
    subcls.rch_ind2, subcls.rch_eps2 = rdt.update_rch_indexes(subcls, subcls.rch_id2)

    print('Aggregating Lake Reaches')
    # Aggregating lake reach types.
    lake_min_dist = 10000
    subcls.rch_id3, subcls.rch_len3,\
     subcls.type3 = rdt.aggregate_lakes(subcls, lake_min_dist)
    # Updating reach indexes.
    subcls.rch_ind3, __ = rdt.update_rch_indexes(subcls, subcls.rch_id3)

    print('Aggregating Dam Reaches')
    # Aggregating dam reach types.
    dam_min_dist = 200
    subcls.rch_id4, subcls.rch_len4,\
     subcls.type4 = rdt.aggregate_dams(subcls, dam_min_dist)
    # Updating reache indexes.
    subcls.rch_ind4, __ = rdt.update_rch_indexes(subcls, subcls.rch_id4)    
    
    # Updating reach flow distance.
    subcls.rch_dist4 = rdt.calc_segDist(subcls.lon, subcls.lat, subcls.rch_id4,
                                        subcls.facc, subcls.rch_ind4)

    print('Second Cutting of Long Reaches')
    subcls.rch_id5, subcls.rch_len5 = rdt.cut_reaches(subcls.rch_id4,
                                                      subcls.rch_len4, subcls.rch_dist4,
                                                      subcls.rch_ind4, 20000)
    subcls.type5 = np.copy(subcls.type4)
    # Updating reach indexes.
    subcls.rch_ind5, \
     subcls.rch_eps5 = rdt.update_rch_indexes(subcls, subcls.rch_id5)
    # Updating reach flow distance.
    subcls.rch_dist5 = rdt.calc_segDist(subcls.lon, subcls.lat, subcls.rch_id5,
                                        subcls.facc, subcls.rch_ind5)

    #########################################################################
    #TOPOLOGY

    print('Creating Reach ID')
    subcls.reach_id, subcls.rch_topo = rdt.reach_topology(subcls)

    # Creating Nodes.
    print('Creating Nodes')
    node_length = 200
    subcls.node_id, subcls.node_len,\
     subcls.node_num = rdt.node_reaches(subcls, node_length)

    print('Creating Ghost Reaches')
    # Defining ghost reaches and nodes.
    subcls.reach_id, subcls.node_id, subcls.rch_len6 = rdt.ghost_reaches(subcls)
    # Updating reach indexes and type.
    subcls.rch_ind6,\
     subcls.rch_eps6 = rdt.update_rch_indexes(subcls, subcls.reach_id)
    # Updating reach flow distance.
    subcls.rch_dist6 = rdt.calc_segDist(subcls.lon, subcls.lat, subcls.reach_id,
                                        subcls.facc, subcls.rch_ind6)
    # Updating type information.
    subcls.type6 = np.zeros(len(subcls.reach_id))
    for j in list(range(len(subcls.reach_id))):
        subcls.type6[j] = int(str(subcls.reach_id[j])[10:11])

    print('Aggregating Data')
    # Append current data to previous data.
    rdt.append_centerlines(centerlines, subcls, cnt)
    cnt = cnt+1

    end=time.time()
    print('Time to Create Reaches and Nodes: ' +
          str(np.round((end-start)/60, 2)) + ' min')

# Update NetCDF File.
print('Updating NetCDF')
rdt.update_netcdf(nc_file, centerlines)

end_all=time.time()
print('Time to Finish All Reaches and Nodes: ' +
      str(np.round((end_all-start_all)/60, 2)) + ' min')

'''
nodes = gp.GeoDataFrame([
    subcls.x,
    subcls.y,
    subcls.reach_id,
    subcls.rch_len6,
    subcls.node_num,
    subcls.node_len,
    subcls.rch_eps6,
    subcls.type6
]).T

#rename columns.
nodes.rename(
    columns={
        0:"x",
        1:"y",
        2:"reach_id",
        3:"reach_len",
        4:"node_num",
        5:"node_len",
        6:"rch_endpts",
        7:"type",
        },inplace=True)

nodes = nodes.apply(pd.to_numeric, errors='ignore') # nodes.dtypes
geom = gp.GeoSeries(map(Point, zip(subcls.x, subcls.y)))
nodes['geometry'] = geom
nodes = gp.GeoDataFrame(nodes)
nodes.set_geometry(col='geometry')
nodes = nodes.set_crs(4326, allow_override=True)

outgpkg = '/Users/ealteanau/Documents/SWORD_Dev/inputs/MHV_SWORD/hb74_mhv_sword_test.gpkg'
nodes.to_file(outgpkg, driver='GPKG', layer='nodes')

end_all=time.time()
print('Time to Write Test Basin: ' +
      str(np.round((end_all-start_all)/60, 2)) + ' min')

'''