###################################################
# Script: init_geom.py
# Date created: 4/24/24
# Usage: Preparing data for topology algorithm. This involves:
#        - Finding initial geometric intersections
###################################################

print('Starting code')

import os
main_dir = os.getcwd()
import sys
import fiona
import shapely
import shapely.ops

import numpy as np
import pandas as pd
import geopandas as gpd
import time
# from timeit import default_timer as timer

# import threading
# import psutil
# process = psutil.Process()

#*******************************************************************************
#Declaration of variables (given as command line arguments)
#*******************************************************************************
# 1 - basin

#*******************************************************************************
#Get command line arguments
#*******************************************************************************
IS_arg=len(sys.argv)
if IS_arg !=3:
     print('ERROR - only 2 argument can be used: Continent and Basin')
     raise SystemExit(22) 

start = time.time()
region=sys.argv[1]
basin=sys.argv[2]

shp_dir = main_dir+'/data/outputs/Reaches_Nodes/v17/shp/'+region+'/'
pts_dir = main_dir+'/data/outputs/Topology/'+region+'/b'+basin+'/'
if os.path.exists(pts_dir) == False:
    os.makedirs(pts_dir)
rrr_riv_shp = shp_dir+region.lower()+'_sword_reaches_hb' + basin + '_v17.shp' 
# rrr_riv_shp = pts_dir+region.lower()+'_sword_reaches_hb' + basin + '_v17_FG1.shp'
rrr_pts_shp = pts_dir+region.lower()+'_sword_reaches_hb' + basin + '_v17_pts.shp'

# log = open('out.txt', 'a')

# basin = str(os.environ['SLURM_ARRAY_TASK_ID'])
# rrr_riv_shp='/nas/longleaf/home/elyssac/SWORD/hb' + basin + '/na_sword_reaches_hb' + basin + '_v17.shp'
# rrr_pts_shp='/nas/longleaf/home/elyssac/SWORD/hb' + basin + '/na_sword_reaches_hb' + basin + '_v17_pts.shp'

print('- The basin is: ' + str(basin))
print('- rrr_riv_shp: ' + str(rrr_riv_shp))
print('- rrr_pts_shp: ' + str(rrr_pts_shp))

# log.write('- The basin is: ' + str(basin) + '\n')
# log.write('- rrr_riv_shp: ' + str(rrr_riv_shp) + '\n')
# log.write('- rrr_pts_shp: ' + str(rrr_pts_shp) + '\n')
# log.flush() #write buffer to file

#**************************************************
# Defining helpful functions
#**************************************************

### Function for reversing coordinates of LineString
def rev_crds(input_geom):
    if input_geom.geom_type.lower() == 'linestring':
        coords = [tuple(coord) for coord in list(input_geom.coords)][::-1]
        out_geom = shapely.geometry.LineString(coords)
    elif input_geom.geom_type.lower() == 'multilinestring':
        coords = [list(this_geom.coords)[::-1] for this_geom in input_geom.geoms][::-1]
        out_geom = shapely.geometry.MultiLineString(coords)
    return out_geom


def find_index_of_point_w_min_distance(list_of_coords, coord):
    temp = [shapely.geometry.Point(c).distance(shapely.geometry.Point(coord)) for c in list_of_coords]
    return(temp.index(min(temp)))


#**************************************************
# Reading in data
#**************************************************

print('Reading in data')
# log.write('Reading in data\n')
# log.flush() #write buffer to file

shp = gpd.read_file(rrr_riv_shp)


#**************************************************
# Finding initial geometric intersections 
#**************************************************
print('Finding intial geometric intersections')
# log.write('Finding intial geometric intersections\n')
# log.flush() #write buffer to file

eps = 5e-07 

crs = shp.crs

### Making a new point layer
pt_schema={'geometry': 'Point', 'properties': {'geom1_rch_id': 'int', 'geom2_rch_id': 'int', 'geom1_n_pnts': 'int', 'ind_intr': 'int'}}
# point_lyr = fiona.open("points.shp", mode='w', driver='ESRI Shapefile', schema=pt_schema, crs =crs)
point_lyr = fiona.open(rrr_pts_shp, mode='w', driver='ESRI Shapefile', schema=pt_schema, crs =crs)

# shp.loc[shp['reach_id'] == 63560900516] # ix =2294
# reaches_win_dist.loc[shp['reach_id'] == 63560900533] # ix2 = 2329

for ix, r in shp.iterrows():
    geom1 = shapely.geometry.shape(shp.geometry[ix])
    geom1_rch_id = shp.reach_id[ix]

    if geom1.geom_type == 'MultiLineString':
        geom1 = shapely.ops.linemerge(geom1)

    selected_reach = shp.loc[shp['reach_id'] == geom1_rch_id]
    #Finding reaches within search distance (11 km)
    reaches_win_dist = shp.assign(distance=shp.apply(lambda x: x.geometry.distance(selected_reach.geometry.iloc[0]), axis=1)).query(f'distance <= {0.1}')
    reaches_win_dist = reaches_win_dist[reaches_win_dist.reach_id != geom1_rch_id]


    #Only comparing selected_reach to reaches within search distance (minimizes computational needs)
    for ix2, r2 in reaches_win_dist.iterrows(): 

        geom2 = shapely.geometry.shape(reaches_win_dist.geometry[ix2])
        geom2_rch_id = reaches_win_dist.reach_id[ix2]

        if geom2.geom_type == 'MultiLineString':
            geom2 = shapely.ops.linemerge(geom2)

        if geom1_rch_id == geom2_rch_id:
            continue

        if geom1.distance(geom2) < eps: 
            point = shapely.ops.nearest_points(geom1, geom2)[0]
            # print(point)

            ## Sometimes the point won't exactly match up, so need to find the nearest point to connect the 2 segments
            found = False
            for i in range(len(geom1.coords.xy[0])):
                x = geom1.coords.xy[0][i]
                y = geom1.coords.xy[1][i]
                tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})

                if(tmp_pt == point):
                    found = True     
                    # print(found)

                    ind = i
                    # print(ind)

                    point_lyr.write({'geometry': shapely.geometry.mapping(tmp_pt), 
                                     'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                    'geom2_rch_id': geom2_rch_id.item(),
                                                    'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                    'ind_intr': ind}})

                    break

            if found == False:
                # print(found)
                dist_point = []
                for i in range(len(geom1.coords.xy[0])):
                    x = geom1.coords.xy[0][i]
                    y = geom1.coords.xy[1][i]
                    tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})
                    dist_point.append(point.distance(tmp_pt))
                
                ind = dist_point.index(min(dist_point))
                # print(ind)

                point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                'geom2_rch_id': geom2_rch_id.item(),
                                                'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                'ind_intr': ind}})

point_lyr.close()


# print('The number of threads is: ' + str(threading.active_count()))
# print('The memory used is: ' + str(process.memory_info().rss))  # in bytes 

end = time.time()
print('Finished in: '+str(np.round((end-start)/60,2))+' mins')

# log.write('Finished')
# log.close()
