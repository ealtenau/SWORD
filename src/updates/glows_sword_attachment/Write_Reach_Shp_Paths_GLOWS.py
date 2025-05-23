'''
This script uses an anxillary data set with pre identified common reaches 
for junctions to try and allow for better connectivity and junctions 
meeting at one point. 
'''
import os
main_dir = os.getcwd()
import numpy as np
import netCDF4 as nc
import geopandas as gp
import geopy.distance
from shapely.geometry import LineString, Point
from geopandas import GeoSeries
import pandas as pd
import time
import argparse

#############################################################################################

def define_geometry(unq_rch, reach_id, cl_x, cl_y, cl_id, common, max_dist, region):
    geom = []
    rm_ind = []
    connections = np.zeros([reach_id.shape[0], reach_id.shape[1]], dtype=int)
    for ind in list(range(len(unq_rch))):
        # print(ind, len(unq_rch)-1)
        in_rch = np.where(reach_id[0,:] == unq_rch[ind])[0]
        sort_ind = in_rch[np.argsort(cl_id[in_rch])]
        x_coords = cl_x[sort_ind]
        y_coords = cl_y[sort_ind]

        #ultimatley don't want to have to use this...
        # if len(in_rch) == 0:
        #     print(unq_rch[ind], 'no centerline points')
        #     continue
     
        #appending neighboring reach endpoints to coordinates
        in_rch_up_dn = []
        for ngh in list(range(1,4)):
            neighbors = np.where(reach_id[ngh,:]==unq_rch[ind])[0]
            keep = np.where(connections[ngh,neighbors] == 0)[0]
            in_rch_up_dn.append(neighbors[keep])
        #formating into single list.
        in_rch_up_dn = np.unique([j for sub in in_rch_up_dn for j in sub]) #reach_id[0,in_rch_up_dn]
            
        #loop through and find what ends each point belong to.
        if len(in_rch_up_dn) > 0:
            end1_dist = []; end2_dist = []
            end1_pt = []; end2_pt = []
            end1_x = []; end2_x = []
            end1_y = []; end2_y = []
            end1_flag = []; end2_flag = []
            for ct in list(range(len(in_rch_up_dn))):
                x_pt = cl_x[in_rch_up_dn[ct]]
                y_pt = cl_y[in_rch_up_dn[ct]]
                if region == 'AS' and x_pt < 0 and np.min([cl_x[sort_ind[0]], cl_x[sort_ind[-1]]]) > 0:
                    print(unq_rch[ind])
                    continue
                elif region == 'AS' and x_pt > 0 and np.min([cl_x[sort_ind[0]], cl_x[sort_ind[-1]]]) < 0:
                    print(unq_rch[ind])
                    continue
                else:
                    #distance to first and last point. 
                    coords_1 = (y_pt, x_pt)
                    coords_2 = (cl_y[sort_ind[0]], cl_x[sort_ind[0]])
                    coords_3 = (cl_y[sort_ind[-1]], cl_x[sort_ind[-1]])
                    d1 = geopy.distance.geodesic(coords_1, coords_2).m
                    d2 = geopy.distance.geodesic(coords_1, coords_3).m
                        
                    if d1 < d2:
                        end1_pt.append(in_rch_up_dn[ct])
                        end1_dist.append(d1)
                        end1_x.append(x_pt)
                        end1_y.append(y_pt)
                        end1_flag.append(common[in_rch_up_dn[ct]])
                    if d1 > d2:
                        end2_pt.append(in_rch_up_dn[ct])
                        end2_dist.append(d2)
                        end2_x.append(x_pt)
                        end2_y.append(y_pt)
                        end2_flag.append(common[in_rch_up_dn[ct]])

            #append coords to ends
            if len(end1_pt) > 0: #reach_id[:,end1_pt]
                #if point has two neighbors it is a common reach and should be skipped.
                if common[sort_ind[0]] == 1: # len(end1_pt) > 1
                    x_coords = x_coords
                    y_coords = y_coords
                
                else:
                    end1_pt = np.array(end1_pt)
                    end1_dist = np.array(end1_dist)
                    end1_x = np.array(end1_x)
                    end1_y = np.array(end1_y)
                    end1_flag = np.array(end1_flag)
                    sort_ind1 = np.argsort(end1_dist)
                    end1_pt = end1_pt[sort_ind1]
                    end1_dist = end1_dist[sort_ind1]
                    end1_x = end1_x[sort_ind1]
                    end1_y = end1_y[sort_ind1]
                    end1_flag = end1_flag[sort_ind1]
                    flag1 = np.where(end1_flag == 1)[0]
                    if len(flag1) > 0: 
                        idx1 = flag1[0] 
                    else: 
                        idx1 = 0
                        
                    if np.min(end1_dist) <= max_dist:
                        x_coords = np.insert(x_coords, 0, end1_x[idx1], axis=0)
                        y_coords = np.insert(y_coords, 0, end1_y[idx1], axis=0)
                    else:
                        ngh_x = cl_x[np.where(reach_id[0,:] == reach_id[0,end1_pt[idx1]])]
                        ngh_y = cl_y[np.where(reach_id[0,:] == reach_id[0,end1_pt[idx1]])]
                        d=[]
                        for c in list(range(len(ngh_x))):
                            temp_coords = (ngh_y[c], ngh_x[c])
                            d.append(geopy.distance.geodesic(coords_2, temp_coords).m)
                        if np.min(d) <= max_dist:
                            append_x = ngh_x[np.where(d == np.min(d))]
                            append_y = ngh_y[np.where(d == np.min(d))]
                            x_coords = np.insert(x_coords, 0, append_x[0], axis=0)
                            y_coords = np.insert(y_coords, 0, append_y[0], axis=0)
                    #flag current reach for neighbors. MAY NEED TO HAPPEN ANYWAY...
                    ngh1 = reach_id[0,end1_pt[idx1]]
                    col1 = np.where(reach_id[1,:]== ngh1)[0]
                    col2 = np.where(reach_id[2,:]== ngh1)[0]
                    col3 = np.where(reach_id[3,:]== ngh1)[0] 
                    if unq_rch[ind] in reach_id[0,col1]:
                        c = np.where(reach_id[0,col1] == unq_rch[ind])[0]
                        connections[1,col1[c]] = 1
                    if unq_rch[ind] in reach_id[0,col2]:
                        c = np.where(reach_id[0,col2] == unq_rch[ind])[0]
                        connections[2,col2[c]] = 1
                    if unq_rch[ind] in reach_id[0,col3]:
                        c = np.where(reach_id[0,col3] == unq_rch[ind])[0]
                        connections[3,col3[c]] = 1

            if len(end2_pt) > 0: #reach_id[:,end2_pt]
                #if point has two neighbors it is a common reach and should be skipped.
                if common[sort_ind[-1]] == 1: # len(end2_pt) > 1
                    x_coords = x_coords
                    y_coords = y_coords
                
                else:
                    end2_pt = np.array(end2_pt)
                    end2_dist = np.array(end2_dist)
                    end2_x = np.array(end2_x)
                    end2_y = np.array(end2_y)
                    end2_flag = np.array(end2_flag)
                    sort_ind2 = np.argsort(end2_dist)
                    end2_pt = end2_pt[sort_ind2]
                    end2_dist = end2_dist[sort_ind2]
                    end2_x = end2_x[sort_ind2]
                    end2_y = end2_y[sort_ind2]
                    end2_flag = end2_flag[sort_ind2]
                    flag2 = np.where(end2_flag == 1)[0]
                    if len(flag2) > 0: 
                        idx2 = flag2[0] 
                    else: 
                        idx2 = 0

                    if np.min(end2_dist) < max_dist:
                        x_coords = np.insert(x_coords, len(x_coords), end2_x[idx2], axis=0)
                        y_coords = np.insert(y_coords, len(y_coords), end2_y[idx2], axis=0)
                    else:
                        ngh_x = cl_x[np.where(reach_id[0,:] == reach_id[0,end2_pt[idx2]])]
                        ngh_y = cl_y[np.where(reach_id[0,:] == reach_id[0,end2_pt[idx2]])]
                        d=[]
                        for c in list(range(len(ngh_x))):
                            temp_coords = (ngh_y[c], ngh_x[c])
                            d.append(geopy.distance.geodesic(coords_3, temp_coords).m)
                        if np.min(d) <= max_dist:
                            append_x = ngh_x[np.where(d == np.min(d))]
                            append_y = ngh_y[np.where(d == np.min(d))]
                            x_coords = np.insert(x_coords, len(x_coords), append_x[0], axis=0)
                            y_coords = np.insert(y_coords, len(y_coords), append_y[0], axis=0)
                    #flag current reach for neighbors.
                    ngh2 = reach_id[0,end2_pt[idx2]]
                    col1 = np.where(reach_id[1,:]== ngh2)[0]
                    col2 = np.where(reach_id[2,:]== ngh2)[0]
                    col3 = np.where(reach_id[3,:]== ngh2)[0] 
                    if unq_rch[ind] in reach_id[0,col1]:
                        c = np.where(reach_id[0,col1] == unq_rch[ind])[0]
                        connections[1,col1[c]] = 1
                    if unq_rch[ind] in reach_id[0,col2]:
                        c = np.where(reach_id[0,col2] == unq_rch[ind])[0]
                        connections[2,col2[c]] = 1
                    if unq_rch[ind] in reach_id[0,col3]:
                        c = np.where(reach_id[0,col3] == unq_rch[ind])[0]
                        connections[3,col3[c]] = 1

        pts = GeoSeries(map(Point, zip(x_coords, y_coords)))
        if len(pts) <= 1:
            rm_ind.append(ind)
            continue
        else:
            line = LineString(pts.tolist())
            geom.append(line) 

    return geom, rm_ind

#############################################################################################

#read in netcdf data. 
parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

outdir = main_dir+'/data/outputs/Reaches_Nodes/'
outpath = outdir+version+'/'
fn1 = outpath+'netcdf/'+region.lower()+'_sword_'+version+'.nc'
fn2 = outpath+'reach_geometry/'+region.lower()+'_sword_'+version+'_connectivity.nc'

data1 = nc.Dataset(fn1)
data2 = nc.Dataset(fn2)

#pull centerline level data. 
reach_id = data2.groups['centerlines'].variables['reach_id'][:]
cl_id = data2.groups['centerlines'].variables['cl_id'][:]
cl_x = data2.groups['centerlines'].variables['x'][:]
cl_y = data2.groups['centerlines'].variables['y'][:]
common = data2.groups['centerlines'].variables['common'][:]

#identify unique reach ids. 
unq_rch = data1.groups['reaches'].variables['reach_id'][:]

#reformat multi-dimensional variables
rch_type = np.array([int(str(rch)[-1]) for rch in unq_rch])
rch_up = np.array(data1.groups['reaches'].variables['rch_id_up'][:]).T
rch_dn = np.array(data1.groups['reaches'].variables['rch_id_dn'][:]).T
swot_orbs = np.array(data1.groups['reaches'].variables['swot_orbits'][:]).T
rch_id_up = []; rch_id_dn = []; swot_orbits = []
for ind in list(range(len(rch_type))):
    rch_id_up.append(str(rch_up[ind,np.where(rch_up[ind,:] > 0)[0]])[1:-1])
    rch_id_dn.append(str(rch_dn[ind,np.where(rch_dn[ind,:] > 0)[0]])[1:-1])
    swot_orbits.append(str(swot_orbs[ind,np.where(swot_orbs[ind,:] > 0)[0]])[1:-1])

#create geometry for each reach. 
print('Creating Reach Geometry')
start = time.time()
geom, rm_ind = define_geometry(unq_rch, reach_id, cl_x, cl_y, 
                               cl_id, common, 200, region) 
end = time.time()
print('Finished Reach Geometry in: '+str(np.round((end-start)/60,2))+' min')

#create initial GeoDataFrame.
reaches = gp.GeoDataFrame([
    np.array(data1.groups['reaches'].variables['x'][:]),
    np.array(data1.groups['reaches'].variables['y'][:]),
    np.array(data1.groups['reaches'].variables['reach_id'][:]),
    np.array(data1.groups['reaches'].variables['reach_length'][:]),
    np.array(data1.groups['reaches'].variables['n_nodes'][:]),
    np.array(data1.groups['reaches'].variables['wse'][:]),
    np.array(data1.groups['reaches'].variables['wse_var'][:]),
    np.array(data1.groups['reaches'].variables['width'][:]),
    np.array(data1.groups['reaches'].variables['width_var'][:]),
    np.array(data1.groups['reaches'].variables['facc'][:]),
    np.array(data1.groups['reaches'].variables['n_chan_max'][:]),
    np.array(data1.groups['reaches'].variables['n_chan_mod'][:]),
    np.array(data1.groups['reaches'].variables['obstr_type'][:]),
    np.array(data1.groups['reaches'].variables['grod_id'][:]),
    np.array(data1.groups['reaches'].variables['hfalls_id'][:]),
    np.array(data1.groups['reaches'].variables['slope'][:]),
    np.array(data1.groups['reaches'].variables['dist_out'][:]),
    np.array(data1.groups['reaches'].variables['lakeflag'][:]),
    np.array(data1.groups['reaches'].variables['max_width'][:]),
    np.array(data1.groups['reaches'].variables['n_rch_up'][:]),
    np.array(data1.groups['reaches'].variables['n_rch_down'][:]),
    rch_id_up,
    rch_id_dn,
    swot_orbits,
    np.array(data1.groups['reaches'].variables['swot_obs'][:]),
    rch_type,
    np.array(data1.groups['reaches'].variables['river_name'][:]),
    np.array(data1.groups['reaches'].variables['edit_flag'][:]),
    np.array(data1.groups['reaches'].variables['trib_flag'][:]),
    np.array(data1.groups['reaches'].variables['path_freq'][:]),
    np.array(data1.groups['reaches'].variables['path_order'][:]),
    np.array(data1.groups['reaches'].variables['path_segs'][:]),
    np.array(data1.groups['reaches'].variables['main_side'][:]),
    np.array(data1.groups['reaches'].variables['stream_order'][:]),
    np.array(data1.groups['reaches'].variables['end_reach'][:]),
    np.array(data1.groups['reaches'].variables['network'][:])
]).T

#rename columns.
reaches.rename(
    columns={
        0:"x",
        1:"y",
        2:"reach_id",
        3:"reach_len",
        4:"n_nodes",
        5:"wse",
        6:"wse_var",
        7:"width",
        8:"width_var",
        9:"facc",
        10:"n_chan_max",
        11:"n_chan_mod",
        12:"obstr_type",
        13:"grod_id",
        14:"hfalls_id",
        15:"slope",
        16:"dist_out",
        17:"lakeflag",
        18:"max_width",
        19:"n_rch_up",
        20:"n_rch_dn",
        21:"rch_id_up",
        22:"rch_id_dn",
        23:"swot_orbit",
        24:"swot_obs",
        25:"type",
        26:"river_name",
        27:"edit_flag",
        28:"trib_flag",
        29:"path_freq",
        30:"path_order",
        31:"path_segs",
        32:"main_side",
        33:"strm_order",
        34:"end_reach",
        35:"network"
        },inplace=True)


#removing rows where reach was only one point.
reaches.drop(rm_ind, inplace=True)
#update data types
reaches = reaches.apply(pd.to_numeric, errors='ignore') # reaches.dtypes
#add geometry column and define crs. 
reaches['geometry'] = geom
reaches = gp.GeoDataFrame(reaches)
reaches.set_geometry(col='geometry') #removed "inplace=True" option on leopold. 
reaches = reaches.set_crs(4326, allow_override=True)

# write geopackage (continental scale)
print('Writing GeoPackage File')
if os.path.exists(outpath+'gpkg/'):
    outgpkg = outpath + 'gpkg/' + region.lower() + '_sword_reaches_' + version + '.gpkg'
else:
    os.makedirs(outpath+'gpkg/')
    outgpkg = outpath + 'gpkg/' + region.lower() + '_sword_reaches_' + version + '.gpkg'

start = time.time()
reaches.to_file(outgpkg, driver='GPKG', layer='reaches')
end = time.time()
print('Finished GPKG in: '+str(np.round((end-start)/60,2))+' min')

# write as shapefile per level2 basin.
print('Writing Shapefiles')
if os.path.exists(outpath + 'shp/' + region + '/'):
    shpdir = outpath + 'shp/' + region + '/'
else:
    os.makedirs(outpath + 'shp/' + region + '/')
    shpdir = outpath + 'shp/' + region + '/'

start = time.time()
level2 = np.array([int(str(r)[0:2]) for r in reaches['reach_id']])
unq_l2 = np.unique(level2)
rch_cp = reaches.copy(); rch_cp['level2'] = level2
for lvl in list(range(len(unq_l2))):
    outshp = shpdir + region.lower() + "_sword_reaches_hb" + str(unq_l2[lvl]) + "_" + version + '.shp'
    subset = rch_cp[rch_cp['level2'] == unq_l2[lvl]]
    subset = subset.drop(columns=['level2'])
    subset.to_file(outshp)
    del(subset)
end = time.time()
print('Finished SHPs in: '+str(np.round((end-start)/60,2))+' min')
