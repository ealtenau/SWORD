import numpy as np
import netCDF4 as nc
import geopandas as gp
import geopy.distance
from shapely.geometry import LineString, Point
from geopandas import GeoSeries
import pandas as pd
import time

#############################################################################################

def define_geometry(unq_rch, reach_id, cl_x, cl_y):
    geom = []
    rm_ind = []
    for ind in list(range(len(unq_rch))):
        # print(ind)
        in_rch = np.where(reach_id[0,:] == unq_rch[ind])[0]
        sort_ind = in_rch[np.argsort(cl_id[in_rch])]
        x_coords = cl_x[sort_ind]
        y_coords = cl_y[sort_ind]

        #appending neighboring reach endpoints to coordinates
        in_rch_up_dn = []
        for ngh in list(range(1,4)):
            in_rch_up_dn.append(np.where(reach_id[ngh,:]==unq_rch[ind])[0])

        in_rch_up_dn = np.unique([j for sub in in_rch_up_dn for j in sub])
        if len(in_rch_up_dn) > 0:
            for ct in list(range(len(in_rch_up_dn))):
                x_pt = cl_x[in_rch_up_dn[ct]]
                y_pt = cl_y[in_rch_up_dn[ct]]
                
                if x_pt < 0 and np.min([cl_x[sort_ind[0]], cl_x[sort_ind[-1]]]) > 0:
                    print(unq_rch[ind])
                    continue
                
                else:
                    #distance to first and last point. 
                    coords_1 = (y_pt, x_pt)
                    coords_2 = (cl_y[sort_ind[0]], cl_x[sort_ind[0]])
                    coords_3 = (cl_y[sort_ind[-1]], cl_x[sort_ind[-1]])
                    d1 = geopy.distance.geodesic(coords_1, coords_2).m
                    d2 = geopy.distance.geodesic(coords_1, coords_3).m
                    #if minimum distance is greater than 200 m then don't attach. 
                    if np.min([d1,d2]) > 200:
                        continue
                    else:
                        if d1 < d2:
                            x_coords = np.insert(x_coords, 0, x_pt, axis=0)
                            y_coords = np.insert(y_coords, 0, y_pt, axis=0)
                        if d1 > d2: 
                            x_coords = np.insert(x_coords, len(x_coords), x_pt, axis=0)
                            y_coords = np.insert(y_coords, len(y_coords), y_pt, axis=0)

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
region = 'NA'
version = 'v14'
outdir = '/Users/ealteanau/Documents/SWORD_Dev/outputs/'
outpath = outdir+version+'/'
fn = outpath+'netcdf/'+region.lower()+'_sword_'+version+'.nc'
# fn = '/Users/ealteanau/Documents/SWORD_Dev/outputs/v14/netcdf/na_sword_v14_subset.nc'

data = nc.Dataset(fn)

#pull centerline level data. 
reach_id = data.groups['centerlines'].variables['reach_id'][:]
cl_id = data.groups['centerlines'].variables['cl_id'][:]
cl_x = data.groups['centerlines'].variables['x'][:]
cl_y = data.groups['centerlines'].variables['y'][:]
#identify unique reach ids. 
unq_rch = data.groups['reaches'].variables['reach_id'][:]

#reformat multi-dimensional variables
rch_type = [int(str(rch)[-1]) for rch in unq_rch]
rch_up = data.groups['reaches'].variables['rch_id_up'][:].T
rch_dn = data.groups['reaches'].variables['rch_id_dn'][:].T
swot_orbs = data.groups['reaches'].variables['swot_orbits'][:].T
rch_id_up = []; rch_id_dn = []; swot_orbits = []
for ind in list(range(len(rch_type))):
    rch_id_up.append(str(rch_up[ind,np.where(rch_up[ind,:] > 0)[0]])[1:-1])
    rch_id_dn.append(str(rch_dn[ind,np.where(rch_dn[ind,:] > 0)[0]])[1:-1])
    swot_orbits.append(str(swot_orbs[ind,np.where(swot_orbs[ind,:] > 0)[0]])[1:-1])

#create geometry for each reach. 
print('Creating Reach Geometry')
start = time.time()
geom, rm_ind = define_geometry(unq_rch, reach_id, cl_x, cl_y)
end = time.time()
print('Finished Reach Geometry in: '+str(np.round((end-start)/60,2))+' min')

#create initial GeoDataFrame.
reaches = gp.GeoDataFrame([
    data.groups['reaches'].variables['x'][:],
    data.groups['reaches'].variables['y'][:],
    data.groups['reaches'].variables['reach_id'][:],
    data.groups['reaches'].variables['reach_length'][:],
    data.groups['reaches'].variables['n_nodes'][:],
    data.groups['reaches'].variables['wse'][:],
    data.groups['reaches'].variables['wse_var'][:],
    data.groups['reaches'].variables['width'][:],
    data.groups['reaches'].variables['width_var'][:],
    data.groups['reaches'].variables['facc'][:],
    data.groups['reaches'].variables['n_chan_max'][:],
    data.groups['reaches'].variables['n_chan_mod'][:],
    data.groups['reaches'].variables['obstr_type'][:],
    data.groups['reaches'].variables['grod_id'][:],
    data.groups['reaches'].variables['hfalls_id'][:],
    data.groups['reaches'].variables['slope'][:],
    data.groups['reaches'].variables['dist_out'][:],
    data.groups['reaches'].variables['lakeflag'][:],
    data.groups['reaches'].variables['max_width'][:],
    data.groups['reaches'].variables['n_rch_up'][:],
    data.groups['reaches'].variables['n_rch_down'][:],
    rch_id_up,
    rch_id_dn,
    swot_orbits,
    data.groups['reaches'].variables['swot_obs'][:],
    rch_type,
    data.groups['reaches'].variables['river_name'][:],
    data.groups['reaches'].variables['edit_flag'][:],
    # data.groups['reaches'].variables['lake_id'][:],
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
        # 28:"lake_id",
        },inplace=True)


#removing rows where reach was only one point.
reaches.drop(rm_ind, inplace=True)
#update data types
reaches = reaches.apply(pd.to_numeric, errors='ignore') # reaches.dtypes
#add geometry column and define crs. 
reaches['geometry'] = geom
reaches.set_geometry(col='geometry', inplace=True)
reaches = reaches.set_crs(4326, allow_override=True)

print('Writing GeoPackage File')
start = time.time()
#write geopackage (continental scale)
outgpkg = outpath + 'gpkg/' + region.lower() + '_sword_reaches_' + version + '.gpkg'
reaches.to_file(outgpkg, driver='GPKG', layer='reaches')
end = time.time()
print('Finished GPKG in: '+str(np.round((end-start)/60,2))+' min')

#write as shapefile per level2 basin.
print('Writing Shapefiles')
start = time.time()
level2 = [int(str(r)[0:2]) for r in reaches['reach_id']]
unq_l2 = np.unique(level2)
rch_cp = reaches.copy(); rch_cp['level2'] = level2
for lvl in list(range(len(unq_l2))):
    outshp = outpath + 'shp/' + region + '/' + region.lower() + "_sword_reaches_hb" + str(unq_l2[lvl]) + "_" + version + '.shp'
    subset = rch_cp[rch_cp['level2'] == unq_l2[lvl]]
    subset = subset.drop(columns=['level2'])
    subset.to_file(outshp)
    del(subset)
end = time.time()
print('Finished SHPs in: '+str(np.round((end-start)/60,2))+' min')
