import netCDF4 as nc
import geopandas as gp
import numpy as np
import pandas as pd
from shapely.geometry import Point

###############################################################################

region = 'EU'
version = 'v17'
sword_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
    +version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
sword = nc.Dataset(sword_fn)

#create variable with new_rch_id and neighbors populated with zeros for now. 
# rch_vars = sword.groups['reaches'].variables.keys()
# node_vars = sword.groups['nodes'].variables.keys()

x = sword.groups['centerlines'].variables['x'][:]
y = sword.groups['centerlines'].variables['y'][:]
cl_id = sword.groups['centerlines'].variables['cl_id'][:]
rch_id = sword.groups['centerlines'].variables['reach_id'][0,:]
node_id = sword.groups['centerlines'].variables['node_id'][0,:]
new_rch_id = np.repeat('NaN', len(x))
new_up_nghs = np.repeat('NaN', len(x))
new_dn_nghs = np.repeat('NaN', len(x))
reverse_topo = np.repeat('NaN', len(x))

level2_basins = np.array([int(str(ind)[0:2]) for ind in rch_id])
unq_l2 = np.unique(level2_basins)

for ind in list(range(len(unq_l2))):
    print(unq_l2[ind])
    pts = np.where(level2_basins == unq_l2[ind])[0]
    nodes = gp.GeoDataFrame([
        x[pts],
        y[pts],
        cl_id[pts],
        rch_id[pts],
        node_id[pts],
        new_rch_id[pts],
        new_up_nghs[pts],
        new_dn_nghs[pts],
        reverse_topo[pts],
    ]).T

    #rename columns.
    nodes.rename(
        columns={
            0:"x",
            1:"y",
            2:"cl_id",
            3:"reach_id",
            4:"node_id",
            5:"new_rch_id",
            6:"new_up_nghs",
            7:"new_dn_nghs",
            8:"reverse_topo",
        },inplace=True)

    nodes = nodes.apply(pd.to_numeric, errors='ignore') # nodes.dtypes
    geom = gp.GeoSeries(map(Point, zip(x[pts], y[pts])))
    nodes['geometry'] = geom
    nodes = gp.GeoDataFrame(nodes)
    nodes.set_geometry(col='geometry')
    nodes = nodes.set_crs(4326, allow_override=True)

    outgpkg = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/gpkg_30m/'\
        +region+'/hb'+str(unq_l2[ind])+'_centerlines_'+version+'.gpkg'
    nodes.to_file(outgpkg, driver='GPKG', layer='nodes')

