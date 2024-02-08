from __future__ import division
import numpy as np
import time
import netCDF4 as nc
import pandas as pd
from scipy import spatial as sp
import utm 
import argparse
from pyproj import Proj

###############################################################################
###############################################################################
###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass 

###############################################################################

def read_data(filename):

    centerlines = Object()
    nodes = Object()
    reaches = Object ()
    
    data = nc.Dataset(filename)
    
    centerlines.cl_id = np.array(data.groups['centerlines'].variables['cl_id'][:])
    centerlines.x = np.array(data.groups['centerlines'].variables['x'][:])
    centerlines.y = np.array(data.groups['centerlines'].variables['y'][:])
    centerlines.reach_id = np.array(data.groups['centerlines'].variables['reach_id'][:])
    centerlines.node_id = np.array(data.groups['centerlines'].variables['node_id'][:])
    
    nodes.id = np.array(data.groups['nodes'].variables['node_id'][:])
    nodes.cl_id = np.array(data.groups['nodes'].variables['cl_ids'][:])
    nodes.x = np.array(data.groups['nodes'].variables['x'][:])
    nodes.y = np.array(data.groups['nodes'].variables['y'][:])
    nodes.len = np.array(data.groups['nodes'].variables['node_length'][:])
    nodes.wse = np.array(data.groups['nodes'].variables['wse'][:])
    nodes.wse_var = np.array(data.groups['nodes'].variables['wse_var'][:])
    nodes.wth = np.array(data.groups['nodes'].variables['width'][:])
    nodes.wth_var = np.array(data.groups['nodes'].variables['width_var'][:])
    nodes.grod = np.array(data.groups['nodes'].variables['obstr_type'][:])
    nodes.grod_fid = np.array(data.groups['nodes'].variables['grod_id'][:])
    nodes.hfalls_fid = np.array(data.groups['nodes'].variables['hfalls_id'][:])
    nodes.nchan_max = np.array(data.groups['nodes'].variables['n_chan_max'][:])
    nodes.nchan_mod = np.array(data.groups['nodes'].variables['n_chan_mod'][:])
    nodes.dist_out = np.array(data.groups['nodes'].variables['dist_out'][:])
    nodes.reach_id = np.array(data.groups['nodes'].variables['reach_id'][:])
    nodes.facc = np.array(data.groups['nodes'].variables['facc'][:])
    nodes.lakeflag = np.array(data.groups['nodes'].variables['lakeflag'][:])
    nodes.wth_coef = np.array(data.groups['nodes'].variables['wth_coef'][:])
    nodes.ext_dist_coef = np.array(data.groups['nodes'].variables['ext_dist_coef'][:])
    nodes.max_wth = np.array(data.groups['nodes'].variables['max_width'][:])
    nodes.meand_len = np.array(data.groups['nodes'].variables['meander_length'][:])
    nodes.river_name = np.array(data.groups['nodes'].variables['river_name'][:])
    nodes.manual_add = np.array(data.groups['nodes'].variables['manual_add'][:])
    nodes.sinuosity = np.array(data.groups['nodes'].variables['sinuosity'][:])
    nodes.edit_flag = np.array(data.groups['nodes'].variables['edit_flag'][:])
    nodes.trib_flag = np.array(data.groups['nodes'].variables['trib_flag'][:])

    reaches.id = np.array(data.groups['reaches'].variables['reach_id'][:])
    reaches.cl_id = np.array(data.groups['reaches'].variables['cl_ids'][:])
    reaches.x = np.array(data.groups['reaches'].variables['x'][:])
    reaches.x_min = np.array(data.groups['reaches'].variables['x_min'][:])
    reaches.x_max = np.array(data.groups['reaches'].variables['x_max'][:])
    reaches.y = np.array(data.groups['reaches'].variables['y'][:])
    reaches.y_min = np.array(data.groups['reaches'].variables['y_min'][:])
    reaches.y_max = np.array(data.groups['reaches'].variables['y_max'][:])
    reaches.len = np.array(data.groups['reaches'].variables['reach_length'][:])
    reaches.wse = np.array(data.groups['reaches'].variables['wse'][:])
    reaches.wse_var = np.array(data.groups['reaches'].variables['wse_var'][:])
    reaches.wth = np.array(data.groups['reaches'].variables['width'][:])
    reaches.wth_var = np.array(data.groups['reaches'].variables['width_var'][:])
    reaches.slope = np.array(data.groups['reaches'].variables['slope'][:])
    reaches.rch_n_nodes = np.array(data.groups['reaches'].variables['n_nodes'][:])
    reaches.grod = np.array(data.groups['reaches'].variables['obstr_type'][:])
    reaches.grod_fid = np.array(data.groups['reaches'].variables['grod_id'][:])
    reaches.hfalls_fid = np.array(data.groups['reaches'].variables['hfalls_id'][:])
    reaches.lakeflag = np.array(data.groups['reaches'].variables['lakeflag'][:])
    reaches.nchan_max = np.array(data.groups['reaches'].variables['n_chan_max'][:])
    reaches.nchan_mod = np.array(data.groups['reaches'].variables['n_chan_mod'][:])
    reaches.dist_out = np.array(data.groups['reaches'].variables['dist_out'][:])
    reaches.n_rch_up = np.array(data.groups['reaches'].variables['n_rch_up'][:])
    reaches.n_rch_down = np.array(data.groups['reaches'].variables['n_rch_down'][:])
    reaches.rch_id_up = np.array(data.groups['reaches'].variables['rch_id_up'][:])
    reaches.rch_id_down = np.array(data.groups['reaches'].variables['rch_id_dn'][:])
    reaches.max_obs = np.array(data.groups['reaches'].variables['swot_obs'][:])
    reaches.orbits = np.array(data.groups['reaches'].variables['swot_orbits'][:])
    reaches.facc = np.array(data.groups['reaches'].variables['facc'][:])
    reaches.iceflag = np.array(data.groups['reaches'].variables['iceflag'][:])
    reaches.max_wth = np.array(data.groups['reaches'].variables['max_width'][:])
    reaches.river_name = np.array(data.groups['reaches'].variables['river_name'][:])
    reaches.low_slope = np.array(data.groups['reaches'].variables['low_slope_flag'][:])
    reaches.edit_flag= np.array(data.groups['reaches'].variables['edit_flag'][:])
    reaches.trib_flag = np.array(data.groups['reaches'].variables['trib_flag'][:])

    data.close()    

    return centerlines, nodes, reaches

###############################################################################
###############################################################################
###############################################################################

start_all = time.time()

#read in netcdf data. 
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
sourceFile = open(outpath+'breaks_log_file.txt', 'w')
fn = outpath+'netcdf/'+region.lower()+'_sword_'+version+'.nc'
centerlines, nodes, reaches = read_data(fn)

no_dwn = reaches.id[np.where(reaches.n_rch_down == 0)[0]]
no_up = reaches.id[np.where(reaches.n_rch_up == 0)[0]]
breaks = np.append(no_up, no_dwn)

#spatial query with all centerline points...
sword_pts = np.vstack((centerlines.x, centerlines.y)).T
kdt = sp.cKDTree(sword_pts)
pt_dist, pt_ind = kdt.query(sword_pts, k = 6, distance_upper_bound=0.005)

#loop through break reaches
print(time.strftime("%d-%b-%Y %H:%M:%S"), file = sourceFile) 
for ind in list(range(len(breaks))):
    r = np.where(reaches.id == breaks[ind])[0]
    if '3' in reaches.edit_flag[r][0]:
        continue
    
    nghs1 = reaches.rch_id_up[:,r]
    nghs2 = reaches.rch_id_down[:,r]
    nghs = np.append(nghs1[np.where(nghs1 > 0)],nghs2[np.where(nghs2 > 0)])
    
    rch = np.where(centerlines.reach_id[0,:] == breaks[ind])[0]
    mn_id = rch[np.where(centerlines.cl_id[rch] == np.min(centerlines.cl_id[rch]))[0]]
    mx_id = rch[np.where(centerlines.cl_id[rch] == np.max(centerlines.cl_id[rch]))[0]]

    good_pts1 = np.where(pt_ind[mn_id,:] != 10236099)[1]
    good_pts2 = np.where(pt_ind[mx_id,:] != 10236099)[1]
    pt1 = np.where(np.unique(centerlines.reach_id[0,pt_ind[mn_id,good_pts1]]) != breaks[ind])[0]
    pt2 = np.where(np.unique(centerlines.reach_id[0,pt_ind[mx_id,good_pts2]]) != breaks[ind])[0]
    cl_nghs = np.unique(np.append(np.unique(centerlines.reach_id[0,pt_ind[mn_id,good_pts1]])[pt1],
                        np.unique(centerlines.reach_id[0,pt_ind[mx_id,good_pts2]])[pt2]))
    
    delete = []
    for n in list(range(len(cl_nghs))):
        if cl_nghs[n] in nghs:
            delete.append(n)
    cl_nghs = np.delete(cl_nghs, delete)
    # print(cl_nghs)

    if len(cl_nghs) > 0:
        print('correcting nghs for reach:', ind, breaks[ind], file = sourceFile)
        for c in list(range(len(cl_nghs))):
            rch_facc = reaches.facc[np.where(reaches.id == cl_nghs[c])]
            ngh_facc = reaches.facc[np.where(reaches.id == breaks[ind])]
            if ngh_facc > rch_facc:
                free = np.where(reaches.rch_id_down[:,r] == 0)[0]
                if len(free) == 0:
                    free = np.where(reaches.rch_id_up[:,r] == 0)[0]
                    reaches.rch_id_up[free[0],r] = cl_nghs[c]
                    reaches.n_rch_up[r] = len(np.where(reaches.rch_id_up[:,r]>0)[0])
                else:
                    reaches.rch_id_down[free[0],r] = cl_nghs[c]
                    reaches.n_rch_down[r] = len(np.where(reaches.rch_id_down[:,r]>0)[0])
            elif ngh_facc < rch_facc:
                free = np.where(reaches.rch_id_up[:,r] == 0)[0]
                if len(free) == 0:
                    free = np.where(reaches.rch_id_down[:,r] == 0)[0]
                    reaches.rch_id_down[free[0],r] = cl_nghs[c]
                    reaches.n_rch_down[r] = len(np.where(reaches.rch_id_down[:,r]>0)[0])
                else:
                    reaches.rch_id_up[free[0],r] = cl_nghs[c]
                    reaches.n_rch_up[r] = len(np.where(reaches.rch_id_up[:,r]>0)[0])
            else:
                r = np.where(reaches.id == breaks[ind])[0]
                free = np.where(reaches.rch_id_down[:,r] == 0)[0]
                if len(free) == 0:
                    free = np.where(reaches.rch_id_up[:,r] == 0)[0]
                    reaches.rch_id_up[free[0],r] = cl_nghs[c]
                    reaches.n_rch_up[r] = len(np.where(reaches.rch_id_up[:,r]>0)[0])
                else:
                    reaches.rch_id_down[free[0],r] = cl_nghs[c]
                    reaches.n_rch_down[r] = len(np.where(reaches.rch_id_down[:,r]>0)[0])


sword = nc.Dataset(fn, 'r+')
sword.groups['reaches'].variables['rch_id_up'][:] = reaches.rch_id_up
sword.groups['reaches'].variables['rch_id_dn'][:] = reaches.rch_id_down
sword.groups['reaches'].variables['n_rch_up'][:] = reaches.n_rch_up
sword.groups['reaches'].variables['n_rch_down'][:] = reaches.n_rch_down
sword.close()                    

# r = np.where(reaches.id == 74100400015)[0]
# reaches.rch_id_up[:,r]
# reaches.n_rch_up[r]
# reaches.rch_id_down[:,r]
# reaches.n_rch_down[r]

