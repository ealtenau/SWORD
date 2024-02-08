from __future__ import division
import numpy as np
import time
import netCDF4 as nc
from scipy import spatial as sp
import utm 
import argparse

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

def format_cl_node_ids_pt2(nodes, centerlines):

    """
    FUNCTION:
        Assigns and formats the node ids along high_resolution centerline
        at the continental scale. This is a subfunction of "format_cl_node_ids"
        in order to speed up the code.

    INPUTS
        nodes -- Object containing attributes for the nodes.
        centerlines -- Object containing attributes for the high-resolution centerline.

    OUTPUTS
        cl_nodes_id - Node IDs for each high-resolution centerline point.
    """

    nodes_x = np.zeros(len(nodes.id))
    nodes_y = np.zeros(len(nodes.id))
    for idx in list(range(len(nodes.id))):
        nodes_x[idx], nodes_y[idx], __ , __ = utm.from_latlon(nodes.y[idx], nodes.x[idx])

    all_pts = np.vstack((nodes_x, nodes_y)).T
    kdt = sp.cKDTree(all_pts)

    cl_nodes_id= np.zeros([4,len(centerlines.cl_id)])
    for ind in list(range(len(nodes.id))):
        #print(ind)

        # converting coordinates for centerlines points.
        cp1 = np.where(centerlines.cl_id == nodes.cl_id[0,ind])[0]
        cp2 = np.where(centerlines.cl_id == nodes.cl_id[1,ind])[0]
        cp1_x, cp1_y, __, __ = utm.from_latlon(centerlines.y[cp1], centerlines.x[cp1])
        cp2_x, cp2_y, __, __ = utm.from_latlon(centerlines.y[cp2], centerlines.x[cp2])

        cp1_pts = np.vstack((cp1_x, cp1_y)).T
        cp1_dist, cp1_ind = kdt.query(cp1_pts, k = 4, distance_upper_bound = 200)
        cp2_pts = np.vstack((cp2_x, cp2_y)).T
        cp2_dist, cp2_ind = kdt.query(cp2_pts, k = 4, distance_upper_bound = 200)

        rmv1 = np.where(cp1_ind[0] == len(nodes.id))[0]
        rmv2 = np.where(cp2_ind[0] == len(nodes.id))[0]
        cp1_ind = np.delete(cp1_ind[0], rmv1)
        cp2_ind = np.delete(cp2_ind[0], rmv2)

        cp1_nodes = nodes.id[np.where(nodes.id[cp1_ind] != nodes.id[ind])[0]]
        cp2_nodes = nodes.id[np.where(nodes.id[cp2_ind] != nodes.id[ind])[0]]

        cl_nodes_id[0:len(cp1_nodes),cp1] = cp1_nodes.reshape((len(cp1_nodes),1))
        cl_nodes_id[0:len(cp2_nodes),cp2] = cp2_nodes.reshape((len(cp2_nodes),1))

    return cl_nodes_id

###############################################################################

def format_cl_node_ids(nodes, centerlines, verbose):

    """
    FUNCTION:
        Assigns and formats the node ids along high_resolution centerline
        at the continental scale.

    INPUTS
        nodes -- Object containing attributes for the nodes.
        centerlines -- Object containing attributes for the high-resolution centerline.

    OUTPUTS
        cl_nodes_id - Node IDs for each high-resolution centerline point.
    """

    cl_nodes_id = np.zeros([4,len(centerlines.cl_id)])
    # divide up into basin level 6 to see if it makes things faster...
    level_nodes = np.array([int(str(ind)[0:6]) for ind in nodes.id])
    level_cl = np.array([int(str(ind)[0:6]) for ind in centerlines.node_id[0,:]])
    uniq_level = np.unique(level_cl)
    for ind in list(range(len(uniq_level))):
        if verbose == True:
            print(str(ind) + ' of ' + str(len(uniq_level)))

        cl_ind = np.where(level_cl == uniq_level[ind])[0]
        nodes_ind = np.where(level_nodes == uniq_level[ind])[0]

        Subnodes = Object()
        Subcls = Object()

        Subnodes.id = nodes.id[nodes_ind]
        Subnodes.x = nodes.x[nodes_ind]
        Subnodes.y = nodes.y[nodes_ind]
        Subnodes.cl_id = nodes.cl_id[:,nodes_ind]
        Subcls.cl_id = centerlines.cl_id[cl_ind]
        Subcls.x = centerlines.x[cl_ind]
        Subcls.y = centerlines.y[cl_ind]

        cl_nodes_id[:,cl_ind] = format_cl_node_ids_pt2(Subnodes, Subcls)

    return cl_nodes_id

################################################################################

def format_cl_rch_ids(reaches, centerlines, verbose):

    """
    FUNCTION:
        Assigns and formats the reach ids along high_resolution centerline
        at the continental scale.

    INPUTS
        reaches -- Object containing attributes for the reaches.
        centerlines -- Object containing attributes for the high-resolution centerline.

    OUTPUTS
        cl_rch_id - Reach IDs for each high-resolution centerline point.
    """

    cl_rch_id = np.zeros([4,len(centerlines.cl_id)])
    for ind in list(range(len(reaches.id))):
        if verbose == True:
            print(str(ind) + ' of ' + str(len(reaches.id)))
        up = np.where(reaches.rch_id_up[:,ind] > 0)[0]
        down = np.where(reaches.rch_id_down[:,ind] > 0)[0]

        # converting coordinates for centerlines points.
        cp1 = np.where(centerlines.cl_id == reaches.cl_id[0,ind])[0]
        cp2 = np.where(centerlines.cl_id == reaches.cl_id[1,ind])[0]

        if len(cp1) > 1:
            cp1 = np.array([cp1[0]])
        if len(cp2) > 1:
            cp2 = np.array([cp2[0]])
        
        cp1_x, cp1_y, __, __ = utm.from_latlon(centerlines.y[cp1], centerlines.x[cp1])
        cp2_x, cp2_y, __, __ = utm.from_latlon(centerlines.y[cp2], centerlines.x[cp2])
        
        vals_up = reaches.rch_id_up[0:4,ind]
        vals_down = reaches.rch_id_down[0:4,ind]

        if len(up) == 0 and len(down) == 0:
            continue

        if len(up) > 0 and len(down) == 0:

            up_lon = reaches.x[np.where(reaches.id == reaches.rch_id_up[:,ind][up[0]])]
            up_lat = reaches.y[np.where(reaches.id == reaches.rch_id_up[:,ind][up[0]])]
            if len(up_lon) == 0 or len(up_lat) == 0:
                continue
            else:
                up_x, up_y, __, __ = utm.from_latlon(up_lat, up_lon)

            d1 = np.sqrt(((cp1_x - up_x)**2 + (cp1_y - up_y)**2))
            d2 = np.sqrt(((cp2_x - up_x)**2 + (cp2_y - up_y)**2))

            if d1 > d2:
                cl_rch_id[:,cp2] = vals_up.reshape((4,1))
            if d1 < d2:
                cl_rch_id[:,cp1] = vals_up.reshape((4,1))

        if len(up) == 0 and len(down) > 0:
            dn_lon = reaches.x[np.where(reaches.id == reaches.rch_id_down[:,ind][down[0]])]
            dn_lat = reaches.y[np.where(reaches.id == reaches.rch_id_down[:,ind][down[0]])]
            if len(dn_lon) == 0 or len(dn_lat) == 0:
                continue
            else:
                dn_x, dn_y, __, __ = utm.from_latlon(dn_lat, dn_lon)

            d1 = np.sqrt(((cp1_x - dn_x)**2 + (cp1_y - dn_y)**2))
            d2 = np.sqrt(((cp2_x - dn_x)**2 + (cp2_y - dn_y)**2))

            if d1 > d2:
                cl_rch_id[:,cp2] = vals_down.reshape((4,1))
            if d1 < d2:
                cl_rch_id[:,cp1] = vals_down.reshape((4,1))

        if len(up) > 0 and len(down) > 0:

            up_lon = reaches.x[np.where(reaches.id == reaches.rch_id_up[:,ind][up[0]])] #changed 0 to ind in rch_id_up rows.
            up_lat = reaches.y[np.where(reaches.id == reaches.rch_id_up[:,ind][up[0]])]
            if len(up_lon) == 0 or len(up_lat) == 0:
                continue
            else:
                up_x, up_y, __, __ = utm.from_latlon(up_lat, up_lon)

            dn_lon = reaches.x[np.where(reaches.id == reaches.rch_id_down[:,ind][down[0]])]
            dn_lat = reaches.y[np.where(reaches.id == reaches.rch_id_down[:,ind][down[0]])]
            if len(dn_lon) == 0 or len(dn_lat) == 0:
                continue
            else:
                dn_x, dn_y, __, __ = utm.from_latlon(dn_lat, dn_lon)

            d1 = np.sqrt(((cp1_x - up_x)**2 + (cp1_y - up_y)**2))
            d2 = np.sqrt(((cp1_x - dn_x)**2 + (cp1_y - dn_y)**2))

            #### need to check!!!!!!!
            if d1 > d2:
                cl_rch_id[:,cp1] = vals_down.reshape((4,1))
                cl_rch_id[:,cp2] = vals_up.reshape((4,1))
            if d1 < d2:
                cl_rch_id[:,cp1] = vals_up.reshape((4,1))
                cl_rch_id[:,cp2] = vals_down.reshape((4,1))

    return cl_rch_id

###############################################################################
###############################################################################
###############################################################################

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
sword_dir = outpath+'netcdf/'+region.lower()+'_sword_'+version+'.nc'
centerlines, nodes, reaches = read_data(sword_dir)

#redo centerline ids for nodes and reaches. 
# cl_nodes_id = format_cl_node_ids(nodes, centerlines, verbose = True)
cl_rch_id = format_cl_rch_ids(reaches, centerlines, verbose = True)
centerlines.reach_id = np.insert(cl_rch_id, 0, centerlines.reach_id[0,:], axis = 0)
centerlines.reach_id = centerlines.reach_id[0:4,:]
# centerlines.node_id =  np.insert(cl_nodes_id, 0, centerlines.node_id[0,:], axis = 0)
# centerlines.node_id = centerlines.node_id[0:4,:]

sword = nc.Dataset(sword_dir, 'r+')
sword.groups['centerlines'].variables['reach_id'][:] = centerlines.reach_id[:]
# sword.groups['centerlines'].variables['node_id'][:] = centerlines.node_id[:]
sword.close()

end_all = time.time()
print('Done Updating Centerline IDs in: ' + str(np.round((end_all-start_all)/60, 2)) + ' min')
