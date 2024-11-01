import netCDF4 as nc
import pandas as pd
import numpy as np

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
    
    centerlines.cl_id = data.groups['centerlines'].variables['cl_id'][:]
    centerlines.x = data.groups['centerlines'].variables['x'][:]
    centerlines.y = data.groups['centerlines'].variables['y'][:]
    centerlines.reach_id = data.groups['centerlines'].variables['reach_id'][:]
    centerlines.node_id = data.groups['centerlines'].variables['node_id'][:]
    
    nodes.id = data.groups['nodes'].variables['node_id'][:]
    nodes.cl_id = data.groups['nodes'].variables['cl_ids'][:]
    nodes.x = data.groups['nodes'].variables['x'][:]
    nodes.y = data.groups['nodes'].variables['y'][:]
    nodes.len = data.groups['nodes'].variables['node_length'][:]
    nodes.wse = data.groups['nodes'].variables['wse'][:]
    nodes.wse_var = data.groups['nodes'].variables['wse_var'][:]
    nodes.wth = data.groups['nodes'].variables['width'][:]
    nodes.wth_var = data.groups['nodes'].variables['width_var'][:]
    nodes.grod = data.groups['nodes'].variables['obstr_type'][:]
    nodes.grod_fid = data.groups['nodes'].variables['grod_id'][:]
    nodes.hfalls_fid = data.groups['nodes'].variables['hfalls_id'][:]
    nodes.nchan_max = data.groups['nodes'].variables['n_chan_max'][:]
    nodes.nchan_mod = data.groups['nodes'].variables['n_chan_mod'][:]
    nodes.dist_out = data.groups['nodes'].variables['dist_out'][:]
    nodes.reach_id = data.groups['nodes'].variables['reach_id'][:]
    nodes.facc = data.groups['nodes'].variables['facc'][:]
    nodes.lakeflag = data.groups['nodes'].variables['lakeflag'][:]
    nodes.wth_coef = data.groups['nodes'].variables['wth_coef'][:]
    nodes.ext_dist_coef = data.groups['nodes'].variables['ext_dist_coef'][:]
    nodes.max_wth = data.groups['nodes'].variables['max_width'][:]
    nodes.meand_len = data.groups['nodes'].variables['meander_length'][:]
    nodes.river_name = data.groups['nodes'].variables['river_name'][:]
    nodes.manual_add = data.groups['nodes'].variables['manual_add'][:]
    nodes.sinuosity = data.groups['nodes'].variables['sinuosity'][:]
    nodes.edit_flag = data.groups['nodes'].variables['edit_flag'][:]
    nodes.trib_flag = data.groups['nodes'].variables['trib_flag'][:]
    nodes.path_freq = data.groups['nodes'].variables['path_freq'][:]
    nodes.path_order = data.groups['nodes'].variables['path_order'][:]
    nodes.path_segs = data.groups['nodes'].variables['path_segs'][:]
    nodes.strm_order = data.groups['nodes'].variables['stream_order'][:]
    nodes.main_side = data.groups['nodes'].variables['main_side'][:]
    nodes.end_rch = data.groups['nodes'].variables['end_reach'][:]
    nodes.network = data.groups['nodes'].variables['network'][:]

    reaches.id = data.groups['reaches'].variables['reach_id'][:]
    reaches.cl_id = data.groups['reaches'].variables['cl_ids'][:]
    reaches.x = data.groups['reaches'].variables['x'][:]
    reaches.x_min = data.groups['reaches'].variables['x_min'][:]
    reaches.x_max = data.groups['reaches'].variables['x_max'][:]
    reaches.y = data.groups['reaches'].variables['y'][:]
    reaches.y_min = data.groups['reaches'].variables['y_min'][:]
    reaches.y_max = data.groups['reaches'].variables['y_max'][:]
    reaches.len = data.groups['reaches'].variables['reach_length'][:]
    reaches.wse = data.groups['reaches'].variables['wse'][:]
    reaches.wse_var = data.groups['reaches'].variables['wse_var'][:]
    reaches.wth = data.groups['reaches'].variables['width'][:]
    reaches.wth_var = data.groups['reaches'].variables['width_var'][:]
    reaches.slope = data.groups['reaches'].variables['slope'][:]
    reaches.rch_n_nodes = data.groups['reaches'].variables['n_nodes'][:]
    reaches.grod = data.groups['reaches'].variables['obstr_type'][:]
    reaches.grod_fid = data.groups['reaches'].variables['grod_id'][:]
    reaches.hfalls_fid = data.groups['reaches'].variables['hfalls_id'][:]
    reaches.lakeflag = data.groups['reaches'].variables['lakeflag'][:]
    reaches.nchan_max = data.groups['reaches'].variables['n_chan_max'][:]
    reaches.nchan_mod = data.groups['reaches'].variables['n_chan_mod'][:]
    reaches.dist_out = data.groups['reaches'].variables['dist_out'][:]
    reaches.n_rch_up = data.groups['reaches'].variables['n_rch_up'][:]
    reaches.n_rch_down = data.groups['reaches'].variables['n_rch_down'][:]
    reaches.rch_id_up = data.groups['reaches'].variables['rch_id_up'][:]
    reaches.rch_id_down = data.groups['reaches'].variables['rch_id_dn'][:]
    reaches.max_obs = data.groups['reaches'].variables['swot_obs'][:]
    reaches.orbits = data.groups['reaches'].variables['swot_orbits'][:]
    reaches.facc = data.groups['reaches'].variables['facc'][:]
    reaches.iceflag = data.groups['reaches'].variables['iceflag'][:]
    reaches.max_wth = data.groups['reaches'].variables['max_width'][:]
    reaches.river_name = data.groups['reaches'].variables['river_name'][:]
    reaches.low_slope = data.groups['reaches'].variables['low_slope_flag'][:]
    reaches.edit_flag= data.groups['reaches'].variables['edit_flag'][:]
    reaches.trib_flag = data.groups['reaches'].variables['trib_flag'][:]
    reaches.path_freq = data.groups['reaches'].variables['path_freq'][:]
    reaches.path_order = data.groups['reaches'].variables['path_order'][:]
    reaches.path_segs = data.groups['reaches'].variables['path_segs'][:]
    reaches.strm_order = data.groups['reaches'].variables['stream_order'][:]
    reaches.main_side = data.groups['reaches'].variables['main_side'][:]
    reaches.end_rch = data.groups['reaches'].variables['end_reach'][:]
    reaches.network = data.groups['reaches'].variables['network'][:]

    data.close()    

    return centerlines, nodes, reaches
    
###############################################################################    
###############################################################################
###############################################################################

region = 'OC'
version = 'v17'
sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
out_dir = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/'+version+'/'+region+'/'
centerlines, nodes, reaches = read_data(sword_dir)

reaches.type = np.array([int(str(rch)[-1]) for rch in reaches.id])
correct = np.where((reaches.n_rch_up > 0)&(reaches.n_rch_down > 0)&(reaches.type == 6))[0]
missing_ghost_headwater = np.where((reaches.n_rch_up == 0)&(reaches.type < 6))[0]
missing_ghost_outlet = np.where((reaches.n_rch_down == 0)&(reaches.type < 6))[0]
all_missing = np.append(missing_ghost_headwater,missing_ghost_outlet)

hw_end = np.repeat(1,len(missing_ghost_headwater))
out_end = np.repeat(2,len(missing_ghost_outlet))
all_ends = np.append(hw_end,out_end)

subreaches = reaches.id[correct]
new_type = []
for r in list(range(len(subreaches))):
    # print(r)
    rch = np.where(reaches.id == subreaches[r])[0]
    up_type = reaches.type[np.where(np.in1d(reaches.id, reaches.rch_id_up[:,rch])==True)[0]]
    dn_type = reaches.type[np.where(np.in1d(reaches.id, reaches.rch_id_down[:,rch])==True)[0]]
    all_types = np.append(up_type,dn_type)
    new_type.append(max(all_types[np.where(all_types<6)[0]]))

ghost = {'reach_id': np.array(reaches.id[correct]).astype('int64'), 'new_type': np.array(new_type).astype('int64')}
ghost = pd.DataFrame(ghost)
ends = {'reach_id': np.array(reaches.id[all_missing]).astype('int64'), 'hw_out': np.array(all_ends).astype('int64')}
ends = pd.DataFrame(ends)

ghost.to_csv(out_dir+region.lower()+'_incorrect_ghost_reaches.csv', index=False)
ends.to_csv(out_dir+region.lower()+'_missing_ghost_reaches.csv', index=False)
print(len(ghost), len(ends))
print('DONE')