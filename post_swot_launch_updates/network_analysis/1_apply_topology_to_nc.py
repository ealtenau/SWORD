import numpy as np
import netCDF4 as nc
import geopandas as gp
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("basin", help="<Required> Level Two Pfafstetter Basin (i.e. 74)", type = str)

args = parser.parse_args()

region = args.region
version = args.version
basin = args.basin

# region = 'NA'
# basin = '73'
# version = 'v17'

# rch_shp_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Topology/'+region+'/b'\
#     +basin+'/'+region.lower()+'_sword_reaches_hb'+basin+'_'+version+'_FG1_LSFix_MS_TopoFix_Acc.shp'
# csv_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Topology/'+region+'/b'\
#     +basin+'/'+region.lower()+'_sword_reaches_hb'+basin+'_'+version+'_rev_LS_MS.csv'
nc_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
    +version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
rch_shp_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
    +version+'/shp/'+region+'/'+region.lower()+'_sword_reaches_hb'+basin+'_'+version+'.shp'

# testing paths:
# rch_shp_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/gpkg copy/na_sword_reaches_v17_orig.gpkg'
# rch_shp_fn ='/Users/ealtenau/Documents/SWORD_Dev/update_requests/v17/SA/topo_fixes/sa_sword_reaches_hb62_v17_FG1_LSFix_MS_TopoFix_Manual.shp'
# csv_fn ='/Users/ealtenau/Documents/SWORD_Dev/update_requests/v17/NA/na_sword_reaches_hb82_v17_rev_LS_FI.csv'

rch_shp = gp.read_file(rch_shp_fn)
netcdf = nc.Dataset(nc_fn,'r+')
# csv = pd.read_csv(csv_fn)

# rev_rchs = np.array(csv['reach_id'])

reaches = np.array(rch_shp['reach_id'])
rch_id_up = np.array(rch_shp['rch_id_up'])
rch_id_dn = np.array(rch_shp['rch_id_dn'])
n_rch_up = np.array(rch_shp['n_rch_up'])
n_rch_dn = np.array(rch_shp['n_rch_dn'])
dist_out = np.array(rch_shp['dist_out'])
rch_len = np.array(rch_shp['reach_len'])
main_side = np.array(rch_shp['main_side'])
end_rch = np.array(rch_shp['end_reach'])

nc_rchs = np.array(netcdf.groups['reaches'].variables['reach_id'][:])
nc_rch_id_up = np.array(netcdf.groups['reaches'].variables['rch_id_up'][:])
nc_rch_id_dn = np.array(netcdf.groups['reaches'].variables['rch_id_dn'][:])
nc_n_rch_up = np.array(netcdf.groups['reaches'].variables['n_rch_up'][:])
nc_n_rch_dn = np.array(netcdf.groups['reaches'].variables['n_rch_down'][:])
nc_dist_out = np.array(netcdf.groups['reaches'].variables['dist_out'][:])
# nc_rch_ms = np.array(netcdf.groups['reaches'].variables['main_side'][:])
# nc_rch_er = np.array(netcdf.groups['reaches'].variables['end_reach'][:])
nc_cl_ids = np.array(netcdf.groups['centerlines'].variables['cl_id'][:])
nc_cl_rchs = np.array(netcdf.groups['centerlines'].variables['reach_id'][:])
nc_node_dist = np.array(netcdf.groups['nodes'].variables['dist_out'][:])
nc_node_len = np.array(netcdf.groups['nodes'].variables['node_length'][:])
nc_nodes = np.array(netcdf.groups['nodes'].variables['node_id'][:])
nc_node_rchs = np.array(netcdf.groups['nodes'].variables['reach_id'][:])
# nc_node_ms = np.array(netcdf.groups['nodes'].variables['main_side'][:])
# nc_node_er = np.array(netcdf.groups['nodes'].variables['end_reach'][:])

zero_out = np.where(np.in1d(nc_rchs, reaches) == True)[0]
cl_zero_out = np.where(np.in1d(nc_cl_rchs[0,:], reaches) == True)[0]
if len(zero_out) < len(reaches):
    diff = len(reaches)-len(zero_out)
    diff_rchs = reaches[np.where(np.in1d(reaches, nc_rchs) == False)[0]]
    print('!WARNING!: netcdf has ' + str(diff) + ' less reaches than shapefile')
    print(diff_rchs)
    sys.exit()

nc_rch_id_up[:,zero_out] = 0
nc_rch_id_dn[:,zero_out] = 0
nc_n_rch_up[zero_out] = 0
nc_n_rch_dn[zero_out] = 0
nc_cl_rchs[1::,cl_zero_out] = 0 
# len(np.unique(nc_cl_rchs[0,cl_zero_out]))
# len(np.unique(nc_cl_rchs[1,cl_zero_out]))
# len(np.unique(nc_cl_rchs[2,cl_zero_out]))
# len(np.unique(nc_cl_rchs[3,cl_zero_out]))

# print('reversing linestrings')
# for rev in list(range(len(rev_rchs))):
#     rch_main = np.where(nc_cl_rchs[0,:] == rev_rchs[rev])[0]
#     sort_inds = np.argsort(nc_cl_ids[rch_main])
#     nc_cl_ids[rch_main[sort_inds]] = nc_cl_ids[rch_main[sort_inds]][::-1]

print('updating topology')
for ind in list(range(len(reaches))):
    # print(ind, reaches[ind], len(reaches)-1)    
    nc_ind = np.where(nc_rchs == reaches[ind])[0]
    cl_ind = np.where(nc_cl_rchs[0,:] == reaches[ind])[0]
    cl_id_up = cl_ind[np.where(nc_cl_ids[cl_ind] == np.max(nc_cl_ids[cl_ind]))]
    cl_id_dn = cl_ind[np.where(nc_cl_ids[cl_ind] == np.min(nc_cl_ids[cl_ind]))]
    
    ###dist_out / main_side / end_reach
    #reaches
    nc_dist_out[nc_ind] = dist_out[ind]
    # nc_rch_ms[nc_ind] = main_side[ind]
    # nc_rch_er[nc_ind] = end_rch[ind]
    #nodes
    nds = np.where(nc_node_rchs == reaches[ind])[0]
    # nc_node_ms[nds] = main_side[ind]
    # min_id = np.where(nc_nodes[nds] == min(nc_nodes[nds]))[0]
    # nc_node_er[nds[min_id]] = end_rch[ind]
    base_val = dist_out[ind] - rch_len[ind]
    node_cs = np.cumsum(nc_node_len[nds])
    nc_node_dist[nds] = node_cs+base_val 

    ###upstream
    if n_rch_up[ind] == 1:
        nc_rch_id_up[0,nc_ind] = int(rch_id_up[ind])
        nc_n_rch_up[nc_ind] = n_rch_up[ind]
        nc_cl_rchs[1,cl_id_up] = int(rch_id_up[ind])
    if n_rch_up[ind] > 1:
        rup = np.array(rch_id_up[ind].split(),dtype=int)
        rup = rup.reshape(len(rup),1)
        nc_rch_id_up[0:len(rup),nc_ind] = rup
        nc_n_rch_up[nc_ind] = n_rch_up[ind]
        if n_rch_up[ind] > 3:
            nc_cl_rchs[1:4,cl_id_up] = rup[0:3]
        else:
            nc_cl_rchs[1:len(rup)+1,cl_id_up] = rup #nc_cl_rchs[:,cl_id_up]
    ###downstream
    if n_rch_dn[ind] == 1:
        nc_rch_id_dn[0,nc_ind] = int(rch_id_dn[ind])
        nc_n_rch_dn[nc_ind] = n_rch_dn[ind]
        nc_cl_rchs[1,cl_id_dn] = int(rch_id_dn[ind])
    if n_rch_dn[ind] > 1:
        rdn = np.array(rch_id_dn[ind].split(),dtype=int)
        rdn = rdn.reshape(len(rdn),1)
        nc_rch_id_dn[0:len(rdn),nc_ind] = rdn
        nc_n_rch_dn[nc_ind] = n_rch_dn[ind]
        if n_rch_dn[ind] > 3:
            nc_cl_rchs[1:4,cl_id_dn] = rdn[0:3]
        else:
            nc_cl_rchs[1:len(rdn)+1,cl_id_dn] = rdn #nc_cl_rchs[:,cl_id_dn]
    
### update netcdf
netcdf.groups['reaches'].variables['rch_id_up'][:] = nc_rch_id_up
netcdf.groups['reaches'].variables['rch_id_dn'][:] = nc_rch_id_dn
netcdf.groups['reaches'].variables['n_rch_up'][:] = nc_n_rch_up
netcdf.groups['reaches'].variables['n_rch_down'][:] = nc_n_rch_dn
netcdf.groups['reaches'].variables['dist_out'][:] = nc_dist_out
netcdf.groups['nodes'].variables['dist_out'][:] = nc_node_dist
netcdf.groups['centerlines'].variables['cl_id'][:] = nc_cl_ids
netcdf.groups['centerlines'].variables['reach_id'][:] = nc_cl_rchs
### not updated at present... 
# netcdf.groups['reaches'].variables['main_side'][:] = nc_rch_ms
# netcdf.groups['reaches'].variables['end_reach'][:] = nc_rch_er
# netcdf.groups['nodes'].variables['main_side'][:] = nc_node_ms
# netcdf.groups['nodes'].variables['end_reach'][:] = nc_node_er
netcdf.close()
print('DONE')


