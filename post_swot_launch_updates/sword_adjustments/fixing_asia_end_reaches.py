import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

region = 'AS'
version = 'v17'

sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
    +version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'
con_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
    +version+'/reach_geometry/'+region.lower()+'_sword_'+version+'_connectivity.nc'

sword = nc.Dataset(sword_dir,'r+')
conn = nc.Dataset(con_dir)

reaches = np.array(sword.groups['reaches'].variables['reach_id'][:])
rch_x = np.array(sword.groups['reaches'].variables['x'][:])
rch_y = np.array(sword.groups['reaches'].variables['y'][:])
node_rchs = np.array(sword.groups['nodes'].variables['reach_id'][:])
nodes = np.array(sword.groups['nodes'].variables['node_id'][:])
node_dist = np.array(sword.groups['nodes'].variables['dist_out'][:])
node_path_order = np.array(sword.groups['nodes'].variables['path_order'][:])

con_cl_ids = np.array(conn.groups['centerlines'].variables['cl_id'][:])
con_rch_ids = np.array(conn.groups['centerlines'].variables['reach_id'][:])
con_node_ids = np.array(conn.groups['centerlines'].variables['node_id'][:])
con_end_ids = np.array(conn.groups['centerlines'].variables['end_reach'][:])
conn.close()


print('Adding End Reach Attribute to Reaches and Nodes') #basin-scale
rch_hw = con_rch_ids[0,np.where(con_end_ids == 1)[0]]
rch_out = con_rch_ids[0,np.where(con_end_ids == 2)[0]]
rch_junc = con_rch_ids[0,np.where(con_end_ids == 3)[0]]
node_hw = con_node_ids[np.where(con_end_ids == 1)[0]]
node_out = con_node_ids[np.where(con_end_ids == 2)[0]]
node_junc = con_node_ids[np.where(con_end_ids == 3)[0]]
rch_ends = np.zeros(len(reaches))
node_ends = np.zeros(len(nodes))
rch_ends[np.where(np.in1d(reaches, rch_junc))] = 3
rch_ends[np.where(np.in1d(reaches, rch_hw))] = 1
rch_ends[np.where(np.in1d(reaches, rch_out))] = 2
node_ends[np.where(np.in1d(nodes, node_junc))] = 3
node_ends[np.where(np.in1d(nodes, node_hw))] = 1
node_ends[np.where(np.in1d(nodes, node_out))] = 2

print('Updating End Reach Values')
node_l2 = np.array([int(str(n)[0:2]) for n in nodes])
unq_l2 = np.unique(node_l2)
ends = nodes[np.where((node_ends > 0)&(node_ends<3))[0]]
for idx in list(range(len(unq_l2))):
    print('Basin:', unq_l2[idx])
    nl2 = np.where(node_l2 == unq_l2[idx])[0]
    subnodes = nodes[nl2]
    subends = node_ends[nl2]
    subndist = node_dist[nl2]
    subnpaths = node_path_order[nl2]
    subnode_rchs = node_rchs[nl2]
    for e in list(range(len(ends))):
        pt = np.where(subnodes == ends[e])[0]
        if len(pt) == 0:
            continue
        rch_pt = np.where(reaches == subnode_rchs[np.where(subnodes == ends[e])[0]])[0]
        end_val = subends[pt]
        pt_dist = subndist[pt]
        pth = subnpaths[pt]
        pth_mx_dist = max(subndist[np.where(subnpaths == pth)])
        pth_mn_dist = min(subndist[np.where(subnpaths == pth)])
        if end_val == 1 and pt_dist == pth_mn_dist:
            # print('headwater to outlet: ', ends[e])
            node_ends[nl2[pt]] = 2
            rch_ends[rch_pt] = 2
        if end_val == 2 and pt_dist == pth_mx_dist:
            # print('outlet to headwater: ', ends[e])
            node_ends[nl2[pt]] = 1
            rch_ends[rch_pt] = 1



sword.groups['reaches'].variables['end_reach'][:] = rch_ends
sword.groups['nodes'].variables['end_reach'][:] = node_ends
sword.close()

# hw = np.where(rch_ends == 1)[0]
# out = np.where(rch_ends == 2)[0]
# plt.scatter(rch_x, rch_y, c = 'grey', s = 4)
# plt.scatter(rch_x[hw], rch_y[hw], c = 'magenta', s = 8)
# plt.scatter(rch_x[out], rch_y[out], c = 'blue', s = 8)
# plt.show()