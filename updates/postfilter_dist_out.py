from __future__ import division
import numpy as np
from scipy import spatial as sp
import netCDF4 as nc
import matplotlib.pyplot as plt
import argparse 

###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass 

###############################################################################

def filter_basin_dist(subreaches):

    new_dist = np.copy(subreaches.dist)
    # start_rch = subreaches.id[np.where(subreaches.dist == np.min(subreaches.dist))[0]][0]
    start_rch = subreaches.id[np.where(subreaches.facc == np.max(subreaches.facc))[0]][0]
    checked = np.zeros(len(subreaches.id))
    loop = 1
    while np.min(checked) < 1:

        # print(loop, int(start_rch))
        if loop > len(subreaches.id) + 500:
            print('LOOP STUCK')

        # if start_rch == 74262300601: #74299700031
        #     break

        rch = np.where(subreaches.id == start_rch)[0]
        rch_pt = np.vstack((subreaches.x[rch], subreaches.y[rch])).T
        pts = np.vstack((subreaches.x, subreaches.y)).T
        kdt = sp.cKDTree(pts)
        eps_dist, eps_ind = kdt.query(rch_pt, k = 20) 
        keep = np.where(eps_dist < 0.3)[1] #keeping neighbors within ~20 km. 
        keep = keep[1::] #need to remove first index because it is the current reach. 
        all_nghs = np.unique(subreaches.id[eps_ind[:,keep]])

        if len(all_nghs) == 0:
            checked[rch] = 1
            z = np.where(checked < 1)[0]
            if len(z) > 0:
                loop = loop+1
                # start_rch = subreaches.id[z[np.where(new_dist[z] == np.min(new_dist[z]))[0]]][0]
                start_rch = subreaches.id[z[np.where(subreaches.facc[z] == np.max(subreaches.facc[z]))[0]]][0]
            else:
                loop = loop+1
                continue
        else:
            ngh_dist = np.array([new_dist[np.where(subreaches.id == n)[0]] for n in all_nghs])
            diff = new_dist[rch] - ngh_dist
            outliers = np.where(diff < -1000000)[0]

            if len(outliers) == 0:
                checked[rch] = 1
                z = np.where(checked < 1)[0]
                if len(z) > 0:
                    loop = loop+1
                    # start_rch = subreaches.id[z[np.where(new_dist[z] == np.min(new_dist[z]))[0]]][0]
                    start_rch = subreaches.id[z[np.where(subreaches.facc[z] == np.max(subreaches.facc[z]))[0]]][0]
                else:
                    loop = loop+1
                    continue 
            
            else:
                max_dist = np.max(ngh_dist[outliers])
                new_dist[rch] = subreaches.len[rch] + max_dist
                checked[rch] = 1
                
                nup = subreaches.ngh_up[np.where(subreaches.ngh_up[:,rch] > 0)[0],rch]
                checked_vals = np.array([checked[np.where(subreaches.id == d)[0]] for d in nup])
                nup_final = nup[np.where(checked_vals == 0)[0]]
                
                if len(nup_final) > 0:
                    loop = loop+1
                    start_rch = nup_final[0]
                else:
                    z = np.where(checked < 1)[0]
                    if len(z) > 0:
                        loop = loop+1
                        # start_rch = subreaches.id[z[np.where(new_dist[z] == np.min(new_dist[z]))[0]]][0]
                        start_rch = subreaches.id[z[np.where(subreaches.facc[z] == np.max(subreaches.facc[z]))[0]]][0]
                    else:
                        loop = loop+1
                        continue           

    return new_dist

###############################################################################

def calc_node_dist_out(subreaches, subnodes):

    node_dist_out = np.unique(subnodes.id)
    uniq_rchs = np.unique(subreaches.id)
    for ind in list(range(len(uniq_rchs))):
        rch = np.where(subreaches.id == uniq_rchs[ind])[0]
        node_locs = np.where(subnodes.reach_id == uniq_rchs[ind])[0]
        node_order = np.argsort(subnodes.id[node_locs])
        node_cdist = np.cumsum(subnodes.len[node_locs[node_order]])

        rch_len = subreaches.len[rch]
        rch_dist_out = subreaches.new_dist_out[rch]
        d = rch_dist_out - rch_len

        node_vals = node_cdist+d
        node_dist_out[node_locs[node_order]] = node_vals

    return node_dist_out

###############################################################################
###############################################################################
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
args = parser.parse_args()

region = args.region
fn_nc = '/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/outputs/Reaches_Nodes/v16/netcdf/'+region.lower()+'_sword_v16.nc'
# version = 'v16'
# fn_nc = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'+region.lower()+'_sword_'+version+'.nc'

sword = nc.Dataset(fn_nc, 'r+')
reaches = sword.groups['reaches'].variables['reach_id'][:]
dist_out = sword.groups['reaches'].variables['dist_out'][:]
rch_len = sword.groups['reaches'].variables['reach_length'][:]
ngh_up = sword.groups['reaches'].variables['rch_id_up'][:]
ngh_dn = sword.groups['reaches'].variables['rch_id_dn'][:]
rch_x = sword.groups['reaches'].variables['x'][:]
rch_y = sword.groups['reaches'].variables['y'][:]
facc = sword.groups['reaches'].variables['facc'][:]

nodes = sword.groups['nodes'].variables['node_id'][:]
node_len = sword.groups['nodes'].variables['node_length'][:]
node_dist = sword.groups['nodes'].variables['dist_out'][:]
node_rchs = sword.groups['nodes'].variables['reach_id'][:]

nlevel2 = np.array([int(str(ind)[0:2]) for ind in nodes])
level2 = np.array([int(str(ind)[0:2]) for ind in reaches])
unq_l2 = np.unique(level2)

#### loop 
reach_dist = np.zeros(len(reaches))
node_dist = np.zeros(len(nodes))
for ind in list(range(len(unq_l2))):
    print('Starting Basin: ' + str(unq_l2[ind]))
    l2 = np.where(level2 == unq_l2[ind])[0]
    subreaches = Object()
    subreaches.x = rch_x[l2]
    subreaches.y = rch_y[l2]
    subreaches.id = reaches[l2]
    subreaches.dist = dist_out[l2]
    subreaches.len = rch_len[l2]
    subreaches.facc = facc[l2]
    subreaches.ngh_up = ngh_up[:,l2]
    subreaches.ngh_dn = ngh_dn[:,l2]

    subnodes = Object()
    nl2 = np.where(nlevel2 == unq_l2[ind])
    subnodes.id = nodes[nl2]
    subnodes.reach_id = node_rchs[nl2]
    subnodes.len = node_len[nl2]

    subreaches.new_dist_out = filter_basin_dist(subreaches)
    subnodes.new_dist_out = calc_node_dist_out(subreaches, subnodes)
    reach_dist[l2] = subreaches.new_dist_out
    node_dist[nl2] = subnodes.new_dist_out

#replace in netcdf 
sword.groups['reaches'].variables['dist_out'][:] = reach_dist
sword.groups['nodes'].variables['dist_out'][:] = node_dist 
sword.close()
print('DONE')

'''
plt.figure(1)
plt.scatter(subreaches.x, subreaches.y, c=subreaches.dist, s = 3)
plt.show()

plt.figure(2)
plt.scatter(subreaches.x, subreaches.y, c=new_dist_out, s = 3)
plt.show()


node_x = sword.groups['nodes'].variables['x'][:]
node_y = sword.groups['nodes'].variables['y'][:]
plt.figure(3)
plt.scatter(node_x[nl2], node_y[nl2], c=subnodes.new_dist_out, s = 3)
plt.show()

plt.scatter(rch_x, rch_y, c=reach_dist, s = 3)
plt.show()

plt.scatter(subreaches.x, subreaches.y, c=new_dist, s = 3)
plt.show()


test = np.where(subreaches.id == 74262300601)[0]
subreaches.dist[test]
new_dist_out[test]
plt.scatter(subreaches.x[test], subreaches.y[test], c='red', s = 8)

'''
