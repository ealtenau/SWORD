import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy import spatial as sp

###############################################################################
################################# Functions ###################################
###############################################################################

class Object(object):
    """
    FUNCTION:
        Creates class object to assign attributes to.
    """
    pass

###############################################################################
###############################################################################
###############################################################################

region = 'OC'
fn_merge_v13 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Merged_Data/v13/'+region+'_Merge_v13.nc'
fn_sword_v16 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf/'+region.lower()+'_sword_v16.nc'
rch_changes_fn = '/Users/ealteanau/Documents/SWORD_Dev/update_requests/v16/'+region+'_rch_id_changes.csv'
m13 = nc.Dataset(fn_merge_v13)
s16 = nc.Dataset(fn_sword_v16)
new_rchs = pd.read_csv(rch_changes_fn)

merge = Object()
merge.x = np.array(m13.groups['centerlines'].variables['x'][:])
merge.y = np.array(m13.groups['centerlines'].variables['y'][:])
merge.basins = np.array(m13.groups['centerlines'].variables['basin_code'][:])
m13.close()

sword = Object()
sword.clx = np.array(s16.groups['centerlines'].variables['x'][:])
sword.cly = np.array(s16.groups['centerlines'].variables['y'][:])
sword.cl_node_id = np.array(s16.groups['centerlines'].variables['node_id'][0,:])
sword.cl_reach_id = np.array(s16.groups['centerlines'].variables['reach_id'][0,:])
s16.close()

s_pts = np.vstack((sword.clx, sword.cly)).T
m_pts = np.vstack((merge.x, merge.y)).T
kdt = sp.cKDTree(m_pts)
eps_dist, eps_ind = kdt.query(s_pts, k = 1)

mb = merge.basins[eps_ind]
sb = np.array([int(str(ind)[0:6]) for ind in sword.cl_reach_id])

# len(np.where(sb != mb)[0])/len(sb)*100
unq_rch = np.array(new_rchs['old_rch_id'])
nsb = np.zeros(len(sb), dtype=int)
for r in list(range(len(unq_rch))):
    print(r, len(unq_rch)-1)
    rch = np.where(sword.cl_reach_id == unq_rch[r])[0]
    if len(rch) == 0:
        continue
    else:
        nsb[rch] = np.repeat(int(str(new_rchs['new_rch_id'][r])[0:6]), len(rch))
    
zeros = np.where(nsb == 0)[0]
nsb[zeros] = sb[zeros]

w1 = np.where(sb != mb)[0]
w2 = np.where(nsb != mb)[0]


perc1 = (len(w1)/len(sb))*100
perc2 = (len(w2)/len(sb))*100
print("Done with " + region + ", improvement: " + str(np.round(perc1,2)) + '% to ' + str(np.round(perc2,2)) + '%')