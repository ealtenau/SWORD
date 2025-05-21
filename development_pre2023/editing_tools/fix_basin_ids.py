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

region = 'NA'
fn_merge_v13 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Merged_Data/v13/'+region+'_Merge_v13.nc'
fn_sword_v16 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf/'+region.lower()+'_sword_v16.nc'

m13 = nc.Dataset(fn_merge_v13)
s16 = nc.Dataset(fn_sword_v16)

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
wrong_rchs = np.where(sb != mb)[0]
unq_rch = np.unique(sword.cl_reach_id[wrong_rchs])
new_rch_id = np.zeros(len(unq_rch), dtype=int)
cp_basins = np.copy(sb)
cp_ids = np.copy(sword.cl_reach_id)
for r in list(range(len(unq_rch))):
    # print(r, len(unq_rch)-1)
    pts = np.where(sword.cl_reach_id == unq_rch[r])[0]
    old_basin = max(set(list(sb[pts])), key=list(sb[pts]).count)
    new_basin = max(set(list(mb[pts])), key=list(mb[pts]).count)
    if old_basin != new_basin:
        cp_basins[pts] = new_basin
        rid = str(unq_rch[r])[-5::]
        nr = int(str(new_basin)+rid)
        check = np.where(cp_ids == nr)[0]
        if len(check) > 0:
            ids = np.where(cp_basins == new_basin)[0]
            rch_nums = [int(str(r)[-5:-1]) for r in cp_ids[ids]]
            max_num = max(rch_nums)
            if len(str(max_num)) == 1:
                new_num = str('000')+str(max_num+1)
            if len(str(max_num)) == 2:
                new_num = str('00')+str(max_num+1)
            if len(str(max_num)) == 3:
                new_num = str('0')+str(max_num+1)
            if len(str(max_num)) == 4:
                new_num = str(max_num+1)
            type = str(unq_rch[r])[-1]
            new_rch_id[r] = int(str(new_basin)+str(new_num)+str(type))
            cp_ids[pts] = int(str(new_basin)+str(new_num)+str(type))
        else:
            new_rch_id[r] = int(str(new_basin)+rid)
            cp_ids[pts] = int(str(new_basin)+rid)

df = pd.DataFrame(np.array([unq_rch, new_rch_id]).T)
df.rename(
    columns={
        0:"old_rch_id",
        1:"new_rch_id",
        },inplace=True)

rm_rows = np.where(new_rch_id == 0)[0]
df = df.drop(rm_rows)
df.to_csv('/Users/ealteanau/Documents/SWORD_Dev/update_requests/v16/'+region+'_rch_id_changes.csv')

perc = (len(df)/len(np.unique(sword.cl_reach_id)))*100
print("Done with " + region + ", reaches updated: " + str(np.round(perc,2)) + '%')