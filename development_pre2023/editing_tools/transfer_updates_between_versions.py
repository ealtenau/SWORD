import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy import spatial as sp
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
args = parser.parse_args()

region = args.region

fn_merge_v12 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Merged_Data/v12/'+region+'_Merge_v12.nc'
fn_merge_v11 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Merged_Data/v11/'+region+'_Merge_v11.nc'
fn_sword_v14 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v14/netcdf/'+region.lower()+'_sword_v14.nc'
fn_pascal_edits = '/Users/ealteanau/Documents/SWORD_Dev/update_requests/v13/completed/lakeflag_updates_hb77hb78.csv'

m12 = nc.Dataset(fn_merge_v12, 'r+')
m11 = nc.Dataset(fn_merge_v11)
s14 = nc.Dataset(fn_sword_v14)
lf = pd.read_csv(fn_pascal_edits)

merge12 = Object()
merge12.x = np.array(m12.groups['centerlines'].variables['x'][:])
merge12.y = np.array(m12.groups['centerlines'].variables['y'][:])
merge12.nchannels = np.array(m12.groups['centerlines'].variables['nchan'][:])
merge12.facc = np.array(m12.groups['centerlines'].variables['flowacc'][:])
merge12.wse = np.array(m12.groups['centerlines'].variables['p_height'][:])
merge12.lakeflag = np.array(m12.groups['centerlines'].variables['lakeflag'][:])
merge12.manual_add = np.array(m12.groups['centerlines'].variables['manual_add'][:])

merge11 = Object()
merge11.x = np.array(m11.groups['centerlines'].variables['x'][:])
merge11.y = np.array(m11.groups['centerlines'].variables['y'][:])
merge11.nchannels = np.array(m11.groups['centerlines'].variables['nchan'][:])
merge11.facc = np.array(m11.groups['centerlines'].variables['flowacc'][:])
merge11.wse = np.array(m11.groups['centerlines'].variables['p_height'][:])
merge11.lakeflag = np.array(m11.groups['centerlines'].variables['lakeflag'][:])
m11.close()

sword14 = Object()
sword14.clx = np.array(s14.groups['centerlines'].variables['x'][:])
sword14.cly = np.array(s14.groups['centerlines'].variables['y'][:])
sword14.cl_node_id = np.array(s14.groups['centerlines'].variables['node_id'][0,:])
sword14.cl_reach_id = np.array(s14.groups['centerlines'].variables['reach_id'][0,:])
sword14.reach_id = np.array(s14.groups['reaches'].variables['reach_id'][:])
sword14.wse = np.array(s14.groups['nodes'].variables['wse'][:])
sword14.facc = np.array(s14.groups['nodes'].variables['facc'][:])
sword14.editflag = np.array(s14.groups['reaches'].variables['edit_flag'][:])
s14.close()

new_lakes = Object()
new_lakes.rchid = np.array(lf.REACHID)
new_lakes.lf = np.array(lf.N_LKFLG)
lf = None

###############################################################################

print('Updating NCHAN Values')
m11_pts = np.vstack((merge11.x, merge11.y)).T
m12_pts = np.vstack((merge12.x, merge12.y)).T
kdt = sp.cKDTree(m11_pts)
eps_dist, eps_ind = kdt.query(m12_pts, k = 2)
merge12.temp_nchan = merge11.nchannels[eps_ind[:,0]]
replace = np.where((merge12.manual_add == 1) & (merge12.lakeflag != 1))[0]
merge12.nchannels[replace] = merge12.temp_nchan[replace]

s14_pts = np.vstack((sword14.clx, sword14.cly)).T
m12_pts = np.vstack((merge12.x, merge12.y)).T
kdt2 = sp.cKDTree(m12_pts)
eps_dist2, eps_ind2 = kdt2.query(s14_pts, k = 2)

if region == 'NA':
    print('Updating LakeFlag Values') 
    for ind in list(range(len(new_lakes.rchid))):
        indexes = np.where(sword14.cl_reach_id == new_lakes.rchid[ind])[0]
        merge_ind = eps_ind2[indexes,0]
        merge12.lakeflag[merge_ind] = new_lakes.lf[ind]

print('Updating FACC and WSE values')
nonnan = np.where(sword14.editflag != 'NaN')[0]
unq_rch = np.unique(sword14.reach_id[nonnan])
for idx in list(range(len(unq_rch))): 
    rind = np.where(sword14.reach_id == unq_rch[idx])[0]
    clind = np.where(sword14.cl_reach_id == unq_rch[idx])[0]
    edit_vals = sword14.editflag[rind][0].split(',')
    if '41' in edit_vals:
        new_facc = sword14.facc[rind]
        merge_ind = eps_ind2[clind,0]
        merge12.facc[merge_ind] = new_facc

    elif '42' in edit_vals:
        new_wse = sword14.wse[rind]
        merge_ind = eps_ind2[clind,0]
        merge12.wse[merge_ind] = new_wse
        
    else:
        continue

print('Updating NetCDF Values')
m12.groups['centerlines'].variables['flowacc'][:] = merge12.facc
m12.groups['centerlines'].variables['p_height'][:] = merge12.wse
m12.groups['centerlines'].variables['nchan'][:] = merge12.nchannels
m12.groups['centerlines'].variables['lakeflag'][:] = merge12.lakeflag
m12.close()
