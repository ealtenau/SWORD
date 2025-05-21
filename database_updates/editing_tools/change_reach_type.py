import netCDF4 as nc
import pandas as pd
import numpy as np

region = 'OC'
version = 'v18'
sword_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+\
    '/netcdf/'+region.lower()+'_sword_'+version+'.nc'
csv_dir = '/Users/ealtenau/Documents/SWORD_Dev/update_requests/'+version+'/'+region+\
    '/'+region.lower()+'_incorrect_ghost_reaches.csv'

sword = nc.Dataset(sword_dir, 'r+')
updates = pd.read_csv(csv_dir)

#loop 
for row in list(range(len(updates))):
    print(row, len(updates)-1)
    rch = np.where(sword.groups['reaches'].variables['reach_id'][:] == updates['reach_id'][row])[0]
    if len(rch) == 0:
        continue
    
    nodes_rch = np.where(sword.groups['nodes'].variables['reach_id'][:] == updates['reach_id'][row])[0]
    if sword.groups['reaches'].variables['edit_flag'][rch] == 'NaN':
        edit_val = '1'
    elif '1' not in str(sword.groups['reaches'].variables['edit_flag'][rch]).split(','):
        edit_val = str(sword.groups['reaches'].variables['edit_flag'][rch])[2:-2] + ',1'
    else:
        edit_val = str(sword.groups['reaches'].variables['edit_flag'][rch])[2:-2]
                    
    rch_up1 = np.where(sword.groups['reaches'].variables['rch_id_up'][0,:] == updates['reach_id'][row])[0]
    rch_up2 = np.where(sword.groups['reaches'].variables['rch_id_up'][1,:] == updates['reach_id'][row])[0]
    rch_up3 = np.where(sword.groups['reaches'].variables['rch_id_up'][2,:] == updates['reach_id'][row])[0]
    rch_up4 = np.where(sword.groups['reaches'].variables['rch_id_up'][3,:] == updates['reach_id'][row])[0]
    rch_dn1 = np.where(sword.groups['reaches'].variables['rch_id_dn'][0,:] == updates['reach_id'][row])[0]
    rch_dn2 = np.where(sword.groups['reaches'].variables['rch_id_dn'][1,:] == updates['reach_id'][row])[0]
    rch_dn3 = np.where(sword.groups['reaches'].variables['rch_id_dn'][2,:] == updates['reach_id'][row])[0]
    rch_dn4 = np.where(sword.groups['reaches'].variables['rch_id_dn'][3,:] == updates['reach_id'][row])[0]
    cl_rch1 = np.where(sword.groups['centerlines'].variables['reach_id'][0,:] == updates['reach_id'][row])[0]
    cl_rch2 = np.where(sword.groups['centerlines'].variables['reach_id'][1,:] == updates['reach_id'][row])[0]
    cl_rch3 = np.where(sword.groups['centerlines'].variables['reach_id'][2,:] == updates['reach_id'][row])[0]
    cl_rch4 = np.where(sword.groups['centerlines'].variables['reach_id'][3,:] == updates['reach_id'][row])[0]
    #create new ids with new type
    rch_id = str(int(sword.groups['reaches'].variables['reach_id'][rch]))
    node_ids = [str(int(n)) for n in sword.groups['nodes'].variables['node_id'][nodes_rch]]
    rch_basin = rch_id[0:-1]
    node_basin = [n[0:-1] for n in node_ids]
    new_rch_id = int(rch_basin + str(updates['new_type'][row]))
    new_node_ids = [int(n + str(updates['new_type'][row])) for n in node_basin]
    #update reach id variables
    sword.groups['reaches'].variables['reach_id'][rch] = new_rch_id
    sword.groups['reaches'].variables['edit_flag'][rch] = edit_val
    sword.groups['nodes'].variables['reach_id'][nodes_rch] = new_rch_id
    sword.groups['nodes'].variables['edit_flag'][nodes_rch] = np.repeat(edit_val, len(nodes_rch))
    if len(cl_rch1) > 0:
        sword.groups['centerlines'].variables['reach_id'][0,cl_rch1] = new_rch_id
    if len(rch_up1) > 0:
        sword.groups['reaches'].variables['rch_id_up'][0,rch_up1] = new_rch_id
    if len(rch_dn1) > 0:
        sword.groups['reaches'].variables['rch_id_dn'][0,rch_dn1] = new_rch_id
    if len(cl_rch2) > 0:
        sword.groups['centerlines'].variables['reach_id'][1,cl_rch2] = new_rch_id
    if len(rch_up2) > 0:
        sword.groups['reaches'].variables['rch_id_up'][1,rch_up2] = new_rch_id
    if len(rch_dn2) > 0:
        sword.groups['reaches'].variables['rch_id_dn'][1,rch_dn2] = new_rch_id
    if len(cl_rch3) > 0:
        sword.groups['centerlines'].variables['reach_id'][2,cl_rch3] = new_rch_id
    if len(rch_up3) > 0:
        sword.groups['reaches'].variables['rch_id_up'][2,rch_up3] = new_rch_id
    if len(rch_dn3) > 0:
        sword.groups['reaches'].variables['rch_id_dn'][2,rch_dn3] = new_rch_id
    if len(cl_rch4) > 0:
        sword.groups['centerlines'].variables['reach_id'][3,cl_rch4] = new_rch_id
    if len(rch_up4) > 0:
        sword.groups['reaches'].variables['rch_id_up'][3,rch_up4] = new_rch_id
    if len(rch_dn4) > 0:
        sword.groups['reaches'].variables['rch_id_dn'][3,rch_dn4] = new_rch_id
    #update node id variables
    for n in list(range(len(node_ids))):
        sword.groups['nodes'].variables['node_id'][nodes_rch[n]] = new_node_ids[n]
        cl_n1 = np.where(sword.groups['centerlines'].variables['node_id'][0,:] == int(node_ids[n]))[0]
        cl_n2 = np.where(sword.groups['centerlines'].variables['node_id'][1,:] == int(node_ids[n]))[0]
        cl_n3 = np.where(sword.groups['centerlines'].variables['node_id'][2,:] == int(node_ids[n]))[0]
        cl_n4 = np.where(sword.groups['centerlines'].variables['node_id'][3,:] == int(node_ids[n]))[0]
        #update netcdf
        if len(cl_n1) > 0:
            sword.groups['centerlines'].variables['node_id'][0,cl_n1] = new_node_ids[n]
        if len(cl_n2) > 0:
            sword.groups['centerlines'].variables['node_id'][1,cl_n2] = new_node_ids[n]
        if len(cl_n3) > 0:
            sword.groups['centerlines'].variables['node_id'][2,cl_n3] = new_node_ids[n]
        if len(cl_n4) > 0:
            sword.groups['centerlines'].variables['node_id'][3,cl_n4] = new_node_ids[n]

print('Cl Dimensions:', len(np.unique(sword.groups['centerlines'].variables['cl_id'][:])), len(sword.groups['centerlines'].variables['cl_id'][:]))
print('Node Dimensions:', len(np.unique(sword.groups['centerlines'].variables['node_id'][0,:])), len(sword.groups['nodes'].variables['node_id'][:]), len(sword.groups['nodes'].variables['node_id'][:]))
print('Rch Dimensions:', len(np.unique(sword.groups['centerlines'].variables['reach_id'][0,:])), len(np.unique(sword.groups['nodes'].variables['reach_id'][:])), len(np.unique(sword.groups['reaches'].variables['reach_id'][:])),len(sword.groups['reaches'].variables['reach_id'][:]))
print('Edit Flag Values:', np.unique(sword.groups['reaches'].variables['edit_flag'][:]))
sword.close()
print('UPDATES DONE')