from importlib.resources import contents
import netCDF4 as nc
import pandas as pd
import numpy as np

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

def apply_updates(rpt_sub, sword):
    rpt_sub = rpt_sub.fillna(0)
    for r in list(range(len(rpt_sub))):  
        print(r, len(rpt_sub)-1)
        row = rpt_sub.index[r]
        flag = rpt_sub['report_index'][row]
        try:
            # Type Change
            if flag == 1:
                rch = np.where(sword.groups['reaches'].variables['reach_id'] == rpt_sub['reach_id'][row])[0]
                nodes_rch = np.where(sword.groups['nodes'].variables['reach_id'] == rpt_sub['reach_id'][row])[0]
                if sword.groups['reaches'].variables['edit_flag'][rch] == 'NaN':
                    edit_val = str(flag)
                else:
                    edit_val = str(sword.groups['reaches'].variables['edit_flag'][rch][0]) + ',' + str(flag)
                
                rch_up1 = np.where(sword.groups['reaches'].variables['rch_id_up'][0,:] == rpt_sub['reach_id'][row])[0]
                rch_up2 = np.where(sword.groups['reaches'].variables['rch_id_up'][1,:] == rpt_sub['reach_id'][row])[0]
                rch_up3 = np.where(sword.groups['reaches'].variables['rch_id_up'][2,:] == rpt_sub['reach_id'][row])[0]
                rch_up4 = np.where(sword.groups['reaches'].variables['rch_id_up'][3,:] == rpt_sub['reach_id'][row])[0]
                rch_dn1 = np.where(sword.groups['reaches'].variables['rch_id_dn'][0,:] == rpt_sub['reach_id'][row])[0]
                rch_dn2 = np.where(sword.groups['reaches'].variables['rch_id_dn'][1,:] == rpt_sub['reach_id'][row])[0]
                rch_dn3 = np.where(sword.groups['reaches'].variables['rch_id_dn'][2,:] == rpt_sub['reach_id'][row])[0]
                rch_dn4 = np.where(sword.groups['reaches'].variables['rch_id_dn'][3,:] == rpt_sub['reach_id'][row])[0]
                cl_rch1 = np.where(sword.groups['centerlines'].variables['reach_id'][0,:] == rpt_sub['reach_id'][row])[0]
                cl_rch2 = np.where(sword.groups['centerlines'].variables['reach_id'][1,:] == rpt_sub['reach_id'][row])[0]
                cl_rch3 = np.where(sword.groups['centerlines'].variables['reach_id'][2,:] == rpt_sub['reach_id'][row])[0]
                cl_rch4 = np.where(sword.groups['centerlines'].variables['reach_id'][3,:] == rpt_sub['reach_id'][row])[0]
                #create new ids with new type
                rch_id = str(int(sword.groups['reaches'].variables['reach_id'][rch]))
                node_ids = [str(int(n)) for n in sword.groups['nodes'].variables['node_id'][nodes_rch]]
                rch_basin = rch_id[0:-1]
                node_basin = [n[0:-1] for n in node_ids]
                new_rch_id = int(rch_basin + str(rpt_sub['data1'][row]))
                new_node_ids = [int(n + str(rpt_sub['data1'][row])) for n in node_basin]
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

            # Node Order Change
            elif flag == 2: 
                rch = np.where(sword.groups['reaches'].variables['reach_id'] == rpt_sub['reach_id'][row])[0]
                rch_up = sword.groups['reaches'].variables['rch_id_up'][:,rch]
                rch_dn = sword.groups['reaches'].variables['rch_id_dn'][:,rch]
                n_rch_up = sword.groups['reaches'].variables['n_rch_up'][rch]
                n_rch_dn = sword.groups['reaches'].variables['n_rch_down'][rch]
                nodes_rch = np.where(sword.groups['nodes'].variables['reach_id'] == rpt_sub['reach_id'][row])[0]
                cl_node_inds = np.where(sword.groups['centerlines'].variables['reach_id'][0,:] == rpt_sub['reach_id'][row])[0]
                if sword.groups['reaches'].variables['edit_flag'][rch] == 'NaN':
                    edit_val = str(flag)
                else:
                    edit_val = str(sword.groups['reaches'].variables['edit_flag'][rch][0]) + ',' + str(flag)
                
                #create new variables
                node_ids = sword.groups['nodes'].variables['node_id'][nodes_rch] 
                dist_out = sword.groups['nodes'].variables['dist_out'][nodes_rch]
                cl_nodes = sword.groups['centerlines'].variables['node_id'][0,cl_node_inds]
                new_node_ids = node_ids[::-1]
                new_dist_out = dist_out[::-1]  
                new_cl_nodes = cl_nodes[::-1]
                #update variables in netcdf
                sword.groups['nodes'].variables['node_id'][nodes_rch] = new_node_ids
                sword.groups['nodes'].variables['dist_out'][nodes_rch] = new_dist_out
                sword.groups['nodes'].variables['edit_flag'][nodes_rch] = np.repeat(edit_val, len(nodes_rch))
                sword.groups['reaches'].variables['rch_id_up'][:,rch] = rch_dn
                sword.groups['reaches'].variables['rch_id_dn'][:,rch] = rch_up
                sword.groups['reaches'].variables['n_rch_up'][rch] = n_rch_dn
                sword.groups['reaches'].variables['n_rch_down'][rch] = n_rch_up
                sword.groups['reaches'].variables['edit_flag'][rch] = edit_val
                sword.groups['centerlines'].variables['node_id'][0,cl_node_inds] = new_cl_nodes

            # Neighbor Change
            elif flag == 3: 
                rch = np.where(sword.groups['reaches'].variables['reach_id'][:] == rpt_sub['reach_id'][row])[0]
                nodes = np.where(sword.groups['nodes'].variables['reach_id'][:] == rpt_sub['reach_id'][row])[0]
                # if sword.groups['reaches'].variables['edit_flag'][rch] == 'NaN':
                #     edit_val = str(flag)
                # else:
                #     edit_val = str(sword.groups['reaches'].variables['edit_flag'][rch][0]) + ',' + str(flag)
                if '3' in sword.groups['reaches'].variables['edit_flag'][rch][0].split(','):
                    edit_val = str(sword.groups['reaches'].variables['edit_flag'][rch][0])
                elif sword.groups['reaches'].variables['edit_flag'][rch][0] == 'NaN':
                    edit_val = str(flag)
                else:
                    edit_val = str(sword.groups['reaches'].variables['edit_flag'][rch][0]) + ',' + str(flag)
                
                up_ngh = rpt_sub['data1'][row]
                if up_ngh == 0:
                    up_ngh = np.zeros(1)
                else:
                    up_ngh = np.array(up_ngh.split(), dtype=int)
                dn_ngh = rpt_sub['data2'][row]
                if dn_ngh == 0:
                    dn_ngh =  np.zeros(1)
                else:
                    dn_ngh = np.array(dn_ngh.split(), dtype=int)
                up_arr = np.zeros(4, dtype=int)
                dn_arr = np.zeros(4, dtype=int)
                up_arr[0:len(up_ngh)] = up_ngh
                dn_arr[0:len(dn_ngh)] = dn_ngh
                up_arr = up_arr.reshape((4,1))
                dn_arr = dn_arr.reshape((4,1))
                sword.groups['reaches'].variables['rch_id_up'][:,rch] = up_arr # dim = (4,1), type = int
                sword.groups['reaches'].variables['rch_id_dn'][:,rch] = dn_arr # dim = (4,1), type = int
                sword.groups['reaches'].variables['n_rch_up'][rch] = len(up_ngh) 
                sword.groups['reaches'].variables['n_rch_down'][rch] = len(dn_ngh)
                sword.groups['reaches'].variables['edit_flag'][rch] = edit_val
                sword.groups['nodes'].variables['edit_flag'][nodes] = np.repeat(edit_val, len(nodes))

            # River Name
            else: #flag == 4
                flag2 = rpt_sub['data1'][row]
                if flag2 == 1:
                    rch = np.where(sword.groups['reaches'].variables['reach_id'] == rpt_sub['reach_id'][row])[0]
                    nodes = np.where(sword.groups['nodes'].variables['reach_id'] == rpt_sub['reach_id'][row])[0]
                    if sword.groups['reaches'].variables['edit_flag'][rch] == 'NaN':
                        edit_val = str(flag) + str(flag2)
                    else:
                        edit_val = str(sword.groups['reaches'].variables['edit_flag'][rch][0]) + ',' + str(flag) + str(flag2)
                    sword.groups['reaches'].variables['facc'][rch] = rpt_sub['data2'][row]
                    sword.groups['reaches'].variables['edit_flag'][rch] = edit_val
                    sword.groups['nodes'].variables['facc'][nodes] = np.repeat(rpt_sub['data2'][row], len(nodes))
                    sword.groups['nodes'].variables['edit_flag'][nodes] = np.repeat(edit_val, len(nodes))
                elif flag2 == 2:
                    rch = np.where(sword.groups['reaches'].variables['reach_id'] == rpt_sub['reach_id'][row])[0]
                    nodes = np.where(sword.groups['nodes'].variables['reach_id'] == rpt_sub['reach_id'][row])[0]
                    if sword.groups['reaches'].variables['edit_flag'][rch] == 'NaN':
                        edit_val = str(flag) + str(flag2)
                    else:
                        edit_val = str(sword.groups['reaches'].variables['edit_flag'][rch][0]) + ',' + str(flag) + str(flag2)
                    sword.groups['reaches'].variables['wse'][rch] = rpt_sub['data2'][row]
                    sword.groups['reaches'].variables['edit_flag'][rch] = edit_val
                    sword.groups['nodes'].variables['wse'][nodes] = np.repeat(rpt_sub['data2'][row], len(nodes))
                    sword.groups['nodes'].variables['edit_flag'][nodes] = np.repeat(edit_val, len(nodes))
                elif flag2 == 3:
                    # FOR NOW NOT CHANGING NODE WIDTHS IF USER INDICATES A REACH WIDTH CHANGE. 
                    rch = np.where(sword.groups['reaches'].variables['reach_id'] == rpt_sub['reach_id'][row])[0]
                    # nodes = np.where(sword.groups['nodes'].variables['reach_id'] == rpt_sub['reach_id'][row])[0]
                    if sword.groups['reaches'].variables['edit_flag'][rch] == 'NaN':
                        edit_val = str(flag) + str(flag2)
                    else:
                        edit_val = str(sword.groups['reaches'].variables['edit_flag'][rch][0]) + ',' + str(flag) + str(flag2)
                    sword.groups['reaches'].variables['width'][rch] = rpt_sub['data2'][row]
                    sword.groups['reaches'].variables['edit_flag'][rch] = edit_val
                    # sword.groups['nodes'].variables['width'][nodes] = np.repeat(rpt_sub['data2'][row], len(nodes))
                    # sword.groups['nodes'].variables['edit_flag'][nodes] = edit_val
                elif flag2 == 4:
                    rch = np.where(sword.groups['reaches'].variables['reach_id'] == rpt_sub['reach_id'][row])[0]
                    if sword.groups['reaches'].variables['edit_flag'][rch] == 'NaN':
                        edit_val = str(flag) + str(flag2)
                    else:
                        edit_val = str(sword.groups['reaches'].variables['edit_flag'][rch][0]) + ',' + str(flag) + str(flag2)
                    sword.groups['reaches'].variables['slope'][rch] = rpt_sub['data2'][row]
                    sword.groups['reaches'].variables['edit_flag'][rch] = edit_val
                else:
                    rch = np.where(sword.groups['reaches'].variables['reach_id'] == rpt_sub['reach_id'][row])[0]
                    nodes = np.where(sword.groups['nodes'].variables['reach_id'] == rpt_sub['reach_id'][row])[0]
                    if sword.groups['reaches'].variables['edit_flag'][rch] == 'NaN':
                        edit_val = str(flag) + str(flag2)
                    else:
                        edit_val = str(sword.groups['reaches'].variables['edit_flag'][rch][0]) + ',' + str(flag) + str(flag2)
                    sword.groups['reaches'].variables['river_name'][rch] = rpt_sub['data2'][row]
                    sword.groups['reaches'].variables['edit_flag'][rch] = edit_val
                    sword.groups['nodes'].variables['river_name'][nodes] = np.repeat(rpt_sub['data2'][row], len(nodes))
                    sword.groups['nodes'].variables['edit_flag'][nodes] = np.repeat(edit_val, len(nodes))    

        except:
            print("Error with row: " + str(row), ", rch: " + str(rpt_sub['reach_id'][row]), ", flag: " + str(flag))

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

version = 'v16'
sword_dir = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'+version+'/netcdf/'
# fn_reports = '/Users/ealteanau/Documents/SWORD_Dev/src/other_src/sword_app/user_reports.csv'
fn_reports = '/Users/ealteanau/Documents/SWORD_Dev/update_requests/v16/user_reports_node_reversals3.csv'

rpt = pd.read_csv(fn_reports)
not_completed = np.where(rpt['updated'] == 0)[0]
rpt_update = rpt.loc[not_completed]
if len(rpt_update) == 0:
    print('NO NEW REPORTS')

else:
    cont = np.array([int(str(r)[0:1]) for r in rpt_update['reach_id']])
    # cont = np.array([int(str(r)[1]) for r in rpt_update['reach_id']])
    NA = np.where(cont >= 7)[0]
    AS = np.where((cont == 3) | (cont == 4))[0]
    EU = np.where(cont == 2)[0]
    SA = np.where(cont == 6)[0]
    AF = np.where(cont == 1)[0]
    OC = np.where(cont == 5)[0]
    regions = np.zeros(len(cont),dtype=object)
    regions[NA] = 'NA'
    regions[SA] = 'SA'
    regions[AS] = 'AS'
    regions[EU] = 'EU'
    regions[OC] = 'OC'
    regions[AF] = 'AF'
    rpt_update['region'] = regions
    
    unq_regions = np.unique(rpt_update['region'])
    for ind in list(range(len(unq_regions))):
        print('Starting: ' + unq_regions[ind])
        rpt_ind = rpt_update.index[np.where(rpt_update['region'] == unq_regions[ind])[0]]
        rpt_sub = rpt_update.loc[rpt_ind]
        sword = nc.Dataset(sword_dir+str(unq_regions[ind]).lower()+'_sword_'+version+'.nc', 'r+')
        apply_updates(rpt_sub, sword)
        print(np.unique(sword.groups['reaches'].variables['edit_flag'][:]))
        sword.close()
    rpt.loc[not_completed,'updated'] = 1
    rpt.to_csv(fn_reports, index=False)
    print('UPDATES DONE')

