import numpy as np
import geopandas as gp

region = 'OC'
fn_gpkg = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/gpkg/'+region.lower()+'_sword_reaches_v16.gpkg'
fn_out = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/TopoFlag/'+region.lower()+'_sword_v16_topoflag.gpkg'

#load sword.
gpkg = gp.read_file(fn_gpkg)

reach_id = np.array(gpkg['reach_id'])
n_rch_up = np.array(gpkg['n_rch_up'])
n_rch_down = np.array(gpkg['n_rch_dn'])
#format neighboring reach ids.
rch_id_up = np.zeros((len(reach_id), 4))
rch_id_dn = np.zeros((len(reach_id), 4))
for r in list(range(len(reach_id))):
    up = np.array(gpkg['rch_id_up'][r].split(),dtype = int)
    dn = np.array(gpkg['rch_id_dn'][r].split(),dtype = int)
    if len(up) > 0:
        rch_id_up[r,0:len(up)] = up
    if len(dn) > 0:
        rch_id_dn[r,0:len(dn)] = dn

#flag topological inconsistencies.
flag = np.zeros(len(reach_id))
for i in list(range(len(reach_id))):
    #check upstream
    for j in list(range(n_rch_up[i])):
        k = np.where(reach_id == rch_id_up[i,j])[0]
        if len(k) == 0:
            continue
        else:
            check = np.where(rch_id_dn[k,:] == reach_id[i])[0]
            if len(check) == 0:
                flag[i] = 1
    #check downstream 
    for j in list(range(n_rch_down[i])):
        k = np.where(reach_id == rch_id_dn[i,j])[0]
        if len(k) == 0:
            continue
        else:
            check = np.where(rch_id_up[k,:] == reach_id[i])[0]
            if len(check) == 0:
                flag[i] = 1

gpkg['topo_flag'] = flag
gpkg.to_file(fn_out, driver='GPKG', layer='reaches')
print('DONE. Percent flagged: ',(np.round(len(np.where(flag == 1)[0])/len(reach_id)*100,3)))