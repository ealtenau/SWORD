from __future__ import division
import os
import numpy as np
import numpy as np
import geopandas as gp
import matplotlib.pyplot as plt

#########################################################################################################################

def get_ngh_info(data):
    ngh_info = np.zeros([len(data.reach_id), 5])
    wse = np.array(data.wse[:])
    facc = np.array(data.facc[:])
    for ind in list(range(len(data.reach_id))):
        wse1 = []; facc1 = []; wse2 = []; facc2 = []
        ngh_info[ind,0] = data.reach_id[ind]
        nghs1 = data.rch_id_up[ind] 
        nghs2 = data.rch_id_dn[ind] 
        if nghs1 is None:
            nghs1 = []
        if nghs2 is None:
            nghs2 = []
        if len(nghs1) > 0:
            nghs1 = nghs1.split()
            nghs1 =  [int(n) for n in nghs1]
            for idx in list(range(len(nghs1))):
                n1 = np.where(data.reach_id == str(nghs1[idx]))[0] #remove string condition if reaches were written using python
                if len(n1) == 0:
                    continue
                else:
                    wse1.append(wse[n1])
                    facc1.append(facc[n1])
                ngh_info[ind,1] = np.max(wse1)
                ngh_info[ind,2] = np.max(facc1)
        
        if len(nghs2) > 0:
            nghs2 = nghs2.split()
            nghs2 =  [int(n) for n in nghs2]
            for idy in list(range(len(nghs2))):
                n2 = np.where(data.reach_id == str(nghs2[idy]))[0] #remove string condition if reaches were written using python
                if len(n2) == 0:
                    continue
                else:
                    wse2.append(wse[n2])
                    facc2.append(facc[n2])
                ngh_info[ind,3] = np.max(wse2)
                ngh_info[ind,4] = np.max(facc2)

        if len(nghs1) == 0:
            ngh_info[ind,1] = -9999
            ngh_info[ind,2] = -9999
        if len(nghs2) == 0:
            ngh_info[ind,3] = -9999
            ngh_info[ind,4] = -9999

    return ngh_info
        
#########################################################################################################################

region = 'NA'
file_dir = '/Users/ealteanau/Documents/SWORD_Dev/outputs/v13/shp/'+region+'/'
files = [f for f in os.listdir(file_dir) if 'shp' in f and 'reaches' in f]

### in larger loop 
for f in list(range(len(files))):
    print(files[f])
    data = gp.read_file(file_dir+files[f])
    #col0 - rch_id, col1 - wse1, col2 - facc1, col3 - wse2, col4 - facc2
    nghs = get_ngh_info(data)
    topo_flag = np.zeros(len(data.reach_id))
    for ind in list(range(len(data.reach_id))):
        rch_facc = data.facc[ind]
        rch_wse = data.wse[ind]
        if nghs[ind,1] == rch_wse and nghs[ind,3] == rch_wse:
            if np.max([nghs[ind,2],nghs[ind,4]]) > rch_facc and np.min([nghs[ind,2],nghs[ind,4]]) < rch_facc:
                topo_flag[ind] = 0
            else:
                topo_flag[ind] = 1
                # print(ind, 'cond 1')
        elif nghs[ind,1] == -9999 and nghs[ind,3] == rch_wse:
            if np.max([nghs[ind,2],nghs[ind,4]]) > rch_facc and np.min([nghs[ind,2],nghs[ind,4]]) < rch_facc:
                topo_flag[ind] = 0
            else:
                topo_flag[ind] = 1
                # print(ind, 'cond 2')
        elif nghs[ind,1] == rch_wse and nghs[ind,3] == -9999:
            if np.max([nghs[ind,2],nghs[ind,4]]) > rch_facc and np.min([nghs[ind,2],nghs[ind,4]]) < rch_facc:
                topo_flag[ind] = 0
            else:
                topo_flag[ind] = 1
                # print(ind, 'cond 3')
        elif nghs[ind,1] >= rch_wse and nghs[ind,3] >= rch_wse:
            if np.max([nghs[ind,2],nghs[ind,4]]) > rch_facc and np.min([nghs[ind,2],nghs[ind,4]]) < rch_facc:
                topo_flag[ind] = 0
            else:
                topo_flag[ind] = 1
                # print(ind, 'cond 4')
        elif nghs[ind,1] <= rch_wse and nghs[ind,3] <= rch_wse: 
            if np.max([nghs[ind,2],nghs[ind,4]]) > rch_facc and np.min([nghs[ind,2],nghs[ind,4]]) < rch_facc:
                topo_flag[ind] = 0
            else:
                topo_flag[ind] = 1
                # print(ind, 'cond 5')
        else:
            continue

    data['topo_flag'] = topo_flag #need to combine with other level 2 basins then wrtie as gpkg.
    if f == 0:
        data_all = data.copy()
    else:
        data_all = data_all.append(data)
    del(data)

data_all = gp.GeoDataFrame(data_all)
data_all.to_file('/Users/ealteanau/Documents/SWORD_Dev/outputs/rch_edits/'+region.lower()+'_sword_v13.gpkg', driver='GPKG', layer='reaches')
flag = np.where(data_all['topo_flag'] == 1)[0]
flag2 = np.where((data_all['topo_flag'] == 1) & (data_all['type'] != 5))[0]
print('DONE', np.round(((len(flag)/len(data_all.reach_id))*100),3), np.round(((len(flag2)/len(data_all.reach_id))*100),3))

# plt.scatter(data_all.x, data_all.y, s = 3, edgecolors=None)
# plt.scatter(np.array(data_all.x)[flag], np.array(data_all.y)[flag], s = 3, c='red', edgecolors=None)
# plt.show()

# 3.6% NA, 3.5% SA, 2.4% AS, 2.3% EU, 3.2% AF, 4.5% OC
# 6.8/5.7% NA, 6.2/4.9% SA, 4.7/3.3% AS, 4.8/4.2% EU, 6.4/4.7% AF, 9.3/5.0% OC