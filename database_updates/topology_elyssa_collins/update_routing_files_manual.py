import os
import geopandas as gpd
import numpy as np
import pandas as pd
import sys
import argparse
import warnings
warnings.filterwarnings("ignore") #if code stops working may need to comment out to check warnings. 

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("basin", help="<Required> Level Two Pfafstetter Basin (i.e. 74)", type = str)
args = parser.parse_args()

b = args.basin
region = args.region

topo_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Topology/'+region+'/b'+b+'/'

# rrr_top_shp= '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/'\
#         +version+'/shp/'+region+'/'+region.lower()+'_sword_reaches_hb'+b+'_'+version+'.shp'
rrr_top_shp= topo_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_LSFix_MS_TopoFix_Acc.shp'
rrr_con_csv= topo_dir + region.lower()+'_sword_reaches_hb' + b + '_'+version+'_FG1_LSFix_MS_TopoFix_con.csv'
rrr_riv_csv= topo_dir + region.lower()+'_sword_reaches_hb' + b + '_'+version+'_FG1_LSFix_MS_TopoFix_riv.csv'

#**************************************************
# Formatting reach ids fields. 
#**************************************************
topo = gpd.read_file(rrr_top_shp)
reach_id = np.array(topo['reach_id'])
rch_up = np.array(topo['rch_id_up'])
rch_dn = np.array(topo['rch_id_dn'])
n_rch_up = np.array(topo['n_rch_up'])
n_rch_dn = np.array(topo['n_rch_dn'])

rch_id_up = np.zeros((len(reach_id), 4), dtype=int)
rch_id_dn = np.zeros((len(reach_id), 4), dtype=int)

for i in list(range(len(reach_id))):
    # print(i, len(reach_id)-1)
    if rch_up[i] == None:
        rup = np.array([0])
    else:
        rup = np.array(rch_up[i].split(), dtype = int)
    if rch_dn[i] == None:
        rdn = np.array([0])
    else:
        rdn = np.array(rch_dn[i].split(), dtype = int)

    rch_id_up[i,0:len(rup)] = rup # rch_id_up[i,:]
    rch_id_dn[i,0:len(rdn)] = rdn # rch_id_dn[i,:]


#**************************************************
# Writing connectivity file: rrr_con_csv
#**************************************************

print('Writing connectivity file: rrr_con_csv')


### FOR NETWORKS WITH SOME MULTI-DOWNSTREAM JUNCTIONS
# - rrr_con_csv 
#   . River ID
#   . Number of downstream rivers
#   . ID of 1st downstream river
#   . (...)
#   . ID of nth downstream river
#   . Number of upstream rivers
#   . ID of 1st upstream river
#   . (...)
#   . ID of nth upstream river

d = {'reach_id': np.array(reach_id).astype('int64'), 
     'n_rch_dn': n_rch_dn.astype('int'),
     'rch_id_dn_1': rch_id_dn[:,0].astype('int64'),
     'rch_id_dn_2': rch_id_dn[:,1].astype('int64'),
     'rch_id_dn_3': rch_id_dn[:,2].astype('int64'),
     'rch_id_dn_4': rch_id_dn[:,3].astype('int64'),
     'n_rch_up': n_rch_up.astype('int'),
     'rch_id_up_1': rch_id_up[:,0].astype('int64'),
     'rch_id_up_2': rch_id_up[:,1].astype('int64'),
     'rch_id_up_3': rch_id_up[:,2].astype('int64'),
     'rch_id_up_4': rch_id_up[:,3].astype('int64')}

df = pd.DataFrame(data=d)

df.to_csv(rrr_con_csv, header=False, index=False)
print('DONE - Connectivity added to:', rrr_con_csv)

#**************************************************
# Writing basin (sorted) file: rrr_bas_csv
#**************************************************

print('Writing basin (sorted) file: rrr_bas_csv')

df = pd.read_csv(rrr_con_csv, header=None)

df.columns = ['reach_id', 'n_rch_dn', 'rch_id_dn_1', 'rch_id_dn_2', 'rch_id_dn_3', 'rch_id_dn_4', 'n_rch_up', 'rch_id_up_1', 'rch_id_up_2', 'rch_id_up_3', 'rch_id_up_4']

df_sub = df[['reach_id', 'n_rch_dn', 'rch_id_dn_1', 'rch_id_dn_2']]
df_sub['reach_id'] = df_sub.reach_id.astype('str')
df_sub['rch_id_dn_1'] = df_sub.rch_id_dn_1.astype('str')
df_sub['rch_id_dn_2'] = df_sub.rch_id_dn_2.astype('str')
ind_no_dn = df_sub.loc[df_sub['rch_id_dn_1'] == '0'].index.to_list()
df_sub.loc[df_sub['rch_id_dn_1'] == '0', 'rch_id_dn_1'] = ''
df_sub.loc[df_sub['rch_id_dn_2'] == '0', 'rch_id_dn_2'] = ''

rch_id_lst = []
rch_id_dn_lst = []
for i in range(len(df_sub)):
    rch_id_lst.append(df_sub.at[i, 'reach_id'])
    rch_id_dn_lst.append([df_sub.at[i, 'rch_id_dn_1']])

    if df_sub.at[i, 'n_rch_dn'] == 2:
        rch_id_lst.append(df_sub.at[i, 'reach_id'])
        rch_id_dn_lst.append([df_sub.at[i, 'rch_id_dn_2']])
     

## This ind_no_dn doesn't match the above because sometimes the same reach_id is added to the list
## multiple times if there is more than one downstream segment, so the indices change
## Therefore, reidentify them here
ind_no_dn = [i for i in range(len(rch_id_dn_lst)) if rch_id_dn_lst[i] == ['']]

for i in ind_no_dn:
    rch_id_dn_lst[i] = []


graph = dict(zip(pd.Series(rch_id_lst), pd.Series(rch_id_dn_lst)))


start = list(graph)
seen = set()
stack = []    # path variable is gone, stack and order are new
order = []    # order will be in reverse order at first
q = start

while q:
    v = q.pop()
    if v not in seen:
        seen.add(v) # no need to append to path any more
        q.extend(graph[v])

        while stack and v not in graph[stack[-1]]: # new stuff here!
            # print('Entered')
            order.append(stack.pop())
        stack.append(v)

stack = stack + order[::-1] 

stack = [int(i) for i in stack]

d = {'reach_id': np.array(stack).astype('int64')}

df = pd.DataFrame(data=d)

df.to_csv(rrr_riv_csv, header=False, index=False)

print('DONE - Topology added to:', rrr_riv_csv)
