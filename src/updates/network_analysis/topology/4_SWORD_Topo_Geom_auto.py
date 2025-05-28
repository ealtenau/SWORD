###################################################
# Script: SWORD_Topo_Geom_auto.py
# Date created: 7/16/24
# Usage: Defining topology in SWORD using polyline geometry
#         - This script is an automatic version of 'SWORD_Topo_Geom.py'
###################################################

import os
main_dir = os.getcwd()
import geopandas as gpd
import numpy as np
import pandas as pd
import sys
import argparse

#*******************************************************************************
#Command Line Variables / Instructions:
#*******************************************************************************
# 1 - SWORD Continent (i.e. AS)
# 2 - Level 2 Pfafstetter Basin (i.e. 36)
# Example Syntax: "python SWORD_Topo_Geom_auto.py AS 36 Main_pts.shp"
#*******************************************************************************

#*******************************************************************************
#Get command line arguments
#*******************************************************************************

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("basin", help="<Required> Level Two Pfafstetter Basin (i.e. 74)", type = str)
parser.add_argument("pts_shp_file", help="<Required> Final Geometric Intersection Point Shapefile for Topology.", type = str)
args = parser.parse_args()

b = args.basin
region = args.region
fn = args.pts_shp_file

data_dir = main_dir+'/data/outputs/Topology/'+region+'/b'+b+'/'

rrr_pts_shp= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_' + fn
rrr_riv_shp= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_LSFix_MS.shp'
rrr_top_shp= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_LSFix_MS_TopoFix.shp'
rrr_con_csv= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_LSFix_MS_TopoFix_con.csv'
rrr_riv_csv= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_LSFix_MS_TopoFix_riv.csv'


#**************************************************
# Writing new shapefile with fixed topology
#**************************************************

print('Writing new shapefile with fixed topology')

geom = gpd.read_file(rrr_pts_shp) 

### Determining if the index of the intersection ('ind_intr') is closer to 0 (geom2 downstream) or the number of vertices in linestring ('geom1_n_pn'; geom2 upstream)
geom['closest'] = -999
for i in geom.index:
    # val = min([0, geom['geom1_n_pn'][i]], key = lambda x: abs(x - geom['ind_intr'][i]))
    if geom["geom1_n_pn"][i] == 2:
        if geom["ind_intr"][i] == 1:
            val = 1
        else:
            val = 0
    else:
        val = min([0, geom["geom1_n_pn"][i]], key = lambda x: abs(x - geom["ind_intr"][i]))
    geom['closest'][i] = val


### Defining the connectivity (upstream/downstream refers to the location of geom2_rch_id)
geom['con'] = np.where(geom['closest'] == 0, 'downstream', 'upstream')
geom = geom.drop_duplicates()



### Looping through each river intersection and populate upstream and downstream connectivity arrays
reach_id = list(set(geom['geom1_rch_']))

rch_id_up = np.zeros((len(reach_id), 4))
rch_id_dn = np.zeros((len(reach_id), 4))

for i in geom.index:
    rch_id_tmp = reach_id.index(geom['geom1_rch_'][i])

    ## First check if this is a river junction, skip this segment if so
    geom2 = geom['geom2_rch_'][i]
    con1 = geom['con'][i]
    ind = geom[(geom['geom1_rch_'] == geom2) & (geom['geom2_rch_'] == geom['geom1_rch_'][i])].index
    if len(ind) == 1:
        con2 = geom['con'][ind].tolist()[0]
        if con1 == con2:
            continue 
    if len(ind) > 1:
        print('More than one matching!')
        break


    if geom['con'][i] == 'upstream':
        ## find index of first non-zero element 
        j = np.nonzero(rch_id_up[rch_id_tmp])

        # if len(j[0]) == 0:
        j = len(j[0])
        
        rch_id_up[rch_id_tmp][j] = geom['geom2_rch_'][i]

    if geom['con'][i] == 'downstream':
        ## find index of first non-zero element 
        j = np.nonzero(rch_id_dn[rch_id_tmp])

        # if len(j[0]) == 0:
        j = len(j[0])
        
        rch_id_dn[rch_id_tmp][j] = geom['geom2_rch_'][i]


### Counting the number of upstream and downstream segments for each river
n_rch_up = np.zeros((len(reach_id)))
n_rch_dn = np.zeros((len(reach_id)))

for i in range(len(reach_id)):
    n_rch_up[i] = np.count_nonzero(rch_id_up[i])
    n_rch_dn[i] = np.count_nonzero(rch_id_dn[i])
   
print('The maximum number of upstream reaches is ' + str(np.max(n_rch_up)))
print('The maximum number of downstream reaches is ' + str(np.max(n_rch_dn)))


SWORD = gpd.read_file(rrr_riv_shp)


reach_id = [str(r) for r in reach_id]

### First looping through and setting everything to 0 
for i in SWORD.index:
    SWORD['n_rch_up'][i] = 0
    SWORD['n_rch_dn'][i] = 0
    SWORD['rch_id_up'][i] = ''
    SWORD['rch_id_dn'][i] = ''

gaps = []
for i in SWORD.index:
    if str(SWORD['reach_id'][i]) in reach_id:
        ind = reach_id.index(str(SWORD['reach_id'][i]))

        SWORD['n_rch_up'][i] = n_rch_up[ind]
        SWORD['n_rch_dn'][i] = n_rch_dn[ind]

        rch_id_up_tmp = rch_id_up[ind][rch_id_up[ind] != 0].tolist()
        rch_id_up_tmp = ' '.join([str(int(i)) for i in rch_id_up_tmp])
        rch_id_dn_tmp = rch_id_dn[ind][rch_id_dn[ind] != 0].tolist()
        rch_id_dn_tmp = ' '.join([str(int(i)) for i in rch_id_dn_tmp])

        SWORD['rch_id_up'][i] = rch_id_up_tmp
        SWORD['rch_id_dn'][i] = rch_id_dn_tmp

    else:
        ## Identify reaches where it has gaps on BOTH sides
        ## NOTE: this does not identify reaches that only have a gap on one side of the segment 
        gaps.append(str(SWORD['reach_id'][i]))


SWORD.to_file(rrr_top_shp)



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

reach_id 
rch_id_up
rch_id_dn
n_rch_up
# n_rch_dn

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