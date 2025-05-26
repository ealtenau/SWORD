###################################################
# Script: SWORD_Geom_fast_v2.py
# Date created: 3/27/24
# Usage: Determining river connectivity using polyline geometry 
#        - Note that this script is an expansion to SWORD_Geom_fast.py to improve efficiency of code:
#           --> Not revisiting potential problem junctions that have already been evaluated and set to be no problem
#           --> Only changing geometry in point intersections layer for reaches where actual changes were made
#        - Added arguments so that I can run script with inputs and store script print statements in text file 
###################################################

###################################################
## TOPOLOGY CODE STEPS:
##  1) Find initial geometric intersections for all reaches
    ##  WHILE LOOP STARTS HERE:
##  2) Extract them into lists
##  3) Find potentially problematic areas (where more than one connected reaches intersect at the same start or end index)   
##  4) Loop through potentially problematic areas and identify which ones are actually problematic
##  5) Reverse LineStrings for problematic areas
##  6) Find new geometric intersections
##          --> Once everything is clean and everything is 'not problematic', use geometric connections with 
##              SWORD_Topo_Geom.py code to label up and downstream reaches for each reach (this code is fast)

## STEPS WHERE CODE IS SLOW:
##  - Steps 1/6 (same code for both steps)
##  - Step 3

## WHY IS CODE SLOW?
##  - Both steps loop through the entire river network, which is slow for large areas

## HOW CAN CODE BE IMPROVED:
##  - Step 1 and 3 will have to be run in full the first time through to get initial intersections/problems
##  - In initial step 3, flag which areas have already been checked and identified as 'not problematic'
##  - For step 6, only modify geometric intersections for areas where LineStrings were reversed (i.e., 'problematic')
##  - Once entering the while loop for step 3, now don't revisit areas that have already been identified as 'not problematic'
###################################################

import sys
import os
main_dir = os.getcwd()
import re
import fiona
import shapely
import shapely.ops

import numpy as np
import pandas as pd
import geopandas as gpd
from timeit import default_timer as timer
import argparse

import warnings
warnings.filterwarnings("ignore") #if code stops working may need to comment out to check warnings. 

# start = timer()
# end = timer()
# end - start # 20.075 seconds


#*******************************************************************************
#Command Line Variables / Instructions:
#*******************************************************************************
# 1 - SWORD Continent (i.e. AS)
# 2 - Level 2 Pfafstetter Basin (i.e. 36)
# Example Syntax: "python SWORD_GEOM_fast_v2.py AS 36 >log_files/out_36.txt"
#*******************************************************************************

#*******************************************************************************
#Get command line arguments
#*******************************************************************************

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("basin", help="<Required> Level Two Pfafstetter Basin (i.e. 74)", type = str)
args = parser.parse_args()

b = args.basin
region = args.region

data_dir = main_dir+'/data/outputs/Topology/'+region+'/b'+b+'/'

line_shp= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_Main.shp'
geom_con_fname= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_Main_pts.shp'
head_junc_csv= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_head_at_junc.csv'
out_junc_csv= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_out_at_junc.csv'
riv_all_shp= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_FG1.shp'
riv_out_shp= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_Main_LSFix.shp'
rev_ids_csv= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_rev_LS.csv'

# #*******************************************************************************
#Print input information
#*******************************************************************************
print('Command line inputs')
print('- '+line_shp)
print('- '+geom_con_fname)
print('- '+head_junc_csv)
print('- '+riv_all_shp)
print('- '+riv_out_shp)
print('- '+rev_ids_csv)

#*******************************************************************************
#Check if files exist 
#*******************************************************************************
try:
     with open(line_shp) as file:
          pass
except IOError as e:
     print('ERROR - Unable to open '+line_shp)
     raise SystemExit(22) 

try:
     with open(geom_con_fname) as file:
          pass
except IOError as e:
     print('ERROR - Unable to open '+geom_con_fname)
     raise SystemExit(22) 

try:
     with open(head_junc_csv) as file:
          pass
except IOError as e:
     print('ERROR - Unable to open '+head_junc_csv)
     raise SystemExit(22) 

try:
     with open(out_junc_csv) as file:
          pass
except IOError as e:
     print('ERROR - Unable to open '+out_junc_csv)
     raise SystemExit(22) 

try:
     with open(riv_all_shp) as file:
          pass
except IOError as e:
     print('ERROR - Unable to open '+riv_all_shp)
     raise SystemExit(22) 


#**************************************************
# Defining helpful functions
#**************************************************

### Function for reversing coordinates of LineString
def rev_crds(input_geom):
    if input_geom.geom_type.lower() == 'linestring':
        coords = [tuple(coord) for coord in list(input_geom.coords)][::-1]
        out_geom = shapely.geometry.LineString(coords)
    elif input_geom.geom_type.lower() == 'multilinestring':
        coords = [list(this_geom.coords)[::-1] for this_geom in input_geom.geoms][::-1]
        out_geom = shapely.geometry.MultiLineString(coords)
    return out_geom


def find_index_of_point_w_min_distance(list_of_coords, coord):
    temp = [shapely.geometry.Point(c).distance(shapely.geometry.Point(coord)) for c in list_of_coords]
    return(temp.index(min(temp)) )


#**************************************************
# Reading in data
#**************************************************
print('Reading in data')

## Basin 81_sub (starting here so code development is faster)
# os.chdir('/Users/elyssac/Documents/SWOT/SWORD_v17/TopologyTests/b81_sub')
# line_shp = "hb81_subset_v17a_FO_Main.shp" # main_side = 0 only, Doing main channels first, deal with divergences after
# geom_con_fname = "hb81_subset_v17a_FO_Main_pts.shp"

## Basin 73
# os.chdir('/Users/elyssac/Documents/SWOT/SWORD_v17/NA/Topology/b73')
# line_shp = "na_sword_reaches_hb73_v17_Main_FG_FO.shp"
# geom_con_fname = 'na_sword_reaches_hb73_v17_Main_FG_FO_pts.shp'

shp = gpd.read_file(line_shp)
crs = shp.crs
eps = 5e-07 # around 1 cm or something in degrees

# head_at_junc = pd.read_csv(head_junc_csv, header=None)
try:
    head_at_junc = pd.read_csv(head_junc_csv, header=None)  #read the CSV file
    head_at_junc = head_at_junc.iloc[:,0].to_list()
    if len(head_at_junc) == 0:  # Check if the DataFrame is empty
        head_at_junc = []

except pd.errors.EmptyDataError:
    head_at_junc = []

head_at_junc = [str(r) for r in head_at_junc]

try:
    outlet_at_junc = pd.read_csv(out_junc_csv, header=None)  #read the CSV file
    outlet_at_junc = outlet_at_junc.iloc[:,0].to_list()
    if len(outlet_at_junc) == 0:  # Check if the DataFrame is empty
        outlet_at_junc = []

except pd.errors.EmptyDataError:
    outlet_at_junc = []

outlet_at_junc = [str(r) for r in outlet_at_junc]

#**************************************************
# Formatting some input data
#**************************************************
### Obtaining reach ids and 'dist_out' for each reach ID
reach_ids = shp['reach_id'].astype('int').astype('str').to_list()
# reach_ids_dist_out = shp['dist_out'].to_list()
reach_ids_dist_out = shp['dist_out'].to_list()


shp['reach_id'] = shp['reach_id'].astype('int')

# riv_out = [str(r) for r in riv_out]
# head_at_junc = []

### Getting linestring geometry and data
lines = shp['geometry'].to_list()




#**************************************************
# Topology algorithm - Main network
#**************************************************
print('Starting topology algorithm for the main network')

start = timer()

store_rch_ids_rev = [] # Only empty if not reversing outlets first


counter = 0
rch_ids_rev = []

pt_schema={'geometry': 'Point', 'properties': {'geom1_rch_id': 'str', 'geom2_rch_id': 'str', 'geom1_n_pnts': 'int', 'ind_intr': 'int'}}

## OLD START PLACE FOR WHILE LOOP

geom_con_shp = gpd.read_file(geom_con_fname)
geom_con_shp['geom1_rch_'] = geom_con_shp['geom1_rch_'].astype('str')
geom_con_shp['geom2_rch_'] = geom_con_shp['geom2_rch_'].astype('str')

# geom_con_shp.loc[geom_con_shp['geom1_rch_'] == '81380100311']

geom_con = fiona.open(geom_con_fname)

con_geom1_rch_id = []
con_geom2_rch_id = []
con_geom1_n_pnts = []
con_ind_intr = []
con_closest = []
for con in geom_con:
    con_geom = shapely.geometry.shape(con.geometry)
    
    con_geom1_rch_id.append(str(con["properties"]["geom1_rch_"]))
    con_geom2_rch_id.append(str(con["properties"]["geom2_rch_"]))
    con_geom1_n_pnts.append(con["properties"]["geom1_n_pn"])
    con_ind_intr.append(con["properties"]["ind_intr"])
    if con["properties"]["geom1_n_pn"] == 2:
        if con["properties"]["ind_intr"] == 1:
            val = 1
        else:
            val = 0
    else:
        val = min([0, con["properties"]["geom1_n_pn"]], key = lambda x: abs(x - con["properties"]["ind_intr"]))
    con_closest.append(val) # If val != 0, then geom2 is upstream


#           --> Not revisiting potential problem junctions that have already been evaluated and set to be no problem
#           --> Only changing geometry in point intersections layer for reaches where actual changes were made

### This code below is slow, figure out how to make it more efficient
###     --> Create this for the whole area before entering the while loop
###     --> Once in the while loop, only make changes where fixes have been applied 

ind0_geom1_rch_id = []
ind0_geom2_rch_id = []

indf_geom1_rch_id = []
indf_geom2_rch_id = []

d_list = []

for ix, r in shp.iterrows():
    # print(ix,r['reach_id'])
    geom1 = lines[ix]
    geom1_rch_id = reach_ids[ix]

    con_geom1_rch_id_ind = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
    
    selected_reach = shp.loc[shp['reach_id'] == int(geom1_rch_id)]
    #Finding reaches within search distance (11 km)
    reaches_win_dist = shp.assign(distance=shp.apply(lambda x: x.geometry.distance(selected_reach.geometry.iloc[0]), axis=1)).query(f'distance <= {0.1}')
    reaches_win_dist = reaches_win_dist[reaches_win_dist.reach_id != geom1_rch_id]

    #Only comparing selected_reach to reaches within search distance (minimizes computational needs)
    for ix2, r2 in reaches_win_dist.iterrows(): 
        # print(ix2,r2['reach_id'])
        geom2 = shapely.geometry.shape(reaches_win_dist.geometry[ix2])
        geom2_rch_id = reaches_win_dist.reach_id.astype('str')[ix2]

        ## Don't need to record intersection if reach ID is the same
        if geom1_rch_id == geom2_rch_id:
            continue

        if geom1.distance(geom2) < eps: 
            con_geom2_rch_id_ind = [i for i in range(len(con_geom2_rch_id)) if con_geom2_rch_id[i] == str(geom2_rch_id)]
            con_geom1_geom2_ind = [i for i in con_geom1_rch_id_ind if i in con_geom2_rch_id_ind]

            ################ added on 8/9/2024 by Elizabeth Altenau. Needed for antimeridian reaches. 
            if len(con_geom1_geom2_ind) == 0:
                continue
            ################

            con_geom2_opp_rch_id_ind = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom2_rch_id)]
            con_geom1_opp_rch_id_ind = [i for i in range(len(con_geom2_rch_id)) if con_geom2_rch_id[i] == str(geom1_rch_id)]
            con_geom1_geom2_opp_ind = [i for i in con_geom1_opp_rch_id_ind if i in con_geom2_opp_rch_id_ind]

            con_geom1_closest = [con_closest[i] for i in con_geom1_geom2_ind]    
            con_geom2_closest = [con_closest[i] for i in con_geom1_geom2_opp_ind]    

            ################ added on 8/10/2024 by Elizabeth Altenau. Needed for antimeridian reaches. 
            if len(con_geom2_closest) == 0:
                continue
            ################

        # if geom1.distance(geom2) < eps: 

            ## Check for locations where 2 segments say they are both downstream of one another 
            if (con_geom1_closest[0] == 0) & (con_geom2_closest[0] == 0):
                # rch_id_dn = geom_dist_out.index(min(geom_dist_out))
                # if rch_id_dn == 0:
                    # ind0_geom1_rch_id.append(str(geom1_rch_id))
                # if (geom1_rch_id not in ind0_geom1_rch_id) | (geom2_rch_id not in ind0_geom2_rch_id):
                ind0_geom1_rch_id.append(str(geom1_rch_id))
                ind0_geom2_rch_id.append(str(geom2_rch_id))

                #Adding reaches to dataframe identifying if it's potentially a problem (1) or not (0)
                d = {"geom1_rch_id": geom1_rch_id, "geom2_rch_id": geom2_rch_id, "con_prob": 1}
                d_list.append(d)

            ## Check for locations where 2 segments say they are both upstream of one another 
            elif (con_geom1_closest[0] != 0) & (con_geom2_closest[0] != 0):
                # if (geom1_rch_id not in indf_geom1_rch_id) | (geom2_rch_id not in indf_geom2_rch_id):
                indf_geom1_rch_id.append(str(geom1_rch_id))
                indf_geom2_rch_id.append(str(geom2_rch_id))

                d = {"geom1_rch_id": geom1_rch_id, "geom2_rch_id": geom2_rch_id, "con_prob": 1}
                d_list.append(d)
            
            else:
                d = {"geom1_rch_id": geom1_rch_id, "geom2_rch_id": geom2_rch_id, "con_prob": 0}
                d_list.append(d)


ind0_df = pd.DataFrame({'geom1_rch_id': ind0_geom1_rch_id, 'geom2_rch_id': ind0_geom2_rch_id})
indf_df = pd.DataFrame({'geom1_rch_id': indf_geom1_rch_id, 'geom2_rch_id': indf_geom2_rch_id})

ind0_df = (ind0_df[~ind0_df.filter(like='geom').apply(frozenset, axis=1).duplicated()].reset_index(drop=True))
indf_df = (indf_df[~indf_df.filter(like='geom').apply(frozenset, axis=1).duplicated()].reset_index(drop=True))

print('The number of possible locations with potential reversed LineString issues is ' + str(len(ind0_df) + len(indf_df)))

df_con_prob = pd.DataFrame(d_list)
print('The number of potentially problematic areas is: ' + str(len(df_con_prob.loc[df_con_prob['con_prob'] == 1])))
df_change = pd.DataFrame({'reach_id': reach_ids, 'change': 0})   


# ind0_df.loc[ind0_df['geom1_rch_id'] == '75330500033'] # ix = 49
# ind0_df.loc[ind0_df['geom1_rch_id'] == '75362200055'] # ix = 82
# ind0_df.loc[ind0_df['geom2_rch_id'] == '75362200025'] # ix = 78

# ind0_df.loc[ind0_df['geom1_rch_id'] == '21602601101'] # ix = 65


rch_ids_rev = [] # Reach IDs that need to have the coordinates of the linestring reversed 

## Now loop through ind0_df and figure out which reach needs to be fixed, if it needs to be fixed at all
for ix in range(len(ind0_df)):
    # print(ix,ind0_df['geom1_rch_id'][ix],ind0_df['geom2_rch_id'][ix])
    geom1_rch_id = ind0_df['geom1_rch_id'][ix]
    geom2_rch_id = ind0_df['geom2_rch_id'][ix]

    geom1_dist_out = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['dist_out'].to_list()[0]
    geom2_dist_out = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['dist_out'].to_list()[0]

    # geom1_type = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['end_reach'].to_list()[0]
    # geom2_type = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['end_reach'].to_list()[0]

    ## Check the number of geometric connections for both reaches 1 and 2 to figure out
    ## if the reaches involve lies at a junction or not. Checking both reaches and taking
    ## the minimum number of geometric connections ensures that I'm looking at the right
    ## area based on the 2 reaches involved (reach 1 could be at a junction and reach 2 could
    ## be adjacent to the junction)
    con_cnt_geom1 = con_geom1_rch_id.count(str(geom1_rch_id))
    con_cnt_geom2 = con_geom2_rch_id.count(str(geom2_rch_id))
    con_cnt = min([con_cnt_geom1, con_cnt_geom2])

    ## If the junction is a headwater a junction, set con_cnt = 3 to ensure junction code is entered below
    ## Previously, the code assumed every junction had an additional reach or end reach attached to all 
    ## segments in the junction, so con_cnt would equal 2 instead of 3 in this case
    ## Check that this code actually works!
    if (str(geom1_rch_id) in head_at_junc) | (str(geom2_rch_id) in head_at_junc):
        con_cnt = 3

    if (str(geom1_rch_id) in outlet_at_junc) | (str(geom2_rch_id) in outlet_at_junc):
        con_cnt = 3

    if con_cnt >= 3:

        # con_pts_pot_junc_ids = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == int(geom1_rch_id)]['geom2_rch_'].astype('str').to_list()
    
        ## Find indices of reach ID of interest in the generated initial connectivity that was created before reversing coordinates of linestrings
        indices = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
        ## Find reach IDs that are geometrically connected to reach ID of interest
        con_geom2_ids_indices = [con_geom2_rch_id[i] for i in indices]
        ## For reach IDs geometrically connected to reach ID of interest, count the number of geometric connections they have
        ## (this narrows it down to those reach IDs that might form the river junction with the reach ID of interest, since 1+
        ##  of the geometrically connected reach IDs might be a segment up/downstream of the reach ID of interest forming
        ##  the river junction)
        con_geom2_ids_cnt = [con_geom2_rch_id.count(i) for i in con_geom2_ids_indices]
        #If one of the associated reaches is an outlet at the junction, then set its count to 3 so that it is includedd
        #in the associated junction IDs (because its cnt will equal 2 instead of 3 and it would be removed otherwise)
        for i in range(len(con_geom2_ids_indices)):
            if (con_geom2_ids_indices[i] in outlet_at_junc) | (con_geom2_ids_indices[i] in head_at_junc):
                con_geom2_ids_cnt[i] = 3
        ## Potential IDs of other reaches associated with the river junction
        ## ("Potential" is used here because a reach ID geometrically connected to the reach ID of interest might have
        ##   another river junction attached to it than the one we are currently looking at)
        con_geom2_pot_junc_indices =  [i for i in range(len(con_geom2_ids_cnt)) if con_geom2_ids_cnt[i] >= 3]
        # con_geom2_pot_junc_indices =  [i for i in range(len(con_geom2_ids_cnt)) if con_geom2_ids_cnt[i] >= 2]
        ## Confirm which potential IDs are actually associated with the river junction of interest
        con_geom2_pot_junc_ids = [con_geom2_ids_indices[i] for i in con_geom2_pot_junc_indices]
        con_geom2_junc_ids = []
        for con_id in con_geom2_pot_junc_ids:
            con_geom1_pot_junc_indices = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == con_id]
            con_geom2_junc_ids_tmp = [con_geom2_rch_id[i] for i in con_geom1_pot_junc_indices]
            if str(geom1_rch_id) in con_geom2_junc_ids_tmp:
                con_geom2_junc_ids.append(con_id)

        ## Only keep reach IDs in the junction that are also connected to geom2_rch_id
        ids_geom2_rch = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == str(geom2_rch_id)]['geom2_rch_'].astype('str').to_list()
        ids_geom2_rch.append(geom2_rch_id)
        con_geom2_junc_ids = [i for i in con_geom2_junc_ids if i in ids_geom2_rch]
        con_geom2_junc_ids.append(str(geom1_rch_id))
        if geom2_rch_id not in con_geom2_junc_ids:
            con_geom2_junc_ids.append(str(geom2_rch_id))

        # junc_ids = [int(i) for i in con_geom2_junc_ids]
        junc_ids = [str(i) for i in con_geom2_junc_ids]
        reach_ids_indices = []
        for junc_id in junc_ids:
            reach_ids_indices.extend([i for i in range(len(reach_ids)) if reach_ids[i] == junc_id])
        junc_dist_out = [reach_ids_dist_out[i] for i in reach_ids_indices]

        ## Write code to determine if junction is multi upstream or multi downstream
        ## The reach with the greatest outlet distance is always going to be an upstream segment(?)
        ## So for each of the remaining segments, loop through and find neighbors that aren't associated
        ## with junctions, get outlet distances, and compare neighbor outlet distances to junction outlet distances
        max_junc_dist_out = junc_dist_out.index(max(junc_dist_out))
        min_junc_dist_out = junc_dist_out.index(min(junc_dist_out))
        up_dn = []
        for i in range(len(junc_ids)):
            if i == max_junc_dist_out:
                up_dn.append('up')
                continue

            if i == min_junc_dist_out:
                up_dn.append('dn')
                continue

            if junc_ids[i] in head_at_junc:
                up_dn.append('up')
                continue

            id_neigh = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
            id_neigh = id_neigh.loc[~id_neigh['geom2_rch_'].isin(junc_ids)]

            dists_neigh = []
            for j in id_neigh['geom2_rch_'].to_list():
                dists_neigh.extend(shp.loc[shp['reach_id'] == int(j)].dist_out.to_list())

            dist_id = junc_dist_out[i]
            # If up_dn_chk is True, this reach is downstream in the junction
            # up_dn_chk = all(dist_id > x for x in dists_neigh)
            # up_dn_chk fails in places where the reach of interest is downstream and at another junction,
            # 1 of the other reaches at the junction is downstream and the other reach is a headwater coming
            # into the junction that has a greater outlet distance than the reach ID of interest. So instead
            # use this code to check if any of the neighbors have a lower outlet distance. This code may break
            # or not work correctly in some locations, but I'm not sure yet.
            bool_comp_dist = [dist_id > x for x in dists_neigh]
            # Create list matching junc_ids that says if each reach is up or dn, max_junc_dist_out is automatically up
            # if up_dn_chk == True:
            if len(dists_neigh) == 1:
                #If there is only 1 other neighbor
                if any(x == True for x in bool_comp_dist):
                    up_dn.append('dn')
                else:
                    up_dn.append('up')
            else:
                #If the reach is attached to another junction
                if all(x == True for x in bool_comp_dist):
                    up_dn.append('dn')
                else:
                    up_dn.append('up')                   
        
        up_cnt = up_dn.count('up')
        dn_cnt = up_dn.count('dn')

        if (up_cnt >= 2) & (dn_cnt == 1):
            ## If it's a multi upstream junction:
            ## Now determine which reach ID is the downstream most reach ID of the junction, and store the reach IDs of every reach but 
            ## the most downstream segment as the ones to reverse coordinates of the linestring
            ##  ACTUALLY THE ABOVE COMMENT IS INCORRECT: reverse the corrdinates of the linestring for the most downstream segment

            ## If the 2 segments involved are one upstream in the junction and one downstream,
            ## reverse the coordinates of the linestring for the most downstream segment

            min_junc_dist_out = junc_dist_out.index(min(junc_dist_out))
            ## Only reversing if the downstream ID of the junction has matching starting coordinates with the other reach IDs
            if (junc_ids[min_junc_dist_out] == geom1_rch_id) | (junc_ids[min_junc_dist_out] == geom2_rch_id):
                rch_ids_rev.append(junc_ids[min_junc_dist_out])
                #Set change to 1 so I know a modification was made
                df_change.loc[df_change['reach_id'] == str(junc_ids[min_junc_dist_out]), 'change'] = 1

            #Set df_con_prob to zero since either there is no problem or it has been fixed in this location (with reversing) 
            df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0

        # elif (up_cnt == 1) & (dn_cnt == 2):
        elif (up_cnt == 1) & (dn_cnt >= 2):
            ## If it's a multi downstream junction:
            geom1_up_dn = up_dn[junc_ids.index(geom1_rch_id)]
            geom2_up_dn = up_dn[junc_ids.index(geom2_rch_id)]

            ## If both segments involved are one up and one down, flip the downstream segment
            if not all(x == 'dn' for x in [geom1_up_dn, geom2_up_dn]):
                if geom1_up_dn == 'dn':
                    rch_ids_rev.append(geom1_rch_id)
                    #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                    df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                    df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1
                elif geom2_up_dn == 'dn':
                    rch_ids_rev.append(geom2_rch_id)
                    #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                    df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                    df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

        else:
            print('New junction configuration exists! Refine code. Reaches involved:')
            print(geom1_rch_id)
            print(geom2_rch_id)
            temp_end = timer()
            print("Time Elapsed: " + str(((temp_end - start) / 60)) + " min")
            raise SystemExit(22) 
                   
                
    ## If it's not a junction, then use outlet distance to figure out which one to reverse
    elif geom1_dist_out < geom2_dist_out:
        rch_ids_rev.append(geom1_rch_id)
        #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
        df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
        df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1

    elif geom1_dist_out > geom2_dist_out:
        rch_ids_rev.append(geom2_rch_id)
        #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
        df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
        df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1



# df_con_prob.loc[df_con_prob['con_prob'] == 1].count() # Potentially problematic areas: 35
# df_change.loc[df_change['change'] == 1].count() # Areas with a modification applied: 4

# indf_df.loc[indf_df['geom2_rch_id'] == '81380100401'] # ix = 3

# indf_df.loc[indf_df['geom2_rch_id'] == '16270200071'] # ix = 1

# indf_df.loc[indf_df['geom1_rch_id'] == '21602601101'] # ix = 78


## Now loop through inf_df and figure out which reach needs to be fixed, if it needs to be fixed at all
for ix in range(len(indf_df)):
    # print(ix,ind0_df['geom1_rch_id'][ix],ind0_df['geom2_rch_id'][ix])
    geom1_rch_id = indf_df['geom1_rch_id'][ix]
    geom2_rch_id = indf_df['geom2_rch_id'][ix]

    geom1_dist_out = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['dist_out'].to_list()[0]
    geom2_dist_out = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['dist_out'].to_list()[0]

    # geom1_type = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['end_reach'].to_list()[0]
    # geom2_type = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['end_reach'].to_list()[0]

    ## Check the number of geometric connections for both reaches 1 and 2 to figure out
    ## if the reaches involve lies at a junction or not. Checking both reaches and taking
    ## the minimum number of geometric connections ensures that I'm looking at the right
    ## area based on the 2 reaches involved (reach 1 could be at a junction and reach 2 could
    ## be adjacent to the junction)
    con_cnt_geom1 = con_geom1_rch_id.count(str(geom1_rch_id))
    con_cnt_geom2 = con_geom2_rch_id.count(str(geom2_rch_id))
    con_cnt = min([con_cnt_geom1, con_cnt_geom2])

    if (str(geom1_rch_id) in head_at_junc) | (str(geom2_rch_id) in head_at_junc):
        con_cnt = 3

    if (str(geom1_rch_id) in outlet_at_junc) | (str(geom2_rch_id) in outlet_at_junc):
        con_cnt = 3

    if con_cnt >= 3:
        ## If the 2 segments involved are one upstream in the junction and one downstream,
        ## reverse the coordinates of the linestring for the upstream segment

        ## Find indices of reach ID of interest in the generated initial connectivity that was created before reversing coordinates of linestrings
        indices = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
        ## Find reach IDs that are geometrically connected to reach ID of interest
        con_geom2_ids_indices = [con_geom2_rch_id[i] for i in indices]
        ## For reach IDs geometrically connected to reach ID of interest, count the number of geometric connections they have
        ## (this narrows it down to those reach IDs that might form the river junction with the reach ID of interest, since 1+
        ##  of the geometrically connected reach IDs might be a segment up/downstream of the reach ID of interest forming
        ##  the river junction)
        con_geom2_ids_cnt = [con_geom2_rch_id.count(i) for i in con_geom2_ids_indices]
        #If one of the associated reaches is an outlet or headwater at the junction, then set its count to 3 so that it is includedd
        #in the associated junction IDs (because its cnt will equal 2 instead of 3 and it would be removed otherwise)
        for i in range(len(con_geom2_ids_indices)):
            if (con_geom2_ids_indices[i] in outlet_at_junc) | (con_geom2_ids_indices[i] in head_at_junc):
                con_geom2_ids_cnt[i] = 3
        ## Potential IDs of other reaches associated with the river junction
        ## ("Potential" is used here because a reach ID geometrically connected to the reach ID of interest might have
        ##   another river junction attached to it than the one we are currently looking at)
        con_geom2_pot_junc_indices =  [i for i in range(len(con_geom2_ids_cnt)) if con_geom2_ids_cnt[i] >= 3]
        # con_geom2_pot_junc_indices =  [i for i in range(len(con_geom2_ids_cnt)) if con_geom2_ids_cnt[i] >= 2]
        ## Confirm which potential IDs are actually associated with the river junction of interest
        con_geom2_pot_junc_ids = [con_geom2_ids_indices[i] for i in con_geom2_pot_junc_indices]
        con_geom2_junc_ids = []
        for con_id in con_geom2_pot_junc_ids:
            con_geom1_pot_junc_indices = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == con_id]
            con_geom2_junc_ids_tmp = [con_geom2_rch_id[i] for i in con_geom1_pot_junc_indices]
            if str(geom1_rch_id) in con_geom2_junc_ids_tmp:
                con_geom2_junc_ids.append(con_id)


        ## Only keep reach IDs in the junction that are also connected to geom2_rch_id
        ids_geom2_rch = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == str(geom2_rch_id)]['geom2_rch_'].astype('str').to_list()
        ids_geom2_rch.append(geom2_rch_id)
        con_geom2_junc_ids = [i for i in con_geom2_junc_ids if i in ids_geom2_rch]
        con_geom2_junc_ids.append(str(geom1_rch_id))
        if geom2_rch_id not in con_geom2_junc_ids:
            con_geom2_junc_ids.append(str(geom2_rch_id))

        # junc_ids = [int(i) for i in con_geom2_junc_ids]
        junc_ids = [str(i) for i in con_geom2_junc_ids]
        reach_ids_indices = []
        for junc_id in junc_ids:
            reach_ids_indices.extend([i for i in range(len(reach_ids)) if reach_ids[i] == junc_id])
        junc_dist_out = [reach_ids_dist_out[i] for i in reach_ids_indices]

        ## Write code to determine if junction is multi upstream or multi downstream
        ## The reach with the greatest outlet distance is always going to be an upstream segment(?)
        ## So for each of the remaining segments, loop through and find neighbors that aren't associated
        ## with junctions, get outlet distances, and compare neighbor outlet distances to junction outlet distances
        max_junc_dist_out = junc_dist_out.index(max(junc_dist_out))
        min_junc_dist_out = junc_dist_out.index(min(junc_dist_out))
        up_dn = []
        for i in range(len(junc_ids)):
            if i == max_junc_dist_out:
                up_dn.append('up')
                continue

            if i == min_junc_dist_out:
                up_dn.append('dn')
                continue

            if junc_ids[i] in head_at_junc:
                up_dn.append('up')
                continue

            id_neigh = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
            id_neigh = id_neigh.loc[~id_neigh['geom2_rch_'].isin(junc_ids)]

            dists_neigh = []
            for j in id_neigh['geom2_rch_'].to_list():
                dists_neigh.extend(shp.loc[shp['reach_id'] == int(j)].dist_out.to_list())

            dist_id = junc_dist_out[i]
            # up_dn_chk = all(dist_id > x for x in dists_neigh)
            bool_comp_dist = [dist_id > x for x in dists_neigh]
            # If True, this reach is downstream in the junction
            # Create list matching junc_ids that says if each reach is up or dn, max_junc_dist_out is automatically up
            if len(dists_neigh) == 1:
                #If there is only 1 other neighbor
                # if up_dn_chk == True:
                if all(x == True for x in bool_comp_dist):
                    up_dn.append('dn')
                else:
                    up_dn.append('up')
            else:
                #If the reach is attached to another junction
                # if up_dn_chk == False:
                if any(x == True for x in bool_comp_dist):
                    up_dn.append('dn')
                else:
                    up_dn.append('up') 
        
        up_cnt = up_dn.count('up')
        dn_cnt = up_dn.count('dn')

                    
        if (up_cnt >= 2) & (dn_cnt == 1):
            ## If it's a multi upstream junction:
            ## Now determine which reach ID is the downstream most reach ID of the junction, and store the reach IDs of every reach but 
            ## the most downstream segment as the ones to reverse coordinates of the linestring
            ## Only reverse if both reaches involved are actually a part of the junction
            if (geom1_rch_id in con_geom2_junc_ids) & (geom2_rch_id in con_geom2_junc_ids):

                # min_junc_dist_out = junc_dist_out.index(min(junc_dist_out))
                # ## Only reversing if the downstream ID of the junction has starting matching starting coordinates with the other reach IDs
                # if (junc_ids[min_junc_dist_out] == geom1_rch_id) | (junc_ids[min_junc_dist_out] == geom2_rch_id):
                #     rch_ids_rev.append(junc_ids[min_junc_dist_out])

                dist_out = [i for i in junc_dist_out if i != min(junc_dist_out)]
                dist_out_ind = []
                for d in dist_out:
                    dist_out_ind.extend([i for i in range(len(junc_dist_out)) if junc_dist_out[i] == d])

                ## Identifying the upstream segments in the junction
                junc_up_ids = [junc_ids[i] for i in dist_out_ind]

                ## Adding 1 or both upstream segments to reverse depending on if the 2 segments involved are 
                ## both the upstream segments
                if geom1_rch_id in junc_up_ids:
                    rch_ids_rev.append(geom1_rch_id)
                    #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                    df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                    df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1

                if geom2_rch_id in junc_up_ids:
                    rch_ids_rev.append(geom2_rch_id)
                    #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                    df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                    df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

                # rch_ids_rev.extend([junc_ids[i] for i in dist_out_ind])
                    
        # elif (up_cnt == 1) & (dn_cnt == 2):
        elif (up_cnt == 1) & (dn_cnt >= 2):
            ## If it's a multi downstream junction:
            geom1_up_dn = up_dn[junc_ids.index(geom1_rch_id)]
            geom2_up_dn = up_dn[junc_ids.index(geom2_rch_id)]

            ## If both segments involved are both the downstream segment do nothing
            ## If both segments involved are one up and one down, flip the upstream segment
            if not all(x == 'dn' for x in [geom1_up_dn, geom2_up_dn]):
                rch_ids_rev.append(junc_ids[up_dn.index('up')])
                #Set change to 1 so I know a modification was made
                df_change.loc[df_change['reach_id'] == str(junc_ids[min_junc_dist_out]), 'change'] = 1

            #Set df_con_prob to zero since either there is no problem or it has been fixed in this location (with reversing) 
            df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0


        else:
            print('New junction configuration exists! Refine code. Reaches involved:')
            print(geom1_rch_id)
            print(geom2_rch_id)
            temp_end = timer()
            print("Time Elapsed: " + str(((temp_end - start) / 60)) + " min")
            raise SystemExit(22) 


    ## If it's not a junction, then use outlet distance to figure out which one to reverse
    elif geom1_dist_out < geom2_dist_out:
        rch_ids_rev.append(geom2_rch_id)
        #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
        df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
        df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

    elif geom1_dist_out > geom2_dist_out:
        rch_ids_rev.append(geom1_rch_id)
        #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
        df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
        df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1


# df_con_prob.loc[df_con_prob['con_prob'] == 1].count() # Potentially problematic areas: 27
# df_change.loc[df_change['change'] == 1].count() # Areas with a modification applied: 10

print('The number of potentially problematic areas remaining is: ' + str(len(df_con_prob.loc[df_con_prob['con_prob'] == 1])))
print('The number of areas modified is: ' + str(len(df_change.loc[df_change['change'] == 1])))



rch_ids_rev = list(set(rch_ids_rev))
# len(rch_ids_rev)

print(rch_ids_rev)
print('len(rch_ids_rev): ' + str(len(rch_ids_rev)))
if len(rch_ids_rev) == 0:
    print('No more reaches to reverse!')
    # raise SystemExit(22) 
else:
    rev_chk = [i for i in rch_ids_rev if i in store_rch_ids_rev]
    if len(rev_chk) > 0:
        print('A reach has been reversed multiple times -- check code!')
        print(rev_chk)
        temp_end = timer()
        print("Time Elapsed: " + str(((temp_end - start) / 60)) + " min")
        raise SystemExit(22) 


    ### Maybe I just need to loop through the reaches that I've already stored as need to be reversed
    for r in rch_ids_rev:
        print('Reversing ' + str(r))
        ix = shp.loc[shp['reach_id'] == int(r)].index.to_list()[0]
        # lines[ix] = rev_crds(geom1)
        shp.geometry[ix] = rev_crds(shp.geometry[ix])
        store_rch_ids_rev.append(r)

    print('Finished reversing initial reaches, finding new geometric intersections')

    ### Make new point layer
    geom_con_fname = geom_con_fname.replace('_pts.shp', '_pts_' + str(counter) + '.shp')
    # geom_con_fname = geom_con_fname.replace('_pts.shp', '_v2code_pts_' + str(counter) + '.shp')
    # geom_con_fname = 'hb81_subset_v17a_FO_Main_v2code_pts_' + str(counter) + '.shp'

    print('geom_con_fname: ' + geom_con_fname)

    point_lyr = fiona.open(geom_con_fname, mode='w', driver='ESRI Shapefile', schema=pt_schema, crs =crs)


    # Now only modify geometric intersections for areas where LineStrings were reversed/changes were made (i.e., 'problematic')
    reach_id_chg = df_change.loc[df_change['change'] == 1]['reach_id'].to_list()
    reach_id_chg = [int(i) for i in reach_id_chg]
    # shp_sub = shp.loc[shp['reach_id'].isin(reach_id_chg)]

    ### NEW WAY

    for ix, r in shp.iterrows():
        geom1 = shapely.geometry.shape(shp.geometry[ix])
        geom1_rch_id = shp.reach_id[ix]

        if geom1_rch_id not in reach_id_chg:
            continue

        selected_reach = shp.loc[shp['reach_id'] == int(geom1_rch_id)]
        #Finding reaches within search distance (11 km)
        reaches_win_dist = shp.assign(distance=shp.apply(lambda x: x.geometry.distance(selected_reach.geometry.iloc[0]), axis=1)).query(f'distance <= {0.1}')
        reaches_win_dist = reaches_win_dist[reaches_win_dist.reach_id != geom1_rch_id]


        #Only comparing selected_reach to reaches within search distance (minimizes computational needs)
        for ix2, r2 in reaches_win_dist.iterrows(): 
            # print(ix2,r2['reach_id'])
            geom2 = shapely.geometry.shape(reaches_win_dist.geometry[ix2])
            geom2_rch_id = reaches_win_dist.reach_id[ix2]

            ## Don't need to record intersection if reach ID is the same
            if geom1_rch_id == geom2_rch_id:
                continue

            if geom1.distance(geom2) < eps: 
                point = shapely.ops.nearest_points(geom1, geom2)[0]
                # print(point)

                ## Sometimes the point won't exactly match up, so need to find the nearest point to connect the 2 segments
                found = False
                for i in range(len(geom1.coords.xy[0])):
                    # print(i)
                    x = geom1.coords.xy[0][i]
                    y = geom1.coords.xy[1][i]
                    tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})

                    if(tmp_pt == point):
                        found = True     

                        ind = i

                        point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                        'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                        'geom2_rch_id': geom2_rch_id.item(),
                                                        'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                        'ind_intr': ind}})

                        break

                if found == False:
                    dist_point = []
                    for i in range(len(geom1.coords.xy[0])):
                        x = geom1.coords.xy[0][i]
                        y = geom1.coords.xy[1][i]
                        tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})
                        dist_point.append(point.distance(tmp_pt))
                    
                    ind = dist_point.index(min(dist_point))

                    point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                    'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                    'geom2_rch_id': geom2_rch_id.item(),
                                                    'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                    'ind_intr': ind}})

    point_lyr.close()
    counter = counter + 1

    geom_con_filt = filter(lambda f: f['properties']['reach_id'] not in reach_id_chg, geom_con)
    geom_con_filt = gpd.GeoDataFrame.from_features([feature for feature in geom_con], crs=crs)
    columns = list(geom_con.meta["schema"]["properties"]) + ["geometry"]
    geom_con_filt = geom_con_filt[columns]
    geom_con_filt = geom_con_filt.loc[~geom_con_filt['geom1_rch_'].isin(reach_id_chg)]
    geom_con_filt.to_file(geom_con_fname, mode="a")

    # end - start # 1.514 seconds

    # tst = gpd.read_file(geom_con_fname)
    # tst.loc[tst['geom1_rch_'] == '81380100311']



    ### Now can start while loop
    # start = timer()
    # end = timer()
    # end - start # 1.514 seconds

    print('Starting while loop for finding further changes')

    while counter < 10:
        print('The while loop is in iteration ' + str(counter))

        geom_con_shp = gpd.read_file(geom_con_fname)
        geom_con_shp['geom1_rch_'] = geom_con_shp['geom1_rch_'].astype('str')
        geom_con_shp['geom2_rch_'] = geom_con_shp['geom2_rch_'].astype('str')

        # geom_con_shp.loc[geom_con_shp['geom1_rch_'] == '81380100311']

        geom_con = fiona.open(geom_con_fname)

        con_geom1_rch_id = []
        con_geom2_rch_id = []
        con_geom1_n_pnts = []
        con_ind_intr = []
        con_closest = []
        for con in geom_con:
            con_geom = shapely.geometry.shape(con.geometry)
            
            con_geom1_rch_id.append(str(con["properties"]["geom1_rch_"]))
            con_geom2_rch_id.append(str(con["properties"]["geom2_rch_"]))
            con_geom1_n_pnts.append(con["properties"]["geom1_n_pn"])
            con_ind_intr.append(con["properties"]["ind_intr"])
            if con["properties"]["geom1_n_pn"] == 2:
                if con["properties"]["ind_intr"] == 1:
                    val = 1
                else:
                    val = 0
            else:
                val = min([0, con["properties"]["geom1_n_pn"]], key = lambda x: abs(x - con["properties"]["ind_intr"]))
            con_closest.append(val) # If val != 0, then geom2 is upstream


        ### NOW ONLY REVISIT PLACES THAT HAVE BEEN IDENTIFIED AS STILL PROBLEMATIC OR HAVE BEEN CHANGED
        reach_id_chg = df_change.loc[df_change['change'] == 1]['reach_id'].to_list()
        reach_id_chg = [int(i) for i in reach_id_chg]
        reach_id_prob = df_con_prob.loc[df_con_prob['con_prob'] == 1]['geom1_rch_id'].to_list()
        reach_id_prob = [int(i) for i in reach_id_prob]
        reach_id_prob.extend(reach_id_chg)

        shp_sub = shp.loc[shp['reach_id'].isin(reach_id_prob)]

        ind0_geom1_rch_id = []
        ind0_geom2_rch_id = []

        indf_geom1_rch_id = []
        indf_geom2_rch_id = []

        d_list = []

        for ix, r in shp_sub.iterrows():
            geom1 = lines[ix]
            geom1_rch_id = reach_ids[ix]

            con_geom1_rch_id_ind = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
            
            selected_reach = shp_sub.loc[shp_sub['reach_id'] == int(geom1_rch_id)]
            #Finding reaches within search distance (11 km)
            reaches_win_dist = shp.assign(distance=shp.apply(lambda x: x.geometry.distance(selected_reach.geometry.iloc[0]), axis=1)).query(f'distance <= {0.1}')
            reaches_win_dist = reaches_win_dist[reaches_win_dist.reach_id != geom1_rch_id]

            #Only comparing selected_reach to reaches within search distance (minimizes computational needs)
            for ix2, r2 in reaches_win_dist.iterrows(): 
                # print(ix2,r2['reach_id'])
                geom2 = shapely.geometry.shape(reaches_win_dist.geometry[ix2])
                geom2_rch_id = reaches_win_dist.reach_id.astype('str')[ix2]

                ## Don't need to record intersection if reach ID is the same
                if geom1_rch_id == geom2_rch_id:
                    continue

                if geom1.distance(geom2) < eps: 
                    con_geom2_rch_id_ind = [i for i in range(len(con_geom2_rch_id)) if con_geom2_rch_id[i] == str(geom2_rch_id)]
                    con_geom1_geom2_ind = [i for i in con_geom1_rch_id_ind if i in con_geom2_rch_id_ind]

                    con_geom2_opp_rch_id_ind = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom2_rch_id)]
                    con_geom1_opp_rch_id_ind = [i for i in range(len(con_geom2_rch_id)) if con_geom2_rch_id[i] == str(geom1_rch_id)]
                    con_geom1_geom2_opp_ind = [i for i in con_geom1_opp_rch_id_ind if i in con_geom2_opp_rch_id_ind]

                    con_geom1_closest = [con_closest[i] for i in con_geom1_geom2_ind]    
                    con_geom2_closest = [con_closest[i] for i in con_geom1_geom2_opp_ind]    

                # if geom1.distance(geom2) < eps: 

                    ## Check for locations where 2 segments say they are both downstream of one another 
                    if (con_geom1_closest[0] == 0) & (con_geom2_closest[0] == 0):
                        # rch_id_dn = geom_dist_out.index(min(geom_dist_out))
                        # if rch_id_dn == 0:
                            # ind0_geom1_rch_id.append(str(geom1_rch_id))
                        # if (geom1_rch_id not in ind0_geom1_rch_id) | (geom2_rch_id not in ind0_geom2_rch_id):
                        ind0_geom1_rch_id.append(str(geom1_rch_id))
                        ind0_geom2_rch_id.append(str(geom2_rch_id))

                        #Adding reaches to dataframe identifying if it's potentially a problem (1) or not (0)
                        d = {"geom1_rch_id": geom1_rch_id, "geom2_rch_id": geom2_rch_id, "con_prob": 1}
                        d_list.append(d)

                    ## Check for locations where 2 segments say they are both upstream of one another 
                    elif (con_geom1_closest[0] != 0) & (con_geom2_closest[0] != 0):
                        # if (geom1_rch_id not in indf_geom1_rch_id) | (geom2_rch_id not in indf_geom2_rch_id):
                        indf_geom1_rch_id.append(str(geom1_rch_id))
                        indf_geom2_rch_id.append(str(geom2_rch_id))

                        d = {"geom1_rch_id": geom1_rch_id, "geom2_rch_id": geom2_rch_id, "con_prob": 1}
                        d_list.append(d)
                    
                    else:
                        d = {"geom1_rch_id": geom1_rch_id, "geom2_rch_id": geom2_rch_id, "con_prob": 0}
                        d_list.append(d)


        ind0_df = pd.DataFrame({'geom1_rch_id': ind0_geom1_rch_id, 'geom2_rch_id': ind0_geom2_rch_id})
        indf_df = pd.DataFrame({'geom1_rch_id': indf_geom1_rch_id, 'geom2_rch_id': indf_geom2_rch_id})

        ind0_df = (ind0_df[~ind0_df.filter(like='geom').apply(frozenset, axis=1).duplicated()].reset_index(drop=True))
        indf_df = (indf_df[~indf_df.filter(like='geom').apply(frozenset, axis=1).duplicated()].reset_index(drop=True))

        print('The number of possible locations with potential reversed LineString issues is ' + str(len(ind0_df) + len(indf_df)))

        df_con_prob = pd.DataFrame(d_list)               
        # df_con_prob.loc[df_con_prob['con_prob'] == 1].count() # Potentially problematic areas: 1
        df_change = pd.DataFrame({'reach_id': reach_ids, 'change': 0})   




        rch_ids_rev = [] # Reach IDs that need to have the coordinates of the linestring reversed 

        ## Now loop through ind0_df and figure out which reach needs to be fixed, if it needs to be fixed at all
        for ix in range(len(ind0_df)):
            geom1_rch_id = ind0_df['geom1_rch_id'][ix]
            geom2_rch_id = ind0_df['geom2_rch_id'][ix]

            geom1_dist_out = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['dist_out'].to_list()[0]
            geom2_dist_out = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['dist_out'].to_list()[0]

            # geom1_type = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['end_reach'].to_list()[0]
            # geom2_type = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['end_reach'].to_list()[0]

            ## Check the number of geometric connections for both reaches 1 and 2 to figure out
            ## if the reaches involve lies at a junction or not. Checking both reaches and taking
            ## the minimum number of geometric connections ensures that I'm looking at the right
            ## area based on the 2 reaches involved (reach 1 could be at a junction and reach 2 could
            ## be adjacent to the junction)
            con_cnt_geom1 = con_geom1_rch_id.count(str(geom1_rch_id))
            con_cnt_geom2 = con_geom2_rch_id.count(str(geom2_rch_id))
            con_cnt = min([con_cnt_geom1, con_cnt_geom2])

            ## If the junction is a headwater a junction, set con_cnt = 3 to ensure junction code is entered below
            ## Previously, the code assumed every junction had an additional reach or end reach attached to all 
            ## segments in the junction, so con_cnt would equal 2 instead of 3 in this case
            ## Check that this code actually works!
            if (str(geom1_rch_id) in head_at_junc) | (str(geom2_rch_id) in head_at_junc):
                con_cnt = 3

            if (str(geom1_rch_id) in outlet_at_junc) | (str(geom2_rch_id) in outlet_at_junc):
                con_cnt = 3

            if con_cnt >= 3:

                # con_pts_pot_junc_ids = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == int(geom1_rch_id)]['geom2_rch_'].astype('str').to_list()
            
                ## Find indices of reach ID of interest in the generated initial connectivity that was created before reversing coordinates of linestrings
                indices = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
                ## Find reach IDs that are geometrically connected to reach ID of interest
                con_geom2_ids_indices = [con_geom2_rch_id[i] for i in indices]
                ## For reach IDs geometrically connected to reach ID of interest, count the number of geometric connections they have
                ## (this narrows it down to those reach IDs that might form the river junction with the reach ID of interest, since 1+
                ##  of the geometrically connected reach IDs might be a segment up/downstream of the reach ID of interest forming
                ##  the river junction)
                con_geom2_ids_cnt = [con_geom2_rch_id.count(i) for i in con_geom2_ids_indices]
                #If one of the associated reaches is an outlet at the junction, then set its count to 3 so that it is includedd
                #in the associated junction IDs (because its cnt will equal 2 instead of 3 and it would be removed otherwise)
                for i in range(len(con_geom2_ids_indices)):
                    if (con_geom2_ids_indices[i] in outlet_at_junc) | (con_geom2_ids_indices[i] in head_at_junc):
                        con_geom2_ids_cnt[i] = 3
                ## Potential IDs of other reaches associated with the river junction
                ## ("Potential" is used here because a reach ID geometrically connected to the reach ID of interest might have
                ##   another river junction attached to it than the one we are currently looking at)
                con_geom2_pot_junc_indices =  [i for i in range(len(con_geom2_ids_cnt)) if con_geom2_ids_cnt[i] >= 3]
                # con_geom2_pot_junc_indices =  [i for i in range(len(con_geom2_ids_cnt)) if con_geom2_ids_cnt[i] >= 2]
                ## Confirm which potential IDs are actually associated with the river junction of interest
                con_geom2_pot_junc_ids = [con_geom2_ids_indices[i] for i in con_geom2_pot_junc_indices]
                con_geom2_junc_ids = []
                for con_id in con_geom2_pot_junc_ids:
                    con_geom1_pot_junc_indices = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == con_id]
                    con_geom2_junc_ids_tmp = [con_geom2_rch_id[i] for i in con_geom1_pot_junc_indices]
                    if str(geom1_rch_id) in con_geom2_junc_ids_tmp:
                        con_geom2_junc_ids.append(con_id)

                ## Only keep reach IDs in the junction that are also connected to geom2_rch_id
                ids_geom2_rch = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == str(geom2_rch_id)]['geom2_rch_'].astype('str').to_list()
                ids_geom2_rch.append(geom2_rch_id)
                con_geom2_junc_ids = [i for i in con_geom2_junc_ids if i in ids_geom2_rch]
                con_geom2_junc_ids.append(str(geom1_rch_id))
                if geom2_rch_id not in con_geom2_junc_ids:
                    con_geom2_junc_ids.append(str(geom2_rch_id))

                # junc_ids = [int(i) for i in con_geom2_junc_ids]
                junc_ids = [str(i) for i in con_geom2_junc_ids]
                reach_ids_indices = []
                for junc_id in junc_ids:
                    reach_ids_indices.extend([i for i in range(len(reach_ids)) if reach_ids[i] == junc_id])
                junc_dist_out = [reach_ids_dist_out[i] for i in reach_ids_indices]

                ## Write code to determine if junction is multi upstream or multi downstream
                ## The reach with the greatest outlet distance is always going to be an upstream segment(?)
                ## So for each of the remaining segments, loop through and find neighbors that aren't associated
                ## with junctions, get outlet distances, and compare neighbor outlet distances to junction outlet distances
                max_junc_dist_out = junc_dist_out.index(max(junc_dist_out))
                min_junc_dist_out = junc_dist_out.index(min(junc_dist_out))
                up_dn = []
                for i in range(len(junc_ids)):
                    if i == max_junc_dist_out:
                        up_dn.append('up')
                        continue

                    if i == min_junc_dist_out:
                        up_dn.append('dn')
                        continue

                    if junc_ids[i] in head_at_junc:
                        up_dn.append('up')
                        continue

                    id_neigh = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
                    id_neigh = id_neigh.loc[~id_neigh['geom2_rch_'].isin(junc_ids)]

                    dists_neigh = []
                    for j in id_neigh['geom2_rch_'].to_list():
                        dists_neigh.extend(shp.loc[shp['reach_id'] == int(j)].dist_out.to_list())

                    dist_id = junc_dist_out[i]
                    # If up_dn_chk is True, this reach is downstream in the junction
                    # up_dn_chk = all(dist_id > x for x in dists_neigh)
                    # up_dn_chk fails in places where the reach of interest is downstream and at another junction,
                    # 1 of the other reaches at the junction is downstream and the other reach is a headwater coming
                    # into the junction that has a greater outlet distance than the reach ID of interest. So instead
                    # use this code to check if any of the neighbors have a lower outlet distance. This code may break
                    # or not work correctly in some locations, but I'm not sure yet.
                    bool_comp_dist = [dist_id > x for x in dists_neigh]
                    # Create list matching junc_ids that says if each reach is up or dn, max_junc_dist_out is automatically up
                    # if up_dn_chk == True:
                    if len(dists_neigh) == 1:
                        #If there is only 1 other neighbor
                        if any(x == True for x in bool_comp_dist):
                            up_dn.append('dn')
                        else:
                            up_dn.append('up')
                    else:
                        #If the reach is attached to another junction
                        if all(x == True for x in bool_comp_dist):
                            up_dn.append('dn')
                        else:
                            up_dn.append('up')                   
                
                up_cnt = up_dn.count('up')
                dn_cnt = up_dn.count('dn')

                if (up_cnt >= 2) & (dn_cnt == 1):
                    ## If it's a multi upstream junction:
                    ## Now determine which reach ID is the downstream most reach ID of the junction, and store the reach IDs of every reach but 
                    ## the most downstream segment as the ones to reverse coordinates of the linestring
                    ##  ACTUALLY THE ABOVE COMMENT IS INCORRECT: reverse the corrdinates of the linestring for the most downstream segment

                    ## If the 2 segments involved are one upstream in the junction and one downstream,
                    ## reverse the coordinates of the linestring for the most downstream segment

                    min_junc_dist_out = junc_dist_out.index(min(junc_dist_out))
                    ## Only reversing if the downstream ID of the junction has matching starting coordinates with the other reach IDs
                    if (junc_ids[min_junc_dist_out] == geom1_rch_id) | (junc_ids[min_junc_dist_out] == geom2_rch_id):
                        rch_ids_rev.append(junc_ids[min_junc_dist_out])
                        #Set change to 1 so I know a modification was made
                        df_change.loc[df_change['reach_id'] == str(junc_ids[min_junc_dist_out]), 'change'] = 1

                    #Set df_con_prob to zero since either there is no problem or it has been fixed in this location (with reversing) 
                    df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0

                # elif (up_cnt == 1) & (dn_cnt == 2):
                elif (up_cnt == 1) & (dn_cnt >= 2):
                    ## If it's a multi downstream junction:
                    geom1_up_dn = up_dn[junc_ids.index(geom1_rch_id)]
                    geom2_up_dn = up_dn[junc_ids.index(geom2_rch_id)]

                    ## If both segments involved are one up and one down, flip the downstream segment
                    if not all(x == 'dn' for x in [geom1_up_dn, geom2_up_dn]):
                        if geom1_up_dn == 'dn':
                            rch_ids_rev.append(geom1_rch_id)
                            #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                            df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                            df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1
                        elif geom2_up_dn == 'dn':
                            rch_ids_rev.append(geom2_rch_id)
                            #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                            df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                            df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

                else:
                    print('New junction configuration exists! Refine code. Reaches involved:')
                    print(geom1_rch_id)
                    print(geom2_rch_id)
                    temp_end = timer()
                    print("Time Elapsed: " + str(((temp_end - start) / 60)) + " min")
                    raise SystemExit(22) 

                        
            ## If it's not a junction, then use outlet distance to figure out which one to reverse
            elif geom1_dist_out < geom2_dist_out:
                rch_ids_rev.append(geom1_rch_id)
                #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1

            elif geom1_dist_out > geom2_dist_out:
                rch_ids_rev.append(geom2_rch_id)
                #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1


        # df_con_prob.loc[df_con_prob['con_prob'] == 1].count() # Potentially problematic areas: 0
        # df_change.loc[df_change['change'] == 1].count() # Areas with a modification applied: 0


        ## Now loop through inf_df and figure out which reach needs to be fixed, if it needs to be fixed at all
        for ix in range(len(indf_df)):
            geom1_rch_id = indf_df['geom1_rch_id'][ix]
            geom2_rch_id = indf_df['geom2_rch_id'][ix]

            geom1_dist_out = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['dist_out'].to_list()[0]
            geom2_dist_out = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['dist_out'].to_list()[0]

            # geom1_type = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['end_reach'].to_list()[0]
            # geom2_type = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['end_reach'].to_list()[0]

            ## Check the number of geometric connections for both reaches 1 and 2 to figure out
            ## if the reaches involve lies at a junction or not. Checking both reaches and taking
            ## the minimum number of geometric connections ensures that I'm looking at the right
            ## area based on the 2 reaches involved (reach 1 could be at a junction and reach 2 could
            ## be adjacent to the junction)
            con_cnt_geom1 = con_geom1_rch_id.count(str(geom1_rch_id))
            con_cnt_geom2 = con_geom2_rch_id.count(str(geom2_rch_id))
            con_cnt = min([con_cnt_geom1, con_cnt_geom2])

            if (str(geom1_rch_id) in head_at_junc) | (str(geom2_rch_id) in head_at_junc):
                con_cnt = 3

            if (str(geom1_rch_id) in outlet_at_junc) | (str(geom2_rch_id) in outlet_at_junc):
                con_cnt = 3

            if con_cnt >= 3:
                ## If the 2 segments involved are one upstream in the junction and one downstream,
                ## reverse the coordinates of the linestring for the upstream segment

                ## Find indices of reach ID of interest in the generated initial connectivity that was created before reversing coordinates of linestrings
                indices = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
                ## Find reach IDs that are geometrically connected to reach ID of interest
                con_geom2_ids_indices = [con_geom2_rch_id[i] for i in indices]
                ## For reach IDs geometrically connected to reach ID of interest, count the number of geometric connections they have
                ## (this narrows it down to those reach IDs that might form the river junction with the reach ID of interest, since 1+
                ##  of the geometrically connected reach IDs might be a segment up/downstream of the reach ID of interest forming
                ##  the river junction)
                con_geom2_ids_cnt = [con_geom2_rch_id.count(i) for i in con_geom2_ids_indices]
                #If one of the associated reaches is an outlet at the junction, then set its count to 3 so that it is includedd
                #in the associated junction IDs (because its cnt will equal 2 instead of 3 and it would be removed otherwise)
                for i in range(len(con_geom2_ids_indices)):
                    if (con_geom2_ids_indices[i] in outlet_at_junc) | (con_geom2_ids_indices[i] in head_at_junc):
                        con_geom2_ids_cnt[i] = 3
                ## Potential IDs of other reaches associated with the river junction
                ## ("Potential" is used here because a reach ID geometrically connected to the reach ID of interest might have
                ##   another river junction attached to it than the one we are currently looking at)
                con_geom2_pot_junc_indices =  [i for i in range(len(con_geom2_ids_cnt)) if con_geom2_ids_cnt[i] >= 3]
                # con_geom2_pot_junc_indices =  [i for i in range(len(con_geom2_ids_cnt)) if con_geom2_ids_cnt[i] >= 2]
                ## Confirm which potential IDs are actually associated with the river junction of interest
                con_geom2_pot_junc_ids = [con_geom2_ids_indices[i] for i in con_geom2_pot_junc_indices]
                con_geom2_junc_ids = []
                for con_id in con_geom2_pot_junc_ids:
                    con_geom1_pot_junc_indices = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == con_id]
                    con_geom2_junc_ids_tmp = [con_geom2_rch_id[i] for i in con_geom1_pot_junc_indices]
                    if str(geom1_rch_id) in con_geom2_junc_ids_tmp:
                        con_geom2_junc_ids.append(con_id)


                ## Only keep reach IDs in the junction that are also connected to geom2_rch_id
                ids_geom2_rch = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == str(geom2_rch_id)]['geom2_rch_'].astype('str').to_list()
                ids_geom2_rch.append(geom2_rch_id)
                con_geom2_junc_ids = [i for i in con_geom2_junc_ids if i in ids_geom2_rch]
                con_geom2_junc_ids.append(str(geom1_rch_id))
                if geom2_rch_id not in con_geom2_junc_ids:
                    con_geom2_junc_ids.append(str(geom2_rch_id))

                # junc_ids = [int(i) for i in con_geom2_junc_ids]
                junc_ids = [str(i) for i in con_geom2_junc_ids]
                reach_ids_indices = []
                for junc_id in junc_ids:
                    reach_ids_indices.extend([i for i in range(len(reach_ids)) if reach_ids[i] == junc_id])
                junc_dist_out = [reach_ids_dist_out[i] for i in reach_ids_indices]

                ## Write code to determine if junction is multi upstream or multi downstream
                ## The reach with the greatest outlet distance is always going to be an upstream segment(?)
                ## So for each of the remaining segments, loop through and find neighbors that aren't associated
                ## with junctions, get outlet distances, and compare neighbor outlet distances to junction outlet distances
                max_junc_dist_out = junc_dist_out.index(max(junc_dist_out))
                min_junc_dist_out = junc_dist_out.index(min(junc_dist_out))
                up_dn = []
                for i in range(len(junc_ids)):
                    if i == max_junc_dist_out:
                        up_dn.append('up')
                        continue

                    if i == min_junc_dist_out:
                        up_dn.append('dn')
                        continue

                    if junc_ids[i] in head_at_junc:
                        up_dn.append('up')
                        continue

                    id_neigh = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
                    id_neigh = id_neigh.loc[~id_neigh['geom2_rch_'].isin(junc_ids)]

                    dists_neigh = []
                    for j in id_neigh['geom2_rch_'].to_list():
                        dists_neigh.extend(shp.loc[shp['reach_id'] == int(j)].dist_out.to_list())

                    dist_id = junc_dist_out[i]
                    # up_dn_chk = all(dist_id > x for x in dists_neigh)
                    bool_comp_dist = [dist_id > x for x in dists_neigh]
                    # If True, this reach is downstream in the junction
                    # Create list matching junc_ids that says if each reach is up or dn, max_junc_dist_out is automatically up
                    if len(dists_neigh) == 1:
                        #If there is only 1 other neighbor
                        # if up_dn_chk == True:
                        if all(x == True for x in bool_comp_dist):
                            up_dn.append('dn')
                        else:
                            up_dn.append('up')
                    else:
                        #If the reach is attached to another junction
                        # if up_dn_chk == False:
                        if any(x == True for x in bool_comp_dist):
                            up_dn.append('dn')
                        else:
                            up_dn.append('up') 
                
                up_cnt = up_dn.count('up')
                dn_cnt = up_dn.count('dn')

                            
                if (up_cnt >= 2) & (dn_cnt == 1):
                    ## If it's a multi upstream junction:
                    ## Now determine which reach ID is the downstream most reach ID of the junction, and store the reach IDs of every reach but 
                    ## the most downstream segment as the ones to reverse coordinates of the linestring
                    ## Only reverse if both reaches involved are actually a part of the junction
                    if (geom1_rch_id in con_geom2_junc_ids) & (geom2_rch_id in con_geom2_junc_ids):

                        # min_junc_dist_out = junc_dist_out.index(min(junc_dist_out))
                        # ## Only reversing if the downstream ID of the junction has starting matching starting coordinates with the other reach IDs
                        # if (junc_ids[min_junc_dist_out] == geom1_rch_id) | (junc_ids[min_junc_dist_out] == geom2_rch_id):
                        #     rch_ids_rev.append(junc_ids[min_junc_dist_out])

                        dist_out = [i for i in junc_dist_out if i != min(junc_dist_out)]
                        dist_out_ind = []
                        for d in dist_out:
                            dist_out_ind.extend([i for i in range(len(junc_dist_out)) if junc_dist_out[i] == d])

                        ## Identifying the upstream segments in the junction
                        junc_up_ids = [junc_ids[i] for i in dist_out_ind]

                        ## Adding 1 or both upstream segments to reverse depending on if the 2 segments involved are 
                        ## both the upstream segments
                        if geom1_rch_id in junc_up_ids:
                            rch_ids_rev.append(geom1_rch_id)
                            #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                            df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                            df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1

                        if geom2_rch_id in junc_up_ids:
                            rch_ids_rev.append(geom2_rch_id)
                            #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                            df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                            df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

                        # rch_ids_rev.extend([junc_ids[i] for i in dist_out_ind])
                            
                # elif (up_cnt == 1) & (dn_cnt == 2):
                elif (up_cnt == 1) & (dn_cnt >= 2):
                    ## If it's a multi downstream junction:
                    geom1_up_dn = up_dn[junc_ids.index(geom1_rch_id)]
                    geom2_up_dn = up_dn[junc_ids.index(geom2_rch_id)]

                    ## If both segments involved are both the downstream segment do nothing
                    ## If both segments involved are one up and one down, flip the upstream segment
                    if not all(x == 'dn' for x in [geom1_up_dn, geom2_up_dn]):
                        rch_ids_rev.append(junc_ids[up_dn.index('up')])
                        #Set change to 1 so I know a modification was made
                        df_change.loc[df_change['reach_id'] == str(junc_ids[min_junc_dist_out]), 'change'] = 1

                    #Set df_con_prob to zero since either there is no problem or it has been fixed in this location (with reversing) 
                    df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0


                else:
                    print('New junction configuration exists! Refine code. Reaches involved:')
                    print(geom1_rch_id)
                    print(geom2_rch_id)
                    temp_end = timer()
                    print("Time Elapsed: " + str(((temp_end - start) / 60)) + " min")
                    raise SystemExit(22) 


            ## If it's not a junction, then use outlet distance to figure out which one to reverse
            elif geom1_dist_out < geom2_dist_out:
                rch_ids_rev.append(geom2_rch_id)
                #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

            elif geom1_dist_out > geom2_dist_out:
                rch_ids_rev.append(geom1_rch_id)
                #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1


        # df_con_prob.loc[df_con_prob['con_prob'] == 1].count() # Potentially problematic areas: 0
        # df_change.loc[df_change['change'] == 1].count() # Areas with a modification applied: 1

        print('The number of potentially problematic areas remaining is: ' + str(len(df_con_prob.loc[df_con_prob['con_prob'] == 1])))
        print('The number of areas modified is: ' + str(len(df_change.loc[df_change['change'] == 1])))


        rch_ids_rev = list(set(rch_ids_rev))
        # len(rch_ids_rev)

        print(rch_ids_rev)
        print('len(rch_ids_rev): ' + str(len(rch_ids_rev)))
        if len(rch_ids_rev) == 0:
            print('No more reaches to reverse!')
            print('The counter is: ' + str(counter))
            break

        rev_chk = [i for i in rch_ids_rev if i in store_rch_ids_rev]
        if len(rev_chk) > 0:
            print('A reach has been reversed multiple times -- check code!')
            print('The counter is: ' + str(counter))
            print(rev_chk)
            temp_end = timer()
            print("Time Elapsed: " + str(((temp_end - start) / 60)) + " min")
            raise SystemExit(22) 


        ### Maybe I just need to loop through the reaches that I've already stored as need to be reversed
        for r in rch_ids_rev:
            print('Reversing ' + str(r))
            ix = shp.loc[shp['reach_id'] == int(r)].index.to_list()[0]
            # lines[ix] = rev_crds(geom1)
            shp.geometry[ix] = rev_crds(shp.geometry[ix])
            store_rch_ids_rev.append(r)

        ### Make new point layer
        # geom_con_fname = geom_con_fname.replace('_pts.shp', '_pts_' + str(counter) + '.shp')
        # geom_con_fname = geom_con_fname.replace('pts_*.shp', '_' + str(counter) + '.shp')
        geom_con_fname = re.sub(r'pts_.*', 'pts_' + str(counter) + '.shp', geom_con_fname)
        # geom_con_fname = 'hb81_subset_v17a_FO_Main_v2code_pts_' + str(counter) + '.shp'

        print('geom_con_fname: ' + geom_con_fname)

        point_lyr = fiona.open(geom_con_fname, mode='w', driver='ESRI Shapefile', schema=pt_schema, crs =crs)


        # Now only modify geometric intersections for areas where LineStrings were reversed/changes were made (i.e., 'problematic')
        reach_id_chg = df_change.loc[df_change['change'] == 1]['reach_id'].to_list()
        reach_id_chg = [int(i) for i in reach_id_chg]
        # shp_sub = shp.loc[shp['reach_id'].isin(reach_id_chg)]

        ### NEW WAY

        for ix, r in shp.iterrows():
            geom1 = shapely.geometry.shape(shp.geometry[ix])
            geom1_rch_id = shp.reach_id[ix]

            if geom1_rch_id not in reach_id_chg:
                continue

            selected_reach = shp.loc[shp['reach_id'] == int(geom1_rch_id)]
            #Finding reaches within search distance (11 km)
            reaches_win_dist = shp.assign(distance=shp.apply(lambda x: x.geometry.distance(selected_reach.geometry.iloc[0]), axis=1)).query(f'distance <= {0.1}')
            reaches_win_dist = reaches_win_dist[reaches_win_dist.reach_id != geom1_rch_id]


            #Only comparing selected_reach to reaches within search distance (minimizes computational needs)
            for ix2, r2 in reaches_win_dist.iterrows(): 

                geom2 = shapely.geometry.shape(reaches_win_dist.geometry[ix2])
                geom2_rch_id = reaches_win_dist.reach_id[ix2]

                ## Don't need to record intersection if reach ID is the same
                if geom1_rch_id == geom2_rch_id:
                    continue

                if geom1.distance(geom2) < eps: 
                    point = shapely.ops.nearest_points(geom1, geom2)[0]
                    # print(point)

                    ## Sometimes the point won't exactly match up, so need to find the nearest point to connect the 2 segments
                    found = False
                    for i in range(len(geom1.coords.xy[0])):
                        x = geom1.coords.xy[0][i]
                        y = geom1.coords.xy[1][i]
                        tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})

                        if(tmp_pt == point):
                            found = True     

                            ind = i

                            point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                            'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                            'geom2_rch_id': geom2_rch_id.item(),
                                                            'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                            'ind_intr': ind}})

                            break

                    if found == False:
                        dist_point = []
                        for i in range(len(geom1.coords.xy[0])):
                            x = geom1.coords.xy[0][i]
                            y = geom1.coords.xy[1][i]
                            tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})
                            dist_point.append(point.distance(tmp_pt))
                        
                        ind = dist_point.index(min(dist_point))

                        point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                        'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                        'geom2_rch_id': geom2_rch_id.item(),
                                                        'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                        'ind_intr': ind}})

        point_lyr.close()
        counter = counter + 1

        geom_con_filt = filter(lambda f: f['properties']['reach_id'] not in reach_id_chg, geom_con)
        geom_con_filt = gpd.GeoDataFrame.from_features([feature for feature in geom_con], crs=crs)
        columns = list(geom_con.meta["schema"]["properties"]) + ["geometry"]
        geom_con_filt = geom_con_filt[columns]
        reach_id_chg_str = [str(r) for r in reach_id_chg]
        geom_con_filt = geom_con_filt.loc[~geom_con_filt['geom1_rch_'].isin(reach_id_chg_str)]
        geom_con_filt.to_file(geom_con_fname, mode="a")

# tst = gpd.read_file(geom_con_fname)
# tst.loc[tst['geom1_rch_'] == '81380100311']
# geom_con_shp.loc[geom_con_shp['geom1_rch_'] == '81380100311']
# geom_con_filt.loc[geom_con_filt['geom1_rch_'] == '81380100311']
# geom_con_filt[~geom_con_filt['geom1_rch_'].isin(reach_id_chg)]


print('The reversed reach IDs are: ' + str(store_rch_ids_rev))
print('The number of reversed reach IDs is: ' + str(len(store_rch_ids_rev)))
#Write output file for reversed LineStrings

print('Writing output river file containing reversed LineStrings')
shp.to_file(riv_out_shp)

print('Writing reach IDs of reversed LineStrings to CSV')
df = pd.DataFrame({"reach_id": store_rch_ids_rev})
# df.to_csv(rev_ids_csv, header=False, index=False)
df.to_csv(rev_ids_csv, index=False)


end_Main = timer()
print('The main network algorithm took ' + str(((end_Main - start) / 60)) + ' minutes or ' + str(((end_Main - start) / 60) / 60) + ' hours to complete.')


# print('- Done')

## CHANGE dist_out TO DIST_OUT WHEN UPDATING DATASET USED
## Maybe not working because adding old and new points to geom_con




#**************************************************
# Topology algorithm - Side network
#**************************************************
print('Starting topology algorithm for the side network')

start_Side = timer()

# store_rch_ids_rev = pd.read_csv(rev_ids_csv, header=None)
store_rch_ids_rev = pd.read_csv(rev_ids_csv)
store_rch_ids_rev = store_rch_ids_rev.iloc[:,0].to_list()
store_rch_ids_rev = [str(r) for r in store_rch_ids_rev]

# geom_con_fname = "hb81_subset_v17a_FO_Main_v2code_pts_0.shp"
# geom_con_fname = '/Users/elyssac/Documents/SWOT/SWORD_v17/NA_2/Topology/b74/na_sword_reaches_hb74_v17_FG1_Main_pts_0.shp'
# geom_con_fname = '/Users/elyssac/Documents/SWOT/SWORD_v17/NA_2/Topology/b81/na_sword_reaches_hb81_v17_FG1_Main_pts_2.shp'
# geom_con_fname = '/Users/elyssac/Documents/SWOT/SWORD_v17/NA_2/Topology/b82/na_sword_reaches_hb82_v17_FG1_Main_pts_0.shp'
# geom_con_fname = '/Users/elyssac/Documents/SWOT/SWORD_v17/NA_2/Topology/b72/na_sword_reaches_hb72_v17_FG1_Main_pts.shp'
# geom_con_fname = '/Users/elyssac/Documents/SWOT/SWORD_v17/NA_2/Topology/b83/na_sword_reaches_hb83_v17_FG1_Main_pts_3.shp'
# geom_con_fname = '/Users/elyssac/Documents/SWOT/SWORD_v17/SA/Topology/b65/sa_sword_reaches_hb65_v17_FG1_Main_pts.shp'
# geom_con_fname = '/Users/elyssac/Documents/SWOT/SWORD_v17/SA/Topology/b63/sa_sword_reaches_hb63_v17_FG1_Main_pts.shp'
# geom_con_fname = '/Users/elyssac/Documents/SWOT/SWORD_v17/SA/Topology/b64/sa_sword_reaches_hb64_v17_FG1_Main_pts_0.shp'
# geom_con_fname = '/Users/elyssac/Documents/SWOT/SWORD_v17/AF/Topology/b11/af_sword_reaches_hb11_v17_FG1_Main_pts_0.shp'
# geom_con_fname = '/Users/elyssac/Documents/SWOT/SWORD_v17/AF/Topology/b16/af_sword_reaches_hb16_v17_FG1_Main_pts_3.shp'

geom_con = gpd.read_file(geom_con_fname)

riv_shp_all = gpd.read_file(riv_all_shp)
riv_shp_main_fix = gpd.read_file(riv_out_shp)

# riv_shp_side = riv_shp_all.loc[riv_shp_all['main_side'] == 1]
riv_shp_side = riv_shp_all.loc[riv_shp_all['main_side'].isin([1,2])]
reach_ids_side = riv_shp_side['reach_id'].to_list()
# reach_ids_side = [str(r) for r in reach_ids_side]
reach_ids_side = [int(r) for r in reach_ids_side]

shp = gpd.GeoDataFrame(pd.concat([riv_shp_main_fix, riv_shp_side], ignore_index=True))
# shp.loc[shp['reach_id'] == 73220300041]


## Need to ensure new point intersections are created as well for main network reaches that are attached to side network reaches


print('Finding new geometric intersections for side channels')
### Making a new point layer
point_shp = riv_out_shp.replace("_LSFix.shp", "_LSFix_Side_pts.shp") 
pt_schema={'geometry': 'Point', 'properties': {'geom1_rch_id': 'int', 'geom2_rch_id': 'int', 'geom1_n_pnts': 'int', 'ind_intr': 'int'}}
# point_lyr = fiona.open("points.shp", mode='w', driver='ESRI Shapefile', schema=pt_schema, crs =crs)
point_lyr = fiona.open(point_shp, mode='w', driver='ESRI Shapefile', schema=pt_schema, crs =crs)

#Only loop through side channel reaches
for ix, r in shp.iterrows():
    geom1 = shapely.geometry.shape(shp.geometry[ix])
    geom1_rch_id = shp.reach_id[ix]

    if geom1_rch_id not in reach_ids_side:
        continue

    selected_reach = shp.loc[shp['reach_id'] == int(geom1_rch_id)]
    #Finding reaches within search distance (11 km)
    reaches_win_dist = shp.assign(distance=shp.apply(lambda x: x.geometry.distance(selected_reach.geometry.iloc[0]), axis=1)).query(f'distance <= {0.1}')
    reaches_win_dist = reaches_win_dist[reaches_win_dist.reach_id != geom1_rch_id]


    #Only comparing selected_reach to reaches within search distance (minimizes computational needs)
    for ix2, r2 in reaches_win_dist.iterrows(): 

        geom2 = shapely.geometry.shape(reaches_win_dist.geometry[ix2])
        geom2_rch_id = reaches_win_dist.reach_id[ix2]

        ## Don't need to record intersection if reach ID is the same
        if geom1_rch_id == geom2_rch_id:
            continue

        if geom1.distance(geom2) < eps: 
            point = shapely.ops.nearest_points(geom1, geom2)[0]
            # print(point)

            ## Sometimes the point won't exactly match up, so need to find the nearest point to connect the 2 segments
            found = False
            for i in range(len(geom1.coords.xy[0])):
                x = geom1.coords.xy[0][i]
                y = geom1.coords.xy[1][i]
                tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})

                if(tmp_pt == point):
                    found = True     

                    ind = i

                    point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                    'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                    'geom2_rch_id': geom2_rch_id.item(),
                                                    'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                    'ind_intr': ind}})

                    break

            if found == False:
                dist_point = []
                for i in range(len(geom1.coords.xy[0])):
                    x = geom1.coords.xy[0][i]
                    y = geom1.coords.xy[1][i]
                    tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})
                    dist_point.append(point.distance(tmp_pt))
                
                ind = dist_point.index(min(dist_point))

                point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                'geom2_rch_id': geom2_rch_id.item(),
                                                'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                'ind_intr': ind}})

point_lyr.close()

geom_con.to_file(point_shp, mode="a")

# tst = gpd.read_file(point_shp)
# tst.loc[tst['geom1_rch_'] == 72608100245]
# tst.loc[tst['geom1_rch_'] == 72608100235]
# tst.loc[tst['geom1_rch_'] == 72608100253]



geom_con_fname = point_shp

# shp = shp.loc[shp['reach_id'].isin(riv_shp_side['reach_id'].to_list())].reset_index()
# reach_ids = shp['reach_id'].astype('int').astype('str').to_list()
# reach_ids_dist_out = shp['dist_out'].to_list()
# shp['reach_id'] = shp['reach_id'].astype('int')
# lines = shp['geometry'].to_list() 

geom_con_shp = gpd.read_file(geom_con_fname)
geom_con_shp['geom1_rch_'] = geom_con_shp['geom1_rch_'].astype('str')
geom_con_shp['geom2_rch_'] = geom_con_shp['geom2_rch_'].astype('str')

# geom_con_shp.loc[geom_con_shp['geom1_rch_'] == '81380100351']

reach_ids_side = [str(r) for r in reach_ids_side]

reach_ids_side_ass = geom_con_shp.loc[(geom_con_shp['geom1_rch_'].isin(reach_ids_side)) | (geom_con_shp['geom2_rch_'].isin(reach_ids_side))]['geom1_rch_'].to_list()
reach_ids_side_ass.extend(geom_con_shp.loc[(geom_con_shp['geom1_rch_'].isin(reach_ids_side)) | (geom_con_shp['geom2_rch_'].isin(reach_ids_side))]['geom2_rch_'].to_list())
reach_ids_side_ass = list(set(reach_ids_side_ass))


print('Redoing new geometric intersections for side channels with associated main network reaches')
### Making a new point layer
pt_schema={'geometry': 'Point', 'properties': {'geom1_rch_id': 'int', 'geom2_rch_id': 'int', 'geom1_n_pnts': 'int', 'ind_intr': 'int'}}
# point_lyr = fiona.open("points.shp", mode='w', driver='ESRI Shapefile', schema=pt_schema, crs =crs)
point_lyr = fiona.open(point_shp, mode='w', driver='ESRI Shapefile', schema=pt_schema, crs =crs)

reach_ids_side_ass_int = [int(r) for r in reach_ids_side_ass]
#Only loop through side channel reaches
for ix, r in shp.iterrows():
    geom1 = shapely.geometry.shape(shp.geometry[ix])
    geom1_rch_id = shp.reach_id[ix]

    if geom1_rch_id not in reach_ids_side_ass_int:
        continue

    selected_reach = shp.loc[shp['reach_id'] == int(geom1_rch_id)]
    #Finding reaches within search distance (11 km)
    reaches_win_dist = shp.assign(distance=shp.apply(lambda x: x.geometry.distance(selected_reach.geometry.iloc[0]), axis=1)).query(f'distance <= {0.1}')
    reaches_win_dist = reaches_win_dist[reaches_win_dist.reach_id != geom1_rch_id]


    #Only comparing selected_reach to reaches within search distance (minimizes computational needs)
    for ix2, r2 in reaches_win_dist.iterrows(): 

        geom2 = shapely.geometry.shape(reaches_win_dist.geometry[ix2])
        geom2_rch_id = reaches_win_dist.reach_id[ix2]

        ## Don't need to record intersection if reach ID is the same
        if geom1_rch_id == geom2_rch_id:
            continue

        if geom1.distance(geom2) < eps: 
            point = shapely.ops.nearest_points(geom1, geom2)[0]
            # print(point)

            ## Sometimes the point won't exactly match up, so need to find the nearest point to connect the 2 segments
            found = False
            for i in range(len(geom1.coords.xy[0])):
                x = geom1.coords.xy[0][i]
                y = geom1.coords.xy[1][i]
                tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})

                if(tmp_pt == point):
                    found = True     

                    ind = i

                    point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                    'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                    'geom2_rch_id': geom2_rch_id.item(),
                                                    'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                    'ind_intr': ind}})

                    break

            if found == False:
                dist_point = []
                for i in range(len(geom1.coords.xy[0])):
                    x = geom1.coords.xy[0][i]
                    y = geom1.coords.xy[1][i]
                    tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})
                    dist_point.append(point.distance(tmp_pt))
                
                ind = dist_point.index(min(dist_point))

                point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                'geom2_rch_id': geom2_rch_id.item(),
                                                'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                'ind_intr': ind}})

point_lyr.close()

geom_con = geom_con.loc[~geom_con['geom1_rch_'].isin(reach_ids_side_ass)]
geom_con.to_file(point_shp, mode="a")

# tst = gpd.read_file(point_shp)
# tst.loc[tst['geom1_rch_'] == 72608100245]
# tst.loc[tst['geom1_rch_'] == 72608100235]
# tst.loc[tst['geom1_rch_'] == 72608100253]

# tst.loc[tst['geom1_rch_'] == 74230100091]
# tst.loc[tst['geom1_rch_'] == 74210000291]
# tst.loc[tst['geom1_rch_'] == 74300400015]
# tst2 = geom_con.loc[~geom_con['geom1_rch_'].isin(reach_ids_side_ass)]
# geom_con.loc[geom_con['geom1_rch_'] == '74210000301']
# tst2.loc[tst2['geom1_rch_'] == '74210000301']


## THIS LAST TIME IS NECESSARY FOR GETTING OUTLET DISTANCES OF NEIGHBORING REACHES TO LABEL UP AND DOWNSTREAM REACHES
geom_con_fname = point_shp

geom_con_shp = gpd.read_file(geom_con_fname)
geom_con_shp['geom1_rch_'] = geom_con_shp['geom1_rch_'].astype('str')
geom_con_shp['geom2_rch_'] = geom_con_shp['geom2_rch_'].astype('str')

reach_ids_side_ass = geom_con_shp.loc[(geom_con_shp['geom1_rch_'].isin(reach_ids_side_ass)) | (geom_con_shp['geom2_rch_'].isin(reach_ids_side_ass))]['geom1_rch_'].to_list()
reach_ids_side_ass.extend(geom_con_shp.loc[(geom_con_shp['geom1_rch_'].isin(reach_ids_side_ass)) | (geom_con_shp['geom2_rch_'].isin(reach_ids_side_ass))]['geom2_rch_'].to_list())
reach_ids_side_ass = list(set(reach_ids_side_ass))



print('Redoing new geometric intersections one last time to find reaches associated')
### Making a new point layer
pt_schema={'geometry': 'Point', 'properties': {'geom1_rch_id': 'int', 'geom2_rch_id': 'int', 'geom1_n_pnts': 'int', 'ind_intr': 'int'}}
# point_lyr = fiona.open("points.shp", mode='w', driver='ESRI Shapefile', schema=pt_schema, crs =crs)
point_lyr = fiona.open(point_shp, mode='w', driver='ESRI Shapefile', schema=pt_schema, crs =crs)

reach_ids_side_ass_int = [int(r) for r in reach_ids_side_ass]
#Only loop through side channel reaches
for ix, r in shp.iterrows():
    geom1 = shapely.geometry.shape(shp.geometry[ix])
    geom1_rch_id = shp.reach_id[ix]

    if geom1_rch_id not in reach_ids_side_ass_int:
        continue

    selected_reach = shp.loc[shp['reach_id'] == int(geom1_rch_id)]
    #Finding reaches within search distance (11 km)
    reaches_win_dist = shp.assign(distance=shp.apply(lambda x: x.geometry.distance(selected_reach.geometry.iloc[0]), axis=1)).query(f'distance <= {0.1}')
    reaches_win_dist = reaches_win_dist[reaches_win_dist.reach_id != geom1_rch_id]


    #Only comparing selected_reach to reaches within search distance (minimizes computational needs)
    for ix2, r2 in reaches_win_dist.iterrows(): 

        geom2 = shapely.geometry.shape(reaches_win_dist.geometry[ix2])
        geom2_rch_id = reaches_win_dist.reach_id[ix2]

        ## Don't need to record intersection if reach ID is the same
        if geom1_rch_id == geom2_rch_id:
            continue

        if geom1.distance(geom2) < eps: 
            point = shapely.ops.nearest_points(geom1, geom2)[0]
            # print(point)

            ## Sometimes the point won't exactly match up, so need to find the nearest point to connect the 2 segments
            found = False
            for i in range(len(geom1.coords.xy[0])):
                x = geom1.coords.xy[0][i]
                y = geom1.coords.xy[1][i]
                tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})

                if(tmp_pt == point):
                    found = True     

                    ind = i

                    point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                    'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                    'geom2_rch_id': geom2_rch_id.item(),
                                                    'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                    'ind_intr': ind}})

                    break

            if found == False:
                dist_point = []
                for i in range(len(geom1.coords.xy[0])):
                    x = geom1.coords.xy[0][i]
                    y = geom1.coords.xy[1][i]
                    tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})
                    dist_point.append(point.distance(tmp_pt))
                
                ind = dist_point.index(min(dist_point))

                point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                'geom2_rch_id': geom2_rch_id.item(),
                                                'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                'ind_intr': ind}})

point_lyr.close()

geom_con = geom_con.loc[~geom_con['geom1_rch_'].isin(reach_ids_side_ass)]
geom_con.to_file(point_shp, mode="a")

# tst = gpd.read_file(point_shp)
# tst.loc[tst["geom1_rch_"] == 74230100091]
# geom_con.loc[geom_con["geom1_rch_"] == '74230100091']

# tst.loc[tst['geom1_rch_'] == 72608100245]
# tst.loc[tst['geom1_rch_'] == 72608100235]
# tst.loc[tst['geom1_rch_'] == 72608100253]



geom_con_fname = point_shp

# shp = shp.loc[shp['reach_id'].isin(riv_shp_side['reach_id'].to_list())].reset_index()
shp = shp.loc[shp['reach_id'].isin(reach_ids_side_ass_int)].reset_index()
reach_ids = shp['reach_id'].astype('int').astype('str').to_list()
reach_ids_dist_out = shp['dist_out'].to_list()
shp['reach_id'] = shp['reach_id'].astype('int')
lines = shp['geometry'].to_list() 

geom_con_shp = gpd.read_file(geom_con_fname)
geom_con_shp['geom1_rch_'] = geom_con_shp['geom1_rch_'].astype('str')
geom_con_shp['geom2_rch_'] = geom_con_shp['geom2_rch_'].astype('str')

# geom_con_side_shp = geom_con_shp.loc[(geom_con_shp['geom1_rch_'].isin(reach_ids_side)) | (geom_con_shp['geom2_rch_'].isin(reach_ids_side))]
geom_con_side_shp = geom_con_shp.loc[(geom_con_shp['geom1_rch_'].isin(reach_ids_side_ass)) | (geom_con_shp['geom2_rch_'].isin(reach_ids_side_ass))]
geom_con_fname = geom_con_fname.replace('pts', 'sub_pts')
geom_con_side_shp.to_file(geom_con_fname)


#Side reaches and main network reaches attached to side network
# reach_ids_side_att = geom_con_side_shp['geom1_rch_'].to_list()
# reach_ids_side_att.extend(geom_con_side_shp['geom2_rch_'].to_list())
# reach_ids_side_att = list(set(reach_ids_side_att))
# geom_con_side_shp.loc[geom_con_side_shp['geom2_rch_'] == '74210000291']

geom_con_side_shp.loc[geom_con_side_shp['geom1_rch_'] == '16220300041']


geom_con_side = fiona.open(geom_con_fname)
# geom_con_all = fiona.open(geom_con_fname)

con_geom1_rch_id = []
con_geom2_rch_id = []
con_geom1_n_pnts = []
con_ind_intr = []
con_closest = []
for con in geom_con_side:
# for con in geom_con_all:
    con_geom = shapely.geometry.shape(con.geometry)
    
    con_geom1_rch_id.append(str(con["properties"]["geom1_rch_"]))
    con_geom2_rch_id.append(str(con["properties"]["geom2_rch_"]))
    con_geom1_n_pnts.append(con["properties"]["geom1_n_pn"])
    con_ind_intr.append(con["properties"]["ind_intr"])
    if con["properties"]["geom1_n_pn"] == 2:
        if con["properties"]["ind_intr"] == 1:
            val = 1
        else:
            val = 0
    else:
        val = min([0, con["properties"]["geom1_n_pn"]], key = lambda x: abs(x - con["properties"]["ind_intr"]))
    con_closest.append(val) # If val != 0, then geom2 is upstream

#           --> Not revisiting potential problem junctions that have already been evaluated and set to be no problem
#           --> Only changing geometry in point intersections layer for reaches where actual changes were made

### This code below is slow, figure out how to make it more efficient
###     --> Create this for the whole area before entering the while loop
###     --> Once in the while loop, only make changes where fixes have been applied 

ind0_geom1_rch_id = []
ind0_geom2_rch_id = []

indf_geom1_rch_id = []
indf_geom2_rch_id = []

d_list = []

for ix, r in shp.iterrows():
    geom1 = lines[ix]
    geom1_rch_id = reach_ids[ix]

    con_geom1_rch_id_ind = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
    
    selected_reach = shp.loc[shp['reach_id'] == int(geom1_rch_id)]
    #Finding reaches within search distance (11 km)
    reaches_win_dist = shp.assign(distance=shp.apply(lambda x: x.geometry.distance(selected_reach.geometry.iloc[0]), axis=1)).query(f'distance <= {0.1}')
    reaches_win_dist = reaches_win_dist[reaches_win_dist.reach_id != geom1_rch_id]

    #Only comparing selected_reach to reaches within search distance (minimizes computational needs)
    for ix2, r2 in reaches_win_dist.iterrows(): 

        geom2 = shapely.geometry.shape(reaches_win_dist.geometry[ix2])
        geom2_rch_id = reaches_win_dist.reach_id.astype('str')[ix2]

        ## Don't need to record intersection if reach ID is the same
        if geom1_rch_id == geom2_rch_id:
            continue

        if geom1.distance(geom2) < eps: 
            con_geom2_rch_id_ind = [i for i in range(len(con_geom2_rch_id)) if con_geom2_rch_id[i] == str(geom2_rch_id)]
            con_geom1_geom2_ind = [i for i in con_geom1_rch_id_ind if i in con_geom2_rch_id_ind]

            con_geom2_opp_rch_id_ind = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom2_rch_id)]
            con_geom1_opp_rch_id_ind = [i for i in range(len(con_geom2_rch_id)) if con_geom2_rch_id[i] == str(geom1_rch_id)]
            con_geom1_geom2_opp_ind = [i for i in con_geom1_opp_rch_id_ind if i in con_geom2_opp_rch_id_ind]

            con_geom1_closest = [con_closest[i] for i in con_geom1_geom2_ind]    
            con_geom2_closest = [con_closest[i] for i in con_geom1_geom2_opp_ind]    

        # if geom1.distance(geom2) < eps: 

            ## Check for locations where 2 segments say they are both downstream of one another 
            if (con_geom1_closest[0] == 0) & (con_geom2_closest[0] == 0):
                # rch_id_dn = geom_dist_out.index(min(geom_dist_out))
                # if rch_id_dn == 0:
                    # ind0_geom1_rch_id.append(str(geom1_rch_id))
                # if (geom1_rch_id not in ind0_geom1_rch_id) | (geom2_rch_id not in ind0_geom2_rch_id):
                ind0_geom1_rch_id.append(str(geom1_rch_id))
                ind0_geom2_rch_id.append(str(geom2_rch_id))

                #Adding reaches to dataframe identifying if it's potentially a problem (1) or not (0)
                d = {"geom1_rch_id": geom1_rch_id, "geom2_rch_id": geom2_rch_id, "con_prob": 1}
                d_list.append(d)

            ## Check for locations where 2 segments say they are both upstream of one another 
            elif (con_geom1_closest[0] != 0) & (con_geom2_closest[0] != 0):
                # if (geom1_rch_id not in indf_geom1_rch_id) | (geom2_rch_id not in indf_geom2_rch_id):
                indf_geom1_rch_id.append(str(geom1_rch_id))
                indf_geom2_rch_id.append(str(geom2_rch_id))

                d = {"geom1_rch_id": geom1_rch_id, "geom2_rch_id": geom2_rch_id, "con_prob": 1}
                d_list.append(d)
            
            else:
                d = {"geom1_rch_id": geom1_rch_id, "geom2_rch_id": geom2_rch_id, "con_prob": 0}
                d_list.append(d)


ind0_df = pd.DataFrame({'geom1_rch_id': ind0_geom1_rch_id, 'geom2_rch_id': ind0_geom2_rch_id})
indf_df = pd.DataFrame({'geom1_rch_id': indf_geom1_rch_id, 'geom2_rch_id': indf_geom2_rch_id})

ind0_df = (ind0_df[~ind0_df.filter(like='geom').apply(frozenset, axis=1).duplicated()].reset_index(drop=True))
indf_df = (indf_df[~indf_df.filter(like='geom').apply(frozenset, axis=1).duplicated()].reset_index(drop=True))

print('The number of possible locations with potential reversed LineString issues is ' + str(len(ind0_df) + len(indf_df)))

df_con_prob = pd.DataFrame(d_list)  
try:               
    print('The number of potentially problematic areas is: ' + str(len(df_con_prob.loc[df_con_prob['con_prob'] == 1])))
except:              
    print('No potentially problematic areas!')
df_change = pd.DataFrame({'reach_id': reach_ids, 'change': 0})   


# ind0_df.loc[ind0_df['geom2_rch_id'] == '74230100101'] # ix = 1
# ind0_df.loc[ind0_df['geom1_rch_id'] == '74300400535'] # ix = 24
# ind0_df.loc[ind0_df['geom2_rch_id'] == '74230800561'] # ix = 7
# ind0_df.loc[ind0_df['geom2_rch_id'] == '74300400025'] # ix = 21

# ind0_df.loc[ind0_df['geom2_rch_id'] == '81190900265'] # ix = 35
# ind0_df.loc[ind0_df['geom2_rch_id'] == '81250700121'] # ix = 91
# ind0_df.loc[ind0_df['geom2_rch_id'] == '81181300281'] # ix = 9
# ind0_df.loc[ind0_df['geom2_rch_id'] == '81160300211'] # ix = 8

# ind0_df.loc[ind0_df['geom1_rch_id'] == '82211000845'] # ix = 98
# ind0_df.loc[ind0_df['geom2_rch_id'] == '82268000591'] # ix = 26

# ind0_df.loc[ind0_df['geom2_rch_id'] == '83160000671'] # ix = 3

# ind0_df.loc[ind0_df['geom1_rch_id'] == '67201000445'] # ix = 16

# ind0_df.loc[ind0_df['geom1_rch_id'] == '11743000191'] # ix = 25
# ind0_df.loc[ind0_df['geom2_rch_id'] == '11743000181'] # ix = 9
# ind0_df.loc[ind0_df['geom2_rch_id'] == '11744500081'] # ix = 14


rch_ids_rev = [] # Reach IDs that need to have the coordinates of the linestring reversed 

## Now loop through ind0_df and figure out which reach needs to be fixed, if it needs to be fixed at all
for ix in range(len(ind0_df)):
    geom1_rch_id = ind0_df['geom1_rch_id'][ix]
    geom2_rch_id = ind0_df['geom2_rch_id'][ix]

    geom1_dist_out = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['dist_out'].to_list()[0]
    geom2_dist_out = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['dist_out'].to_list()[0]

    if (str(geom1_rch_id) not in reach_ids_side) & (str(geom2_rch_id) not in reach_ids_side):
        continue

    # geom1_type = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['end_reach'].to_list()[0]
    # geom2_type = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['end_reach'].to_list()[0]

    ## Check the number of geometric connections for both reaches 1 and 2 to figure out
    ## if the reaches involve lies at a junction or not. Checking both reaches and taking
    ## the minimum number of geometric connections ensures that I'm looking at the right
    ## area based on the 2 reaches involved (reach 1 could be at a junction and reach 2 could
    ## be adjacent to the junction)
    con_cnt_geom1 = con_geom1_rch_id.count(str(geom1_rch_id))
    con_cnt_geom2 = con_geom2_rch_id.count(str(geom2_rch_id))
    con_cnt = min([con_cnt_geom1, con_cnt_geom2])

    ## For identifying cases where the 2 "problem" reaches are both connected to a junction, but the junctions
    ## aren't a part of the problem
    geom1_idx = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
    geom1_idx_geom2 = [con_geom2_rch_id[i] for i in geom1_idx]
    geom2_idx = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom2_rch_id)]
    geom2_idx_geom2 = [con_geom2_rch_id[i] for i in geom2_idx]
    geom1_geom2_comm = list(set(geom1_idx_geom2) & set(geom2_idx_geom2))
    if len(geom1_geom2_comm) == 1:
        #Check if the reach in common is associated with both junctions: e.g., 11743000191
        #This would be the case if it's associated with both geom1_rch_id and geom2_rch_id,
        #but attached at opposite ends of the reach
        comm_idx = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_geom2_comm[0])]
        comm_idx_geom2 = [con_geom2_rch_id[i] for i in comm_idx]
        comm_idx_geom2_geom1 = comm_idx_geom2.index(str(geom1_rch_id))
        comm_idx_geom2_geom2 = comm_idx_geom2.index(str(geom2_rch_id))

        geom1_comm_idx = comm_idx[comm_idx_geom2_geom1]
        geom1_comm_idx_closest = con_closest[geom1_comm_idx]
 
        geom2_comm_idx  =comm_idx[comm_idx_geom2_geom2]
        geom2_comm_idx_closest = con_closest[geom2_comm_idx]

        if geom1_comm_idx_closest != geom2_comm_idx_closest: # Maybe need to modify this if statement?
            con_cnt = 2
 

    if (con_cnt >= 3) & (len(geom1_geom2_comm) == 0):
        con_cnt = 2

    ## If the junction is a headwater a junction, set con_cnt = 3 to ensure junction code is entered below
    ## Previously, the code assumed every junction had an additional reach or end reach attached to all 
    ## segments in the junction, so con_cnt would equal 2 instead of 3 in this case
    ## Check that this code actually works!
    if (str(geom1_rch_id) in head_at_junc) | (str(geom2_rch_id) in head_at_junc):
        con_cnt = 3

    if con_cnt >= 3:

        # con_pts_pot_junc_ids = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == int(geom1_rch_id)]['geom2_rch_'].astype('str').to_list()
    
        ## Find indices of reach ID of interest in the generated initial connectivity that was created before reversing coordinates of linestrings
        indices = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
        ## Find reach IDs that are geometrically connected to reach ID of interest
        con_geom2_ids_indices = [con_geom2_rch_id[i] for i in indices]
        ## For reach IDs geometrically connected to reach ID of interest, count the number of geometric connections they have
        ## (this narrows it down to those reach IDs that might form the river junction with the reach ID of interest, since 1+
        ##  of the geometrically connected reach IDs might be a segment up/downstream of the reach ID of interest forming
        ##  the river junction)
        # con_geom2_ids_cnt
        # con_geom2_ids_indices
        # [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == '74300400015']
        # [con_geom2_rch_id[i] for i in [88, 89, 90, 91]]
        # [i for i in range(len(con_geom2_rch_id)) if con_geom2_rch_id[i] == '74300400015']
        # [con_geom1_rch_id[i] for i in [97, 254]]

        con_geom2_ids_cnt = [con_geom2_rch_id.count(i) for i in con_geom2_ids_indices]
        ## Potential IDs of other reaches associated with the river junction
        ## ("Potential" is used here because a reach ID geometrically connected to the reach ID of interest might have
        ##   another river junction attached to it than the one we are currently looking at)
        con_geom2_pot_junc_indices =  [i for i in range(len(con_geom2_ids_cnt)) if con_geom2_ids_cnt[i] >= 3]
        # con_geom2_pot_junc_indices =  [i for i in range(len(con_geom2_ids_cnt)) if con_geom2_ids_cnt[i] >= 2]
        ## Confirm which potential IDs are actually associated with the river junction of interest
        con_geom2_pot_junc_ids = [con_geom2_ids_indices[i] for i in con_geom2_pot_junc_indices]
        con_geom2_junc_ids = []
        for con_id in con_geom2_pot_junc_ids:
            con_geom1_pot_junc_indices = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == con_id]
            con_geom2_junc_ids_tmp = [con_geom2_rch_id[i] for i in con_geom1_pot_junc_indices]
            if str(geom1_rch_id) in con_geom2_junc_ids_tmp:
                con_geom2_junc_ids.append(con_id)

        ## Only keep reach IDs in the junction that are also connected to geom2_rch_id
        ids_geom2_rch = geom_con_side_shp.loc[geom_con_side_shp['geom1_rch_'] == str(geom2_rch_id)]['geom2_rch_'].astype('str').to_list()
        # ids_geom2_rch = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == str(geom2_rch_id)]['geom2_rch_'].astype('str').to_list()
        ids_geom2_rch.append(geom2_rch_id)
        con_geom2_junc_ids = [i for i in con_geom2_junc_ids if i in ids_geom2_rch]
        con_geom2_junc_ids.append(str(geom1_rch_id))
        if geom2_rch_id not in con_geom2_junc_ids:
            con_geom2_junc_ids.append(str(geom2_rch_id))

        # junc_ids = [int(i) for i in con_geom2_junc_ids]
        junc_ids = [str(i) for i in con_geom2_junc_ids]
        reach_ids_indices = []
        for junc_id in junc_ids:
            reach_ids_indices.extend([i for i in range(len(reach_ids)) if reach_ids[i] == junc_id])
        junc_dist_out = [reach_ids_dist_out[i] for i in reach_ids_indices]

        ## Check if reaches are forming a loop (e.g., 74230100021, 74230100101, and 74230100061 form a loop)
        ## Code below identifies reaches that don't meet at a common point and therefore likely form a loop
        r1 = []
        r2 = []
        pt = []
        att = []
        for i in junc_ids:
            for j in junc_ids:

                if i == j:
                    continue

                tmp1 = shp.loc[shp['reach_id'] == int(i)]['geometry'].to_list()[0]
                tmp2 = shp.loc[shp['reach_id'] == int(j)]['geometry'].to_list()[0]

                if tmp1.distance(tmp2) < eps: 
                    # point = shapely.ops.nearest_points(tmp1, tmp2)[0]
                    point = shapely.intersection(tmp1, tmp2)

                    #The 2 reaches are attached at both ends
                    if point.geom_type == 'MultiPoint':
                        ## If shapely is returning 2 points when it shouldn't (e.g., 11744600021 and 11744600011)
                        mp_pts = [(point.geoms[p].x, point.geoms[p].y) for p in range(len(point.geoms))]
                        mp_pts_rnd = [(round(mp_pts[p][0], 4), round(mp_pts[p][1], 4)) for p in range(len(mp_pts))]
                        mp_pts_rnd_uniq = list(set(mp_pts_rnd))
                        if len(mp_pts_rnd_uniq) == 1:
                            point = shapely.ops.nearest_points(tmp1, tmp2)[0]
                            att.append(False)
                        else:
                            att.append(True)
                    else:
                        point = shapely.ops.nearest_points(tmp1, tmp2)[0]
                        att.append(False)

                    r1.append(i)
                    r2.append(j)
                    pt.append(point)

        df_chk_loop = pd.DataFrame({"r1": r1, "r2": r2, "point": pt, "att": att})
        att_ids = df_chk_loop.loc[df_chk_loop["att"] == True]
        att_ids = att_ids["r1"].to_list()
        comm_pt = df_chk_loop["point"].mode().to_list()[0]
        df_chk_yes_loop = df_chk_loop.loc[df_chk_loop["point"] != comm_pt]
        df_chk_yes_loop = df_chk_yes_loop.loc[~df_chk_loop["r1"].isin(att_ids)]
        df_chk_yes_loop = df_chk_yes_loop.loc[~df_chk_loop["r2"].isin(att_ids)]
        loop_ids = df_chk_yes_loop["r1"].to_list()
        loop_ids.extend(df_chk_yes_loop["r2"].to_list())
        loop_ids = list(set(loop_ids))

        df_chk_not_loop = df_chk_loop.loc[df_chk_loop["point"] == comm_pt]
        not_loop_ids = df_chk_not_loop["r1"].to_list()
        not_loop_ids.extend(df_chk_not_loop["r2"].to_list())
        not_loop_ids = list(set(not_loop_ids))

        junc_ids = not_loop_ids
        reach_ids_indices = []
        for junc_id in junc_ids:
            reach_ids_indices.extend([i for i in range(len(reach_ids)) if reach_ids[i] == junc_id])
        junc_dist_out = [reach_ids_dist_out[i] for i in reach_ids_indices]


        #Identifying what's forming the loop that's not a part of the junction
        loop_not_junc_id = list(set(loop_ids) - set(not_loop_ids))
        if len(loop_not_junc_id) > 1:
            print("There is more than one reach that is part of the loop, but not part of the junction. Check code.")
            break
            # raise SystemExit(22) 

        ## If geom1_rch_id or geom2_rch_id are not in the "junction" IDs, then it's probably not a junction?
        ## Treat it like 2 segments connected to one another instead
        if (str(geom1_rch_id) not in not_loop_ids) | (str(geom2_rch_id) not in not_loop_ids):
                ## If it's not a junction, then use outlet distance to figure out which one to reverse
            if geom1_dist_out < geom2_dist_out:
                rch_ids_rev.append(geom2_rch_id)
                #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1
                continue

            elif geom1_dist_out > geom2_dist_out:
                rch_ids_rev.append(geom1_rch_id)
                #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1
                continue

        ## If reaches form a loop, identify up and downstream reaches differently
        if len(loop_ids) == 3:
            junc_ids = not_loop_ids
            reach_ids_indices = []
            for junc_id in junc_ids:
                reach_ids_indices.extend([i for i in range(len(reach_ids)) if reach_ids[i] == junc_id])
            junc_dist_out = [reach_ids_dist_out[i] for i in reach_ids_indices]

            #Getting outlet distance for reach that's not part of the junction
            loop_not_junc_id_idx = reach_ids.index(loop_not_junc_id[0])
            loop_not_junc_id_dist_out = reach_ids_dist_out[loop_not_junc_id_idx]

        # elif len(loop_ids) > 0:
        elif len(loop_ids) > 3:
            # print('The number of IDs in the loop is greater than 0, but not equal to 3. Double check code.')
            print('The number of IDs in the loop is greater than 3. Double check code.')
            break
            # raise SystemExit(22) 

        # elif len(loop_ids) == 0:
        ## Write code to determine if junction is multi upstream or multi downstream
        ## The reach with the greatest outlet distance is always going to be an upstream segment(?)
        ## So for each of the remaining segments, loop through and find neighbors that aren't associated
        ## with junctions, get outlet distances, and compare neighbor outlet distances to junction outlet distances
        max_junc_dist_out = junc_dist_out.index(max(junc_dist_out))
        min_junc_dist_out = junc_dist_out.index(min(junc_dist_out))
        up_dn = []
        for i in range(len(junc_ids)):
            # if i == max_junc_dist_out:
            #     up_dn.append('up')
            #     continue

            # if i == min_junc_dist_out:
            #     up_dn.append('dn')
            #     continue

            if junc_ids[i] in head_at_junc:
                up_dn.append('up')
                continue

            if shp.loc[shp['reach_id'] == int(junc_ids[i])].main_side.to_list()[0] == 0:
                def_con = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
                def_con_id_neigh = def_con['geom2_rch_'].to_list()
                def_con_id_neigh = [int(j) for j in def_con_id_neigh]
                def_con_neigh_main_side = shp.loc[shp['reach_id'].isin(def_con_id_neigh)]
                def_con_neigh_main = def_con_neigh_main_side.loc[def_con_neigh_main_side['main_side'] == 0].reach_id.to_list()
                def_con_neigh_main = [str(j) for j in def_con_neigh_main]
                def_con_neigh_main_con = geom_con_shp.loc[geom_con_shp['geom1_rch_'].isin(def_con_neigh_main)] 
                def_con_neigh_main_con = def_con_neigh_main_con.loc[def_con_neigh_main_con['geom2_rch_'] == junc_ids[i]]
                def_con_neigh_main_con = def_con_neigh_main_con.loc[def_con_neigh_main_con['geom1_rch_'].isin(junc_ids)]
                # if len(def_con_neigh_main_con) > 1:
                #     print('There is more than 1 attached main network reach. Check code.')
                #     print(geom1_rch_id)
                #     print(geom2_rch_id)
                #     break

                if len(def_con_neigh_main_con) >= 1:
                    val = min([0, def_con_neigh_main_con["geom1_n_pn"].to_list()[0]], key = lambda x: abs(x - def_con_neigh_main_con["ind_intr"].to_list()[0]))
                    if val == 0:
                        up_dn.append('dn')
                        continue
                    else:
                        up_dn.append('up')
                        continue

            id_neigh = geom_con_side_shp.loc[geom_con_side_shp['geom1_rch_'] == junc_ids[i]] 
            # id_neigh = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
            id_neigh = id_neigh.loc[~id_neigh['geom2_rch_'].isin(junc_ids)]

            dists_neigh = []
            main_side_neigh = []
            for j in id_neigh['geom2_rch_'].to_list():
                dists_neigh.extend(shp.loc[shp['reach_id'] == int(j)].dist_out.to_list())
                main_side_neigh.extend(shp.loc[shp['reach_id'] == int(j)].main_side.to_list())


            dist_id = junc_dist_out[i]
            # If up_dn_chk is True, this reach is downstream in the junction
            # up_dn_chk = all(dist_id > x for x in dists_neigh)
            # up_dn_chk fails in places where the reach of interest is downstream and at another junction,
            # 1 of the other reaches at the junction is downstream and the other reach is a headwater coming
            # into the junction that has a greater outlet distance than the reach ID of interest. So instead
            # use this code to check if any of the neighbors have a lower outlet distance. This code may break
            # or not work correctly in some locations, but I'm not sure yet.
            bool_comp_dist = [dist_id > x for x in dists_neigh]
            # Create list matching junc_ids that says if each reach is up or dn, max_junc_dist_out is automatically up
            # if up_dn_chk == True:
            if len(dists_neigh) == 1:
                #If there is only 1 other neighbor
                # if any(x == True for x in bool_comp_dist):
                if any(x == True for x in bool_comp_dist) & any(y != 0 for y in main_side_neigh):
                    up_dn.append('dn')
                elif any(x == True for x in bool_comp_dist) & any(y == 0 for y in main_side_neigh):
                    up_dn.append('up')
                else:
                    up_dn.append('up')
            else:
                #If the reach is attached to another junction
                # if all(x == True for x in bool_comp_dist):
                if all(x == True for x in bool_comp_dist) & (i != max_junc_dist_out):
                    up_dn.append('dn')
                # elif any(x == True for x in bool_comp_dist) & all(y == 0 for y in main_side_neigh):
                #     up_dn.append('dn')
                else:
                    up_dn.append('up')                   
        
        up_cnt = up_dn.count('up')
        dn_cnt = up_dn.count('dn')

        ## If all of the reaches are labeled as upstream or downstream, that isn't possible
        ## so now use min and max outlet distance to label at least 1 up and 1 down
        if (all(x == "up" for x in up_dn)) | (all(x == "dn" for x in up_dn)):
            up_dn = []
            for i in range(len(junc_ids)):
                if i == max_junc_dist_out:
                    up_dn.append('up')
                    continue

                if i == min_junc_dist_out:
                    up_dn.append('dn')
                    continue

                if junc_ids[i] in head_at_junc:
                    up_dn.append('up')
                    continue

                if shp.loc[shp['reach_id'] == int(junc_ids[i])].main_side.to_list()[0] == 0:
                    def_con = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
                    def_con_id_neigh = def_con['geom2_rch_'].to_list()
                    def_con_id_neigh = [int(j) for j in def_con_id_neigh]
                    def_con_neigh_main_side = shp.loc[shp['reach_id'].isin(def_con_id_neigh)]
                    def_con_neigh_main = def_con_neigh_main_side.loc[def_con_neigh_main_side['main_side'] == 0].reach_id.to_list()
                    def_con_neigh_main = [str(j) for j in def_con_neigh_main]
                    def_con_neigh_main_con = geom_con_shp.loc[geom_con_shp['geom1_rch_'].isin(def_con_neigh_main)] 
                    def_con_neigh_main_con = def_con_neigh_main_con.loc[def_con_neigh_main_con['geom2_rch_'] == junc_ids[i]]
                    def_con_neigh_main_con = def_con_neigh_main_con.loc[def_con_neigh_main_con['geom1_rch_'].isin(junc_ids)]
                    # if len(def_con_neigh_main_con) > 1:
                    #     print('There is more than 1 attached main network reach. Check code.')
                    #     print(geom1_rch_id)
                    #     print(geom2_rch_id)
                    #     break

                    if len(def_con_neigh_main_con) >= 1:
                        val = min([0, def_con_neigh_main_con["geom1_n_pn"].to_list()[0]], key = lambda x: abs(x - def_con_neigh_main_con["ind_intr"].to_list()[0]))
                        if val == 0:
                            up_dn.append('dn')
                            continue
                        else:
                            up_dn.append('up')
                            continue

                id_neigh = geom_con_side_shp.loc[geom_con_side_shp['geom1_rch_'] == junc_ids[i]] 
                # id_neigh = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
                id_neigh = id_neigh.loc[~id_neigh['geom2_rch_'].isin(junc_ids)]

                dists_neigh = []
                main_side_neigh = []
                for j in id_neigh['geom2_rch_'].to_list():
                    dists_neigh.extend(shp.loc[shp['reach_id'] == int(j)].dist_out.to_list())
                    main_side_neigh.extend(shp.loc[shp['reach_id'] == int(j)].main_side.to_list())

                dist_id = junc_dist_out[i]
                # If up_dn_chk is True, this reach is downstream in the junction
                # up_dn_chk = all(dist_id > x for x in dists_neigh)
                # up_dn_chk fails in places where the reach of interest is downstream and at another junction,
                # 1 of the other reaches at the junction is downstream and the other reach is a headwater coming
                # into the junction that has a greater outlet distance than the reach ID of interest. So instead
                # use this code to check if any of the neighbors have a lower outlet distance. This code may break
                # or not work correctly in some locations, but I'm not sure yet.
                bool_comp_dist = [dist_id > x for x in dists_neigh]
                # Create list matching junc_ids that says if each reach is up or dn, max_junc_dist_out is automatically up
                # if up_dn_chk == True:
                if len(dists_neigh) == 1:
                    #If there is only 1 other neighbor
                    # if any(x == True for x in bool_comp_dist):
                    if any(x == True for x in bool_comp_dist) & any(y != 0 for y in main_side_neigh):
                        up_dn.append('dn')
                    elif any(x == True for x in bool_comp_dist) & any(y == 0 for y in main_side_neigh):
                        up_dn.append('up')
                    else:
                        up_dn.append('up')
                else:
                    #If the reach is attached to another junction
                    if all(x == True for x in bool_comp_dist):
                        up_dn.append('dn')
                    else:
                        up_dn.append('up')                   
            
            up_cnt = up_dn.count('up')
            dn_cnt = up_dn.count('dn')
            
        # if (up_cnt == 2) & (dn_cnt == 1):
        if (up_cnt >= 2) & (dn_cnt == 1):
            ## If it's a multi upstream junction:
            ## Now determine which reach ID is the downstream most reach ID of the junction, and store the reach IDs of every reach but 
            ## the most downstream segment as the ones to reverse coordinates of the linestring
            ##  ACTUALLY THE ABOVE COMMENT IS INCORRECT: reverse the corrdinates of the linestring for the most downstream segment

            ## If the 2 segments involved are one upstream in the junction and one downstream,
            ## reverse the coordinates of the linestring for the most downstream segment

            min_junc_dist_out = junc_dist_out.index(min(junc_dist_out))
            ## Only reversing if the downstream ID of the junction has matching starting coordinates with the other reach IDs
            if (junc_ids[min_junc_dist_out] == geom1_rch_id) | (junc_ids[min_junc_dist_out] == geom2_rch_id):
                rch_ids_rev.append(junc_ids[min_junc_dist_out])
                #Set change to 1 so I know a modification was made
                df_change.loc[df_change['reach_id'] == str(junc_ids[min_junc_dist_out]), 'change'] = 1

            #Set df_con_prob to zero since either there is no problem or it has been fixed in this location (with reversing) 
            df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0

        # elif (up_cnt == 1) & (dn_cnt == 2):
        elif (up_cnt == 1) & (dn_cnt >= 2):
            ## If it's a multi downstream junction:
            geom1_up_dn = up_dn[junc_ids.index(geom1_rch_id)]
            geom2_up_dn = up_dn[junc_ids.index(geom2_rch_id)]

            ## If both segments involved are one up and one down, flip the downstream segment
            if not all(x == 'dn' for x in [geom1_up_dn, geom2_up_dn]):
                if geom1_up_dn == 'dn':
                    rch_ids_rev.append(geom1_rch_id)
                    #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                    df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                    df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1
                elif geom2_up_dn == 'dn':
                    rch_ids_rev.append(geom2_rch_id)
                    #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                    df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                    df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

        elif (up_cnt == 2) & (dn_cnt == 2):
            ## If it's a multi downstream junction:
            geom1_up_dn = up_dn[junc_ids.index(geom1_rch_id)]
            geom2_up_dn = up_dn[junc_ids.index(geom2_rch_id)]

            ## Checking for locations where 2 reaches are connected at both ends (e.g., 81340500201 and 81340500011)
            tmp1 = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['geometry'].to_list()[0]
            tmp2 = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['geometry'].to_list()[0]
            if tmp1.distance(tmp2) < eps: 
                point = shapely.ops.nearest_points(tmp1, tmp2)
                if len(point) > 1:
                    if all(x == 'dn' for x in [geom1_up_dn, geom2_up_dn]):
                        df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                        continue
    
            ## If both segments involved are both the upstream segment do nothing
            ## If both segments involved are both the downstream segment, flip both
            ## If both segments involved are one up and one down, flip the downstream segment
            if all(x == 'dn' for x in [geom1_up_dn, geom2_up_dn]):
                rch_ids_rev.append(geom1_rch_id)
                rch_ids_rev.append(geom2_rch_id)
                df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1
                df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

            elif (not all(x == 'up' for x in [geom1_up_dn, geom2_up_dn])) & (not all(x == 'dn' for x in [geom1_up_dn, geom2_up_dn])):
                if geom1_up_dn == 'dn':
                    rch_ids_rev.append(geom1_rch_id)
                    df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1

                elif geom2_up_dn == 'dn':
                    rch_ids_rev.append(geom2_rch_id)
                    df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

            df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0

        else:
            print('New junction configuration exists! Refine code. Reaches involved:')
            print(geom1_rch_id)
            print(geom2_rch_id)
            break
            # raise SystemExit(22) 

                
    ## If it's not a junction, then use outlet distance to figure out which one to reverse
    elif geom1_dist_out < geom2_dist_out:
        rch_ids_rev.append(geom1_rch_id)
        #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
        df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
        df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1

    elif geom1_dist_out > geom2_dist_out:
        rch_ids_rev.append(geom2_rch_id)
        #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
        df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
        df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1



# df_con_prob.loc[df_con_prob['con_prob'] == 1].count() # Potentially problematic areas: 35
# df_change.loc[df_change['change'] == 1].count() # Areas with a modification applied: 4


# indf_df.loc[indf_df['geom2_rch_id'] == '74230100101'] # ix = 11
# indf_df.loc[indf_df['geom2_rch_id'] == '74300500041'] # ix = 14
# indf_df.loc[indf_df['geom1_rch_id'] == '74300400545'] # ix = 17
# indf_df.loc[indf_df['geom1_rch_id'] == '74300400535'] # ix = 15
# indf_df.loc[indf_df['geom2_rch_id'] == '74100900045'] # ix = 10

# indf_df.loc[indf_df['geom2_rch_id'] == '81210500551'] # ix = 30
# indf_df.loc[indf_df['geom2_rch_id'] == '81310400431'] # ix = 52

# indf_df.loc[indf_df['geom2_rch_id'] == '82211000775'] # ix = 63
# indf_df.loc[indf_df['geom2_rch_id'] == '82270000441'] # ix = 18

# indf_df.loc[indf_df['geom1_rch_id'] == '63103500065'] # ix = 8

# indf_df.loc[indf_df['geom1_rch_id'] == '64232000241'] # ix = 28

# indf_df.loc[indf_df['geom2_rch_id'] == '11743000181'] # ix = 3


## Now loop through inf_df and figure out which reach needs to be fixed, if it needs to be fixed at all
for ix in range(len(indf_df)):
    geom1_rch_id = indf_df['geom1_rch_id'][ix]
    geom2_rch_id = indf_df['geom2_rch_id'][ix]

    geom1_dist_out = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['dist_out'].to_list()[0]
    geom2_dist_out = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['dist_out'].to_list()[0]

    if (str(geom1_rch_id) not in reach_ids_side) & (str(geom2_rch_id) not in reach_ids_side):
        continue

    # geom1_type = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['end_reach'].to_list()[0]
    # geom2_type = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['end_reach'].to_list()[0]

    ## Check the number of geometric connections for both reaches 1 and 2 to figure out
    ## if the reaches involve lies at a junction or not. Checking both reaches and taking
    ## the minimum number of geometric connections ensures that I'm looking at the right
    ## area based on the 2 reaches involved (reach 1 could be at a junction and reach 2 could
    ## be adjacent to the junction)
    con_cnt_geom1 = con_geom1_rch_id.count(str(geom1_rch_id))
    con_cnt_geom2 = con_geom2_rch_id.count(str(geom2_rch_id))
    con_cnt = min([con_cnt_geom1, con_cnt_geom2])

    ## For identifying cases where the 2 "problem" reaches are both connected to a junction, but the junctions
    ## aren't a part of the problem
    geom1_idx = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
    geom1_idx_geom2 = [con_geom2_rch_id[i] for i in geom1_idx]
    geom2_idx = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom2_rch_id)]
    geom2_idx_geom2 = [con_geom2_rch_id[i] for i in geom2_idx]
    geom1_geom2_comm = list(set(geom1_idx_geom2) & set(geom2_idx_geom2))
    if len(geom1_geom2_comm) == 1:
        #Check if the reach in common is associated with both junctions: e.g., 11743000191
        #This would be the case if it's associated with both geom1_rch_id and geom2_rch_id,
        #but attached at opposite ends of the reach
        comm_idx = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_geom2_comm[0])]
        comm_idx_geom2 = [con_geom2_rch_id[i] for i in comm_idx]
        comm_idx_geom2_geom1 = comm_idx_geom2.index(str(geom1_rch_id))
        comm_idx_geom2_geom2 = comm_idx_geom2.index(str(geom2_rch_id))

        geom1_comm_idx = comm_idx[comm_idx_geom2_geom1]
        geom1_comm_idx_closest = con_closest[geom1_comm_idx]
 
        geom2_comm_idx  =comm_idx[comm_idx_geom2_geom2]
        geom2_comm_idx_closest = con_closest[geom2_comm_idx]

        if geom1_comm_idx_closest != geom2_comm_idx_closest: # Maybe need to modify this if statement?
            con_cnt = 2

    if (con_cnt >= 3) & (len(geom1_geom2_comm) == 0):
        con_cnt = 2

    # tst = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
    # [con_geom2_rch_id[i] for i in tst]
    # tst = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom2_rch_id)]
    # [con_geom2_rch_id[i] for i in tst]

    # '74300400425'
    # ['74300300045', '74300400185', '74300500041']
    # '74300500041'
    # ['74300300045', '74300400425', '74300500035']

    if (str(geom1_rch_id) in head_at_junc) | (str(geom2_rch_id) in head_at_junc):
        con_cnt = 3

    if con_cnt >= 3:
        ## If the 2 segments involved are one upstream in the junction and one downstream,
        ## reverse the coordinates of the linestring for the upstream segment

        ## Find indices of reach ID of interest in the generated initial connectivity that was created before reversing coordinates of linestrings
        indices = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
        ## Find reach IDs that are geometrically connected to reach ID of interest
        con_geom2_ids_indices = [con_geom2_rch_id[i] for i in indices]
        ## For reach IDs geometrically connected to reach ID of interest, count the number of geometric connections they have
        ## (this narrows it down to those reach IDs that might form the river junction with the reach ID of interest, since 1+
        ##  of the geometrically connected reach IDs might be a segment up/downstream of the reach ID of interest forming
        ##  the river junction)
        con_geom2_ids_cnt = [con_geom2_rch_id.count(i) for i in con_geom2_ids_indices]
        ## Potential IDs of other reaches associated with the river junction
        ## ("Potential" is used here because a reach ID geometrically connected to the reach ID of interest might have
        ##   another river junction attached to it than the one we are currently looking at)
        con_geom2_pot_junc_indices =  [i for i in range(len(con_geom2_ids_cnt)) if con_geom2_ids_cnt[i] >= 3]
        # con_geom2_pot_junc_indices =  [i for i in range(len(con_geom2_ids_cnt)) if con_geom2_ids_cnt[i] >= 2]
        ## Confirm which potential IDs are actually associated with the river junction of interest
        con_geom2_pot_junc_ids = [con_geom2_ids_indices[i] for i in con_geom2_pot_junc_indices]
        con_geom2_junc_ids = []
        for con_id in con_geom2_pot_junc_ids:
            con_geom1_pot_junc_indices = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == con_id]
            con_geom2_junc_ids_tmp = [con_geom2_rch_id[i] for i in con_geom1_pot_junc_indices]
            if str(geom1_rch_id) in con_geom2_junc_ids_tmp:
            # if (str(geom1_rch_id) in con_geom2_junc_ids_tmp) & (str(geom2_rch_id) in con_geom2_junc_ids_tmp):
                con_geom2_junc_ids.append(con_id)


        ## Only keep reach IDs in the junction that are also connected to geom2_rch_id
        ids_geom2_rch = geom_con_side_shp.loc[geom_con_side_shp['geom1_rch_'] == str(geom2_rch_id)]['geom2_rch_'].astype('str').to_list()
        # ids_geom2_rch = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == str(geom2_rch_id)]['geom2_rch_'].astype('str').to_list()
        ids_geom2_rch.append(geom2_rch_id)
        con_geom2_junc_ids = [i for i in con_geom2_junc_ids if i in ids_geom2_rch]
        con_geom2_junc_ids.append(str(geom1_rch_id))
        if geom2_rch_id not in con_geom2_junc_ids:
            con_geom2_junc_ids.append(str(geom2_rch_id))

        # junc_ids = [int(i) for i in con_geom2_junc_ids]
        junc_ids = [str(i) for i in con_geom2_junc_ids]
        reach_ids_indices = []
        for junc_id in junc_ids:
            reach_ids_indices.extend([i for i in range(len(reach_ids)) if reach_ids[i] == junc_id])
        junc_dist_out = [reach_ids_dist_out[i] for i in reach_ids_indices]

        ## Check if reaches are forming a loop (e.g., 74230100021, 74230100101, and 74230100061 form a loop)
        ## Code below identifies reaches that don't meet at a common point and therefore likely form a loop
        r1 = []
        r2 = []
        pt = []
        att = []
        for i in junc_ids:
            for j in junc_ids:

                if i == j:
                    continue

                tmp1 = shp.loc[shp['reach_id'] == int(i)]['geometry'].to_list()[0]
                tmp2 = shp.loc[shp['reach_id'] == int(j)]['geometry'].to_list()[0]

                if tmp1.distance(tmp2) < eps: 
                    # point = shapely.ops.nearest_points(tmp1, tmp2)[0]
                    point = shapely.intersection(tmp1, tmp2)

                    #The 2 reaches are attached at both ends
                    if point.geom_type == 'MultiPoint':
                        ## If shapely is returning 2 points when it shouldn't (e.g., 11744600021 and 11744600011)
                        mp_pts = [(point.geoms[p].x, point.geoms[p].y) for p in range(len(point.geoms))]
                        mp_pts_rnd = [(round(mp_pts[p][0], 4), round(mp_pts[p][1], 4)) for p in range(len(mp_pts))]
                        mp_pts_rnd_uniq = list(set(mp_pts_rnd))
                        if len(mp_pts_rnd_uniq) == 1:
                            point = shapely.ops.nearest_points(tmp1, tmp2)[0]
                            att.append(False)
                        else:
                            att.append(True)
                    else:
                        point = shapely.ops.nearest_points(tmp1, tmp2)[0]
                        att.append(False)

                    r1.append(i)
                    r2.append(j)
                    pt.append(point)

        df_chk_loop = pd.DataFrame({"r1": r1, "r2": r2, "point": pt, "att": att})
        att_ids = df_chk_loop.loc[df_chk_loop["att"] == True]
        att_ids = att_ids["r1"].to_list()
        comm_pt = df_chk_loop["point"].mode().to_list()[0]
        df_chk_yes_loop = df_chk_loop.loc[df_chk_loop["point"] != comm_pt]
        df_chk_yes_loop = df_chk_yes_loop.loc[~df_chk_loop["r1"].isin(att_ids)]
        df_chk_yes_loop = df_chk_yes_loop.loc[~df_chk_loop["r2"].isin(att_ids)]
        loop_ids = df_chk_yes_loop["r1"].to_list()
        loop_ids.extend(df_chk_yes_loop["r2"].to_list())
        loop_ids = list(set(loop_ids))

        df_chk_not_loop = df_chk_loop.loc[df_chk_loop["point"] == comm_pt]
        not_loop_ids = df_chk_not_loop["r1"].to_list()
        not_loop_ids.extend(df_chk_not_loop["r2"].to_list())
        not_loop_ids = list(set(not_loop_ids))

        junc_ids = not_loop_ids
        reach_ids_indices = []
        for junc_id in junc_ids:
            reach_ids_indices.extend([i for i in range(len(reach_ids)) if reach_ids[i] == junc_id])
        junc_dist_out = [reach_ids_dist_out[i] for i in reach_ids_indices]


        #Identifying what's forming the loop that's not a part of the junction
        loop_not_junc_id = list(set(loop_ids) - set(not_loop_ids))
        if len(loop_not_junc_id) > 1:
            print("There is more than one reach that is part of the loop, but not part of the junction. Check code.")
            break
            # raise SystemExit(22) 

        ## If geom1_rch_id or geom2_rch_id are not in the "junction" IDs, then it's probably not a junction?
        ## Treat it like 2 segments connected to one another instead
        if (str(geom1_rch_id) not in not_loop_ids) | (str(geom2_rch_id) not in not_loop_ids):
                ## If it's not a junction, then use outlet distance to figure out which one to reverse
            if geom1_dist_out < geom2_dist_out:
                rch_ids_rev.append(geom2_rch_id)
                #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1
                continue

            elif geom1_dist_out > geom2_dist_out:
                rch_ids_rev.append(geom1_rch_id)
                #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1
                continue
                    

        ## If reaches form a loop, identify up and downstream reaches differently
        if len(loop_ids) == 3:
            junc_ids = not_loop_ids
            reach_ids_indices = []
            for junc_id in junc_ids:
                reach_ids_indices.extend([i for i in range(len(reach_ids)) if reach_ids[i] == junc_id])
            junc_dist_out = [reach_ids_dist_out[i] for i in reach_ids_indices]

            #Getting outlet distance for reach that's not part of the junction
            loop_not_junc_id_idx = reach_ids.index(loop_not_junc_id[0])
            loop_not_junc_id_dist_out = reach_ids_dist_out[loop_not_junc_id_idx]

        # elif len(loop_ids) > 0:
        elif len(loop_ids) > 3:
            # print('The number of IDs in the loop is greater than 0, but not equal to 3. Double check code.')
            print('The number of IDs in the loop is greater than 3. Double check code.')
            break
            # raise SystemExit(22) 

        # elif len(loop_ids) == 0:
        ## Write code to determine if junction is multi upstream or multi downstream
        ## The reach with the greatest outlet distance is always going to be an upstream segment(?)
        ## So for each of the remaining segments, loop through and find neighbors that aren't associated
        ## with junctions, get outlet distances, and compare neighbor outlet distances to junction outlet distances
        max_junc_dist_out = junc_dist_out.index(max(junc_dist_out))
        min_junc_dist_out = junc_dist_out.index(min(junc_dist_out))
        up_dn = []
        for i in range(len(junc_ids)):
            # if i == max_junc_dist_out:
            #     up_dn.append('up')
            #     continue

            # if i == min_junc_dist_out:
            #     up_dn.append('dn')
            #     continue

            if junc_ids[i] in head_at_junc:
                up_dn.append('up')
                continue

            ##This code is for cases when some reaches in the junction are part of the main network
            ##and we already know the flow direction on the main network, so we can now use that info
            ##to define the flow direction for the associated side network reaches
            if shp.loc[shp['reach_id'] == int(junc_ids[i])].main_side.to_list()[0] == 0:
                def_con = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
                def_con_id_neigh = def_con['geom2_rch_'].to_list()
                def_con_id_neigh = [int(j) for j in def_con_id_neigh]
                def_con_neigh_main_side = shp.loc[shp['reach_id'].isin(def_con_id_neigh)]
                def_con_neigh_main = def_con_neigh_main_side.loc[def_con_neigh_main_side['main_side'] == 0].reach_id.to_list()
                def_con_neigh_main = [str(j) for j in def_con_neigh_main]
                def_con_neigh_main_con = geom_con_shp.loc[geom_con_shp['geom1_rch_'].isin(def_con_neigh_main)] 
                def_con_neigh_main_con = def_con_neigh_main_con.loc[def_con_neigh_main_con['geom2_rch_'] == junc_ids[i]]
                def_con_neigh_main_con = def_con_neigh_main_con.loc[def_con_neigh_main_con['geom1_rch_'].isin(junc_ids)]
                # if len(def_con_neigh_main_con) > 1:
                #     print('There is more than 1 attached main network reach. Check code.')
                #     print(geom1_rch_id)
                #     print(geom2_rch_id)
                #     break

                if len(def_con_neigh_main_con) >= 1:
                    val = min([0, def_con_neigh_main_con["geom1_n_pn"].to_list()[0]], key = lambda x: abs(x - def_con_neigh_main_con["ind_intr"].to_list()[0]))
                    if val == 0:
                        up_dn.append('dn')
                        continue
                    else:
                        up_dn.append('up')
                        continue

            id_neigh = geom_con_side_shp.loc[geom_con_side_shp['geom1_rch_'] == junc_ids[i]] 
            # id_neigh = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
            id_neigh = id_neigh.loc[~id_neigh['geom2_rch_'].isin(junc_ids)]

            dists_neigh = []
            main_side_neigh = []
            for j in id_neigh['geom2_rch_'].to_list():
                dists_neigh.extend(shp.loc[shp['reach_id'] == int(j)].dist_out.to_list())
                main_side_neigh.extend(shp.loc[shp['reach_id'] == int(j)].main_side.to_list())

            dist_id = junc_dist_out[i]
            # up_dn_chk = all(dist_id > x for x in dists_neigh)
            bool_comp_dist = [dist_id > x for x in dists_neigh]
            # If True, this reach is downstream in the junction
            # Create list matching junc_ids that says if each reach is up or dn, max_junc_dist_out is automatically up
            if len(dists_neigh) == 1:
                #If there is only 1 other neighbor
                # if any(x == True for x in bool_comp_dist):
                if any(x == True for x in bool_comp_dist) & any(y != 0 for y in main_side_neigh):
                    up_dn.append('dn')
                elif any(x == True for x in bool_comp_dist) & any(y == 0 for y in main_side_neigh):
                    up_dn.append('up')
                else:
                    up_dn.append('up')
            else:
                #If the reach is attached to another junction
                # if up_dn_chk == False:
                if any(x == True for x in bool_comp_dist):
                    up_dn.append('dn')
                # elif any(x == True for x in bool_comp_dist) & all(y == 0 for y in main_side_neigh):
                #     up_dn.append('dn')
                else:
                    up_dn.append('up') 
        
        up_cnt = up_dn.count('up')
        dn_cnt = up_dn.count('dn')


        ## If all of the reaches are labeled as upstream or downstream, that isn't possible
        ## so now use min and max outlet distance to label at least 1 up and 1 down
        if (all(x == "up" for x in up_dn)) | (all(x == "dn" for x in up_dn)):
            up_dn = []
            for i in range(len(junc_ids)):
                if i == max_junc_dist_out:
                    up_dn.append('up')
                    continue

                if i == min_junc_dist_out:
                    up_dn.append('dn')
                    continue

                if junc_ids[i] in head_at_junc:
                    up_dn.append('up')
                    continue

                if shp.loc[shp['reach_id'] == int(junc_ids[i])].main_side.to_list()[0] == 0:
                    def_con = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
                    def_con_id_neigh = def_con['geom2_rch_'].to_list()
                    def_con_id_neigh = [int(j) for j in def_con_id_neigh]
                    def_con_neigh_main_side = shp.loc[shp['reach_id'].isin(def_con_id_neigh)]
                    def_con_neigh_main = def_con_neigh_main_side.loc[def_con_neigh_main_side['main_side'] == 0].reach_id.to_list()
                    def_con_neigh_main = [str(j) for j in def_con_neigh_main]
                    def_con_neigh_main_con = geom_con_shp.loc[geom_con_shp['geom1_rch_'].isin(def_con_neigh_main)] 
                    def_con_neigh_main_con = def_con_neigh_main_con.loc[def_con_neigh_main_con['geom2_rch_'] == junc_ids[i]]
                    def_con_neigh_main_con = def_con_neigh_main_con.loc[def_con_neigh_main_con['geom1_rch_'].isin(junc_ids)]
                    # if len(def_con_neigh_main_con) > 1:
                    #     print('There is more than 1 attached main network reach. Check code.')
                    #     print(geom1_rch_id)
                    #     print(geom2_rch_id)
                    #     break

                    if len(def_con_neigh_main_con) >= 1:
                        val = min([0, def_con_neigh_main_con["geom1_n_pn"].to_list()[0]], key = lambda x: abs(x - def_con_neigh_main_con["ind_intr"].to_list()[0]))
                        if val == 0:
                            up_dn.append('dn')
                            continue
                        else:
                            up_dn.append('up')
                            continue

                id_neigh = geom_con_side_shp.loc[geom_con_side_shp['geom1_rch_'] == junc_ids[i]] 
                # id_neigh = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
                id_neigh = id_neigh.loc[~id_neigh['geom2_rch_'].isin(junc_ids)]

                dists_neigh = []
                main_side_neigh = []
                for j in id_neigh['geom2_rch_'].to_list():
                    dists_neigh.extend(shp.loc[shp['reach_id'] == int(j)].dist_out.to_list())
                    main_side_neigh.extend(shp.loc[shp['reach_id'] == int(j)].main_side.to_list())

                dist_id = junc_dist_out[i]
                # If up_dn_chk is True, this reach is downstream in the junction
                # up_dn_chk = all(dist_id > x for x in dists_neigh)
                # up_dn_chk fails in places where the reach of interest is downstream and at another junction,
                # 1 of the other reaches at the junction is downstream and the other reach is a headwater coming
                # into the junction that has a greater outlet distance than the reach ID of interest. So instead
                # use this code to check if any of the neighbors have a lower outlet distance. This code may break
                # or not work correctly in some locations, but I'm not sure yet.
                bool_comp_dist = [dist_id > x for x in dists_neigh]
                # Create list matching junc_ids that says if each reach is up or dn, max_junc_dist_out is automatically up
                # if up_dn_chk == True:
                if len(dists_neigh) == 1:
                    #If there is only 1 other neighbor
                    # if any(x == True for x in bool_comp_dist):
                    if any(x == True for x in bool_comp_dist) & any(y != 0 for y in main_side_neigh):
                        up_dn.append('dn')
                    elif any(x == True for x in bool_comp_dist) & any(y == 0 for y in main_side_neigh):
                        up_dn.append('up')
                    else:
                        up_dn.append('up')
                else:
                    #If the reach is attached to another junction
                    if all(x == True for x in bool_comp_dist):
                        up_dn.append('dn')
                    else:
                        up_dn.append('up')                   
            
            up_cnt = up_dn.count('up')
            dn_cnt = up_dn.count('dn')
                    
        if (up_cnt >= 2) & (dn_cnt == 1):
        # if (up_cnt == 2) & (dn_cnt == 1):
            ## If it's a multi upstream junction:
            ## Now determine which reach ID is the downstream most reach ID of the junction, and store the reach IDs of every reach but 
            ## the most downstream segment as the ones to reverse coordinates of the linestring
            ## Only reverse if both reaches involved are actually a part of the junction
            # if (geom1_rch_id in con_geom2_junc_ids) & (geom2_rch_id in con_geom2_junc_ids):
            if (geom1_rch_id in junc_ids) & (geom2_rch_id in junc_ids):

                # min_junc_dist_out = junc_dist_out.index(min(junc_dist_out))
                # ## Only reversing if the downstream ID of the junction has starting matching starting coordinates with the other reach IDs
                # if (junc_ids[min_junc_dist_out] == geom1_rch_id) | (junc_ids[min_junc_dist_out] == geom2_rch_id):
                #     rch_ids_rev.append(junc_ids[min_junc_dist_out])

                dist_out = [i for i in junc_dist_out if i != min(junc_dist_out)]
                dist_out_ind = []
                for d in dist_out:
                    dist_out_ind.extend([i for i in range(len(junc_dist_out)) if junc_dist_out[i] == d])

                ## Identifying the upstream segments in the junction
                junc_up_ids = [junc_ids[i] for i in dist_out_ind]

                ## Adding 1 or both upstream segments to reverse depending on if the 2 segments involved are 
                ## both the upstream segments
                if geom1_rch_id in junc_up_ids:
                    rch_ids_rev.append(geom1_rch_id)
                    #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                    df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                    df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1

                if geom2_rch_id in junc_up_ids:
                    rch_ids_rev.append(geom2_rch_id)
                    #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                    df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                    df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

                # rch_ids_rev.extend([junc_ids[i] for i in dist_out_ind])
                    
        # elif (up_cnt == 1) & (dn_cnt == 2):
        elif (up_cnt == 1) & (dn_cnt >= 2):
            ## If it's a multi downstream junction:
            geom1_up_dn = up_dn[junc_ids.index(geom1_rch_id)]
            geom2_up_dn = up_dn[junc_ids.index(geom2_rch_id)]

            ## If both segments involved are both the downstream segment do nothing
            ## If both segments involved are one up and one down, flip the upstream segment
            if not all(x == 'dn' for x in [geom1_up_dn, geom2_up_dn]):
                rch_ids_rev.append(junc_ids[up_dn.index('up')])
                #Set change to 1 so I know a modification was made
                df_change.loc[df_change['reach_id'] == str(junc_ids[min_junc_dist_out]), 'change'] = 1

            #Set df_con_prob to zero since either there is no problem or it has been fixed in this location (with reversing) 
            df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0

        elif (up_cnt == 2) & (dn_cnt == 2):
            ## If it's a multi downstream junction:
            geom1_up_dn = up_dn[junc_ids.index(geom1_rch_id)]
            geom2_up_dn = up_dn[junc_ids.index(geom2_rch_id)]

            ## Checking for locations where 2 reaches are connected at both ends (e.g., 81340500201 and 81340500011)
            tmp1 = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['geometry'].to_list()[0]
            tmp2 = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['geometry'].to_list()[0]
            if tmp1.distance(tmp2) < eps: 
                point = shapely.ops.nearest_points(tmp1, tmp2)
                if len(point) > 1:
                    if all(x == 'up' for x in [geom1_up_dn, geom2_up_dn]):
                        df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                        continue           

            ## If both segments involved are both the downstream segment do nothing
            ## If both segments involved are both the upstream segment, flip both
            ## If both segments involved are one up and one down, flip the upstream segment
            if all(x == 'up' for x in [geom1_up_dn, geom2_up_dn]):
                rch_ids_rev.append(geom1_rch_id)
                rch_ids_rev.append(geom2_rch_id)
                df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1
                df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

            elif (not all(x == 'dn' for x in [geom1_up_dn, geom2_up_dn])) & (not all(x == 'up' for x in [geom1_up_dn, geom2_up_dn])):
                if geom1_up_dn == 'up':
                    rch_ids_rev.append(geom1_rch_id)
                    df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1

                elif geom2_up_dn == 'up':
                    rch_ids_rev.append(geom2_rch_id)
                    df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

            df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0

        else:
            print('New junction configuration exists! Refine code. Reaches involved:')
            print(geom1_rch_id)
            print(geom2_rch_id)
            break
            # raise SystemExit(22) 


    ## If it's not a junction, then use outlet distance to figure out which one to reverse
    elif geom1_dist_out < geom2_dist_out:
        if (geom2_rch_id in store_rch_ids_rev) & (int(geom1_rch_id) in head_at_junc):
            rch_ids_rev.append(geom1_rch_id) 
        elif (geom1_rch_id in store_rch_ids_rev) & (int(geom2_rch_id) in head_at_junc):
            rch_ids_rev.append(geom2_rch_id)    
        else:
            rch_ids_rev.append(geom2_rch_id)
            #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
            df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
            df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

    elif geom1_dist_out > geom2_dist_out:
        rch_ids_rev.append(geom1_rch_id)
        #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
        df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
        df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1


# df_con_prob.loc[df_con_prob['con_prob'] == 1].count() # Potentially problematic areas: 27
# df_change.loc[df_change['change'] == 1].count() # Areas with a modification applied: 10

try:               
    print('The number of potentially problematic areas remaining is: ' + str(len(df_con_prob.loc[df_con_prob['con_prob'] == 1])))
except:              
    print('No potentially problematic areas remaining!')

print('The number of areas modified is: ' + str(len(df_change.loc[df_change['change'] == 1])))

# MANUALLY DO THIS FOR THIS BASIN
# 73220300021 needs to be flipped 
# rch_ids_rev = ['73220300021']

rch_ids_rev = list(set(rch_ids_rev))
# len(rch_ids_rev)
rch_ids_rev = [r for r in rch_ids_rev if (r in reach_ids_side) |  (int(r) in head_at_junc)]

print(rch_ids_rev)
print('len(rch_ids_rev): ' + str(len(rch_ids_rev)))
if len(rch_ids_rev) == 0:
    print('No more reaches to reverse!')
    # end_Main_Side = timer()
    # print('The side network algorithm took ' + str(((end_Main_Side - start_Side) / 60)) + ' seconds or ' + str(((end_Main_Side - start_Side) / 60) / 60) + ' hours to complete.')
    # print('The whole algorithm took ' + str(((end_Main_Side - start) / 60)) + ' seconds or ' + str(((end_Main_Side - start) / 60) / 60) + ' hours to complete.')
    # raise SystemExit(22) 
else:
    rev_chk = [i for i in rch_ids_rev if i in store_rch_ids_rev]
    if len(rev_chk) > 0:
        print('A reach has been reversed multiple times -- check code!')
        print('rev_chk IDs: ')
        print(rev_chk)
        print('Writing reach IDs that been reversed multiple times to CSV -- may need to do manual topology in these locations')
        df = pd.DataFrame({"reach_id": rev_chk})
        rev_ids_csv = rev_ids_csv.replace('LS.csv', 'multi.csv')
        df.to_csv(rev_ids_csv, index=False)
        temp_end = timer()
        print("Time Elapsed: " + str(((temp_end - start) / 60)) + " min")
        raise SystemExit(22) 


    ### Maybe I just need to loop through the reaches that I've already stored as need to be reversed
    for r in rch_ids_rev:
        # if (r in reach_ids_side) | (int(r) in head_at_junc):  
            print('Reversing ' + str(r))
            ix = shp.loc[shp['reach_id'] == int(r)].index.to_list()[0]
            # lines[ix] = rev_crds(geom1)
            shp.geometry[ix] = rev_crds(shp.geometry[ix])
            store_rch_ids_rev.append(r)

    print('Finished reversing initial reaches, finding new geometric intersections')

    ### Make new point layer
    counter = 0
    geom_con_fname = geom_con_fname.replace('_pts.shp', '_pts_' + str(counter) + '.shp')

    print('geom_con_fname: ' + geom_con_fname)

    point_lyr = fiona.open(geom_con_fname, mode='w', driver='ESRI Shapefile', schema=pt_schema, crs =crs)

    riv_shp_main_fix_sub = riv_shp_main_fix.loc[~riv_shp_main_fix['reach_id'].isin(reach_ids_side_ass_int)]
    shp = gpd.GeoDataFrame(pd.concat([riv_shp_main_fix_sub, shp], ignore_index=True))
    # shp.loc[shp['reach_id'] == 74230100091]

    # Now only modify geometric intersections for areas where LineStrings were reversed/changes were made (i.e., 'problematic')
    reach_id_chg = df_change.loc[df_change['change'] == 1]['reach_id'].to_list()
    reach_id_chg = [int(i) for i in reach_id_chg]
    # shp_sub = shp.loc[shp['reach_id'].isin(reach_id_chg)]
    # reach_id_chg = [73220300021]

    ### NEW WAY
    # reach_ids_side_att_int = [int(r) for r in reach_ids_side_att]
    for ix, r in shp.iterrows():
        geom1 = shapely.geometry.shape(shp.geometry[ix])
        geom1_rch_id = shp.reach_id[ix]

        # if geom1_rch_id not in reach_id_chg:
        if geom1_rch_id not in reach_ids_side_ass_int:
            continue

        selected_reach = shp.loc[shp['reach_id'] == int(geom1_rch_id)]
        #Finding reaches within search distance (11 km)
        reaches_win_dist = shp.assign(distance=shp.apply(lambda x: x.geometry.distance(selected_reach.geometry.iloc[0]), axis=1)).query(f'distance <= {0.1}')
        reaches_win_dist = reaches_win_dist[reaches_win_dist.reach_id != geom1_rch_id]


        #Only comparing selected_reach to reaches within search distance (minimizes computational needs)
        for ix2, r2 in reaches_win_dist.iterrows(): 

            geom2 = shapely.geometry.shape(reaches_win_dist.geometry[ix2])
            geom2_rch_id = reaches_win_dist.reach_id[ix2]

            ## Don't need to record intersection if reach ID is the same
            if geom1_rch_id == geom2_rch_id:
                continue

            if geom1.distance(geom2) < eps: 
                point = shapely.ops.nearest_points(geom1, geom2)[0]
                # print(point)

                ## Sometimes the point won't exactly match up, so need to find the nearest point to connect the 2 segments
                found = False
                for i in range(len(geom1.coords.xy[0])):
                    x = geom1.coords.xy[0][i]
                    y = geom1.coords.xy[1][i]
                    tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})

                    if(tmp_pt == point):
                        found = True     

                        ind = i

                        point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                        'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                        'geom2_rch_id': geom2_rch_id.item(),
                                                        'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                        'ind_intr': ind}})

                        break

                if found == False:
                    dist_point = []
                    for i in range(len(geom1.coords.xy[0])):
                        x = geom1.coords.xy[0][i]
                        y = geom1.coords.xy[1][i]
                        tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})
                        dist_point.append(point.distance(tmp_pt))
                    
                    ind = dist_point.index(min(dist_point))

                    point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                    'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                    'geom2_rch_id': geom2_rch_id.item(),
                                                    'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                    'ind_intr': ind}})

    point_lyr.close()
    counter = counter + 1

    geom_con = geom_con.loc[~geom_con['geom1_rch_'].isin(reach_ids_side_ass_int)]
    geom_con.to_file(geom_con_fname, mode="a")


    # tst = gpd.read_file(geom_con_fname)
    # tst.loc[tst['geom1_rch_'] == 74230100091]
    # tst.loc[tst['geom1_rch_'] == 73220300011]
    # tst.loc[tst['geom1_rch_'] == 73220300021]
    # geom_con.loc[geom_con['geom1_rch_'] == '74230100091']
    # geom_con.loc[geom_con['geom1_rch_'] == '73220300011']



    print('Starting while loop for finding further changes')

    while counter < 20:
        print('The while loop is in iteration ' + str(counter))

        # shp = shp.loc[shp['reach_id'].isin(riv_shp_side['reach_id'].to_list())].reset_index()
        # reach_ids = shp['reach_id'].astype('int').astype('str').to_list()
        # reach_ids_dist_out = shp['dist_out'].to_list()
        # shp['reach_id'] = shp['reach_id'].astype('int')
        # lines = shp['geometry'].to_list() 

        # geom_con_shp = gpd.read_file(geom_con_fname)
        # geom_con_shp['geom1_rch_'] = geom_con_shp['geom1_rch_'].astype('str')
        # geom_con_shp['geom2_rch_'] = geom_con_shp['geom2_rch_'].astype('str')

        # reach_ids_side = [str(r) for r in reach_ids_side]

        # geom_con_shp.loc[geom_con_shp['geom1_rch_'].isin(reach_ids_side)]

        # geom_con_side_shp = geom_con_shp.loc[(geom_con_shp['geom1_rch_'].isin(reach_ids_side)) | (geom_con_shp['geom2_rch_'].isin(reach_ids_side))]
        # geom_con_fname = geom_con_fname.replace('pts', 'sub_pts')
        # geom_con_side_shp.to_file(geom_con_fname)

        # shp = shp.loc[shp['reach_id'].isin(riv_shp_side['reach_id'].to_list())].reset_index()
        # shp = shp.loc[shp['reach_id'].isin(reach_ids_side_ass_int)].reset_index()
        shp = shp.loc[shp['reach_id'].isin(reach_ids_side_ass_int)].reset_index(drop=True)
        reach_ids = shp['reach_id'].astype('int').astype('str').to_list()
        reach_ids_dist_out = shp['dist_out'].to_list()
        shp['reach_id'] = shp['reach_id'].astype('int')
        lines = shp['geometry'].to_list() 

        geom_con_shp = gpd.read_file(geom_con_fname)
        geom_con_shp['geom1_rch_'] = geom_con_shp['geom1_rch_'].astype('str')
        geom_con_shp['geom2_rch_'] = geom_con_shp['geom2_rch_'].astype('str')


        # geom_con_side_shp = geom_con_shp.loc[(geom_con_shp['geom1_rch_'].isin(reach_ids_side)) | (geom_con_shp['geom2_rch_'].isin(reach_ids_side))]
        geom_con_side_shp = geom_con_shp.loc[(geom_con_shp['geom1_rch_'].isin(reach_ids_side_ass)) | (geom_con_shp['geom2_rch_'].isin(reach_ids_side_ass))]
        geom_con_fname = geom_con_fname.replace('pts', 'sub_pts') #  CHECK THIS NAMING!!
        geom_con_side_shp.to_file(geom_con_fname)

        geom_con_side = fiona.open(geom_con_fname)
        # geom_con_all = fiona.open(geom_con_fname)

        # #Side reaches and main network reaches attached to side network
        # reach_ids_side_att = geom_con_side_shp['geom1_rch_'].to_list()
        # reach_ids_side_att.extend(geom_con_side_shp['geom2_rch_'].to_list())
        # reach_ids_side_att = list(set(reach_ids_side_att))

    # geom_con_shp.loc[geom_con_shp['geom1_rch_'] == '81380100351']

        con_geom1_rch_id = []
        con_geom2_rch_id = []
        con_geom1_n_pnts = []
        con_ind_intr = []
        con_closest = []
        for con in geom_con_side:
            con_geom = shapely.geometry.shape(con.geometry)
            
            con_geom1_rch_id.append(str(con["properties"]["geom1_rch_"]))
            con_geom2_rch_id.append(str(con["properties"]["geom2_rch_"]))
            con_geom1_n_pnts.append(con["properties"]["geom1_n_pn"])
            con_ind_intr.append(con["properties"]["ind_intr"])
            if con["properties"]["geom1_n_pn"] == 2:
                if con["properties"]["ind_intr"] == 1:
                    val = 1
                else:
                    val = 0
            else:
                val = min([0, con["properties"]["geom1_n_pn"]], key = lambda x: abs(x - con["properties"]["ind_intr"]))
            con_closest.append(val) # If val != 0, then geom2 is upstream


        ### NOW ONLY REVISIT PLACES THAT HAVE BEEN IDENTIFIED AS STILL PROBLEMATIC OR HAVE BEEN CHANGED
        reach_id_chg = df_change.loc[df_change['change'] == 1]['reach_id'].to_list()
        reach_id_chg = [int(i) for i in reach_id_chg]
        reach_id_prob = df_con_prob.loc[df_con_prob['con_prob'] == 1]['geom1_rch_id'].to_list()
        reach_id_prob = [int(i) for i in reach_id_prob]
        reach_id_prob.extend(reach_id_chg)

        shp_sub = shp.loc[shp['reach_id'].isin(reach_id_prob)]

        ind0_geom1_rch_id = []
        ind0_geom2_rch_id = []

        indf_geom1_rch_id = []
        indf_geom2_rch_id = []

        d_list = []

        for ix, r in shp_sub.iterrows():
            geom1 = lines[ix]
            geom1_rch_id = reach_ids[ix]

            con_geom1_rch_id_ind = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
            
            selected_reach = shp_sub.loc[shp_sub['reach_id'] == int(geom1_rch_id)]
            #Finding reaches within search distance (11 km)
            reaches_win_dist = shp.assign(distance=shp.apply(lambda x: x.geometry.distance(selected_reach.geometry.iloc[0]), axis=1)).query(f'distance <= {0.1}')
            reaches_win_dist = reaches_win_dist[reaches_win_dist.reach_id != geom1_rch_id]

            #Only comparing selected_reach to reaches within search distance (minimizes computational needs)
            for ix2, r2 in reaches_win_dist.iterrows(): 

                geom2 = shapely.geometry.shape(reaches_win_dist.geometry[ix2])
                geom2_rch_id = reaches_win_dist.reach_id.astype('str')[ix2]

                ## Don't need to record intersection if reach ID is the same
                if geom1_rch_id == geom2_rch_id:
                    continue

                if geom1.distance(geom2) < eps: 
                    con_geom2_rch_id_ind = [i for i in range(len(con_geom2_rch_id)) if con_geom2_rch_id[i] == str(geom2_rch_id)]
                    con_geom1_geom2_ind = [i for i in con_geom1_rch_id_ind if i in con_geom2_rch_id_ind]

                    con_geom2_opp_rch_id_ind = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom2_rch_id)]
                    con_geom1_opp_rch_id_ind = [i for i in range(len(con_geom2_rch_id)) if con_geom2_rch_id[i] == str(geom1_rch_id)]
                    con_geom1_geom2_opp_ind = [i for i in con_geom1_opp_rch_id_ind if i in con_geom2_opp_rch_id_ind]

                    con_geom1_closest = [con_closest[i] for i in con_geom1_geom2_ind]    
                    con_geom2_closest = [con_closest[i] for i in con_geom1_geom2_opp_ind]    

                # if geom1.distance(geom2) < eps: 

                    ## Check for locations where 2 segments say they are both downstream of one another 
                    if (con_geom1_closest[0] == 0) & (con_geom2_closest[0] == 0):
                        # rch_id_dn = geom_dist_out.index(min(geom_dist_out))
                        # if rch_id_dn == 0:
                            # ind0_geom1_rch_id.append(str(geom1_rch_id))
                        # if (geom1_rch_id not in ind0_geom1_rch_id) | (geom2_rch_id not in ind0_geom2_rch_id):
                        ind0_geom1_rch_id.append(str(geom1_rch_id))
                        ind0_geom2_rch_id.append(str(geom2_rch_id))

                        #Adding reaches to dataframe identifying if it's potentially a problem (1) or not (0)
                        d = {"geom1_rch_id": geom1_rch_id, "geom2_rch_id": geom2_rch_id, "con_prob": 1}
                        d_list.append(d)

                    ## Check for locations where 2 segments say they are both upstream of one another 
                    elif (con_geom1_closest[0] != 0) & (con_geom2_closest[0] != 0):
                        # if (geom1_rch_id not in indf_geom1_rch_id) | (geom2_rch_id not in indf_geom2_rch_id):
                        indf_geom1_rch_id.append(str(geom1_rch_id))
                        indf_geom2_rch_id.append(str(geom2_rch_id))

                        d = {"geom1_rch_id": geom1_rch_id, "geom2_rch_id": geom2_rch_id, "con_prob": 1}
                        d_list.append(d)
                    
                    else:
                        d = {"geom1_rch_id": geom1_rch_id, "geom2_rch_id": geom2_rch_id, "con_prob": 0}
                        d_list.append(d)


        # ind0_geom1_rch_id = []
        # ind0_geom2_rch_id = []

        # indf_geom1_rch_id = []
        # indf_geom2_rch_id = []

        # d_list = []

        # for ix, r in shp.iterrows():
        #     geom1 = lines[ix]
        #     geom1_rch_id = reach_ids[ix]

        #     con_geom1_rch_id_ind = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
            
        #     selected_reach = shp.loc[shp['reach_id'] == int(geom1_rch_id)]
        #     #Finding reaches within search distance (11 km)
        #     reaches_win_dist = shp.assign(distance=shp.apply(lambda x: x.geometry.distance(selected_reach.geometry.iloc[0]), axis=1)).query(f'distance <= {0.1}')
        #     reaches_win_dist = reaches_win_dist[reaches_win_dist.reach_id != geom1_rch_id]

        #     #Only comparing selected_reach to reaches within search distance (minimizes computational needs)
        #     for ix2, r2 in reaches_win_dist.iterrows(): 

        #         geom2 = shapely.geometry.shape(reaches_win_dist.geometry[ix2])
        #         geom2_rch_id = reaches_win_dist.reach_id.astype('str')[ix2]

        #         ## Don't need to record intersection if reach ID is the same
        #         if geom1_rch_id == geom2_rch_id:
        #             continue

        #         if geom1.distance(geom2) < eps: 
        #             con_geom2_rch_id_ind = [i for i in range(len(con_geom2_rch_id)) if con_geom2_rch_id[i] == str(geom2_rch_id)]
        #             con_geom1_geom2_ind = [i for i in con_geom1_rch_id_ind if i in con_geom2_rch_id_ind]

        #             con_geom2_opp_rch_id_ind = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom2_rch_id)]
        #             con_geom1_opp_rch_id_ind = [i for i in range(len(con_geom2_rch_id)) if con_geom2_rch_id[i] == str(geom1_rch_id)]
        #             con_geom1_geom2_opp_ind = [i for i in con_geom1_opp_rch_id_ind if i in con_geom2_opp_rch_id_ind]

        #             con_geom1_closest = [con_closest[i] for i in con_geom1_geom2_ind]    
        #             con_geom2_closest = [con_closest[i] for i in con_geom1_geom2_opp_ind]    

        #         # if geom1.distance(geom2) < eps: 

        #             ## Check for locations where 2 segments say they are both downstream of one another 
        #             if (con_geom1_closest[0] == 0) & (con_geom2_closest[0] == 0):
        #                 # rch_id_dn = geom_dist_out.index(min(geom_dist_out))
        #                 # if rch_id_dn == 0:
        #                     # ind0_geom1_rch_id.append(str(geom1_rch_id))
        #                 # if (geom1_rch_id not in ind0_geom1_rch_id) | (geom2_rch_id not in ind0_geom2_rch_id):
        #                 ind0_geom1_rch_id.append(str(geom1_rch_id))
        #                 ind0_geom2_rch_id.append(str(geom2_rch_id))

        #                 #Adding reaches to dataframe identifying if it's potentially a problem (1) or not (0)
        #                 d = {"geom1_rch_id": geom1_rch_id, "geom2_rch_id": geom2_rch_id, "con_prob": 1}
        #                 d_list.append(d)

        #             ## Check for locations where 2 segments say they are both upstream of one another 
        #             elif (con_geom1_closest[0] != 0) & (con_geom2_closest[0] != 0):
        #                 # if (geom1_rch_id not in indf_geom1_rch_id) | (geom2_rch_id not in indf_geom2_rch_id):
        #                 indf_geom1_rch_id.append(str(geom1_rch_id))
        #                 indf_geom2_rch_id.append(str(geom2_rch_id))

        #                 d = {"geom1_rch_id": geom1_rch_id, "geom2_rch_id": geom2_rch_id, "con_prob": 1}
        #                 d_list.append(d)
                    
        #             else:
        #                 d = {"geom1_rch_id": geom1_rch_id, "geom2_rch_id": geom2_rch_id, "con_prob": 0}
        #                 d_list.append(d)


        ind0_df = pd.DataFrame({'geom1_rch_id': ind0_geom1_rch_id, 'geom2_rch_id': ind0_geom2_rch_id})
        indf_df = pd.DataFrame({'geom1_rch_id': indf_geom1_rch_id, 'geom2_rch_id': indf_geom2_rch_id})

        ind0_df = (ind0_df[~ind0_df.filter(like='geom').apply(frozenset, axis=1).duplicated()].reset_index(drop=True))
        indf_df = (indf_df[~indf_df.filter(like='geom').apply(frozenset, axis=1).duplicated()].reset_index(drop=True))

        print('The number of possible locations with potential reversed LineString issues is ' + str(len(ind0_df) + len(indf_df)))

        df_con_prob = pd.DataFrame(d_list)               
        print('The number of potentially problematic areas is: ' + str(len(df_con_prob.loc[df_con_prob['con_prob'] == 1])))
        df_change = pd.DataFrame({'reach_id': reach_ids, 'change': 0})   


        # ind0_df.loc[ind0_df['geom1_rch_id'] == '82268000591'] # ix = 85

        rch_ids_rev = [] # Reach IDs that need to have the coordinates of the linestring reversed 

        ## Now loop through ind0_df and figure out which reach needs to be fixed, if it needs to be fixed at all
        for ix in range(len(ind0_df)):
            geom1_rch_id = ind0_df['geom1_rch_id'][ix]
            geom2_rch_id = ind0_df['geom2_rch_id'][ix]

            geom1_dist_out = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['dist_out'].to_list()[0]
            geom2_dist_out = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['dist_out'].to_list()[0]

            if (str(geom1_rch_id) not in reach_ids_side) & (str(geom2_rch_id) not in reach_ids_side):
                continue

            # geom1_type = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['end_reach'].to_list()[0]
            # geom2_type = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['end_reach'].to_list()[0]

            ## Check the number of geometric connections for both reaches 1 and 2 to figure out
            ## if the reaches involve lies at a junction or not. Checking both reaches and taking
            ## the minimum number of geometric connections ensures that I'm looking at the right
            ## area based on the 2 reaches involved (reach 1 could be at a junction and reach 2 could
            ## be adjacent to the junction)
            con_cnt_geom1 = con_geom1_rch_id.count(str(geom1_rch_id))
            con_cnt_geom2 = con_geom2_rch_id.count(str(geom2_rch_id))
            con_cnt = min([con_cnt_geom1, con_cnt_geom2])

            ## For identifying cases where the 2 "problem" reaches are both connected to a junction, but the junctions
            ## aren't a part of the problem
            geom1_idx = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
            geom1_idx_geom2 = [con_geom2_rch_id[i] for i in geom1_idx]
            geom2_idx = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom2_rch_id)]
            geom2_idx_geom2 = [con_geom2_rch_id[i] for i in geom2_idx]
            geom1_geom2_comm = list(set(geom1_idx_geom2) & set(geom2_idx_geom2))
            if len(geom1_geom2_comm) == 1:
                #Check if the reach in common is associated with both junctions: e.g., 11743000191
                #This would be the case if it's associated with both geom1_rch_id and geom2_rch_id,
                #but attached at opposite ends of the reach
                comm_idx = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_geom2_comm[0])]
                comm_idx_geom2 = [con_geom2_rch_id[i] for i in comm_idx]
                comm_idx_geom2_geom1 = comm_idx_geom2.index(str(geom1_rch_id))
                comm_idx_geom2_geom2 = comm_idx_geom2.index(str(geom2_rch_id))

                geom1_comm_idx = comm_idx[comm_idx_geom2_geom1]
                geom1_comm_idx_closest = con_closest[geom1_comm_idx]
        
                geom2_comm_idx  =comm_idx[comm_idx_geom2_geom2]
                geom2_comm_idx_closest = con_closest[geom2_comm_idx]

                if geom1_comm_idx_closest != geom2_comm_idx_closest: # Maybe need to modify this if statement?
                    con_cnt = 2

            if (con_cnt >= 3) & (len(geom1_geom2_comm) == 0):
                con_cnt = 2

            ## If the junction is a headwater a junction, set con_cnt = 3 to ensure junction code is entered below
            ## Previously, the code assumed every junction had an additional reach or end reach attached to all 
            ## segments in the junction, so con_cnt would equal 2 instead of 3 in this case
            ## Check that this code actually works!
            if (str(geom1_rch_id) in head_at_junc) | (str(geom2_rch_id) in head_at_junc):
                con_cnt = 3

            if con_cnt >= 3:

                # con_pts_pot_junc_ids = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == int(geom1_rch_id)]['geom2_rch_'].astype('str').to_list()
            
                ## Find indices of reach ID of interest in the generated initial connectivity that was created before reversing coordinates of linestrings
                indices = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
                ## Find reach IDs that are geometrically connected to reach ID of interest
                con_geom2_ids_indices = [con_geom2_rch_id[i] for i in indices]
                ## For reach IDs geometrically connected to reach ID of interest, count the number of geometric connections they have
                ## (this narrows it down to those reach IDs that might form the river junction with the reach ID of interest, since 1+
                ##  of the geometrically connected reach IDs might be a segment up/downstream of the reach ID of interest forming
                ##  the river junction)
                con_geom2_ids_cnt = [con_geom2_rch_id.count(i) for i in con_geom2_ids_indices]
                ## Potential IDs of other reaches associated with the river junction
                ## ("Potential" is used here because a reach ID geometrically connected to the reach ID of interest might have
                ##   another river junction attached to it than the one we are currently looking at)
                con_geom2_pot_junc_indices =  [i for i in range(len(con_geom2_ids_cnt)) if con_geom2_ids_cnt[i] >= 3]
                # con_geom2_pot_junc_indices =  [i for i in range(len(con_geom2_ids_cnt)) if con_geom2_ids_cnt[i] >= 2]
                ## Confirm which potential IDs are actually associated with the river junction of interest
                con_geom2_pot_junc_ids = [con_geom2_ids_indices[i] for i in con_geom2_pot_junc_indices]
                con_geom2_junc_ids = []
                for con_id in con_geom2_pot_junc_ids:
                    con_geom1_pot_junc_indices = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == con_id]
                    con_geom2_junc_ids_tmp = [con_geom2_rch_id[i] for i in con_geom1_pot_junc_indices]
                    if str(geom1_rch_id) in con_geom2_junc_ids_tmp:
                        con_geom2_junc_ids.append(con_id)

                ## Only keep reach IDs in the junction that are also connected to geom2_rch_id
                ids_geom2_rch = geom_con_side_shp.loc[geom_con_side_shp['geom1_rch_'] == str(geom2_rch_id)]['geom2_rch_'].astype('str').to_list()
                ids_geom2_rch.append(geom2_rch_id)
                con_geom2_junc_ids = [i for i in con_geom2_junc_ids if i in ids_geom2_rch]
                con_geom2_junc_ids.append(str(geom1_rch_id))
                if geom2_rch_id not in con_geom2_junc_ids:
                    con_geom2_junc_ids.append(str(geom2_rch_id))

                # junc_ids = [int(i) for i in con_geom2_junc_ids]
                junc_ids = [str(i) for i in con_geom2_junc_ids]
                reach_ids_indices = []
                for junc_id in junc_ids:
                    reach_ids_indices.extend([i for i in range(len(reach_ids)) if reach_ids[i] == junc_id])
                junc_dist_out = [reach_ids_dist_out[i] for i in reach_ids_indices]

                ## Check if reaches are forming a loop (e.g., 74230100021, 74230100101, and 74230100061 form a loop)
                ## Code below identifies reaches that don't meet at a common point and therefore likely form a loop
                r1 = []
                r2 = []
                pt = []
                att = []
                for i in junc_ids:
                    for j in junc_ids:

                        if i == j:
                            continue

                        tmp1 = shp.loc[shp['reach_id'] == int(i)]['geometry'].to_list()[0]
                        tmp2 = shp.loc[shp['reach_id'] == int(j)]['geometry'].to_list()[0]

                        if tmp1.distance(tmp2) < eps: 
                            # point = shapely.ops.nearest_points(tmp1, tmp2)[0]
                            point = shapely.intersection(tmp1, tmp2)

                            #The 2 reaches are attached at both ends
                            if point.geom_type == 'MultiPoint':
                                ## If shapely is returning 2 points when it shouldn't (e.g., 11744600021 and 11744600011)
                                mp_pts = [(point.geoms[p].x, point.geoms[p].y) for p in range(len(point.geoms))]
                                mp_pts_rnd = [(round(mp_pts[p][0], 4), round(mp_pts[p][1], 4)) for p in range(len(mp_pts))]
                                mp_pts_rnd_uniq = list(set(mp_pts_rnd))
                                if len(mp_pts_rnd_uniq) == 1:
                                    point = shapely.ops.nearest_points(tmp1, tmp2)[0]
                                    att.append(False)
                                else:
                                    att.append(True)
                            else:
                                point = shapely.ops.nearest_points(tmp1, tmp2)[0]
                                att.append(False)

                            r1.append(i)
                            r2.append(j)
                            pt.append(point)

                df_chk_loop = pd.DataFrame({"r1": r1, "r2": r2, "point": pt, "att": att})
                att_ids = df_chk_loop.loc[df_chk_loop["att"] == True]
                att_ids = att_ids["r1"].to_list()
                comm_pt = df_chk_loop["point"].mode().to_list()[0]
                df_chk_yes_loop = df_chk_loop.loc[df_chk_loop["point"] != comm_pt]
                df_chk_yes_loop = df_chk_yes_loop.loc[~df_chk_loop["r1"].isin(att_ids)]
                df_chk_yes_loop = df_chk_yes_loop.loc[~df_chk_loop["r2"].isin(att_ids)]
                loop_ids = df_chk_yes_loop["r1"].to_list()
                loop_ids.extend(df_chk_yes_loop["r2"].to_list())
                loop_ids = list(set(loop_ids))

                df_chk_not_loop = df_chk_loop.loc[df_chk_loop["point"] == comm_pt]
                not_loop_ids = df_chk_not_loop["r1"].to_list()
                not_loop_ids.extend(df_chk_not_loop["r2"].to_list())
                not_loop_ids = list(set(not_loop_ids))

                junc_ids = not_loop_ids
                reach_ids_indices = []
                for junc_id in junc_ids:
                    reach_ids_indices.extend([i for i in range(len(reach_ids)) if reach_ids[i] == junc_id])
                junc_dist_out = [reach_ids_dist_out[i] for i in reach_ids_indices]


                #Identifying what's forming the loop that's not a part of the junction
                loop_not_junc_id = list(set(loop_ids) - set(not_loop_ids))
                if len(loop_not_junc_id) > 1:
                    print("There is more than one reach that is part of the loop, but not part of the junction. Check code.")
                    break
                    # raise SystemExit(22) 

                ## If geom1_rch_id or geom2_rch_id are not in the "junction" IDs, then it's probably not a junction?
                ## Treat it like 2 segments connected to one another instead
                if (str(geom1_rch_id) not in not_loop_ids) | (str(geom2_rch_id) not in not_loop_ids):
                        ## If it's not a junction, then use outlet distance to figure out which one to reverse
                    if geom1_dist_out < geom2_dist_out:
                        rch_ids_rev.append(geom2_rch_id)
                        #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                        df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                        df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1
                        continue

                    elif geom1_dist_out > geom2_dist_out:
                        rch_ids_rev.append(geom1_rch_id)
                        #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                        df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                        df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1
                        continue

                ## If reaches form a loop, identify up and downstream reaches differently
                if len(loop_ids) == 3:
                    junc_ids = not_loop_ids
                    reach_ids_indices = []
                    for junc_id in junc_ids:
                        reach_ids_indices.extend([i for i in range(len(reach_ids)) if reach_ids[i] == junc_id])
                    junc_dist_out = [reach_ids_dist_out[i] for i in reach_ids_indices]

                    #Getting outlet distance for reach that's not part of the junction
                    loop_not_junc_id_idx = reach_ids.index(loop_not_junc_id[0])
                    loop_not_junc_id_dist_out = reach_ids_dist_out[loop_not_junc_id_idx]

                # elif len(loop_ids) > 0:
                elif len(loop_ids) > 3:
                    # print('The number of IDs in the loop is greater than 0, but not equal to 3. Double check code.')
                    print('The number of IDs in the loop is greater than 3. Double check code.')
                    break
                    # raise SystemExit(22) 

                # elif len(loop_ids) == 0:  
                ## Write code to determine if junction is multi upstream or multi downstream
                ## The reach with the greatest outlet distance is always going to be an upstream segment(?)
                ## So for each of the remaining segments, loop through and find neighbors that aren't associated
                ## with junctions, get outlet distances, and compare neighbor outlet distances to junction outlet distances
                max_junc_dist_out = junc_dist_out.index(max(junc_dist_out))
                min_junc_dist_out = junc_dist_out.index(min(junc_dist_out))
                up_dn = []
                for i in range(len(junc_ids)):
                    # if i == max_junc_dist_out:
                    #     up_dn.append('up')
                    #     continue

                    # if i == min_junc_dist_out:
                    #     up_dn.append('dn')
                    #     continue

                    if junc_ids[i] in head_at_junc:
                        up_dn.append('up')
                        continue

                    if shp.loc[shp['reach_id'] == int(junc_ids[i])].main_side.to_list()[0] == 0:
                        def_con = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
                        def_con_id_neigh = def_con['geom2_rch_'].to_list()
                        def_con_id_neigh = [int(j) for j in def_con_id_neigh]
                        def_con_neigh_main_side = shp.loc[shp['reach_id'].isin(def_con_id_neigh)]
                        def_con_neigh_main = def_con_neigh_main_side.loc[def_con_neigh_main_side['main_side'] == 0].reach_id.to_list()
                        def_con_neigh_main = [str(j) for j in def_con_neigh_main]
                        def_con_neigh_main_con = geom_con_shp.loc[geom_con_shp['geom1_rch_'].isin(def_con_neigh_main)] 
                        def_con_neigh_main_con = def_con_neigh_main_con.loc[def_con_neigh_main_con['geom2_rch_'] == junc_ids[i]]
                        def_con_neigh_main_con = def_con_neigh_main_con.loc[def_con_neigh_main_con['geom1_rch_'].isin(junc_ids)]
                        # if len(def_con_neigh_main_con) > 1:
                        #     print('There is more than 1 attached main network reach. Check code.')
                        #     print(geom1_rch_id)
                        #     print(geom2_rch_id)
                        #     break

                        if len(def_con_neigh_main_con) >= 1:
                            val = min([0, def_con_neigh_main_con["geom1_n_pn"].to_list()[0]], key = lambda x: abs(x - def_con_neigh_main_con["ind_intr"].to_list()[0]))
                            if val == 0:
                                up_dn.append('dn')
                                continue
                            else:
                                up_dn.append('up')
                                continue

                    id_neigh = geom_con_side_shp.loc[geom_con_side_shp['geom1_rch_'] == junc_ids[i]] 
                    id_neigh = id_neigh.loc[~id_neigh['geom2_rch_'].isin(junc_ids)]

                    dists_neigh = []
                    main_side_neigh = []
                    for j in id_neigh['geom2_rch_'].to_list():
                        dists_neigh.extend(shp.loc[shp['reach_id'] == int(j)].dist_out.to_list())
                        main_side_neigh.extend(shp.loc[shp['reach_id'] == int(j)].main_side.to_list())

                    dist_id = junc_dist_out[i]
                    # If up_dn_chk is True, this reach is downstream in the junction
                    # up_dn_chk = all(dist_id > x for x in dists_neigh)
                    # up_dn_chk fails in places where the reach of interest is downstream and at another junction,
                    # 1 of the other reaches at the junction is downstream and the other reach is a headwater coming
                    # into the junction that has a greater outlet distance than the reach ID of interest. So instead
                    # use this code to check if any of the neighbors have a lower outlet distance. This code may break
                    # or not work correctly in some locations, but I'm not sure yet.
                    bool_comp_dist = [dist_id > x for x in dists_neigh]
                    # Create list matching junc_ids that says if each reach is up or dn, max_junc_dist_out is automatically up
                    # if up_dn_chk == True:
                    if len(dists_neigh) == 1:
                        #If there is only 1 other neighbor
                        # if any(x == True for x in bool_comp_dist):
                        if any(x == True for x in bool_comp_dist) & any(y != 0 for y in main_side_neigh):
                            up_dn.append('dn')
                        elif any(x == True for x in bool_comp_dist) & any(y == 0 for y in main_side_neigh):
                            up_dn.append('up')
                        else:
                            up_dn.append('up')
                    else:
                        #If the reach is attached to another junction
                        # if all(x == True for x in bool_comp_dist):
                        if all(x == True for x in bool_comp_dist) & (i != max_junc_dist_out):
                            up_dn.append('dn')
                        # elif any(x == True for x in bool_comp_dist) & all(y == 0 for y in main_side_neigh):
                        #     up_dn.append('dn')
                        else:
                            up_dn.append('up')                   
                
                up_cnt = up_dn.count('up')
                dn_cnt = up_dn.count('dn')

                ## If all of the reaches are labeled as upstream or downstream, that isn't possible
                ## so now use min and max outlet distance to label at least 1 up and 1 down
                if (all(x == "up" for x in up_dn)) | (all(x == "dn" for x in up_dn)):
                    up_dn = []
                    for i in range(len(junc_ids)):
                        if i == max_junc_dist_out:
                            up_dn.append('up')
                            continue

                        if i == min_junc_dist_out:
                            up_dn.append('dn')
                            continue

                        if junc_ids[i] in head_at_junc:
                            up_dn.append('up')
                            continue

                        if shp.loc[shp['reach_id'] == int(junc_ids[i])].main_side.to_list()[0] == 0:
                            def_con = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
                            def_con_id_neigh = def_con['geom2_rch_'].to_list()
                            def_con_id_neigh = [int(j) for j in def_con_id_neigh]
                            def_con_neigh_main_side = shp.loc[shp['reach_id'].isin(def_con_id_neigh)]
                            def_con_neigh_main = def_con_neigh_main_side.loc[def_con_neigh_main_side['main_side'] == 0].reach_id.to_list()
                            def_con_neigh_main = [str(j) for j in def_con_neigh_main]
                            def_con_neigh_main_con = geom_con_shp.loc[geom_con_shp['geom1_rch_'].isin(def_con_neigh_main)] 
                            def_con_neigh_main_con = def_con_neigh_main_con.loc[def_con_neigh_main_con['geom2_rch_'] == junc_ids[i]]
                            def_con_neigh_main_con = def_con_neigh_main_con.loc[def_con_neigh_main_con['geom1_rch_'].isin(junc_ids)]
                            # if len(def_con_neigh_main_con) > 1:
                            #     print('There is more than 1 attached main network reach. Check code.')
                            #     print(geom1_rch_id)
                            #     print(geom2_rch_id)
                            #     break

                            if len(def_con_neigh_main_con) >= 1:
                                val = min([0, def_con_neigh_main_con["geom1_n_pn"].to_list()[0]], key = lambda x: abs(x - def_con_neigh_main_con["ind_intr"].to_list()[0]))
                                if val == 0:
                                    up_dn.append('dn')
                                    continue
                                else:
                                    up_dn.append('up')
                                    continue


                        id_neigh = geom_con_side_shp.loc[geom_con_side_shp['geom1_rch_'] == junc_ids[i]] 
                        # id_neigh = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
                        id_neigh = id_neigh.loc[~id_neigh['geom2_rch_'].isin(junc_ids)]

                        dists_neigh = []
                        main_side_neigh = []
                        for j in id_neigh['geom2_rch_'].to_list():
                            dists_neigh.extend(shp.loc[shp['reach_id'] == int(j)].dist_out.to_list())
                            main_side_neigh.extend(shp.loc[shp['reach_id'] == int(j)].main_side.to_list())

                        dist_id = junc_dist_out[i]
                        # If up_dn_chk is True, this reach is downstream in the junction
                        # up_dn_chk = all(dist_id > x for x in dists_neigh)
                        # up_dn_chk fails in places where the reach of interest is downstream and at another junction,
                        # 1 of the other reaches at the junction is downstream and the other reach is a headwater coming
                        # into the junction that has a greater outlet distance than the reach ID of interest. So instead
                        # use this code to check if any of the neighbors have a lower outlet distance. This code may break
                        # or not work correctly in some locations, but I'm not sure yet.
                        bool_comp_dist = [dist_id > x for x in dists_neigh]
                        # Create list matching junc_ids that says if each reach is up or dn, max_junc_dist_out is automatically up
                        # if up_dn_chk == True:
                        if len(dists_neigh) == 1:
                            #If there is only 1 other neighbor
                            # if any(x == True for x in bool_comp_dist):
                            if any(x == True for x in bool_comp_dist) & any(y != 0 for y in main_side_neigh):
                                up_dn.append('dn')
                            elif any(x == True for x in bool_comp_dist) & any(y == 0 for y in main_side_neigh):
                                up_dn.append('up')
                            else:
                                up_dn.append('up')
                        else:
                            #If the reach is attached to another junction
                            if all(x == True for x in bool_comp_dist):
                                up_dn.append('dn')
                            else:
                                up_dn.append('up')                   
                    
                    up_cnt = up_dn.count('up')
                    dn_cnt = up_dn.count('dn')
                    
                if (up_cnt >= 2) & (dn_cnt == 1):
                    ## If it's a multi upstream junction:
                    ## Now determine which reach ID is the downstream most reach ID of the junction, and store the reach IDs of every reach but 
                    ## the most downstream segment as the ones to reverse coordinates of the linestring
                    ##  ACTUALLY THE ABOVE COMMENT IS INCORRECT: reverse the corrdinates of the linestring for the most downstream segment

                    ## If the 2 segments involved are one upstream in the junction and one downstream,
                    ## reverse the coordinates of the linestring for the most downstream segment

                    min_junc_dist_out = junc_dist_out.index(min(junc_dist_out))
                    ## Only reversing if the downstream ID of the junction has matching starting coordinates with the other reach IDs
                    if (junc_ids[min_junc_dist_out] == geom1_rch_id) | (junc_ids[min_junc_dist_out] == geom2_rch_id):
                        rch_ids_rev.append(junc_ids[min_junc_dist_out])
                        #Set change to 1 so I know a modification was made
                        df_change.loc[df_change['reach_id'] == str(junc_ids[min_junc_dist_out]), 'change'] = 1

                    #Set df_con_prob to zero since either there is no problem or it has been fixed in this location (with reversing) 
                    df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0

                # elif (up_cnt == 1) & (dn_cnt == 2):
                elif (up_cnt == 1) & (dn_cnt >= 2):
                    ## If it's a multi downstream junction:
                    geom1_up_dn = up_dn[junc_ids.index(geom1_rch_id)]
                    geom2_up_dn = up_dn[junc_ids.index(geom2_rch_id)]

                    ## If both segments involved are one up and one down, flip the downstream segment
                    if not all(x == 'dn' for x in [geom1_up_dn, geom2_up_dn]):
                        if geom1_up_dn == 'dn':
                            rch_ids_rev.append(geom1_rch_id)
                            #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                            df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                            df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1
                        elif geom2_up_dn == 'dn':
                            rch_ids_rev.append(geom2_rch_id)
                            #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                            df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                            df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

                elif (up_cnt == 2) & (dn_cnt == 2):
                    ## If it's a multi downstream junction:
                    geom1_up_dn = up_dn[junc_ids.index(geom1_rch_id)]
                    geom2_up_dn = up_dn[junc_ids.index(geom2_rch_id)]

                    ## Checking for locations where 2 reaches are connected at both ends (e.g., 81340500201 and 81340500011)
                    tmp1 = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['geometry'].to_list()[0]
                    tmp2 = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['geometry'].to_list()[0]
                    if tmp1.distance(tmp2) < eps: 
                        point = shapely.ops.nearest_points(tmp1, tmp2)
                        if len(point) > 1:
                            if all(x == 'dn' for x in [geom1_up_dn, geom2_up_dn]):
                                df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                                continue
            
                    ## If both segments involved are both the upstream segment do nothing
                    ## If both segments involved are both the downstream segment, flip both
                    ## If both segments involved are one up and one down, flip the downstream segment
                    if all(x == 'dn' for x in [geom1_up_dn, geom2_up_dn]):
                        rch_ids_rev.append(geom1_rch_id)
                        rch_ids_rev.append(geom2_rch_id)
                        df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1
                        df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

                    elif (not all(x == 'up' for x in [geom1_up_dn, geom2_up_dn])) & (not all(x == 'dn' for x in [geom1_up_dn, geom2_up_dn])):
                        if geom1_up_dn == 'dn':
                            rch_ids_rev.append(geom1_rch_id)
                            df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1

                        elif geom2_up_dn == 'dn':
                            rch_ids_rev.append(geom2_rch_id)
                            df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

                    df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0

                else:
                    print('New junction configuration exists! Refine code. Reaches involved:')
                    print(geom1_rch_id)
                    print(geom2_rch_id)
                    break
                    # raise SystemExit(22) 

                        
            ## If it's not a junction, then use outlet distance to figure out which one to reverse
            elif geom1_dist_out < geom2_dist_out:
                rch_ids_rev.append(geom1_rch_id)
                #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1

            elif geom1_dist_out > geom2_dist_out:
                rch_ids_rev.append(geom2_rch_id)
                #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1



        # df_con_prob.loc[df_con_prob['con_prob'] == 1].count() # Potentially problematic areas: 35
        # df_change.loc[df_change['change'] == 1].count() # Areas with a modification applied: 4


        # indf_df.loc[indf_df['geom2_rch_id'] == '81210700171'] # ix = 11
        # indf_df.loc[indf_df['geom2_rch_id'] == '65321000791'] # ix = 6
        # indf_df.loc[indf_df['geom1_rch_id'] == '65321000791'] # ix = 0

        ## Now loop through inf_df and figure out which reach needs to be fixed, if it needs to be fixed at all
        for ix in range(len(indf_df)):
            geom1_rch_id = indf_df['geom1_rch_id'][ix]
            geom2_rch_id = indf_df['geom2_rch_id'][ix]

            geom1_dist_out = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['dist_out'].to_list()[0]
            geom2_dist_out = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['dist_out'].to_list()[0]

            if (str(geom1_rch_id) not in reach_ids_side) & (str(geom2_rch_id) not in reach_ids_side):
                continue

            # geom1_type = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['end_reach'].to_list()[0]
            # geom2_type = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['end_reach'].to_list()[0]

            ## Check the number of geometric connections for both reaches 1 and 2 to figure out
            ## if the reaches involve lies at a junction or not. Checking both reaches and taking
            ## the minimum number of geometric connections ensures that I'm looking at the right
            ## area based on the 2 reaches involved (reach 1 could be at a junction and reach 2 could
            ## be adjacent to the junction)
            con_cnt_geom1 = con_geom1_rch_id.count(str(geom1_rch_id))
            con_cnt_geom2 = con_geom2_rch_id.count(str(geom2_rch_id))
            con_cnt = min([con_cnt_geom1, con_cnt_geom2])

            ## For identifying cases where the 2 "problem" reaches are both connected to a junction, but the junctions
            ## aren't a part of the problem
            geom1_idx = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
            geom1_idx_geom2 = [con_geom2_rch_id[i] for i in geom1_idx]
            geom2_idx = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom2_rch_id)]
            geom2_idx_geom2 = [con_geom2_rch_id[i] for i in geom2_idx]
            geom1_geom2_comm = list(set(geom1_idx_geom2) & set(geom2_idx_geom2))
            if len(geom1_geom2_comm) == 1:
                #Check if the reach in common is associated with both junctions: e.g., 11743000191
                #This would be the case if it's associated with both geom1_rch_id and geom2_rch_id,
                #but attached at opposite ends of the reach
                comm_idx = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_geom2_comm[0])]
                comm_idx_geom2 = [con_geom2_rch_id[i] for i in comm_idx]
                comm_idx_geom2_geom1 = comm_idx_geom2.index(str(geom1_rch_id))
                comm_idx_geom2_geom2 = comm_idx_geom2.index(str(geom2_rch_id))

                geom1_comm_idx = comm_idx[comm_idx_geom2_geom1]
                geom1_comm_idx_closest = con_closest[geom1_comm_idx]
        
                geom2_comm_idx  =comm_idx[comm_idx_geom2_geom2]
                geom2_comm_idx_closest = con_closest[geom2_comm_idx]

                if geom1_comm_idx_closest != geom2_comm_idx_closest: # Maybe need to modify this if statement?
                    con_cnt = 2

            if (con_cnt >= 3) & (len(geom1_geom2_comm) == 0):
                con_cnt = 2

            if (str(geom1_rch_id) in head_at_junc) | (str(geom2_rch_id) in head_at_junc):
                con_cnt = 3

            if con_cnt >= 3:
                ## If the 2 segments involved are one upstream in the junction and one downstream,
                ## reverse the coordinates of the linestring for the upstream segment

                ## Find indices of reach ID of interest in the generated initial connectivity that was created before reversing coordinates of linestrings
                indices = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == str(geom1_rch_id)]
                ## Find reach IDs that are geometrically connected to reach ID of interest
                con_geom2_ids_indices = [con_geom2_rch_id[i] for i in indices]
                ## For reach IDs geometrically connected to reach ID of interest, count the number of geometric connections they have
                ## (this narrows it down to those reach IDs that might form the river junction with the reach ID of interest, since 1+
                ##  of the geometrically connected reach IDs might be a segment up/downstream of the reach ID of interest forming
                ##  the river junction)
                con_geom2_ids_cnt = [con_geom2_rch_id.count(i) for i in con_geom2_ids_indices]
                ## Potential IDs of other reaches associated with the river junction
                ## ("Potential" is used here because a reach ID geometrically connected to the reach ID of interest might have
                ##   another river junction attached to it than the one we are currently looking at)
                con_geom2_pot_junc_indices =  [i for i in range(len(con_geom2_ids_cnt)) if con_geom2_ids_cnt[i] >= 3]
                # con_geom2_pot_junc_indices =  [i for i in range(len(con_geom2_ids_cnt)) if con_geom2_ids_cnt[i] >= 2]
                ## Confirm which potential IDs are actually associated with the river junction of interest
                con_geom2_pot_junc_ids = [con_geom2_ids_indices[i] for i in con_geom2_pot_junc_indices]
                con_geom2_junc_ids = []
                for con_id in con_geom2_pot_junc_ids:
                    con_geom1_pot_junc_indices = [i for i in range(len(con_geom1_rch_id)) if con_geom1_rch_id[i] == con_id]
                    con_geom2_junc_ids_tmp = [con_geom2_rch_id[i] for i in con_geom1_pot_junc_indices]
                    if str(geom1_rch_id) in con_geom2_junc_ids_tmp:
                        con_geom2_junc_ids.append(con_id)


                ## Only keep reach IDs in the junction that are also connected to geom2_rch_id
                ids_geom2_rch = geom_con_side_shp.loc[geom_con_side_shp['geom1_rch_'] == str(geom2_rch_id)]['geom2_rch_'].astype('str').to_list()
                ids_geom2_rch.append(geom2_rch_id)
                con_geom2_junc_ids = [i for i in con_geom2_junc_ids if i in ids_geom2_rch]
                con_geom2_junc_ids.append(str(geom1_rch_id))
                if geom2_rch_id not in con_geom2_junc_ids:
                    con_geom2_junc_ids.append(str(geom2_rch_id))

                # junc_ids = [int(i) for i in con_geom2_junc_ids]
                junc_ids = [str(i) for i in con_geom2_junc_ids]
                reach_ids_indices = []
                for junc_id in junc_ids:
                    reach_ids_indices.extend([i for i in range(len(reach_ids)) if reach_ids[i] == junc_id])
                junc_dist_out = [reach_ids_dist_out[i] for i in reach_ids_indices]

                ## Check if reaches are forming a loop (e.g., 74230100021, 74230100101, and 74230100061 form a loop)
                ## Code below identifies reaches that don't meet at a common point and therefore likely form a loop
                r1 = []
                r2 = []
                pt = []
                att = []
                for i in junc_ids:
                    for j in junc_ids:

                        if i == j:
                            continue

                        tmp1 = shp.loc[shp['reach_id'] == int(i)]['geometry'].to_list()[0]
                        tmp2 = shp.loc[shp['reach_id'] == int(j)]['geometry'].to_list()[0]

                        if tmp1.distance(tmp2) < eps: 
                            # point = shapely.ops.nearest_points(tmp1, tmp2)[0]
                            point = shapely.intersection(tmp1, tmp2)

                            #The 2 reaches are attached at both ends
                            if point.geom_type == 'MultiPoint':
                                ## If shapely is returning 2 points when it shouldn't (e.g., 11744600021 and 11744600011)
                                mp_pts = [(point.geoms[p].x, point.geoms[p].y) for p in range(len(point.geoms))]
                                mp_pts_rnd = [(round(mp_pts[p][0], 4), round(mp_pts[p][1], 4)) for p in range(len(mp_pts))]
                                mp_pts_rnd_uniq = list(set(mp_pts_rnd))
                                if len(mp_pts_rnd_uniq) == 1:
                                    point = shapely.ops.nearest_points(tmp1, tmp2)[0]
                                    att.append(False)
                                else:
                                    att.append(True)
                            else:
                                point = shapely.ops.nearest_points(tmp1, tmp2)[0]
                                att.append(False)

                            r1.append(i)
                            r2.append(j)
                            pt.append(point)

                df_chk_loop = pd.DataFrame({"r1": r1, "r2": r2, "point": pt, "att": att})
                att_ids = df_chk_loop.loc[df_chk_loop["att"] == True]
                att_ids = att_ids["r1"].to_list()
                comm_pt = df_chk_loop["point"].mode().to_list()[0]
                df_chk_yes_loop = df_chk_loop.loc[df_chk_loop["point"] != comm_pt]
                df_chk_yes_loop = df_chk_yes_loop.loc[~df_chk_loop["r1"].isin(att_ids)]
                df_chk_yes_loop = df_chk_yes_loop.loc[~df_chk_loop["r2"].isin(att_ids)]
                loop_ids = df_chk_yes_loop["r1"].to_list()
                loop_ids.extend(df_chk_yes_loop["r2"].to_list())
                loop_ids = list(set(loop_ids))

                df_chk_not_loop = df_chk_loop.loc[df_chk_loop["point"] == comm_pt]
                not_loop_ids = df_chk_not_loop["r1"].to_list()
                not_loop_ids.extend(df_chk_not_loop["r2"].to_list())
                not_loop_ids = list(set(not_loop_ids))

                junc_ids = not_loop_ids
                reach_ids_indices = []
                for junc_id in junc_ids:
                    reach_ids_indices.extend([i for i in range(len(reach_ids)) if reach_ids[i] == junc_id])
                junc_dist_out = [reach_ids_dist_out[i] for i in reach_ids_indices]

                # ## Check if reaches are forming a loop (e.g., 74230100021, 74230100101, and 74230100061 form a loop)
                # ## Code below identifies reaches that don't meet at a common point and therefore likely form a loop
                # r1 = []
                # r2 = []
                # pt = []
                # for i in junc_ids:
                #     for j in junc_ids:

                #         if i == j:
                #             continue

                #         tmp1 = shp.loc[shp['reach_id'] == int(i)]['geometry'].to_list()[0]
                #         tmp2 = shp.loc[shp['reach_id'] == int(j)]['geometry'].to_list()[0]

                #         if tmp1.distance(tmp2) < eps: 
                #             point = shapely.ops.nearest_points(tmp1, tmp2)[0]

                #             r1.append(i)
                #             r2.append(j)
                #             pt.append(point)

                # df_chk_loop = pd.DataFrame({"r1": r1, "r2": r2, "point": pt})
                # comm_pt = df_chk_loop["point"].mode().to_list()[0]
                # df_chk_yes_loop = df_chk_loop.loc[df_chk_loop["point"] != comm_pt]
                # loop_ids = df_chk_yes_loop["r1"].to_list()
                # loop_ids.extend(df_chk_yes_loop["r2"].to_list())
                # loop_ids = list(set(loop_ids))

                # df_chk_not_loop = df_chk_loop.loc[df_chk_loop["point"] == comm_pt]
                # not_loop_ids = df_chk_not_loop["r1"].to_list()
                # not_loop_ids.extend(df_chk_not_loop["r2"].to_list())
                # not_loop_ids = list(set(not_loop_ids))

                #Identifying what's forming the loop that's not a part of the junction
                loop_not_junc_id = list(set(loop_ids) - set(not_loop_ids))
                if len(loop_not_junc_id) > 1:
                    print("There is more than one reach that is part of the loop, but not part of the junction. Check code.")
                    break
                    # raise SystemExit(22) 

                ## If geom1_rch_id or geom2_rch_id are not in the "junction" IDs, then it's probably not a junction?
                ## Treat it like 2 segments connected to one another instead
                if (str(geom1_rch_id) not in not_loop_ids) | (str(geom2_rch_id) not in not_loop_ids):
                        ## If it's not a junction, then use outlet distance to figure out which one to reverse
                    if geom1_dist_out < geom2_dist_out:
                        rch_ids_rev.append(geom2_rch_id)
                        #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                        df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                        df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1
                        continue

                    elif geom1_dist_out > geom2_dist_out:
                        rch_ids_rev.append(geom1_rch_id)
                        #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                        df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                        df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1
                        continue

                ## If reaches form a loop, identify up and downstream reaches differently
                if len(loop_ids) == 3:
                    junc_ids = not_loop_ids
                    reach_ids_indices = []
                    for junc_id in junc_ids:
                        reach_ids_indices.extend([i for i in range(len(reach_ids)) if reach_ids[i] == junc_id])
                    junc_dist_out = [reach_ids_dist_out[i] for i in reach_ids_indices]

                    #Getting outlet distance for reach that's not part of the junction
                    loop_not_junc_id_idx = reach_ids.index(loop_not_junc_id[0])
                    loop_not_junc_id_dist_out = reach_ids_dist_out[loop_not_junc_id_idx]

                # elif len(loop_ids) > 0:
                elif len(loop_ids) > 3:
                    # print('The number of IDs in the loop is greater than 0, but not equal to 3. Double check code.')
                    print('The number of IDs in the loop is greater than 3. Double check code.')
                    break
                    # raise SystemExit(22) 

                # elif len(loop_ids) == 0:  
                ## Write code to determine if junction is multi upstream or multi downstream
                ## The reach with the greatest outlet distance is always going to be an upstream segment(?)
                ## So for each of the remaining segments, loop through and find neighbors that aren't associated
                ## with junctions, get outlet distances, and compare neighbor outlet distances to junction outlet distances
                max_junc_dist_out = junc_dist_out.index(max(junc_dist_out))
                min_junc_dist_out = junc_dist_out.index(min(junc_dist_out))
                up_dn = []
                for i in range(len(junc_ids)):
                    # if i == max_junc_dist_out:
                    #     up_dn.append('up')
                    #     continue

                    # if i == min_junc_dist_out:
                    #     up_dn.append('dn')
                    #     continue

                    if junc_ids[i] in head_at_junc:
                        up_dn.append('up')
                        continue

                    if shp.loc[shp['reach_id'] == int(junc_ids[i])].main_side.to_list()[0] == 0:
                        def_con = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
                        def_con_id_neigh = def_con['geom2_rch_'].to_list()
                        def_con_id_neigh = [int(j) for j in def_con_id_neigh]
                        def_con_neigh_main_side = shp.loc[shp['reach_id'].isin(def_con_id_neigh)]
                        def_con_neigh_main = def_con_neigh_main_side.loc[def_con_neigh_main_side['main_side'] == 0].reach_id.to_list()
                        def_con_neigh_main = [str(j) for j in def_con_neigh_main]
                        def_con_neigh_main_con = geom_con_shp.loc[geom_con_shp['geom1_rch_'].isin(def_con_neigh_main)] 
                        def_con_neigh_main_con = def_con_neigh_main_con.loc[def_con_neigh_main_con['geom2_rch_'] == junc_ids[i]]
                        def_con_neigh_main_con = def_con_neigh_main_con.loc[def_con_neigh_main_con['geom1_rch_'].isin(junc_ids)]
                        # if len(def_con_neigh_main_con) > 1:
                        #     print('There is more than 1 attached main network reach. Check code.')
                        #     print(geom1_rch_id)
                        #     print(geom2_rch_id)
                        #     break

                        if len(def_con_neigh_main_con) >= 1:
                            val = min([0, def_con_neigh_main_con["geom1_n_pn"].to_list()[0]], key = lambda x: abs(x - def_con_neigh_main_con["ind_intr"].to_list()[0]))
                            if val == 0:
                                up_dn.append('dn')
                                continue
                            else:
                                up_dn.append('up')
                                continue

                        
                    id_neigh = geom_con_side_shp.loc[geom_con_side_shp['geom1_rch_'] == junc_ids[i]] 
                    id_neigh = id_neigh.loc[~id_neigh['geom2_rch_'].isin(junc_ids)]

                    dists_neigh = []
                    main_side_neigh = []
                    for j in id_neigh['geom2_rch_'].to_list():
                        dists_neigh.extend(shp.loc[shp['reach_id'] == int(j)].dist_out.to_list())
                        main_side_neigh.extend(shp.loc[shp['reach_id'] == int(j)].main_side.to_list())

                    dist_id = junc_dist_out[i]
                    # up_dn_chk = all(dist_id > x for x in dists_neigh)
                    bool_comp_dist = [dist_id > x for x in dists_neigh]
                    # If True, this reach is downstream in the junction
                    # Create list matching junc_ids that says if each reach is up or dn, max_junc_dist_out is automatically up
                    if len(dists_neigh) == 1:
                        #If there is only 1 other neighbor
                        # if any(x == True for x in bool_comp_dist):
                        if any(x == True for x in bool_comp_dist) & any(y != 0 for y in main_side_neigh):
                            up_dn.append('dn')
                        elif any(x == True for x in bool_comp_dist) & any(y == 0 for y in main_side_neigh):
                            up_dn.append('up')
                        else:
                            up_dn.append('up')
                    else:
                        #If the reach is attached to another junction
                        # if up_dn_chk == False:
                        if any(x == True for x in bool_comp_dist):
                            up_dn.append('dn')
                        # elif any(x == True for x in bool_comp_dist) & all(y == 0 for y in main_side_neigh):
                        #     up_dn.append('dn')
                        else:
                            up_dn.append('up') 
                
                up_cnt = up_dn.count('up')
                dn_cnt = up_dn.count('dn')

                ## If all of the reaches are labeled as upstream or downstream, that isn't possible
                ## so now use min and max outlet distance to label at least 1 up and 1 down
                if (all(x == "up" for x in up_dn)) | (all(x == "dn" for x in up_dn)):
                    up_dn = []
                    for i in range(len(junc_ids)):
                        if i == max_junc_dist_out:
                            up_dn.append('up')
                            continue

                        if i == min_junc_dist_out:
                            up_dn.append('dn')
                            continue

                        if junc_ids[i] in head_at_junc:
                            up_dn.append('up')
                            continue

                        if shp.loc[shp['reach_id'] == int(junc_ids[i])].main_side.to_list()[0] == 0:
                            def_con = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
                            def_con_id_neigh = def_con['geom2_rch_'].to_list()
                            def_con_id_neigh = [int(j) for j in def_con_id_neigh]
                            def_con_neigh_main_side = shp.loc[shp['reach_id'].isin(def_con_id_neigh)]
                            def_con_neigh_main = def_con_neigh_main_side.loc[def_con_neigh_main_side['main_side'] == 0].reach_id.to_list()
                            def_con_neigh_main = [str(j) for j in def_con_neigh_main]
                            def_con_neigh_main_con = geom_con_shp.loc[geom_con_shp['geom1_rch_'].isin(def_con_neigh_main)] 
                            def_con_neigh_main_con = def_con_neigh_main_con.loc[def_con_neigh_main_con['geom2_rch_'] == junc_ids[i]]
                            def_con_neigh_main_con = def_con_neigh_main_con.loc[def_con_neigh_main_con['geom1_rch_'].isin(junc_ids)]
                            # if len(def_con_neigh_main_con) > 1:
                            #     print('There is more than 1 attached main network reach. Check code.')
                            #     print(geom1_rch_id)
                            #     print(geom2_rch_id)
                            #     break

                            if len(def_con_neigh_main_con) >= 1:
                                val = min([0, def_con_neigh_main_con["geom1_n_pn"].to_list()[0]], key = lambda x: abs(x - def_con_neigh_main_con["ind_intr"].to_list()[0]))
                                if val == 0:
                                    up_dn.append('dn')
                                    continue
                                else:
                                    up_dn.append('up')
                                    continue


                        id_neigh = geom_con_side_shp.loc[geom_con_side_shp['geom1_rch_'] == junc_ids[i]] 
                        # id_neigh = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == junc_ids[i]] 
                        id_neigh = id_neigh.loc[~id_neigh['geom2_rch_'].isin(junc_ids)]

                        dists_neigh = []
                        main_side_neigh = []
                        for j in id_neigh['geom2_rch_'].to_list():
                            dists_neigh.extend(shp.loc[shp['reach_id'] == int(j)].dist_out.to_list())
                            main_side_neigh.extend(shp.loc[shp['reach_id'] == int(j)].main_side.to_list())

                        dist_id = junc_dist_out[i]
                        # If up_dn_chk is True, this reach is downstream in the junction
                        # up_dn_chk = all(dist_id > x for x in dists_neigh)
                        # up_dn_chk fails in places where the reach of interest is downstream and at another junction,
                        # 1 of the other reaches at the junction is downstream and the other reach is a headwater coming
                        # into the junction that has a greater outlet distance than the reach ID of interest. So instead
                        # use this code to check if any of the neighbors have a lower outlet distance. This code may break
                        # or not work correctly in some locations, but I'm not sure yet.
                        bool_comp_dist = [dist_id > x for x in dists_neigh]
                        # Create list matching junc_ids that says if each reach is up or dn, max_junc_dist_out is automatically up
                        # if up_dn_chk == True:
                        if len(dists_neigh) == 1:
                            #If there is only 1 other neighbor
                            # if any(x == True for x in bool_comp_dist):
                            if any(x == True for x in bool_comp_dist) & any(y != 0 for y in main_side_neigh):
                                up_dn.append('dn')
                            elif any(x == True for x in bool_comp_dist) & any(y == 0 for y in main_side_neigh):
                                up_dn.append('up')
                            else:
                                up_dn.append('up')
                        else:
                            #If the reach is attached to another junction
                            if all(x == True for x in bool_comp_dist):
                                up_dn.append('dn')
                            else:
                                up_dn.append('up')                   
                    
                    up_cnt = up_dn.count('up')
                    dn_cnt = up_dn.count('dn')
                
                            
                if (up_cnt >= 2) & (dn_cnt == 1):
                    ## If it's a multi upstream junction:
                    ## Now determine which reach ID is the downstream most reach ID of the junction, and store the reach IDs of every reach but 
                    ## the most downstream segment as the ones to reverse coordinates of the linestring
                    ## Only reverse if both reaches involved are actually a part of the junction
                    if (geom1_rch_id in con_geom2_junc_ids) & (geom2_rch_id in con_geom2_junc_ids):

                        # min_junc_dist_out = junc_dist_out.index(min(junc_dist_out))
                        # ## Only reversing if the downstream ID of the junction has starting matching starting coordinates with the other reach IDs
                        # if (junc_ids[min_junc_dist_out] == geom1_rch_id) | (junc_ids[min_junc_dist_out] == geom2_rch_id):
                        #     rch_ids_rev.append(junc_ids[min_junc_dist_out])

                        dist_out = [i for i in junc_dist_out if i != min(junc_dist_out)]
                        dist_out_ind = []
                        for d in dist_out:
                            dist_out_ind.extend([i for i in range(len(junc_dist_out)) if junc_dist_out[i] == d])

                        ## Identifying the upstream segments in the junction
                        junc_up_ids = [junc_ids[i] for i in dist_out_ind]

                        ## Adding 1 or both upstream segments to reverse depending on if the 2 segments involved are 
                        ## both the upstream segments
                        if geom1_rch_id in junc_up_ids:
                            rch_ids_rev.append(geom1_rch_id)
                            #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                            df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                            df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1

                        if geom2_rch_id in junc_up_ids:
                            rch_ids_rev.append(geom2_rch_id)
                            #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                            df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                            df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

                        # rch_ids_rev.extend([junc_ids[i] for i in dist_out_ind])
                            
                # elif (up_cnt == 1) & (dn_cnt == 2):
                elif (up_cnt == 1) & (dn_cnt >= 2):
                    ## If it's a multi downstream junction:
                    geom1_up_dn = up_dn[junc_ids.index(geom1_rch_id)]
                    geom2_up_dn = up_dn[junc_ids.index(geom2_rch_id)]

                    ## If both segments involved are both the downstream segment do nothing
                    ## If both segments involved are one up and one down, flip the upstream segment
                    if not all(x == 'dn' for x in [geom1_up_dn, geom2_up_dn]):
                        rch_ids_rev.append(junc_ids[up_dn.index('up')])
                        #Set change to 1 so I know a modification was made
                        df_change.loc[df_change['reach_id'] == str(junc_ids[min_junc_dist_out]), 'change'] = 1

                    #Set df_con_prob to zero since either there is no problem or it has been fixed in this location (with reversing) 
                    df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0


                elif (up_cnt == 2) & (dn_cnt == 2):
                    ## If it's a multi downstream junction:
                    geom1_up_dn = up_dn[junc_ids.index(geom1_rch_id)]
                    geom2_up_dn = up_dn[junc_ids.index(geom2_rch_id)]

                    ## Checking for locations where 2 reaches are connected at both ends (e.g., 81340500201 and 81340500011)
                    tmp1 = shp.loc[shp['reach_id'] == int(geom1_rch_id)]['geometry'].to_list()[0]
                    tmp2 = shp.loc[shp['reach_id'] == int(geom2_rch_id)]['geometry'].to_list()[0]
                    if tmp1.distance(tmp2) < eps: 
                        point = shapely.ops.nearest_points(tmp1, tmp2)
                        if len(point) > 1:
                            if all(x == 'up' for x in [geom1_up_dn, geom2_up_dn]):
                                df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                                continue           

                    ## If both segments involved are both the downstream segment do nothing
                    ## If both segments involved are both the upstream segment, flip both
                    ## If both segments involved are one up and one down, flip the upstream segment
                    if all(x == 'up' for x in [geom1_up_dn, geom2_up_dn]):
                        rch_ids_rev.append(geom1_rch_id)
                        rch_ids_rev.append(geom2_rch_id)
                        df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1
                        df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

                    elif (not all(x == 'dn' for x in [geom1_up_dn, geom2_up_dn])) & (not all(x == 'up' for x in [geom1_up_dn, geom2_up_dn])):
                        if geom1_up_dn == 'up':
                            rch_ids_rev.append(geom1_rch_id)
                            df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1

                        elif geom2_up_dn == 'up':
                            rch_ids_rev.append(geom2_rch_id)
                            df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

                    df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0

                else:
                    print('New junction configuration exists! Refine code. Reaches involved:')
                    print(geom1_rch_id)
                    print(geom2_rch_id)
                    break
                    # raise SystemExit(22) 


            ## If it's not a junction, then use outlet distance to figure out which one to reverse
            elif geom1_dist_out < geom2_dist_out:
                if (geom2_rch_id in store_rch_ids_rev) & (int(geom1_rch_id) in head_at_junc):
                    rch_ids_rev.append(geom1_rch_id) 
                elif (geom1_rch_id in store_rch_ids_rev) & (int(geom2_rch_id) in head_at_junc):
                    rch_ids_rev.append(geom2_rch_id)     
                else:        
                    rch_ids_rev.append(geom2_rch_id)
                    #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                    df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                    df_change.loc[df_change['reach_id'] == str(geom2_rch_id), 'change'] = 1

            elif geom1_dist_out > geom2_dist_out:
                rch_ids_rev.append(geom1_rch_id)
                #Set df_con_prob to zero since problem has been fixed in this location and set change to 1 so I know a modification was made
                df_con_prob.loc[(df_con_prob['geom1_rch_id'] == str(geom1_rch_id)) & (df_con_prob['geom2_rch_id'] == str(geom2_rch_id)), 'con_prob'] = 0
                df_change.loc[df_change['reach_id'] == str(geom1_rch_id), 'change'] = 1


        # df_con_prob.loc[df_con_prob['con_prob'] == 1].count() # Potentially problematic areas: 27
        # df_change.loc[df_change['change'] == 1].count() # Areas with a modification applied: 10

        print('The number of potentially problematic areas remaining is: ' + str(len(df_con_prob.loc[df_con_prob['con_prob'] == 1])))
        print('The number of areas modified is: ' + str(len(df_change.loc[df_change['change'] == 1])))


        rch_ids_rev = list(set(rch_ids_rev))
        # len(rch_ids_rev)
        rch_ids_rev = [r for r in rch_ids_rev if (r in reach_ids_side) |  (int(r) in head_at_junc)]

        print(rch_ids_rev)
        print('len(rch_ids_rev): ' + str(len(rch_ids_rev)))
        if len(rch_ids_rev) == 0:
            print('No more reaches to reverse!')
            print('The counter is: ' + str(counter))
            break

        rev_chk = [i for i in rch_ids_rev if i in store_rch_ids_rev]
        if len(rev_chk) > 0:
            print('A reach has been reversed multiple times -- check code!')
            print('The counter is: ' + str(counter))
            print(rev_chk)
            print('Writing reach IDs that been reversed multiple times to CSV -- may need to do manual topology in these locations')
            df = pd.DataFrame({"reach_id": rev_chk})
            rev_ids_multi_csv = rev_ids_csv.replace('LS.csv', 'multi.csv')
            df.to_csv(rev_ids_multi_csv, index=False)
            break
            # raise SystemExit(22) 


        ### Maybe I just need to loop through the reaches that I've already stored as need to be reversed
        for r in rch_ids_rev:
            # if (r in reach_ids_side) | (int(r) in head_at_junc):  
                print('Reversing ' + str(r))
                ix = shp.loc[shp['reach_id'] == int(r)].index.to_list()[0]
                # lines[ix] = rev_crds(geom1)
                shp.geometry[ix] = rev_crds(shp.geometry[ix])
                store_rch_ids_rev.append(r)

        print('Finished reversing initial reaches, finding new geometric intersections')

        ### Make new point layer
        # geom_con_fname = geom_con_fname.replace('_pts.shp', '_pts_' + str(counter) + '.shp')
        geom_con_fname = re.sub(r'pts_.*', 'pts_' + str(counter) + '.shp', geom_con_fname)

        print('geom_con_fname: ' + geom_con_fname)

        point_lyr = fiona.open(geom_con_fname, mode='w', driver='ESRI Shapefile', schema=pt_schema, crs =crs)

        # shp = gpd.GeoDataFrame(pd.concat([riv_shp_main_fix, shp], ignore_index=True))
        riv_shp_main_fix_sub = riv_shp_main_fix.loc[~riv_shp_main_fix['reach_id'].isin(reach_ids_side_ass_int)]
        shp = gpd.GeoDataFrame(pd.concat([riv_shp_main_fix_sub, shp], ignore_index=True))
        # shp.loc[shp['reach_id'] == 74230100091]

        # Now only modify geometric intersections for areas where LineStrings were reversed/changes were made (i.e., 'problematic')
        reach_id_chg = df_change.loc[df_change['change'] == 1]['reach_id'].to_list()
        reach_id_chg = [int(i) for i in reach_id_chg]
        # shp_sub = shp.loc[shp['reach_id'].isin(reach_id_chg)]

        ### NEW WAY
        # reach_ids_side_att_int = [int(r) for r in reach_ids_side_att]
        for ix, r in shp.iterrows():
            geom1 = shapely.geometry.shape(shp.geometry[ix])
            geom1_rch_id = shp.reach_id[ix]

            # if geom1_rch_id not in reach_id_chg:
            if geom1_rch_id not in reach_ids_side_ass_int:
                continue

            selected_reach = shp.loc[shp['reach_id'] == int(geom1_rch_id)]
            #Finding reaches within search distance (11 km)
            reaches_win_dist = shp.assign(distance=shp.apply(lambda x: x.geometry.distance(selected_reach.geometry.iloc[0]), axis=1)).query(f'distance <= {0.1}')
            reaches_win_dist = reaches_win_dist[reaches_win_dist.reach_id != geom1_rch_id]


            #Only comparing selected_reach to reaches within search distance (minimizes computational needs)
            for ix2, r2 in reaches_win_dist.iterrows(): 

                geom2 = shapely.geometry.shape(reaches_win_dist.geometry[ix2])
                geom2_rch_id = reaches_win_dist.reach_id[ix2]

                ## Don't need to record intersection if reach ID is the same
                if geom1_rch_id == geom2_rch_id:
                    continue

                if geom1.distance(geom2) < eps: 
                    point = shapely.ops.nearest_points(geom1, geom2)[0]
                    # print(point)

                    ## Sometimes the point won't exactly match up, so need to find the nearest point to connect the 2 segments
                    found = False
                    for i in range(len(geom1.coords.xy[0])):
                        x = geom1.coords.xy[0][i]
                        y = geom1.coords.xy[1][i]
                        tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})

                        if(tmp_pt == point):
                            found = True     

                            ind = i

                            point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                            'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                            'geom2_rch_id': geom2_rch_id.item(),
                                                            'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                            'ind_intr': ind}})

                            break

                    if found == False:
                        dist_point = []
                        for i in range(len(geom1.coords.xy[0])):
                            x = geom1.coords.xy[0][i]
                            y = geom1.coords.xy[1][i]
                            tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})
                            dist_point.append(point.distance(tmp_pt))
                        
                        ind = dist_point.index(min(dist_point))

                        point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                        'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                        'geom2_rch_id': geom2_rch_id.item(),
                                                        'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                        'ind_intr': ind}})

        point_lyr.close()
        counter = counter + 1

        geom_con = geom_con.loc[~geom_con['geom1_rch_'].isin(reach_ids_side_ass_int)]
        geom_con.to_file(geom_con_fname, mode="a")


## If the code stops because it can't figure out what direction a reach should be going
## and there are some reaches that need to be reversed that are not included in the
## list of reaches that the code can't figure: revserse those final reaches and
## find new geometric intersections for them

rev_not_multi = [i for i in rch_ids_rev if i not in store_rch_ids_rev]
if len(rev_not_multi) > 0:
    print('Reversing last reaches not involved in reaches that have been reversed multiple times')

    # for r in rch_ids_rev:
    for r in rev_not_multi:
    # if (r in reach_ids_side) | (int(r) in head_at_junc):  
        print('Reversing ' + str(r))
        ix = shp.loc[shp['reach_id'] == int(r)].index.to_list()[0]
        # lines[ix] = rev_crds(geom1)
        shp.geometry[ix] = rev_crds(shp.geometry[ix])
        store_rch_ids_rev.append(r)

    print('Finished reversing reaches, finding new geometric intersections')

    ### Make new point layer
    # geom_con_fname = geom_con_fname.replace('_pts.shp', '_pts_' + str(counter) + '.shp')
    geom_con_fname = re.sub(r'pts_.*', 'pts_final.shp', geom_con_fname)

    print('geom_con_fname: ' + geom_con_fname)

    point_lyr = fiona.open(geom_con_fname, mode='w', driver='ESRI Shapefile', schema=pt_schema, crs =crs)

    # shp = gpd.GeoDataFrame(pd.concat([riv_shp_main_fix, shp], ignore_index=True))
    riv_shp_main_fix_sub = riv_shp_main_fix.loc[~riv_shp_main_fix['reach_id'].isin(reach_ids_side_ass_int)]
    shp = gpd.GeoDataFrame(pd.concat([riv_shp_main_fix_sub, shp], ignore_index=True))
    # shp.loc[shp['reach_id'] == 74230100091]

    # Now only modify geometric intersections for areas where LineStrings were reversed/changes were made (i.e., 'problematic')
    reach_id_chg = df_change.loc[df_change['change'] == 1]['reach_id'].to_list()
    reach_id_chg = [int(i) for i in reach_id_chg]
    # shp_sub = shp.loc[shp['reach_id'].isin(reach_id_chg)]

    ### NEW WAY
    # reach_ids_side_att_int = [int(r) for r in reach_ids_side_att]
    for ix, r in shp.iterrows():
        geom1 = shapely.geometry.shape(shp.geometry[ix])
        geom1_rch_id = shp.reach_id[ix]

        # if geom1_rch_id not in reach_id_chg:
        if geom1_rch_id not in reach_ids_side_ass_int:
            continue

        selected_reach = shp.loc[shp['reach_id'] == int(geom1_rch_id)]
        #Finding reaches within search distance (11 km)
        reaches_win_dist = shp.assign(distance=shp.apply(lambda x: x.geometry.distance(selected_reach.geometry.iloc[0]), axis=1)).query(f'distance <= {0.1}')
        reaches_win_dist = reaches_win_dist[reaches_win_dist.reach_id != geom1_rch_id]


        #Only comparing selected_reach to reaches within search distance (minimizes computational needs)
        for ix2, r2 in reaches_win_dist.iterrows(): 

            geom2 = shapely.geometry.shape(reaches_win_dist.geometry[ix2])
            geom2_rch_id = reaches_win_dist.reach_id[ix2]

            ## Don't need to record intersection if reach ID is the same
            if geom1_rch_id == geom2_rch_id:
                continue

            if geom1.distance(geom2) < eps: 
                point = shapely.ops.nearest_points(geom1, geom2)[0]
                # print(point)

                ## Sometimes the point won't exactly match up, so need to find the nearest point to connect the 2 segments
                found = False
                for i in range(len(geom1.coords.xy[0])):
                    x = geom1.coords.xy[0][i]
                    y = geom1.coords.xy[1][i]
                    tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})

                    if(tmp_pt == point):
                        found = True     

                        ind = i

                        point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                        'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                        'geom2_rch_id': geom2_rch_id.item(),
                                                        'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                        'ind_intr': ind}})

                        break

                if found == False:
                    dist_point = []
                    for i in range(len(geom1.coords.xy[0])):
                        x = geom1.coords.xy[0][i]
                        y = geom1.coords.xy[1][i]
                        tmp_pt = shapely.geometry.shape({'type': 'Point', 'coordinates': [(x, y)]})
                        dist_point.append(point.distance(tmp_pt))
                    
                    ind = dist_point.index(min(dist_point))

                    point_lyr.write({'geometry': shapely.geometry.mapping(point), 
                                    'properties': {'geom1_rch_id': geom1_rch_id.item(), 
                                                    'geom2_rch_id': geom2_rch_id.item(),
                                                    'geom1_n_pnts': len(geom1.coords.xy[0]),
                                                    'ind_intr': ind}})

    point_lyr.close()

    geom_con = geom_con.loc[~geom_con['geom1_rch_'].isin(reach_ids_side_ass_int)]
    geom_con.to_file(geom_con_fname, mode="a")



print('The reversed reach IDs are: ' + str(store_rch_ids_rev))
print('The number of reversed reach IDs is: ' + str(len(store_rch_ids_rev)))
#Write output file for reversed LineStrings

print('Writing output river file containing reversed LineStrings')
#MS = Main/Side
riv_out_shp = riv_out_shp.replace("_Main_LSFix.shp", "_LSFix_MS.shp") 
riv_shp_main_fix_sub = riv_shp_main_fix.loc[~riv_shp_main_fix['reach_id'].isin(reach_ids_side_ass_int)]
shp = gpd.GeoDataFrame(pd.concat([riv_shp_main_fix_sub, shp], ignore_index=True))
shp.to_file(riv_out_shp)

print('Writing reach IDs of reversed LineStrings to CSV')
df = pd.DataFrame({"reach_id": store_rch_ids_rev})
rev_ids_csv = rev_ids_csv.replace('LS.csv', 'LS_MS.csv')
# df.to_csv(rev_ids_csv, header=False, index=False)
df.to_csv(rev_ids_csv, index=False)


end_Main_Side = timer()

print('The side network algorithm took ' + str(((end_Main_Side - start_Side) / 60)) + ' minutes or ' + str(((end_Main_Side - start_Side) / 60) / 60) + ' hours to complete.')
print('The whole algorithm took ' + str(((end_Main_Side - start) / 60)) + ' minutes or ' + str(((end_Main_Side - start) / 60) / 60) + ' hours to complete.')

print('- Done')