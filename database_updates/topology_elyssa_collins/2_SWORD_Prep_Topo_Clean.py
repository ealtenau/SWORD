###################################################
# Script: SWORD_Prep_Topo_Clean.py
# Date created: 7/13/24
# Usage: Same as SWORD_Prep_Topo.py, but cleaner and with more comments

# Modified: 7/19/24 by E.Altenau
# Updates to make code less line-by-line dependant. 
# Broken into 4 Sections:
# Section 1: Identify and fix geometry problems at junctions. 
# Section 2: Identify and fix geometry problems at reach intersections between junctions.
# Section 3: Identify potential gaps in the network. 
# Section 4: Update geometric intersections. 

# Manual updates can be made to the outputs between sections. 
# The code can be re-started at the section that required
# the manual adjustments.

###################################################

import os
import fiona
import shapely
import shapely.ops
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import argparse

import warnings
warnings.filterwarnings("ignore") #if code stops working may need to comment out to check warnings. 

#*******************************************************************************
#Command Line Variables / Instructions:
#*******************************************************************************
# 1 - SWORD Continent (i.e. AS)
# 2 - Level 2 Pfafstetter Basin (i.e. 36)
# 3 = Section of code to start at (integer value 1-5)
    # 1 - Finding and fixing geometry problems - Junctions.
    # 2 - Finding and fixing geometry problems - Standard Reaches.
    # 3 - Identifying Gaps.
    # 4 - Updating Geometric Intersections.
    # 5 - Run all sections from beginning.

# Example Syntax: "python SWORD_Topo_Geom_auto.py AS 36 4"
#*******************************************************************************

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
    return(temp.index(min(temp)))

#**************************************************
# Defining basin
#**************************************************

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("basin", help="<Required> Level Two Pfafstetter Basin (i.e. 74)", type = str)
parser.add_argument("section", help="<Required> Section of Code to Start (1-5). \
                    Four code sections total. Section = 5 runs code from the start.", type = int)
args = parser.parse_args()

b = args.basin
region = args.region
section = args.section

shp_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17/shp/'+region+'/'
out_dir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Topology/'+region+'/b'+b+'/'
if os.path.exists(out_dir+'intermediate/') == False:
    os.makedirs(out_dir+'intermediate/')
os.chdir(out_dir)

#**************************************************
# Reading in data
#**************************************************
if section in [5]:

    #### Read in river shapefile
    line_shp = shp_dir+region.lower()+"_sword_reaches_hb" + b + "_v17.shp"
    # line_shp = out_dir+region.lower()+"_sword_reaches_hb" + b + "_v17_FG1.shp"
    shp = gpd.read_file(line_shp)
    eps = 5e-07 
    crs = shp.crs

    #-------------------------------------------------------------------------------
    #Section 1: Finding and fixing geometry problems
    #-------------------------------------------------------------------------------

    #**************************************************
    # Fixing geometry for headwaters/outlets or side channels that are connected to 2 other reaches
    #**************************************************

    #### Read in points shapefile containing initial geometric intersections (from init_geom.py)
    point_shp = out_dir+region.lower()+'_sword_reaches_hb'+b+'_v17_pts.shp'
    geom_con_shp = gpd.read_file(point_shp)
    reach_ids = geom_con_shp['geom1_rch_'].unique()
    print('Number of Reaches in Basin:',len(reach_ids))

    #### Finding reaches with problematic geometry
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('FINDING GEOMETRY PROBLEMS - JUNCTIONS')
    con_cnt = []
    end_reach = []
    main_side = []
    for r in reach_ids:
        con_cnt.append(len(geom_con_shp.loc[geom_con_shp['geom1_rch_'] == r]))
        end_reach.append(shp.loc[shp['reach_id'] == r]['end_reach'].to_list()[0])
        main_side.append(shp.loc[shp['reach_id'] == r]['main_side'].to_list()[0])

    reach_num_con = pd.DataFrame({'reach_id': reach_ids, 'con_cnt': con_cnt, 'end_reach': end_reach, 'main_side': main_side})
    reach_num_con_2 = reach_num_con.loc[reach_num_con['con_cnt'] == 2]
    reach_num_con_2_out_head = reach_num_con_2.loc[reach_num_con_2['end_reach'].isin([1,2,3])]
    reach_num_con_2_out_head = reach_num_con_2_out_head['reach_id'].astype('str').to_list()
    reach_num_con_2_out_head_int = [int(i) for i in reach_num_con_2_out_head]

    reach_num_con_2_side = reach_num_con_2.loc[reach_num_con_2['main_side'] == 1]
    reach_num_con_2_side = reach_num_con_2_side.loc[reach_num_con_2_side['end_reach'].isin([1,2,3])]
    reach_num_con_2_side = reach_num_con_2_side['reach_id'].astype('str').to_list()

    reach_num_con_2_out_head_side_int = [int(i) for i in reach_num_con_2_side]
    reach_num_con_2_out_head_side_int.extend(reach_num_con_2_out_head_int)
    # print('Problems Identified:',len(reach_num_con_2_out_head_side_int))

    if isinstance(geom_con_shp['geom1_rch_'].to_list()[0], str):
        reach_num_con_2_out_head_side_int = [str(i) for i in reach_num_con_2_out_head_side_int]

    ## Checking if the reaches identified are incorrect neighbors or headwaters at junctions
    ## Only need to fix geometry of incorrect neighbors, but need to have check in topology
    ## code for identifying headwaters at junctions
    prob_juncs = []
    head_at_junc = []
    for r in reach_num_con_2_out_head_side_int:
        ## CHECK IF IT SHOULD BE STRING OR INT(R)
        geom_con_sub = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == r]
        geom_con_sub_geom2_ids = geom_con_sub['geom2_rch_'].to_list()

        found = False
        for r2 in geom_con_sub_geom2_ids:
            geom_r1_line = shp.loc[shp['reach_id'] == int(r)].geometry.to_list()[0]
            geom_r2_line = shp.loc[shp['reach_id'] == int(r2)].geometry.to_list()[0]

            ## If the potential problem reach has more than one point intersection with 
            ## geometrically connected reaches, then it's probably a problem reach
            if geom_r1_line.intersection(geom_r2_line).geom_type == 'MultiPoint':
                prob_juncs.append(r)
                found = True

        ## If there were no MultiPoint intersections for the reach, then this 
        ## reach is likely a headwater at a junction
        if found == False:
            
            ## Now check if the Point intersections are each at the opposite end of the 
            ## reach of interest
            ##      - If so, then this reach doesn't need any modifications (i.e., it is
            ##        a reach labeled as a headwater that is actually just a normal reach
            ##        connected to 2 other reaches)
            ##      - If not, then this reach is likely a headwater at a junction
            geom_str_end = []
            for r2 in geom_con_sub_geom2_ids:
                geom_con_sub_geom2 = geom_con_sub.loc[geom_con_sub['geom2_rch_'] == r2].reset_index()
                if geom_con_sub_geom2["geom1_n_pn"][0] == 2:
                    if geom_con_sub_geom2["ind_intr"][0] == 1:
                        val = 1
                    else:
                        val = 0
                else:
                    val = min([0, geom_con_sub_geom2["geom1_n_pn"][0]], key = lambda x: abs(x - geom_con_sub_geom2["ind_intr"][0]))

                if val == 0:
                    geom_str_end.append('start')
                else:
                    geom_str_end.append('end')
        
            if len(list(set(geom_str_end))) == 1:
                head_at_junc.append(r)

    ## prob_juncs contains the list of reach ids that have problematic geometry
    ## head_at_junc contains the list of reach ids that are headwaters at junctions (i.e., doesn't have an attached ghost reach)
    prob_juncs = list(set(prob_juncs))
    head_at_junc = list(set(head_at_junc))
    print('Problems Identified:', len(prob_juncs))
    print('Headwaters at Junctions Identified:', len(head_at_junc)) 
    if len(prob_juncs) == 0:
        rch_ass = []

if section in [1,5]:
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('FIXING GEOMETRY PROBLEMS - JUNCTIONS') 
    ### If you get an error in the below code, you might need to manually fix geometry.
    ### In this case, remove the reach ID from 'intermediate/prob_juncs.csv' and re-run below code for fixing:

    if section < 5: 
        shp = gpd.read_file(out_dir+region.lower()+'_sword_reaches_hb' + b + '_v17_FG1.shp')
        geom_con_shp = gpd.read_file(out_dir+region.lower()+'_sword_reaches_hb'+b+'_v17_pts.shp')
        eps = 5e-07 
        crs = shp.crs
        prob_juncs = pd.read_csv(out_dir+'intermediate/prob_juncs.csv')
        prob_juncs = list(prob_juncs['reach_id'])
        head_at_junc = pd.read_csv(out_dir+'intermediate/headwater_juncs.csv')
        head_at_junc = list(head_at_junc['reach_id'])
        outlet_at_junc = pd.read_csv(out_dir+'intermediate/outlet_juncs.csv')
        outlet_at_junc = list(outlet_at_junc['reach_id'])
        rch_ass = pd.read_csv(out_dir+'intermediate/associated_rchs.csv')
        rch_ass = list(rch_ass['reach_id'])
    try:
        #### Fixing reaches with problematic geometry
        geom_con_shp['geom1_rch_'] = geom_con_shp['geom1_rch_'].astype('str')
        geom_con_shp['geom2_rch_'] = geom_con_shp['geom2_rch_'].astype('str')

        rch_ass = [] # List of associated reaches to reach with modified geometry
        for r in prob_juncs:
            shp_sub = shp.loc[shp['reach_id'] == r]
            geom_con_sub = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == str(r)]
            geom_con_sub_geom2_ids = geom_con_sub['geom2_rch_'].to_list()

            geom2_1_dist_out = shp.loc[shp['reach_id'] == int(geom_con_sub_geom2_ids[0])]['dist_out'].to_list()[0]
            geom2_2_dist_out = shp.loc[shp['reach_id'] == int(geom_con_sub_geom2_ids[1])]['dist_out'].to_list()[0]

            ## The reach of interest should be in between the 2 reaches it's connected to 
            if geom2_1_dist_out < geom2_2_dist_out:
                down_id = geom_con_sub_geom2_ids[0]
                up_id = geom_con_sub_geom2_ids[1]
            else:
                down_id = geom_con_sub_geom2_ids[1]
                up_id = geom_con_sub_geom2_ids[0]

            geom_con_sub_geom2_ids.append(str(r))

            ## Find where the middle reach connects to the upstream reach
            geom_con_up_id = geom_con_sub.loc[geom_con_sub['geom2_rch_'] == up_id]
            val = min([0, geom_con_up_id["geom1_n_pn"].to_list()[0]], key = lambda x: abs(x - geom_con_up_id["ind_intr"].to_list()[0]))

            if val == 0:
                ## If the index of intersection with the upstream segment is closest to 0, 
                ## then need to reconnect the upstream segment to the coordinate at the last
                ## index of the middle reach

                geom_up_id = shp.loc[shp['reach_id'] == int(up_id)]['geometry'].to_list()[0]
                geom_mid_id = shp.loc[shp['reach_id'] == int(r)]['geometry'].to_list()[0]

                ## Set the first coordinate of the LineString for the upstream ID to the
                ## last coordinate of the LineString for the middle ID
                coords = list(geom_up_id.coords)
                ind_min = find_index_of_point_w_min_distance(coords, shapely.Point((geom_mid_id.coords.xy[0][len(geom_mid_id.coords.xy[0]) - 1], geom_mid_id.coords.xy[1][len(geom_mid_id.coords.xy[0]) - 1])))        
                del coords[0:ind_min + 1]
                coords.insert(0, (geom_mid_id.coords.xy[0][len(geom_mid_id.coords.xy[0]) - 1], geom_mid_id.coords.xy[1][len(geom_mid_id.coords.xy[0]) - 1]))
                geom_fix = shapely.geometry.LineString(coords)
                shp.loc[shp['reach_id'] == int(up_id), 'geometry'] = shapely.geometry.LineString(geom_fix)

            else:
                ## If the index of intersection with the upstream segment is closest to the last index, 
                ## then need to reconnect the upstream segment at the last coordinate to the coordinate 
                ## at the last index of the middle reach

                geom_up_id = shp.loc[shp['reach_id'] == int(up_id)]['geometry'].to_list()[0]
                geom_mid_id = shp.loc[shp['reach_id'] == int(r)]['geometry'].to_list()[0]

                ## Set the last coordinate of the LineString for the upstream ID to the
                ## last coordinate of the LineString for the middle ID
                coords = list(geom_up_id.coords)
                ind_min = find_index_of_point_w_min_distance(coords, shapely.Point((geom_mid_id.coords.xy[0][len(geom_mid_id.coords.xy[0]) - 1], geom_mid_id.coords.xy[1][len(geom_mid_id.coords.xy[0]) - 1])))        
                del coords[ind_min:len(coords) - 1]
                coords.insert(0, (geom_mid_id.coords.xy[0][len(geom_mid_id.coords.xy[0]) - 1], geom_mid_id.coords.xy[1][len(geom_mid_id.coords.xy[0]) - 1]))
                geom_fix = shapely.geometry.LineString(coords)
                shp.loc[shp['reach_id'] == int(up_id), 'geometry'] = shapely.geometry.LineString(geom_fix)

            rch_ass.append(up_id)
            rch_ass.append(down_id)

        shp.to_file(out_dir+region.lower()+'_sword_reaches_hb' + b + '_v17_FG1.shp')
        prob_csv = pd.DataFrame({"reach_id": prob_juncs})
        ar_csv = pd.DataFrame({"reach_id": rch_ass})
        hw_csv = pd.DataFrame({"reach_id": head_at_junc})
        prob_csv.to_csv(out_dir+'intermediate/prob_juncs.csv', index = False)
        hw_csv.to_csv(out_dir+'intermediate/headwater_juncs.csv', index = False)
        ar_csv.to_csv(out_dir+'intermediate/associated_rchs.csv', index = False)
        if 'outlet_at_junc' in locals():
            ol_csv = pd.DataFrame({"reach_id": outlet_at_junc})
            ol_csv.to_csv(out_dir+'intermediate/outlet_juncs.csv', index = False)
        else:
            outlet_at_junc = []
            ol_csv = pd.DataFrame({"reach_id": outlet_at_junc})
            ol_csv.to_csv(out_dir+'intermediate/outlet_juncs.csv', index = False)
        
        print("All Problems Fixed - Section 1: Junction Reaches")

    except Exception as e:
        shp.to_file(out_dir+region.lower()+'_sword_reaches_hb' + b + '_v17_FG1.shp')
        prob_csv = pd.DataFrame({"reach_id": prob_juncs})
        ar_csv = pd.DataFrame({"reach_id": rch_ass})
        hw_csv = pd.DataFrame({"reach_id": head_at_junc})
        prob_csv.to_csv(out_dir+'intermediate/prob_juncs.csv', index = False)
        hw_csv.to_csv(out_dir+'intermediate/headwater_juncs.csv', index = False)
        ar_csv.to_csv(out_dir+'intermediate/associated_rchs.csv', index = False)
        if 'outlet_at_junc' in locals():
            ol_csv = pd.DataFrame({"reach_id": outlet_at_junc})
            ol_csv.to_csv(out_dir+'intermediate/outlet_juncs.csv', index = False)
        else:
            outlet_at_junc = []
            ol_csv = pd.DataFrame({"reach_id": outlet_at_junc})
            ol_csv.to_csv(out_dir+'intermediate/outlet_juncs.csv', index = False)
        print(f"An error occurred: {e}")
        print("!! Manual Corrections Required - Section 1: Junction Reach Problems !!")
        print(print('File Reference -> intermediate/prob_juncs.csv'))
        print("Exiting")
        sys.exit()

#**************************************************
# Fixing geometry for all other reaches that are connected to 2 other reaches
#**************************************************
if section in [1,5]:
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('FINDING GEOMETRY PROBLEMS - STANDARD REACHES')

    shp['last_digit'] = shp['reach_id'].astype('str').str.strip().str[-1]

    #### Read in points shapefile containing initial geometric intersections (from init_geom.py)
    # point_shp = out_dir+region.lower()+'_sword_reaches_hb'+b+'_v17_pts.shp'
    # geom_con_shp = gpd.read_file(point_shp)
    reach_ids = geom_con_shp['geom1_rch_'].unique()
    # len(reach_ids)

    #### Finding reaches with problematic geometry
    con_cnt = []
    last_digit = []
    for r in reach_ids:
        con_cnt.append(len(geom_con_shp.loc[geom_con_shp['geom1_rch_'] == r]))
        try:
            last_digit.append(shp.loc[shp['reach_id'] == r]['last_digit'].to_list()[0])
        except:
            last_digit.append(shp.loc[shp['reach_id'] == int(r)]['last_digit'].to_list()[0])

    reach_num_con = pd.DataFrame({'reach_id': reach_ids, 'con_cnt': con_cnt, 'last_digit': last_digit})
    reach_num_con_2 = reach_num_con.loc[reach_num_con['con_cnt'] == 2]
    reach_num_con_2_out_dig = reach_num_con_2.loc[reach_num_con_2['last_digit'].isin(['1', '2', '3', '4', '5'])]
    reach_num_con_2_out_dig = reach_num_con_2_out_dig['reach_id'].astype('str').to_list()
    reach_num_con_2_out_dig_int = [int(i) for i in reach_num_con_2_out_dig]
    # len(reach_num_con_2_out_dig_int)
    # reach_num_con_2_out_dig_int.index(74222800021)

    prob_rch = []
    for r1 in reach_num_con_2_out_dig_int:
        geom_con_sub = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == r1]

        con_cnt = []
        for r2 in geom_con_sub['geom2_rch_'].to_list():
            con_cnt.append(len(geom_con_shp.loc[geom_con_shp['geom1_rch_'] == r2]))
        
        # if all(x == 3 for x in con_cnt):
        if all(x >= 3 for x in con_cnt):
            prob_rch.append(r1)

    # len(prob_rch)
    # prob_rch.index(73112001231)

    prob_rch_fix = []
    for r in prob_rch:
        ## MIGHT NEED TO CHECK IF IT SHOULD BE STRING OR INT(R)
        geom_con_sub = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == r]
        geom_con_sub_geom2_ids = geom_con_sub['geom2_rch_'].to_list()

        for r2 in geom_con_sub_geom2_ids:
            geom_r1_line = shp.loc[shp['reach_id'] == int(r)].geometry.to_list()[0]
            geom_r2_line = shp.loc[shp['reach_id'] == int(r2)].geometry.to_list()[0]

            ## If the potential problem reach has more than one point intersection with 
            ## geometrically connected reaches, then it's probably a problem reach
            if geom_r1_line.intersection(geom_r2_line).geom_type == 'MultiPoint':
                prob_rch_fix.append(r)
                break

    ## prob_rch_fix contains the list of reach ids that have problematic geometry
    prob_rch_fix = list(set(prob_rch_fix))
    print('Problems Found:', len(prob_rch_fix))

if section in [1,2,5]:
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('FIXING GEOMETRY PROBLEMS - STANDARD REACHES') 
    ### If you get an error in the below code, you might need to manually fix geometry.
    ### In this case, remove the reach ID from 'intermediate/prob_rchs.csv' and re-run below code for fixing:
    
    if section < 5: #re-read files when re-starting later sections after manual edits. 
        shp = gpd.read_file(out_dir+region.lower()+'_sword_reaches_hb' + b + '_v17_FG1.shp')
        geom_con_shp = gpd.read_file(out_dir+region.lower()+'_sword_reaches_hb'+b+'_v17_pts.shp')
        eps = 5e-07 
        crs = shp.crs
        prob_juncs = pd.read_csv(out_dir+'intermediate/prob_juncs.csv')
        prob_juncs = list(prob_juncs['reach_id'])
        head_at_junc = pd.read_csv(out_dir+'intermediate/headwater_juncs.csv')
        head_at_junc = list(head_at_junc['reach_id'])
        outlet_at_junc = pd.read_csv(out_dir+'intermediate/outlet_juncs.csv')
        outlet_at_junc = list(outlet_at_junc['reach_id'])
        rch_ass = pd.read_csv(out_dir+'intermediate/associated_rchs.csv')
        rch_ass = list(rch_ass['reach_id'])
        if 'prob_rch_fix' in locals() == False:
            prob_rch_fix = pd.read_csv(out_dir+'intermediate/prob_rchs.csv')
            prob_rch_fix = list(prob_rch_fix['reach_id'])        

    try:
        #### Fixing reaches with problematic geometry
        geom_con_shp['geom1_rch_'] = geom_con_shp['geom1_rch_'].astype('str')
        geom_con_shp['geom2_rch_'] = geom_con_shp['geom2_rch_'].astype('str')

        for r in prob_rch_fix:
            shp_sub = shp.loc[shp['reach_id'] == r]
            geom_con_sub = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == str(r)]
            geom_con_sub_geom2_ids = geom_con_sub['geom2_rch_'].to_list()

            geom2_1_dist_out = shp.loc[shp['reach_id'] == int(geom_con_sub_geom2_ids[0])]['dist_out'].to_list()[0]
            geom2_2_dist_out = shp.loc[shp['reach_id'] == int(geom_con_sub_geom2_ids[1])]['dist_out'].to_list()[0]

            ## The reach of interest should be in between the 2 reaches it's connected to 
            if geom2_1_dist_out < geom2_2_dist_out:
                down_id = geom_con_sub_geom2_ids[0]
                up_id = geom_con_sub_geom2_ids[1]
            else:
                down_id = geom_con_sub_geom2_ids[1]
                up_id = geom_con_sub_geom2_ids[0]

            geom_con_sub_geom2_ids.append(str(r))

            ## Find where the middle reach connects to the upstream reach
            geom_con_up_id = geom_con_sub.loc[geom_con_sub['geom2_rch_'] == up_id]
            val = min([0, geom_con_up_id["geom1_n_pn"].to_list()[0]], key = lambda x: abs(x - geom_con_up_id["ind_intr"].to_list()[0]))

            geom_con_up_id_geom1 = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == up_id]
            geom_con_up_id_geom1 = geom_con_up_id_geom1.loc[geom_con_up_id_geom1['geom2_rch_'] == str(r)]
            val_up = min([0, geom_con_up_id_geom1["geom1_n_pn"].to_list()[0]], key = lambda x: abs(x - geom_con_up_id_geom1["ind_intr"].to_list()[0]))

            if val == 0:
                ## If the index of intersection with the upstream segment is closest to 0, 
                ## then need to reconnect the upstream segment to the coordinate at the last
                ## index of the middle reach

                geom_up_id = shp.loc[shp['reach_id'] == int(up_id)]['geometry'].to_list()[0]
                geom_mid_id = shp.loc[shp['reach_id'] == int(r)]['geometry'].to_list()[0]

                if val_up != 0:
                    ## Set the last coordinate of the LineString for the upstream ID to the
                    ## last coordinate of the LineString for the middle ID
                    coords = list(geom_up_id.coords)
                    ind_min = find_index_of_point_w_min_distance(coords, shapely.Point((geom_mid_id.coords.xy[0][len(geom_mid_id.coords.xy[0]) - 1], geom_mid_id.coords.xy[1][len(geom_mid_id.coords.xy[0]) - 1])))        
                    # del coords[ind_min:len(coords) - 1]
                    del coords[ind_min:len(coords)]
                    # del coords[0:ind_min + 1]
                    coords.insert(len(coords), (geom_mid_id.coords.xy[0][len(geom_mid_id.coords.xy[0]) - 1], geom_mid_id.coords.xy[1][len(geom_mid_id.coords.xy[0]) - 1]))
                    geom_fix = shapely.geometry.LineString(coords)
                    shp.loc[shp['reach_id'] == int(up_id), 'geometry'] = shapely.geometry.LineString(geom_fix)
                
                else:
                    ## Set the first coordinate of the LineString for the upstream ID to the
                    ## last coordinate of the LineString for the middle ID
                    coords = list(geom_up_id.coords)
                    ind_min = find_index_of_point_w_min_distance(coords, shapely.Point((geom_mid_id.coords.xy[0][len(geom_mid_id.coords.xy[0]) - 1], geom_mid_id.coords.xy[1][len(geom_mid_id.coords.xy[0]) - 1])))        
                    del coords[0:ind_min + 1]
                    coords.insert(0, (geom_mid_id.coords.xy[0][len(geom_mid_id.coords.xy[0]) - 1], geom_mid_id.coords.xy[1][len(geom_mid_id.coords.xy[0]) - 1]))
                    geom_fix = shapely.geometry.LineString(coords)
                    shp.loc[shp['reach_id'] == int(up_id), 'geometry'] = shapely.geometry.LineString(geom_fix)

            else:
                ## If the index of intersection with the upstream segment is closest to the last index, 
                ## then need to reconnect the upstream segment at the last coordinate to the coordinate 
                ## at the last index of the middle reach

                geom_up_id = shp.loc[shp['reach_id'] == int(up_id)]['geometry'].to_list()[0]
                geom_mid_id = shp.loc[shp['reach_id'] == int(r)]['geometry'].to_list()[0]

                ## Set the last coordinate of the LineString for the upstream ID to the
                ## last coordinate of the LineString for the middle ID
                coords = list(geom_up_id.coords)
                ind_min = find_index_of_point_w_min_distance(coords, shapely.Point((geom_mid_id.coords.xy[0][len(geom_mid_id.coords.xy[0]) - 1], geom_mid_id.coords.xy[1][len(geom_mid_id.coords.xy[0]) - 1])))        
                # del coords[ind_min:len(coords) - 1]
                del coords[ind_min:len(coords)]
                coords.insert(0, (geom_mid_id.coords.xy[0][len(geom_mid_id.coords.xy[0]) - 1], geom_mid_id.coords.xy[1][len(geom_mid_id.coords.xy[0]) - 1]))
                geom_fix = shapely.geometry.LineString(coords)
                shp.loc[shp['reach_id'] == int(up_id), 'geometry'] = shapely.geometry.LineString(geom_fix)

            rch_ass.append(up_id)
            rch_ass.append(down_id)

        #### Write fixed geometry problems to new shapefile so the original file isn't overwritten
        shp.to_file(out_dir+region.lower()+'_sword_reaches_hb' + b + '_v17_FG1.shp')
        prob_csv = pd.DataFrame({"reach_id": prob_rch_fix})
        ar_csv = pd.DataFrame({"reach_id": rch_ass})
        prob_csv.to_csv(out_dir+'intermediate/prob_rchs.csv', index = False)
        ar_csv.to_csv(out_dir+'intermediate/associated_rchs.csv', index = False)
        if 'outlet_at_junc' in locals():
            ol_csv = pd.DataFrame({"reach_id": outlet_at_junc})
            ol_csv.to_csv(out_dir+'intermediate/outlet_juncs.csv', index = False)
        else:
            outlet_at_junc = []
            ol_csv = pd.DataFrame({"reach_id": outlet_at_junc})
            ol_csv.to_csv(out_dir+'intermediate/outlet_juncs.csv', index = False)
        print('All Problems Fixed - Section 2: Standard Reaches')

    except Exception as e:
        shp.to_file(out_dir+region.lower()+'_sword_reaches_hb' + b + '_v17_FG1.shp')
        prob_csv = pd.DataFrame({"reach_id": prob_rch_fix})
        ar_csv = pd.DataFrame({"reach_id": rch_ass})
        prob_csv.to_csv(out_dir+'intermediate/prob_rchs.csv', index = False)
        ar_csv.to_csv(out_dir+'intermediate/associated_rchs.csv', index = False)
        if 'outlet_at_junc' in locals():
            ol_csv = pd.DataFrame({"reach_id": outlet_at_junc})
            ol_csv.to_csv(out_dir+'intermediate/outlet_juncs.csv', index = False)
        else:
            outlet_at_junc = []
            ol_csv = pd.DataFrame({"reach_id": outlet_at_junc})
            ol_csv.to_csv(out_dir+'intermediate/outlet_juncs.csv', index = False)
        print(f"An error occurred: {e}")
        print("!! Manual Corrections Required - Section 2: Standard Reach Problems !!")
        print('File Reference -> intermediate/prob_rchs.csv')
        print("Exiting")
        sys.exit()

if section in [1,2,3,5]:
    #**************************************************
    # Reaches that are not headwaters/outlets that are only attached on 1 end (i.e., might be a gap)
    #**************************************************
    ## This indicates locations where there might be a gap
    # os.chdir('/Users/elyssac/Documents/SWOT/SWORD_v17/EU/Topology/b' + b)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('FINDING GAP REACHES') 

    if section < 5: #re-read files when re-starting later sections after manual edits. 
        shp = gpd.read_file(out_dir+region.lower()+'_sword_reaches_hb' + b + '_v17_FG1.shp')
        geom_con_shp = gpd.read_file(out_dir+region.lower()+'_sword_reaches_hb'+b+'_v17_pts.shp')
        eps = 5e-07 
        crs = shp.crs
        prob_juncs = pd.read_csv(out_dir+'intermediate/prob_juncs.csv')
        prob_juncs = list(prob_juncs['reach_id'])
        head_at_junc = pd.read_csv(out_dir+'intermediate/headwater_juncs.csv')
        head_at_junc = list(head_at_junc['reach_id'])
        outlet_at_junc = pd.read_csv(out_dir+'intermediate/outlet_juncs.csv')
        outlet_at_junc = list(outlet_at_junc['reach_id'])
        rch_ass = pd.read_csv(out_dir+'intermediate/associated_rchs.csv')
        rch_ass = list(rch_ass['reach_id'])
        prob_rch_fix = pd.read_csv(out_dir+'intermediate/prob_rchs.csv')
        prob_rch_fix = list(prob_rch_fix['reach_id'])

    shp['last_digit'] = shp['reach_id'].astype('str').str.strip().str[-1]
    shp_sub = shp.loc[shp['last_digit'] != '6']

    #### Finding potential gap reaches
    gap_rch = []
    for r in shp_sub.reach_id.to_list():
        if isinstance(geom_con_shp['geom1_rch_'][0], str):
            r = str(r)
        else:
            r = int(r)

        geom_con_sub = geom_con_shp.loc[geom_con_shp['geom1_rch_'] == r]
        geom_con_sub_geom2 = geom_con_sub['geom2_rch_'].to_list()
        geom_con_sub_intr = geom_con_sub['ind_intr'].to_list()

        geom2_con_sub = geom_con_shp.loc[geom_con_shp['geom2_rch_'] == r]
        geom2_con_sub_geom1 = geom2_con_sub['geom1_rch_'].to_list()

        if len(geom_con_sub) == 0:
            print('Dataframe is empty!')
            break

        if len(geom_con_sub_geom2) >= len(geom2_con_sub_geom1):
            diff = list(set(geom_con_sub_geom2) - set(geom2_con_sub_geom1))
        elif len(geom2_con_sub_geom1) > len(geom_con_sub_geom2):
            diff = list(set(geom2_con_sub_geom1) - set(geom_con_sub_geom2))

        if all(i == geom_con_sub_intr[0] for i in geom_con_sub_intr):
            gap_rch.append(r)
        
        if len(diff) > 0:
            gap_rch.append(r)

    gap_rch = list(set(gap_rch))
    # len(gap_rch)
    gap_rch = [int(r) for r in gap_rch]

    #### Store all reach IDs with geometry problems in one variable
    prob_juncs.extend(prob_rch_fix)

    #### Remove the reaches with geometry problems from the list of potential reach gaps
    #### because it has hopefully already been fixed
    for r in prob_juncs:
        if r in gap_rch:
            gap_rch.remove(r)

    prob_csv = pd.DataFrame({"reach_id": prob_juncs})
    prob_csv.to_csv(out_dir+'intermediate/prob_rchs_all.csv', index = False)
    gap_csv = pd.DataFrame({"reach_id": gap_rch})
    gap_csv.to_csv(out_dir+'intermediate/gap_rchs.csv', index = False)
    print('Gap Reaches Identified:',len(gap_rch))

    if len(gap_rch) > 0:
        print("!! Manual Corrections Required - Section 3: Gap Reaches !!")
        print("Exiting")
        sys.exit()

    #### Now go manually check reaches in 'intermediate/gap_rchs.csv' in QGIS to see if they are just headwaters/outlets
    #### Remove from the list if it is a headwater or outlet. 
    ####        - Also add the reach to the 'intermediate/headwater_juncs.csv' or 'intermediate/outlet_juncs.csv' files if it's attached to a junction:
    ####        - Fix the geometry in the '_FG1.shp' file using QGIS if there is some sort of geometry
    ####          problem (gap or otherwise). If this is done, be sure to add reaches to 'intermediate/prob_rchs_all.csv' and
    ####          'intermediate/associated_rchs.csv' (reaches associated with the problem junction) lists. 
    ####          *** Remember that this is done so that finding the new geometric intersections runs much faster ***.

if section in [1,2,3,4,5]:
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('RE-RUNNING GEOMETRIC INTERSECTIONS') 
    #-------------------------------------------------------------------------------
    #Section 2: Finding new geomtric intersections after fixing geometry problems, writing files
    #-------------------------------------------------------------------------------

    #**************************************************
    # Now fix geometrics point intersections at reaches that were geometrically modified
    #**************************************************
    #### Read in river shapefile with fixed geometry and points shapefile containing initial 
    #### geometric intersections (from init_geom.py)
    #### Read in juctions and associated reach, etc. files. 
    #### When the topology algorithm breaks and new geometry errors are found and fixed, 
    #### update the 'intermediate/prob_rchs_all.csv','intermediate/prob_rchs_all.csv',
    #### and 'intermediate/headwater_juncs.csv' files. Then start this code with 
    #### section = 4 (i.e.: python SWORD_Prep_Topo_Clean.py AS 36 4).

    if section < 5: #re-read files when re-starting later sections after manual edits. 
        shp = gpd.read_file(out_dir+region.lower()+'_sword_reaches_hb' + b + '_v17_FG1.shp')
        eps = 5e-07 
        crs = shp.crs
        prob_juncs = pd.read_csv(out_dir+'intermediate/prob_rchs_all.csv')
        prob_juncs = list(prob_juncs['reach_id'])
        head_at_junc = pd.read_csv(out_dir+'intermediate/headwater_juncs.csv')
        head_at_junc = list(head_at_junc['reach_id'])
        outlet_at_junc = pd.read_csv(out_dir+'intermediate/outlet_juncs.csv')
        outlet_at_junc = list(outlet_at_junc['reach_id'])
        rch_ass = pd.read_csv(out_dir+'intermediate/associated_rchs.csv')
        rch_ass = list(rch_ass['reach_id'])

    #### Now finding new geomtric intersections where edits were made
    rch_ass_int = [int(r) for r in rch_ass]
    rch_ass_int.extend(prob_juncs)

    point_shp = out_dir+region.lower()+'_sword_reaches_hb'+b+'_v17_pts.shp'
    geom_con = fiona.open(point_shp)

    ## Name of output points shapefile containing new geometric intersections
    # geom_con_fname = "eu_sword_reaches_hb" + b + "_v17_FG1_pts.shp"
    geom_con_fname = out_dir+region.lower()+'_sword_reaches_hb'+b+'_v17_FG1_pts.shp'

    ### Making a new point layer
    pt_schema={'geometry': 'Point', 'properties': {'geom1_rch_id': 'int', 'geom2_rch_id': 'int', 'geom1_n_pnts': 'int', 'ind_intr': 'int'}}
    point_lyr = fiona.open(geom_con_fname, mode='w', driver='ESRI Shapefile', schema=pt_schema, crs =crs)

    for ix, r in shp.iterrows():
        geom1 = shapely.geometry.shape(shp.geometry[ix])
        geom1_rch_id = shp.reach_id[ix]

        # if geom1_rch_id not in prob_juncs:
        if geom1_rch_id not in rch_ass_int:
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


    geom_con_filt = filter(lambda f: f['properties']['reach_id'] not in rch_ass_int, geom_con)
    geom_con_filt = gpd.GeoDataFrame.from_features([feature for feature in geom_con], crs=crs)
    columns = list(geom_con.meta["schema"]["properties"]) + ["geometry"]
    geom_con_filt = geom_con_filt[columns]
    geom_con_filt = geom_con_filt.loc[~geom_con_filt['geom1_rch_'].isin(rch_ass_int)]
    geom_con_filt.to_file(geom_con_fname, mode="a")


    #**************************************************
    # Writing modified reaches to CSV 
    #**************************************************
    d = {'reach_id': np.array(prob_juncs).astype('int64')}
    df = pd.DataFrame(d)
    df.to_csv(out_dir+region.lower()+'_sword_reaches_hb' + b + '_v17_fixed_geom.csv', index=False)


    #**************************************************
    # Writing headwater at junction reaches to CSV 
    #**************************************************
    if 'head_at_junc' in locals():
        d = {'reach_id': np.array(head_at_junc).astype('int64')}
        df = pd.DataFrame(d)
        df.to_csv(out_dir+region.lower()+'_sword_reaches_hb' + b + '_v17_head_at_junc.csv', header=False, index=False)
    else:
        head_at_junc = []
        d = {'reach_id': np.array(head_at_junc).astype('int64')}
        df = pd.DataFrame(d)
        df.to_csv(out_dir+region.lower()+'_sword_reaches_hb' + b + '_v17_head_at_junc.csv', header=False, index=False)

    #**************************************************
    # Writing outlet at junction reaches to CSV 
    #**************************************************
    ## Make sure outlet_at_junc is set to 'outlet_at_junc = []' if there aren't any found
    if 'outlet_at_junc' in locals():
        d = {'reach_id': np.array(outlet_at_junc).astype('int64')}
        df = pd.DataFrame(d)
        df.to_csv(out_dir+region.lower()+'_sword_reaches_hb' + b + '_v17_out_at_junc.csv', header=False, index=False)
    else:
        outlet_at_junc = []
        d = {'reach_id': np.array(outlet_at_junc).astype('int64')}
        df = pd.DataFrame(d)
        df.to_csv(out_dir+region.lower()+'_sword_reaches_hb' + b + '_v17_out_at_junc.csv', header=False, index=False)

    #**************************************************
    # Subsetting river shapefile and point connections to main network, input to topo algorithm
    #**************************************************

    shp_Main = shp.loc[shp['main_side'] == 0]
    reach_ids_Main = shp_Main['reach_id'].to_list()

    geom_con_shp = gpd.read_file(geom_con_fname)
    geom_con_shp_Main = geom_con_shp.loc[(geom_con_shp['geom1_rch_'].isin(reach_ids_Main)) & 
                                        (geom_con_shp['geom2_rch_'].isin(reach_ids_Main))]

    shp_Main.to_file(out_dir+region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_Main.shp')
    geom_con_shp_Main.to_file(out_dir+region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_Main_pts.shp')

    print('DONE - New Intersections Completed')
    print(region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_Main_pts.shp')

