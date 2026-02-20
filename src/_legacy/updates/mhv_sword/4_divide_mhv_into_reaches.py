"""
Creating SWORD-like reaches for the MHV-SWORD 
dataset (4_divide_mhv_into_reaches.py).
===================================================

This script divides the MHV-SWORD database into 
SWORD-like reaches and nodes. 

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA).

Execution example (terminal):
    python path/to/4_divide_mhv_into_reaches.py NA

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import time
import numpy as np
import argparse
import glob
import src.updates.auxillary_utils as aux 
import src.updates.mhv_sword.mhv_reach_def_tools as rdt
import warnings
warnings.filterwarnings("ignore") #if code stops working may need to comment out to check warnings. 

###############################################################################
##################### Defining Reach and Node Locations #######################
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Region", type = str)
args = parser.parse_args()

start_all = time.time()
region = args.region

#input file(s).
main_dir = main_dir+'/data/inputs/'
nc_file_list = np.sort(glob.glob(os.path.join(main_dir+'MHV_SWORD/netcdf/'+region+'/', '*.nc')))

#loop through mhv-sword netcdfs and create sword-like reaches. 
for f in list(range(len(nc_file_list))):
    print('STARTING BASIN: ' + nc_file_list[f][-13:-11])
    start = time.time()

    # Reading in data.
    subcls = rdt.read_merge_netcdf(nc_file_list[f])

    # Making sure flow accumulation minimum isn't zero.
    subcls.facc[np.where(subcls.facc == 0)[0]] = 0.001
    # Cutting basins to 6 digits.
    subcls.basins = np.array([int(str(ind)[0:6]) for ind in subcls.basins])
    # Correcting for antimeridian.
    subcls.lon[np.where(subcls.lon <= -180)] = -179.9999
    subcls.lon[np.where(subcls.lon >= 180)] = 179.9999
    subcls.x = np.copy(subcls.lon)
    subcls.y = np.copy(subcls.lat)

    # Creating empty objects to fill with attributes.
    centerlines = rdt.Object()
    centerlines.reach_id = np.zeros(len(subcls.seg))
    centerlines.node_id = np.zeros(len(subcls.seg))
    centerlines.rch_len = np.zeros(len(subcls.seg))
    centerlines.node_num = np.zeros(len(subcls.seg))
    centerlines.node_len = np.zeros(len(subcls.seg))
    centerlines.rch_eps = np.zeros(len(subcls.seg))
    centerlines.type = np.zeros(len(subcls.seg))
    centerlines.rch_ind = np.zeros(len(subcls.seg))
    centerlines.rch_num = np.zeros(len(subcls.seg))
    centerlines.rch_dist = np.zeros(len(subcls.seg))
    centerlines.rch_flag = np.zeros(len(subcls.seg))
        
    #########################################################################
    #REACH DEFINITION

    print('Defining Intial Reaches')
    # Define inital reaches.
    subcls.rch_id0, subcls.type0, subcls.rch_len0 = rdt.find_boundaries(subcls) #heavily modified for index ordering and odd basin boundaries... 

    print('Cutting Long Reaches')
    subcls.rch_id1, subcls.rch_len1 = rdt.cut_reaches(subcls.rch_id0,
                                                      subcls.rch_len0, subcls.dist,
                                                      subcls.ind, 20000)

    # Make sure there is only one "type" value per reach.
    subcls.type1 = np.zeros(len(subcls.rch_id1))
    uniq_rch = np.unique(subcls.rch_id1)
    for idx in list(range(len(uniq_rch))):
        rch = np.where(subcls.rch_id1 == uniq_rch[idx])[0]
        if len(rch) <=2:
            subcls.type1[rch] = np.min(subcls.type0[rch])
        else:
            subcls.type1[rch] = max(set(list(subcls.type0[rch])),
                                    key=list(subcls.type0[rch]).count)

    print('Aggregating River Reaches')
    # Aggregating river reach types.
    river_min_dist = 8000
    subcls.rch_id2, subcls.rch_len2,\
     subcls.type2 = rdt.aggregate_rivers(subcls, river_min_dist)
    # Updating reach indexes.
    subcls.rch_ind2, subcls.rch_eps2 = rdt.update_rch_indexes(subcls, subcls.rch_id2)

    print('Aggregating Lake Reaches')
    # Aggregating lake reach types.
    lake_min_dist = 8000
    subcls.rch_id3, subcls.rch_len3,\
     subcls.type3 = rdt.aggregate_lakes(subcls, lake_min_dist)
    # Updating reach indexes.
    subcls.rch_ind3, __ = rdt.update_rch_indexes(subcls, subcls.rch_id3)

    print('Aggregating Dam Reaches')
    # Aggregating dam reach types.
    dam_min_dist = 200
    subcls.rch_id4, subcls.rch_len4,\
     subcls.type4 = rdt.aggregate_dams(subcls, dam_min_dist)
    # Updating reache indexes.
    subcls.rch_ind4, __ = rdt.update_rch_indexes(subcls, subcls.rch_id4)    
    
    # Updating reach flow distance.
    subcls.rch_dist4 = aux.calc_geodesic_dist(subcls.lon, subcls.lat, 
                                              subcls.rch_id4, subcls.rch_ind4)

    print('Second Cutting of Long Reaches')
    subcls.rch_id5, subcls.rch_len5 = rdt.cut_reaches(subcls.rch_id4,
                                                      subcls.rch_len4, subcls.rch_dist4,
                                                      subcls.rch_ind4, 20000)
    subcls.type5 = np.copy(subcls.type4)
    # Updating reach indexes.
    subcls.rch_ind5, \
     subcls.rch_eps5 = rdt.update_rch_indexes(subcls, subcls.rch_id5)
    # Updating reach flow distance.
    subcls.rch_dist5 = rdt.calc_segDist(subcls.lon, subcls.lat, subcls.rch_id5,
                                        subcls.facc, subcls.rch_ind5)

    print('Checking Reaches')
    #Check and correct reaches with odd index problems. 
    issues = rdt.check_rchs(subcls.rch_id5, subcls.rch_dist5, subcls.rch_ind5)
    if len(issues) > 0:
        print('~correcting reach length issues')
        rdt.correct_rchs(subcls, issues)

    #########################################################################
    #TOPOLOGY

    print('Creating Reach ID')
    subcls.reach_id, subcls.rch_topo = rdt.reach_topology(subcls)

    # Creating Nodes.
    print('Creating Nodes')
    node_length = 200
    subcls.node_id, subcls.node_len,\
     subcls.node_num = rdt.node_reaches(subcls, node_length)

    print('Creating Ghost Reaches')
    # Defining ghost reaches and nodes.
    subcls.reach_id, subcls.node_id, subcls.rch_len6 = rdt.ghost_reaches(subcls)
    # Updating reach indexes and type.
    subcls.rch_ind6,\
     subcls.rch_eps6 = rdt.update_rch_indexes(subcls, subcls.reach_id)
    # Updating reach flow distance.
    subcls.rch_dist6 = rdt.calc_segDist(subcls.lon, subcls.lat, subcls.reach_id,
                                        subcls.facc, subcls.rch_ind6)
    # Updating type information.
    subcls.type6 = np.zeros(len(subcls.reach_id))
    for j in list(range(len(subcls.reach_id))):
        subcls.type6[j] = int(str(subcls.reach_id[j])[-1])

    # Checking Reaches again for final percentage of issues.
    issues_final = rdt.check_rchs(subcls.reach_id, subcls.rch_dist6, subcls.rch_ind6)
    subcls.rch_flag_final = np.zeros(len(subcls.seg))
    subcls.rch_flag_final[np.where(np.in1d(subcls.reach_id, issues_final)==True)[0]] = 1

    print('Aggregating Data')
    # Append current data to previous data.
    centerlines.reach_id[:] = subcls.reach_id
    centerlines.node_id[:] = subcls.node_id
    centerlines.rch_len[:] = subcls.rch_len6
    centerlines.node_num[:] = subcls.node_num
    centerlines.node_len[:] = subcls.node_len
    centerlines.rch_eps[:] = subcls.rch_eps6
    centerlines.type[:] = subcls.type6
    centerlines.rch_ind[:] = subcls.rch_ind6
    centerlines.rch_num[:] = subcls.rch_topo
    centerlines.rch_dist[:] = subcls.rch_dist6
    centerlines.rch_flag[:] = subcls.rch_flag_final

    # Update NetCDF File.
    print('Updating NetCDF')
    rdt.update_netcdf(nc_file_list[f], centerlines)

    end=time.time()
    print('Time to Create Reaches and Nodes: ' +
          str(np.round((end-start)/60, 2)) + ' min. Percent issues: ' 
          + str(np.round(len(issues_final)/len(np.unique(subcls.reach_id))*100,2)))

end_all=time.time()
print('Time to Finish All Reaches and Nodes: ' +
      str(np.round((end_all-start_all)/60, 2)) + ' min')