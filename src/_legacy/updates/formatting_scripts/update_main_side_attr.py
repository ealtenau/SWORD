"""
Updating Main-Side Attribute after centerline additions.
(update_main_side_attr.py)
===============================================================

This scripts updates the main-side attribute based on new
centerline additions in SWORD.  

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python path/to/update_main_side_attr.py NA v17

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import argparse
import geopandas as gp
from src.updates.sword_duckdb import SWORD

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

#read data
db_path = os.path.join(main_dir, f'data/duckdb/sword_{version}.duckdb')
sword = SWORD(db_path, region, version)
sword.copy() #copies original file for version control.

gpkg_fn = sword.paths['topo_dir']+region.lower()+'_sword_reaches_'+version+'_acc.gpkg'
gpkg = gp.read_file(gpkg_fn)
gpkg_net = np.array(gpkg['network'])
gpkg_rchs = np.array(gpkg['reach_id'])
gpkg_acc = np.array(gpkg['acc'])
gpkg_ms = np.array(gpkg['main_side'])

#find networks that don't have a main channel going to coast. 
unq_nets = np.unique(sword.reaches.network)
check_net = []
for net in list(range(len(unq_nets))):
    ntwk_rchs = np.where(sword.reaches.network == unq_nets[net])[0]
    outlets = np.where(sword.reaches.end_rch[ntwk_rchs] == 2)[0]
    if len(outlets) == 0:
        continue
    ms_out = np.min(sword.reaches.main_side[ntwk_rchs[outlets]])
    if ms_out > 0:
        check_net.append(unq_nets[net])

#find the downstream most reach on the main channel network
#based of accumulated reaches. 
start_rchs = []
for c in list(range(len(check_net))):
    ntwk_rchs = np.where(gpkg_net == check_net[c])[0]
    main_chan = np.where(gpkg_ms[ntwk_rchs] == 0)[0]
    main_rchs = gpkg_rchs[ntwk_rchs[main_chan]]
    for r in list(range(len(main_rchs))):
        rch = np.where(sword.reaches.id == main_rchs[r])[0]
        dn_rchs = np.unique(sword.reaches.rch_id_down[:,rch])
        dn_rchs = dn_rchs[dn_rchs>0]
        min_ds = np.min(sword.reaches.main_side[
            np.where(np.in1d(sword.reaches.id, dn_rchs)==True)[0]])
        if min_ds > 0:
            start_rchs.append(main_rchs[r])

#trace downstream from the identified reaches using topology 
#and number of reaches accumulated to update main_side values to 0. 
main_update = []
for ind in list(range(len(start_rchs))):
    next_rch = np.array([start_rchs[ind]])
    loop = 1
    while len(next_rch) > 0:
        # print(ind, loop)
        rch = np.where(gpkg_rchs == next_rch)[0]
        dn_rchs = np.unique(sword.reaches.rch_id_down[:,rch])
        dn_rchs = dn_rchs[dn_rchs>0]
        if len(dn_rchs) > 0:
            dn_ms = np.array([gpkg_ms[np.where(gpkg_rchs == d)[0]] for d in dn_rchs])
            dn_acc = np.array([gpkg_acc[np.where(gpkg_rchs == d)[0]] for d in dn_rchs])
            #find downstream reach with greatest accumulation
            #mark for updating and make next reach. 
            if np.min(dn_ms) > 0:
                max_rch = np.where(dn_acc == np.max(dn_acc))[0][0] 
                main_update.append(dn_rchs[max_rch])
                next_rch = np.array([dn_rchs[max_rch]])
                loop = loop+1
            else:
                next_rch = np.array([])
                loop = loop+1
        else:
            next_rch = np.array([])
            loop = loop+1

#update attribute values. 
rind = np.where(np.in1d(sword.reaches.id, main_update)==True)[0]
nind = np.where(np.in1d(sword.nodes.reach_id, main_update)==True)[0]
sword.reaches.main_side[rind] = 0
sword.nodes.main_side[nind] = 0

#write data. 
sword.save_nc()

########### PLOTS
# import matplotlib.pyplot as plt
# n = np.where(np.in1d(sword.reaches.network, check_net)==True)[0]
# s = np.where(np.in1d(sword.reaches.id, start_rchs)==True)[0]
# u = np.where(np.in1d(sword.reaches.id, main_update)==True)[0]
# plt.scatter(sword.reaches.x, sword.reaches.y, c='blue', s = 3)
# plt.scatter(sword.reaches.x[n], sword.reaches.y[n], c='red', s = 3)
# plt.scatter(sword.reaches.x[s], sword.reaches.y[s], c='gold', s = 8)
# plt.scatter(sword.reaches.x[u], sword.reaches.y[u], c='black', s = 8)
# plt.show()