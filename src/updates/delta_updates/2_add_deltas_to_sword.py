"""
Adding Deltas to SWORD (2_add_deltas_to_sword)
===================================================

This script reads in the delta centerlines and 
auxillary attributes and formats them to be added 
to the SWOT River Database (SWORD).

Output is a new SWORD netCDF file with all delta 
location and hydrologic attributes added.

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA), SWORD version (i.e. v17), 
and the directory path containing the delta netCDF file.

Execution example (terminal):
    python path/to/2_add_deltas_to_sword.py NA v17 path/to/delta_file.nc

"""
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import time
import numpy as np
import shutil
import argparse
import src.updates.delta_updates.delta_utils as dlt
from src.updates.sword import SWORD

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("filename", help="directory to delta file", type = str)
args = parser.parse_args()

region = args.region
version = args.version
delta_dir = args.filename
mv_dir = os.path.dirname(delta_dir)+'/added_sword_v18/'
if os.path.isdir(mv_dir) is False:
    os.makedirs(mv_dir)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Starting File:', os.path.basename(delta_dir))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

#read delta data and SWORD. 
delta_cls = dlt.read_delta(delta_dir)
sword = SWORD(main_dir, region, version)
#copy data file in case errors occur in overwrite. 
sword.copy() 
#add max sword cl_id to delta cl_ids. 
max_id = np.max(sword.centerlines.cl_id)
delta_cls.cl_id = delta_cls.cl_id+max_id

print('--> Cutting long delta reaches')
#cut delta reaches greater than 16km.
thresh = 16000
dlt.cut_delta_segments(delta_cls, thresh) #np.max(delta_cls.new_len)

print('--> Finding and breaking SWORD reaches at delta headwaters')
#find closest upstream sword reach
#determine if it needs to be split/broken. 
break_rchs, break_ids = dlt.find_sword_breaks(delta_cls, sword)
#break sword reaches. 
if len(break_rchs) > 0:
    print('----> breaking', len(break_rchs), 'reaches')
    sword.break_reaches(break_rchs, break_ids) #verbose=True

print('--> Creating SWORD formatted IDs')
#order segment numbers.
dlt.number_segments(delta_cls) #np.unique(delta_cls.rch_num)
#order node numbers. 
dlt.number_nodes(delta_cls)
#create SWORD formatted reach and node IDs 
#**Note: need to create SWORD IDs after breaks to make sure all 
#reach IDs are included**. 
dlt.create_sword_ids(delta_cls, sword) #np.unique(delta_cls.sword_rch_id_down)

print('--> Finding and deleting SWORD reaches')
#find sword reaches to delete.
pt_radius = 25 #number of points to search around delta junction. 
del_rchs = dlt.find_ds_sword_rchs(delta_cls, sword, pt_radius)
#delete sword reaches.
if len(del_rchs) > 0:
    print('----> deleting', len(del_rchs), 'reaches')
    sword.delete_data(del_rchs)
#find remaining sword reaches to remove or flag as tributaries
#into the delta. 
rmv_rchs, delta_tribs = dlt.find_delta_tribs(delta_cls, sword)
#save plot of what was deleted for future checks. 
dlt.plot_sword_deletions(sword, 
                         delta_cls, 
                         rmv_rchs, 
                         delta_tribs, 
                         delta_dir)
#delete sword reaches.
if len(rmv_rchs) > 0:
    print('----> deleting', len(rmv_rchs), 'additional reaches')
    sword.delete_data(rmv_rchs)
#save plot of what was added for future checks. 
dlt.plot_sword_additions(sword, 
                         delta_cls, 
                         delta_dir)

print('--> Creating node and reach dimensions and attributes')
#defining nodes and attributes.
subnodes = dlt.create_nodes(delta_cls)
#defining reaches and attributes.
subreaches = dlt.create_reaches(delta_cls)
#formatting topology attributes.
dlt.format_sword_topo_attributes(delta_cls, subreaches)
#format path_segs, end_rch, and network for new delta reaches.
dlt.format_fill_attributes(delta_cls, subnodes, subreaches, sword)

print('--> Adding delta reaches to SWORD')
#append delta data to sword. 
sword.append_data(delta_cls, subnodes, subreaches)
if len(delta_tribs) > 0:
    #find which reaches to break for delta tributaries 
    trib_rchs, trib_ids = dlt.find_tributary_junctions(sword, delta_tribs)
    #break reaches at tributary junctions. 
    sword.break_reaches(trib_rchs, trib_ids) #verbose=True
    #format topology at new tributary junctions. 
    basin = int(str(subreaches.id[0])[0:2])
    dlt.tributary_topo(sword, delta_tribs, basin)

# print('--> Writing NetCDF')
sword.save_nc() 
# Move delta file to added directory. 
shutil.move(delta_dir, mv_dir)

#checking dimensions
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Cl Dimensions:', len(np.unique(sword.centerlines.cl_id)), len(sword.centerlines.cl_id))
print('Node Dimensions:', len(np.unique(sword.centerlines.node_id[0,:])), len(np.unique(sword.nodes.id)), len(sword.nodes.id))
print('Rch Dimensions:', len(np.unique(sword.centerlines.reach_id[0,:])), len(np.unique(sword.nodes.reach_id)), len(np.unique(sword.reaches.id)), len(sword.reaches.id))
print('min node char len:', len(str(np.min(sword.nodes.id))))
print('max node char len:', len(str(np.max(sword.nodes.id))))
print('min reach char len:', len(str(np.min(sword.reaches.id))))
print('max reach char len:', len(str(np.max(sword.reaches.id))))
print('Edit flag values:', np.unique(sword.reaches.edit_flag))

end = time.time()
print('Finished Adding Delta in:', str(np.round((end-start)/60, 2)) + ' min')
