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
the directory path containing the delta netCDF file, 
and a True/False statment indicating if you want to write 
the new netCDF files. If False, only the plots will be output
which allows for users to inspect for any potential issues 
before altering the SWORD database. 

Execution example (terminal):
    python path/to/2_add_deltas_to_sword.py NA v17 path/to/delta_file.nc True

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
parser.add_argument("write_data", help="True / False statement for whether or not to write the new netCDF file.", type = str)
args = parser.parse_args()

region = args.region
version = args.version
delta_dir = args.filename
write_data = args.write_data

# delta_dir = 'data/inputs/Deltas/delta_updates/netcdf/Amazon_delta_sword.nc'

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
if write_data == 'True':
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
# manual delete list for specific deltas
manual_delete = []
manual_tributaries = []
if 'Parana' in delta_dir or 'parana' in delta_dir.lower():
    manual_delete = [64212000165, 64212000155, 64212000145, 64212000135]
elif 'Orinoco' in delta_dir or 'orinoco' in delta_dir.lower():
    manual_tributaries = []  # 61309000375 was deleted in earlier step
elif 'Niger' in delta_dir or 'niger' in delta_dir.lower():
    manual_tributaries = [14301000995, 14301001006, 14301001065, 14301001076, 14301000675]
elif 'Amazon' in delta_dir or 'amazon' in delta_dir.lower():
    manual_tributaries = [62100100125, 62100100135, 62100100145, 62100100155, 62100100165, 
                         62100100175, 62100100185, 62100100195, 62100100205, 62100100275]
    manual_delete = [62210000495, 62210000506, 62210000255, 62210000275, 62210000285, 
                    62210000455, 62210000465, 62210000475, 62210000486, 62210000446, 
                    62210000435, 62302000375, 62302000385, 62210000225, 62210000235, 
                    62210000245, 62210000265, 62210000686, 62305000656, 61690000016,
                    62305000796, 62305000826, 62305000846, 62305000446, 62304000486,
                    62304000596,62210000616,62100900416]

rmv_rchs, delta_tribs = dlt.find_delta_tribs(delta_cls, sword, delete_ids=manual_delete, tributary_ids=manual_tributaries)
# DEBUG: print what's in each list
print(f"DEBUG: rmv_rchs contains {len(rmv_rchs)} reaches")
print(f"DEBUG: delta_tribs contains {len(delta_tribs)} reaches")
if len(manual_tributaries) > 0:
    print(f"DEBUG: manual_tributaries: {manual_tributaries}")
    for trib_id in manual_tributaries:
        if trib_id in rmv_rchs:
            print(f"DEBUG: WARNING - {trib_id} is in rmv_rchs (will be deleted)")
        if trib_id in delta_tribs:
            print(f"DEBUG: {trib_id} is in delta_tribs (will be tributary)")
#save plot of what was deleted for future checks. 
dlt.plot_sword_deletions(sword, 
                         delta_cls, 
                         rmv_rchs, 
                         delta_tribs, 
                         delta_dir)
#delete sword reaches.
if len(rmv_rchs) > 0:
    # remove manual tributaries from delete list
    if len(manual_tributaries) > 0:
        rmv_rchs = np.array([r for r in rmv_rchs if r not in manual_tributaries])
        print(f'----> removed {len(manual_tributaries)} manual tributaries from delete list')
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

if write_data == 'True':
    print('--> Writing NetCDF')
    sword.save_nc() 
    #Move delta file to added directory. 
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
