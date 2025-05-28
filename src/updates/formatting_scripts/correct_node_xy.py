# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import time
import argparse
import src.updates.sword_utils as swd 

###############################################################################
###############################################################################
###############################################################################

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
args = parser.parse_args()

region = args.region
version = args.version

# region = 'NA'
# version = 'v16'

#read data
paths = swd.prepare_paths(main_dir, region, version)
sword_fn = paths['nc_dir']+paths['nc_fn']
centerlines, nodes, reaches = swd.read_nc(sword_fn)

#recalculating node x-y values. 
node_x = np.zeros(len(nodes))
node_y = np.zeros(len(nodes))
for n in list(range(len(nodes.id))):
    print(n, len(nodes.id)-1)
    pts = np.where(centerlines.node_id[0,:] == nodes.id[n])[0]
    nodes.x[n] = np.median(centerlines.x[pts])
    nodes.y[n] = np.median(centerlines.y[pts])

#filler variables
swd.discharge_attr_nc(reaches)
#write data
swd.write_nc(centerlines, reaches, nodes, region, sword_fn)

end = time.time()
print('DONE IN:', str(np.round((end-start)/60, 2)), 'mins')