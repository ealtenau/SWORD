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
from src.updates.sword import SWORD

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
sword = SWORD(main_dir, region, version)

#recalculating node x-y values. 
for n in list(range(len(sword.nodes.id))):
    print(n, len(sword.nodes.id)-1)
    pts = np.where(sword.centerlines.node_id[0,:] == sword.nodes.id[n])[0]
    sword.nodes.x[n] = np.median(sword.centerlines.x[pts])
    sword.nodes.y[n] = np.median(sword.centerlines.y[pts])

#write data
sword.save_nc()

end = time.time()
print('DONE IN:', str(np.round((end-start)/60, 2)), 'mins')