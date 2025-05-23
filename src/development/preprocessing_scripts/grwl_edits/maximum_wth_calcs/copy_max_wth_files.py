# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 08:45:13 2021
"""
import os
main_dir = os.getcwd()
import shutil
import numpy as np

###############################################################################

def getListOfFiles(dirName):

    """
    FUNCTION:
        For the given path, gets a recursive list of all files in the directory tree.

    INPUTS
        dirName -- Input directory

    OUTPUTS
        allFiles -- list of files under directory
    """

    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

###############################################################################

region = 'NA'
region_dir = main_dir+'/data/outputs/Merged_Data_v8/'+region+'/'
region_paths = np.array([f for f in getListOfFiles(region_dir) if '.shp' in f])
region_names = np.array([name[-17:-10] for name in region_paths])
dest_folder = main_dir+'/data/inputs/GRWL/bank_widths/'+region+'/'

global_dir = main_dir+'/data/inputs/GRWL/bank_widths/'
global_paths = np.array([f for f in os.listdir(global_dir) if '.csv' in f])
global_names = np.array([name[0:7] for name in global_paths])

for ind in list(range(len(region_names))):
    index = np.where(global_names == region_names[ind])[0]
    if len(index) == 0:
        continue
    else:
        shutil.copy(global_dir+global_paths[int(index)], dest_folder)