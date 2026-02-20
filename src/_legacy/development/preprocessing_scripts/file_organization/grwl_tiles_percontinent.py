# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:07:00 2019
"""

import os
main_dir = os.getcwd()
import shutil
import numpy as np
import pandas as pd

###############################################################################

def getListOfFiles(dirName):
    '''
    For the given path, get the List of all files in the directory tree 
    '''
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

grwl_paths = getListOfFiles(main_dir+'/data/inputs/GRWL/shps/GRWL_vector_V01.01_LatLonNames/')
cont_path = main_dir+'/data/inputs/GRWL/EDITS/GRWL_Tiles_Continents.csv'

data = pd.read_csv(cont_path, sep=',', delimiter=None, header='infer')
continents = data.assignment
names = data.tile
names = np.asarray(names)
region = np.unique(data.assignment)
region = np.delete(region, np.where(region == 'NorthAmerica'))
region = np.delete(region, np.where(region == 'No GRWL Data'))

for ind in list(range(len(region))):
        
    patterns = names[np.where(continents == region[ind])]
    
    copy_files = []
    for idx in list(range(len(patterns))):                
        copy_files.append([file for file in grwl_paths if patterns[idx] in file])
    
    cp_files = np.array(copy_files).flatten()    
    for f in cp_files:    
        out_dir = main_dir+'/data/inputs/GRWL/' + region[ind] + '/' 
        shutil.copy(f, out_dir)
        
    
   