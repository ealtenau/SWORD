from __future__ import division
import os
main_dir = os.getcwd()
import numpy as np
import geopandas as gp
import pandas as pd
import time

start_all = time.time()
tile_fn = main_dir+'/data/swot_data/Tiles_ScienceOrbit/nom_tile_bounds_v1.shp'
# tile_fn = main_dir+'/data/swot_data/Tiles_CalVal/tile_bounds_v4.shp'

tiles = gp.GeoDataFrame.from_file(tile_fn)   
tile_list = []
reg_id = []
    
start = time.time()
basin_fn = main_dir+'/data/inputs/HydroBASINS/hb_level2_global.gpkg'
basins = gp.GeoDataFrame.from_file(basin_fn)
intersect = gp.sjoin(basins, tiles, how="inner") 

basin_tiles = []
bsn_id = []
unq_basins = np.unique(intersect.PFAF_ID[:])
for ind in list(range(len(unq_basins))):
    basin_tiles.append(list(intersect.loc[ind].pass_tile))
    bsn_id.append(list(np.repeat(np.unique(intersect.loc[ind].PFAF_ID), len(intersect.loc[ind].pass_tile))))

basin_tiles = [item for sublist in basin_tiles for item in sublist]
bsn_id = [item for sublist in bsn_id for item in sublist]
        
tile_csv = pd.DataFrame([bsn_id, basin_tiles]).T
tile_csv.rename(columns={0:"PFAF_ID",1:"pass_tile",},inplace=True)
tile_csv.to_csv(main_dir+'/data/swot_data/SWORD_v16_PassTile/nominal/csv/level2_basins_pass_tile_nominal.csv', index=False)



    
