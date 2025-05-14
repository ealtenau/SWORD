from __future__ import division
import numpy as np
import geopandas as gp
import pandas as pd
import time

start_all = time.time()
# regions = 'na', 'sa', 'as', 'eu', 'af', 'oc'
regions = ['na']
tile_fn = '/Users/ealtenau/Documents/SWORD_Dev/swot_data/Tiles_ScienceOrbit/nom_tile_bounds_v1.shp'
# tile_fn = '/Users/ealtenau/Documents/SWORD_Dev/swot_data/Tiles_CalVal/tile_bounds_v4.shp'

tiles = gp.GeoDataFrame.from_file(tile_fn)   
tile_list = []
reg_id = []
for r in regions:
    start = time.time()
    print(r.upper())
    sword_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v18/gpkg/'+r+'_sword_reaches_v18.gpkg'
    reaches = gp.GeoDataFrame.from_file(sword_fn)
    intersect = gp.sjoin(reaches, tiles, how="inner") 
    tiles_all = np.unique(intersect.pass_tile)
    tile_list.append(np.unique(tiles_all))
    reg_id.append(np.repeat(r, len(tiles_all)))
    
    rch_tiles = []
    for ind in list(range(reaches.shape[0])):
        if reaches.reach_id[ind] in np.array(intersect.reach_id):
            in_rch = len(np.where(intersect.reach_id == reaches.reach_id[ind])[0])
            if in_rch > 1:
                rch_tiles.append(str(list(intersect.pass_tile[ind]))[1:-1])
            else:
                rch_tiles.append(intersect.pass_tile[ind])
        else:
            rch_tiles.append('NaN')
    rchs = np.array(reaches.reach_id)
    rch_tiles = np.array(rch_tiles, dtype=object)
    tile_csv = pd.DataFrame([rchs, rch_tiles]).T
    tile_csv.rename(columns={0:"reach_id",1:"pass_tile",},inplace=True)
    tile_csv.to_csv('/Users/ealtenau/Documents/SWORD_Dev/swot_data/SWORD_v18_PassTile/nominal/csv/'+r+'_sword_reaches_v18_pass_tile_nominal.csv', index=False)
    # tile_csv.to_csv('/Users/ealtenau/Documents/SWORD_Dev/swot_data/SWORD_v18_PassTile/calval/csv/'+r+'_sword_reaches_v18_pass_tile_calval.csv', index=False)

    reaches['pass_tile'] = rch_tiles
    outgpkg = '/Users/ealtenau/Documents/SWORD_Dev/swot_data/SWORD_v18_PassTile/nominal/gpkg/'+r+'_sword_reaches_v18_pass_tile_nominal.gpkg'
    # outgpkg = '/Users/ealtenau/Documents/SWORD_Dev/swot_data/SWORD_v18_PassTile/calval/gpkg/'+r+'_sword_reaches_v18_pass_tile_calval.gpkg'
    reaches.to_file(outgpkg, driver='GPKG', layer='reaches')

    end = time.time()
    print('Finished '+r.upper()+' in: '+str(np.round((end-start)/60,2))+' min')


tile_list = np.array([item for items in tile_list for item in items])
reg_id = np.array([item for items in reg_id for item in items])

csv_df = pd.DataFrame([tile_list, reg_id]).T
csv_df.rename(columns={0:"pass_tile",1:"sword_region",},inplace=True)
csv_df.to_csv('/Users/ealtenau/Documents/SWORD_Dev/swot_data/SWORD_v18_PassTile/nominal/csv/Pass_Tile_Nominal_SWORD_v18_Region.csv', index=False)
# csv_df.to_csv('/Users/ealtenau/Documents/SWORD_Dev/swot_data/SWORD_v18_PassTile/calval/csv/Pass_Tile_CalVal_SWORD_v18_Region.csv', index=False)

end_all = time.time() 
print('Finished all in: '+str(np.round((end_all-start_all)/60,2))+' min')