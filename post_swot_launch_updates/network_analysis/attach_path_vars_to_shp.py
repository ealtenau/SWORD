import geopandas as gp
import pandas as pd
import numpy as np

csv_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17a/network_testing/Elyssa_basin_hb77/Colorado_hb77_paths.csv'
gpkg_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17a/network_testing/Elyssa_basin_hb77/hb77_Colorado_SWORD_v17a1_GF.gpkg'

gpkg = gp.read_file(gpkg_fn)
csv = pd.read_csv(csv_fn)

rch_id = np.array(csv['reach_id'])
path = np.array(csv['path_order_by_length'])
path_freq = np.array(csv['cumulative_path'])
dist_out = np.array(csv['dist_out'])
main_side = np.array(csv['main_side'])

gpkg_path = np.zeros(gpkg.shape[0])
gpkg_path_freq = np.zeros(gpkg.shape[0])
gpkg_dist_out = np.zeros(gpkg.shape[0])
gpkg_main_side = np.zeros(gpkg.shape[0])
unq_rch = np.unique([rch_id])
for ind in list(range(len(unq_rch))):
    rch = np.where(gpkg['reach_id'] == unq_rch[ind])[0]
    if len(rch) == 0:
        continue
    pts = np.where(rch_id == unq_rch[ind])[0]

    gpkg_path[rch] = np.max(path[pts])
    gpkg_path_freq[rch] = np.max(path_freq[pts])
    gpkg_dist_out[rch] = np.max(dist_out[pts])
    gpkg_main_side[rch] = np.max(main_side[pts])

gpkg['path_order'] = gpkg_path
gpkg['path_freq'] = gpkg_path_freq
gpkg['dist_out2'] = gpkg_dist_out
gpkg['main_side'] = gpkg_main_side

gpkg.to_file(gpkg_fn, driver='GPKG', layer='reaches')