import os 
main_dir = os.getcwd()
import numpy as np
import pandas as pd
import netCDF4 as nc

fn = main_dir+'/data/swot_data/Amazon/007_151R/503/'\
    'SWOT_L2_HR_RiverTile_503_007_151R/SWOT_L2_HR_RiverTile_503_007_151R_amazon_20230503/'\
    'SWOT_L2_HR_PIXCVecRiver_503_007_151R_20230426T230153_20230426T230204_PIA1_01_01.nc'

pixc = nc.Dataset(fn)
keep = np.where(pixc.variables['height_vectorproc'][:] != -9999)[0]
df = pd.DataFrame(np.array([pixc.variables['longitude_vectorproc'][keep], 
                            pixc.variables['latitude_vectorproc'][keep],
                            pixc.variables['height_vectorproc'][keep],
                            pixc.variables['reach_id'][keep],
                            pixc.variables['node_id'][keep],
                            pixc.variables['segmentation_label'][keep],
                            pixc.variables['distance_to_node'][keep],
                            pixc.variables['pixc_index'][keep]]).T)

df.rename(
    columns={
        0:'lon',
        1:'lat',
        2:'height',
        3:'reach_id',
        4:'node_id',
        5:'seg_label',
        6:'dist_to_node',
        7:'pixc_index'
        },inplace=True)

df.to_csv(fn[:-3]+'.csv',index=False)