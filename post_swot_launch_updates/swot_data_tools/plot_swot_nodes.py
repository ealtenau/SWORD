import numpy as np
import netCDF4 as nc
import geopandas as gp
import matplotlib.pyplot as plt

# fn = '/Users/ealteanau/Documents/SWORD_Dev/swot_data/Amazon/007_151R/503/'\
#     'SWOT_L2_HR_RiverTile_503_007_151R/SWOT_L2_HR_RiverTile_503_007_151R_amazon_20230503/'\
#     'SWOT_L2_HR_RiverTile_Node_503_007_151R_20230426T230153_20230426T230204_PIA1_01_01.shp'

fn = '/Users/ealteanau/Documents/SWORD_Dev/swot_data/Amazon/022_159R/503/'\
    'SWOT_L2_HR_RiverTile_503_022_159R_amazon_20230503/'\
    'SWOT_L2_HR_RiverTile_Node_503_022_159R_20230427T114933_20230427T114944_PIA1_01_01.shp'

tile = fn[-59:-47]

data = gp.read_file(fn)
rchs = np.array(data['reach_id'])
nodes = np.array(data['node_id'])
wse = np.array(data['wse'])
p_wse = np.array(data['p_wse'])
node_num = np.array([int(str(ind)[-4:-1]) for ind in nodes])

unq_rchs = np.unique(rchs)
for ind in list(range(len(unq_rchs))):
    title = tile + '_' + str(unq_rchs[ind])
    pts = np.where(rchs == unq_rchs[ind])[0]
    plt.scatter(node_num[pts], wse[pts], s=3, c='blue')
    plt.scatter(node_num[pts], p_wse[pts], s=3, c='red')
    plt.title(title)
    plt.xlabel('node number')
    plt.ylabel('wse')
    # plt.show()
    plt.savefig('/Users/ealteanau/Documents/SWORD_Dev/src/other_src/swot_data_testing/plots/'+title+'.png')
    plt.close()