from __future__ import division
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import requests
from io import StringIO
import pandas as pd

##################################################################################
##################################################################################
##################################################################################

# Function to call multiple hydrocron lake CSVs
def fetch_hydrocron_list(
    feature="YourFeature",
    feature_ids="YourFieldIDs",
    start_time="YourStartTime",
    end_time="YourEndTime",
    fields="YourFields",
    output=None
):
    # Default if not set by user
    if output is None:
        output = ["csv"]
    # Loop through all URL combinations
    for ind in list(range(len(feature_ids))):
        # Construct the URL for each feature_id
        url = (
            f"https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
            f"?feature={feature}&feature_id={feature_ids[ind]}"
            f"&start_time={start_time}&end_time={end_time}"
            f"&fields={fields}&output={output}"
        )
        # Make the request
        response = requests.get(url).json()
        if "error" in response:
            continue
        else:
            response_csv = response['results']['csv']
            df = pd.read_csv(StringIO(response_csv))
            if ind == 0:
                df_all = df.copy()
            else:
                df_all = df_all.append(df)
    return df_all

##################################################################################
##################################################################################
##################################################################################

sword_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf/na_sword_v16.nc'
sword = nc.Dataset(sword_fn)
nc_nodes = np.array(sword['/nodes/node_id'][:])
nc_nrchs = np.array(sword['/nodes/reach_id'][:])
rch_list = np.array([77449500091, 77449500081, 77449300071, 77449300061, 77449300051, 77449300041, 77449300031, 77449300021, 77449300011, 77449300251])
#in this case feature ids are a list of nodes corresponding to specific reaches. 
feature_ids = nc_nodes[np.where(np.in1d(nc_nrchs, rch_list)==True)[0]]
# feature_ids = feature_ids[0:5] #subset for testing.
feature="Node"
start_time="2024-01-01T00:00:00Z"
end_time="2024-11-01T00:00:00Z"
fields="reach_id,node_id,time_str,wse,wse_u,node_q,p_dist_out" #"reach_id,node_id,time_str,wse,wse_u,node_q"

# Build hydrocron_csv_list
# Fetch data for multiple feature_ids
hydrocron_csv = fetch_hydrocron_list(
    feature=feature, feature_ids=feature_ids, start_time=start_time, end_time=end_time, fields=fields,
    output="csv"
)

subset = hydrocron_csv[hydrocron_csv['wse']>0]
subset2 = subset[subset['node_q']<2]
# subset2.to_csv('/Users/ealtenau/Documents/SWORD_Dev/swot_data/Sacramento_River/hydrocron_wse_nodes.csv',index=False)


unq_nodes = np.unique(subset2['node_id'])
median = np.zeros(len(unq_nodes))
dist_med = np.zeros(len(unq_nodes))
for n in list(range(len(unq_nodes))):
    print(n, len(unq_nodes)-1)
    pts = np.where(subset2['node_id'] == unq_nodes[n])[0]
    wse_all = subset2['wse'].iloc[pts]
    median[n] = np.median(wse_all)
    dist_med[n] = max(subset2['p_dist_out'].iloc[pts])


df_med = pd.DataFrame(np.array([unq_nodes, median, dist_med]).T)
# df_med.to_csv('/Users/ealtenau/Documents/SWORD_Dev/swot_data/Sacramento_River/hydrocron_wse_nodes_median.csv',index=False)

###############################################################################################
###############################################################################################
###############################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gp

############ read and plot data from exported data
subset2 = pd.read_csv('/Users/ealtenau/Documents/SWORD_Dev/swot_data/Sacramento_River/hydrocron_wse_nodes.csv')
med_data = pd.read_csv('/Users/ealtenau/Documents/SWORD_Dev/swot_data/Sacramento_River/hydrocron_wse_nodes_median.csv')
riffles_df = gp.read_file('/Users/ealtenau/Documents/SWORD_Dev/swot_data/Sacramento_River/sac_riffle_pooles.gpkg')
median = np.array(med_data['1'])
dist_med = np.array(med_data['2'])
nodes = np.array(med_data['0'])
rifflepool = np.array(riffles_df['riffle_pool'])
# good = np.where(median > 0)[0]

riffles = np.array(riffles_df['node_id'].iloc[np.where(rifflepool == 'riffle')[0]])
pooles = np.array(riffles_df['node_id'].iloc[np.where(rifflepool == 'pool')[0]])
r = np.where(np.in1d(nodes, riffles) == True)[0]
p = np.where(np.in1d(nodes, pooles) == True)[0]

pt1 = np.where(nodes == 77449500080181)[0]
pt2 = np.where(nodes == 77449500080301)[0]
plt.scatter(subset2['p_dist_out'], subset2['wse'], c='lightgrey', s=2)
plt.scatter(dist_med, median, c='black', s=8)
plt.scatter(dist_med[pt1], median[pt1], c='red', s=10)
plt.scatter(dist_med[pt2], median[pt2], c='blue', s=10)
plt.xlabel('Distance From Outlet (m)')
plt.ylabel('SWOT WSE (m)')
plt.title('Upper Sacramento River')
plt.show()
