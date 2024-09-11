import numpy as np
import netCDF4 as nc 
import geopandas as gp
import pandas as pd
import argparse

#read in netcdf data. 
parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("basin", help="<Required> Level Two Pfafstetter Basin (i.e. 74)", type = str)
args = parser.parse_args()

region = args.region
version = args.version
basin = args.basin

# region = 'AS'
# version = 'v17'
# basin = '31'

maindir = '/Users/ealtenau/Documents/SWORD_Dev/outputs/'
fn1 = maindir+'Topology/'+region+'/b'+basin+'/'+region.lower()+'_sword_reaches_hb'+basin+'_'+version+'_FG1_pts.shp'
fn2 = maindir+'Reaches_Nodes/'+version+'/reach_geometry/'+region.lower()+'_sword_'+version+'_connectivity.nc'
fn3 = maindir+'Topology/'+region+'/b'+basin+'/intermediate/prob_rchs_all.csv'
outfile = maindir+'Topology/'+region+'/b'+basin+'/intermediate/check_junctions.csv'

shp = gp.read_file(fn1)
pts = nc.Dataset(fn2)
current = pd.read_csv(fn3)

reaches = np.array(pts.groups['centerlines'].variables['reach_id'][:,:])
con = np.array(pts.groups['centerlines'].variables['end_reach'][:])

level2 = np.array([int(str(r)[0:2]) for r in reaches[0,:]])
bsn_ind = np.where(level2 == int(basin))[0]
subset = reaches[:,bsn_ind]
con_sub = con[bsn_ind]

unq_rchs = np.unique(subset[0,np.where(con_sub == 3)[0]])
check_juncs = []
### loop
for ind in list(range(len(unq_rchs))):
    print(ind, len(unq_rchs)-1)
    r = np.where(subset[0,:] == unq_rchs[ind])[0]
    nghs = np.unique(subset[1::,r])
    nghs = nghs[nghs!=0]
    shp_ind = np.where(np.array(shp['geom1_rch_']) == unq_rchs[ind])[0]
    if len(nghs) != len(shp_ind):
        check_juncs.append(unq_rchs[ind])

rmv = np.where(np.in1d(np.array(check_juncs), np.array(current['reach_id'])) == True)[0]
check_juncs = np.array(check_juncs)
check_juncs = np.delete(check_juncs, rmv)

check_csv = pd.DataFrame({"reach_id": check_juncs})
check_csv.to_csv(outfile, index = False)
print('DONE')
print('Junctions to check:', len(check_juncs))