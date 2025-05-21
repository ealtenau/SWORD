import numpy as np
import netCDF4 as nc
from scipy import spatial as sp
import matplotlib.pyplot as plt
import argparse 
import time

parser = argparse.ArgumentParser()
parser.add_argument("region", help="continental region", type = str)
args = parser.parse_args()

region = args.region
fn_merge11 = '/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/outputs/Merged_Data/v11/'+region.lower()+'_Merge_v11.nc'
fn_sword16 = '/afs/cas.unc.edu/depts/geological_sciences/pavelsky/students/ealtenau/SWORD_dev/outputs/Reaches_Nodes/v16/netcdf/'+region.lower()+'_sword_v16.nc'

# fn_merge11 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Merged_Data/v11/'+region.lower()+'_Merge_v11.nc'
# fn_sword16 = '/Users/ealteanau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf/'+region.lower()+'_sword_v16.nc'

start = time.time()

m11 = nc.Dataset(fn_merge11)
s16 = nc.Dataset(fn_sword16, 'r+')

mx = m11.groups['centerlines'].variables['x'][:]
my = m11.groups['centerlines'].variables['y'][:]
mfacc = m11.groups['centerlines'].variables['flowacc'][:]

clx = s16.groups['centerlines'].variables['x'][:]
cly = s16.groups['centerlines'].variables['y'][:]
cl_nodes = s16.groups['centerlines'].variables['node_id'][0,:]
cl_rchs = s16.groups['centerlines'].variables['reach_id'][0,:]
reaches = s16.groups['reaches'].variables['reach_id'][:]
nodes = s16.groups['nodes'].variables['node_id'][:]
rch_facc = s16.groups['reaches'].variables['facc'][:]
node_facc = s16.groups['nodes'].variables['facc'][:]

m11_pts = np.vstack((mx, my)).T
s16_pts = np.vstack((clx, cly)).T
kdt = sp.cKDTree(m11_pts)
eps_dist, eps_ind = kdt.query(s16_pts, k = 1)

rfacc = np.zeros(len(reaches))
nfacc = np.zeros(len(nodes))
for r in list(range(len(reaches))):
    print(r, len(reaches)-1)
    pts = np.where(cl_rchs == reaches[r])[0]
    unq_nodes = np.unique(cl_nodes[pts])
    max_dist = np.max(eps_dist[pts])
    if max_dist > 0.001:
        rfacc[r] = -9999
        nplace = np.where(np.in1d(nodes, unq_nodes))[0]
        nfacc[nplace] = -9999
    else:
        rfacc[r] = np.max(mfacc[eps_ind[pts]])
        for n in list(range(len(unq_nodes))):
            npts = np.where(cl_nodes == unq_nodes[n])[0]
            nplace = np.where(nodes == unq_nodes[n])[0]
            nfacc[nplace] = np.max(mfacc[eps_ind[npts]])
        
rfacc[np.where(rfacc == -9999)] = rch_facc[np.where(rfacc == -9999)]
nfacc[np.where(nfacc == -9999)] = node_facc[np.where(nfacc == -9999)]

### Update in NetCDFs
s16.groups['reaches'].variables['facc'][:] = rfacc
s16.groups['nodes'].variables['facc'][:] = nfacc
m11.close()
s16.close()

end = time.time()
print('Finished in: '+str(np.round((end-start)/3600,2))+' hrs')



# rx = s16.groups['reaches'].variables['x'][:]
# ry = s16.groups['reaches'].variables['y'][:]

# plt.scatter(rx, ry, s=3, c=np.log(rfacc))
# plt.scatter(rx, ry, s=3, c=np.log(rch_facc))
# plt.show()

# plt.scatter(clx[pts], cly[pts], s=3, c='blue')
# plt.scatter(mx[eps_ind[pts]], my[eps_ind[pts]], s=3, c='red')
# plt.show()

# plt.scatter(rx16, ry16, c='blue', s=2)
# plt.scatter(rx14, ry14, c='magenta', s=2)
# plt.scatter(rx16[far], ry16[far], c='green', s=2)
# plt.show()

# plt.scatter(nx16, ny16, c='blue', s=2)
# plt.scatter(nx14, ny14, c='magenta', s=2)
# plt.scatter(nx16[far2], ny16[far2], c='green', s=2)
# plt.show()

# plt.scatter(rx16, ry16, c=np.log(rfacc16), s=2)
# plt.scatter(rx14, ry14, c=np.log(rfacc14), s=2)
# plt.show()

