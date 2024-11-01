import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import splprep, BSpline, splev
import random
import os
import pandas as pd
# import geopandas as gp
# from scipy import stats
# from scipy.interpolate import UnivariateSpline

#############################################################################################

def read_pixc_data(pixc_dir, files, rch):

    for f in list(range(len(files))):
        pixc_df = nc.Dataset(pixc_dir+files[f])
        subset = np.where(pixc_df.variables['reach_id'][:] == rch)[0]
        if f == 0:
            pixc_x = np.array(pixc_df.variables['longitude_vectorproc'][subset])
            pixc_y = np.array(pixc_df.variables['latitude_vectorproc'][subset])
            pixc_nodes = np.array(pixc_df.variables['node_id'][subset])
        else:
            add_x = np.array(pixc_df.variables['longitude_vectorproc'][subset])
            add_y = np.array(pixc_df.variables['latitude_vectorproc'][subset])
            add_n = np.array(pixc_df.variables['node_id'][subset])
            pixc_x = np.append(pixc_x, add_x)
            pixc_y = np.append(pixc_y, add_y)
            pixc_nodes = np.append(pixc_nodes, add_n)

        # print(len(pixc_x))

    return pixc_x, pixc_y, pixc_nodes
    
#############################################################################################

sword_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16/netcdf/na_sword_v16.nc'
pixc_dir = '/Users/ealtenau/Documents/SWORD_Dev/swot_data/shifting_tests/412_079L/'
files = os.listdir(pixc_dir)
files =[f for f in files if '.nc' in f] 
outdir = '/Users/ealtenau/Documents/SWORD_Dev/swot_data/shifting_tests/plots/'

tile = pixc_dir[-9:-1]

# pixc = nc.Dataset(pixc_dir+files[0])
# xmax = np.nanmax(np.array(pixc.variables['longitude_vectorproc'][:]))
# xmin = np.nanmin(np.array(pixc.variables['longitude_vectorproc'][:]))
# ymax = np.nanmax(np.array(pixc.variables['latitude_vectorproc'][:]))
# ymin = np.nanmin(np.array(pixc.variables['latitude_vectorproc'][:]))
# ll = np.array([xmin, ymin])  # lower-left
# ur = np.array([xmax, ymax])  # upper-right
# pixc.close()

sword = nc.Dataset(sword_fn)
sword_lon = sword.groups['centerlines'].variables['x'][:]
sword_lat = sword.groups['centerlines'].variables['y'][:]
sword_rchs =  sword.groups['centerlines'].variables['reach_id'][0,:]
node_x_all = sword.groups['nodes'].variables['x'][:]
node_y_all = sword.groups['nodes'].variables['y'][:]
node_id_all = sword.groups['nodes'].variables['node_id'][:]
node_rch = sword.groups['nodes'].variables['reach_id'][:]
sword_points = [(sword_lon[i], sword_lat[i]) for i in range(len(sword_lon))]
sword_pts = np.array(sword_points)
# sword_idx = np.all(np.logical_and(ll <= sword_pts, sword_pts <= ur), axis=1)
# reaches = np.unique(sword_rchs[sword_idx])
sword.close()


flagged = pd.read_csv('/Users/ealtenau/Desktop/sword_shift_flag_test3.csv')
reaches_all = np.array(flagged['3'])
f = np.where(flagged['2'] == 1)[0]
reaches = np.unique(reaches_all[f])

for r in list(range(len(reaches))): # r = 4 is a good test, 74298900671

    rch = reaches[r]
    pixc_x,pixc_y,pixc_nodes = read_pixc_data(pixc_dir,files, rch)

    if len(pixc_x) == 0:
        continue
    
    cl_pts = np.where(sword_rchs == rch)[0]
    node_pts = np.where(node_rch == rch)[0]
    node_x = node_x_all[node_pts]
    node_y = node_y_all[node_pts]
    node_id = node_id_all[node_pts]

    if len(node_pts) < 10:
        continue 

    new_x_med = np.zeros(len(node_id))
    new_y_med = np.zeros(len(node_id))
    new_x_mean = np.zeros(len(node_id))
    new_y_mean = np.zeros(len(node_id))
    flag = np.zeros(len(node_id))
    for n in list(range(len(node_id))):
        px = np.where(pixc_nodes == node_id[n])[0]
        # print(len(px))
        if len(px) < 10:
            new_x_med[n] = node_x[n]
            new_y_med[n] = node_y[n]
            new_x_mean[n] = node_x[n]
            new_y_mean[n] = node_y[n]
            flag[n] = 1
        else:
            new_x_med[n] = np.median(pixc_x[px])
            new_y_med[n] = np.median(pixc_y[px])
            new_x_mean[n] = np.mean(pixc_x[px])
            new_y_mean[n] = np.mean(pixc_y[px])

    x_diff = np.max(node_x) - np.min(node_x)
    y_diff = np.max(node_y) - np.min(node_y)
    if y_diff < x_diff:
        s = np.var(new_y_med)/3
    else:
        s = np.var(new_x_med)/3
        

    pts = np.vstack((new_x_med, new_y_med))
    # Find the B-spline representation of an N-dimensional curve
    tck, u = splprep(pts, s=s) #0.0001, 0.000075
    n_new = np.linspace(u.min(), u.max(), len(node_pts))
    cl_new = np.linspace(u.min(), u.max(), len(cl_pts))
    # Evaluate a B-spline
    node_x_smooth, node_y_smooth = splev(n_new, tck)
    cl_x_smooth, cl_y_smooth = splev(cl_new, tck)

    # plt.scatter(pixc_x, pixc_y, c = 'grey', alpha = 0.1, s=3)
    # plt.scatter(cl_x_smooth, cl_y_smooth, c = 'cyan', s=3)
    # plt.scatter(node_x_smooth, node_y_smooth, c = 'blue', s=8)
    # # plt.scatter(node_x, node_y, c = 'black', s=8)
    # plt.show()

    ### plot
    outfile = outdir+'rch_'+tile+'_'+str(rch)
    na = np.where(flag == 1)[0]
    plt.figure(figsize=(5.5,7.5))
    plt.scatter(pixc_x, pixc_y, c = 'grey', alpha = 0.1, s=3)
    plt.plot(node_x, node_y, c = 'blue')
    plt.scatter(node_x, node_y, c = 'blue', s=8)
    # plt.ylim(-43.365, -43.315)
    # plt.plot(new_x_med, new_y_med, c = 'cyan')
    # plt.scatter(new_x_med, new_y_med, c = 'cyan', s=8)
    # plt.plot(node_x_smooth, node_y_smooth, 'b--', c = 'cyan', linewidth=2, alpha = 0.75)
    plt.plot(node_x_smooth, node_y_smooth, c = 'cyan')
    plt.scatter(node_x_smooth, node_y_smooth, c = 'cyan', s=8)
    plt.scatter(new_x_med[na], new_y_med[na], c = 'gold', s=8)
    plt.xlabel('lon')
    plt.ylabel('lat')
    plt.title('Smoothed, Shifted Centerline')
    plt.savefig(outfile)
    plt.close()
    # plt.show()


'''
plt.scatter(pixc_x, pixc_y, c = 'magenta', s=3)
plt.scatter(node_x, node_y, c = 'blue', s=3)
plt.show()

na = np.where(flag == 1)[0]

plt.figure(1, figsize=(5.5,7.5))
plt.scatter(pixc_x, pixc_y, c = 'grey', alpha = 0.2, s=3)
plt.scatter(node_x, node_y, c = node_id, cmap = 'jet', s=8)
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Original SWORD')
plt.show()

plt.figure(2, figsize=(5.5,7.5))
plt.scatter(pixc_x, pixc_y, c = 'grey', alpha = 0.2, s=3)
plt.scatter(new_x_med, new_y_med, c = node_id, cmap = 'jet', s=8)
plt.scatter(new_x_med[na], new_y_med[na], c = 'black', s=8)
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Median Location of Node Pixels')
plt.show()

plt.figure(3, figsize=(5.5,7.5))
plt.scatter(pixc_x, pixc_y, c = 'grey', alpha = 0.2, s=3)
plt.scatter(new_x_mean, new_y_mean, c = node_id, cmap = 'jet',s=8)
plt.scatter(new_x_med[na], new_y_med[na], c = 'black', s=8)
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Mean Location of Node Pixels')
plt.show()

plt.figure(4, figsize=(5.5,7.5))
plt.scatter(pixc_x, pixc_y, c = 'grey', alpha = 0.2, s=3)
plt.scatter(new_x_shift, new_y_shift, c = node_id, cmap = 'jet',s=8)
plt.scatter(new_x_shift[na], new_y_shift[na], c = 'black', s=8)
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Shifted Location of Node Pixels, alpha = 0.75')
plt.show()

####
# plt.figure(5, figsize=(5.5,7.5))
# plt.scatter(pixc_x, pixc_y, c = 'grey', alpha = 0.2, s=3)
# plt.scatter(shift02_x, shift02_y, c = 'firebrick',s=8)
# plt.scatter(shift04_x, shift04_y, c = 'gold',s=8)
# plt.scatter(shift06_x, shift06_y, c = 'limegreen',s=8)
# plt.scatter(shift08_x, shift08_y, c = 'darkturquoise',s=8)
# plt.scatter(shift1_x, shift1_y, c = 'royalblue',s=8)
# plt.scatter(new_x_shift[na], new_y_shift[na], c = 'black', s=8)
# plt.xlabel('lon')
# plt.ylabel('lat')
# plt.title('Shifted Locations of Node Pixels')
# plt.show()


####node colors

unq_nodes = np.unique(pixc_nodes)
number_of_colors = len(unq_nodes)
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]

plt.figure(figsize=(5.5,7.5))
for i in list(range(len(unq_nodes))):
    seg = np.where(pixc_nodes == unq_nodes[i])
    plt.scatter(pixc_x[seg], pixc_y[seg], c=color[i], s = 3, alpha = 0.1)
plt.plot(new_x_med, new_y_med, c = 'black')
plt.scatter(new_x_med, new_y_med, c = 'black', s=8)
plt.plot(node_x_smooth, node_y_smooth, c = 'white', linewidth=1)
plt.scatter(node_x_smooth, node_y_smooth, c = 'white', s=3)
plt.xlabel('lon')
plt.ylabel('lat')
plt.title('Raw vs. Smoothed Centerline')
plt.show()

'''



# xx, yy, zz = kde2D(pixc_x, pixc_y, 1.0)

# plt.pcolormesh(xx, yy, zz)
# plt.scatter(pixc_x, pixc_y, s=2, facecolor='white')
# plt.show()