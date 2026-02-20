# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:16:10 2018
"""

#Practice with Shapefiles

#Useful functions
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


"""
FIGURES
"""
#nn_final[np.where(nn_final > 2000000)] = np.nan
#nn_upa_filt[np.where(nn_upa_filt > 2000000)] = np.nan

plt.figure(1, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Flow Accumulation (km2)', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl.lon, grwl.lat, c=np.log(grwl.facc_filt), edgecolors='none', s = 3)
plt.colorbar()

plt.figure(10, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Raw Flow Accumulation (km2)', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl.lon, grwl.lat, c=np.log(grwl.facc), edgecolors='none', s = 3)
plt.colorbar()

plt.figure(2, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Elevation (m)', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl.lon, grwl.lat, c=np.log(grwl.elv), edgecolors='none', s = 1)
plt.colorbar()

plt.figure(4, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Delta Flag', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl.lon, grwl.lat, c=grwl.delta, edgecolors='none', s = 1)
plt.colorbar()

plt.figure(5, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Segments', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl.lon, grwl.lat, c=grwl.seg, edgecolors='none', s = 1)
plt.colorbar()

plt.figure(7, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Segment Distance', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl.lon, grwl.lat, c=np.log((grwl.dist/1000)), edgecolors='none', s = 1)
plt.colorbar()

plt.figure(8, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('GFDM Flow Accumulation (km2)', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(mh.lon, mh.lat, c=np.log(mh.facc), edgecolors='none', s = 3)
plt.colorbar()

plt.figure(9, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('GFDM vs. GRWL', fontsize=16)
plt.xlabel('longitude', fontsize=14)
plt.ylabel('latitude', fontsize=14)
plt.scatter(mh.lon, mh.lat, c=np.log(mh.facc), edgecolors='none', s = 5)
plt.scatter(grwl.lon, grwl.lat, c='red', edgecolors='none', s = 3)
#plt.colorbar()

plt.figure(10, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('MH vs. GRWL', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.scatter(mh.x, mh.y, c='blue', edgecolors='none', s = 5)
plt.scatter(grwl.x, grwl.y, c='red', edgecolors='none', s = 3)
plt.show()

count = 1
bid = np.zeros(len(grwl_id))
for basin in np.unique(basin_code):
    bid[np.where(basin_code == basin)] = count
    count = count+1

plt.figure(12, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Basins', fontsize=16)
plt.xlabel('lon', fontsize=14)
plt.ylabel('lat', fontsize=14)
plt.scatter(grwl.lon, grwl.lat, c=bid, edgecolors='none', s = 3)
plt.show()

plt.figure(13, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Basins', fontsize=16)
plt.xlabel('lon', fontsize=14)
plt.ylabel('lat', fontsize=14)
plt.scatter(grwl.lon, grwl.lat, c=grwl.manual, edgecolors='none', s = 3)

plt.figure(15, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Smoothed Points', fontsize=16)
plt.xlabel('lon', fontsize=14)
plt.ylabel('lat', fontsize=14)
plt.scatter(grwl.longitude, grwl.latitude, c='blue', edgecolors='none', s = 5)
plt.scatter(grwl.lon, grwl.lat, c='red', edgecolors='none', s = 5)


###############################################################################
w = np.where(grwl.y > 10000000)
plt.scatter(grwl.x, grwl.y, c = 'blue', s = 5, edgecolors = None)
plt.scatter(grwl.x[w], grwl.y[w], c = 'red', s = 5, edgecolors = None)

grwl.y[w] = 10000000




plt.scatter(grod_lon, grod_lat, c = 'blue', s = 5, edgecolors = None)
plt.scatter(grod_lon_clip, grod_lat_clip, c = 'red', s = 5, edgecolors = None)

z = np.where(grod_ID > 0)[0]
plt.scatter(grwl.x, grwl.y, c = 'grey', s = 5, edgecolors = None)
plt.scatter(grwl.x[z], grwl.y[z], c = 'blue', s = 5, edgecolors = None)
plt.scatter(grod_x, grod_y, c = 'cyan', s = 5, edgecolors = None)
