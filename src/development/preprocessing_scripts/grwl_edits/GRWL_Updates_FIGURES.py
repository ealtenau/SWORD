# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:33:36 2019
"""
import matplotlib.pyplot as plt
import random

"""
FIGURES
"""
plt.figure(1, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Locations', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl.x, grwl.y, c='blue', edgecolors='none', s = 5)
plt.scatter(edits.x[seg], edits.y[seg], c='red', edgecolors='none', s = 5)
plt.scatter(grwl.x[pt_ind], grwl.y[pt_ind], c='green', edgecolors='none', s = 5)
plt.scatter(edits.x2[pt_ind2], edits.y2[pt_ind2], c='orange', edgecolors='none', s = 5)

plt.figure(2, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Manual Locations', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl.x, grwl.y, c=grwl.manual, edgecolors='none', s = 5)

plt.figure(3, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('New GRWL Ind', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl.lon, grwl.lat, c=np.log(grwl.finalInd), edgecolors='none', s = 5)

plt.figure(4, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Locations', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl.lon, grwl.lat, c=grwl.finalID, edgecolors='none', s = 5)

plt.figure(5, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Lake IDs', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl.x, grwl.y, c=grwl.lake, edgecolors='none', s = 5)

ep = np.where(grwl.finaleps>0)[0]
plt.figure(6, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('End Points', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl.x, grwl.y, c='blue', edgecolors='none', s = 3)
plt.scatter(grwl.x[ep], grwl.y[ep], c='red', edgecolors='none', s = 5)

plt.figure(7, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Segments', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(edits.x, edits.y, c=edits.seg, edgecolors='none', s = 5)

plt.figure(8, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Segment IDs', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl.x, grwl.y, c=np.log(grwl.finalInd), edgecolors='none', s = 5)
z = np.where(grwl.finaleps > 0)[0]
plt.scatter(grwl.x[z], grwl.y[z], c='pink', s= 20, edgecolors = None)

plt.figure(9, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Lake IDs', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(edits.x, edits.y, c=edits.lake, edgecolors='none', s = 5)

plt.figure(10, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Data', fontsize=16)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.scatter(grwl.x, grwl.y, c='blue', edgecolors='none', s = 5)
plt.scatter(edits.x, edits.y, c='red', edgecolors='none', s = 5)
#plt.scatter(edits_x[neighbors_id], edits_y[neighbors_id], c='yellow', edgecolors='none', s = 3)

###############################################################################
### Plotting GRWL IDs    
unq_id = np.unique(grwl.finalID)
number_of_colors2 = len(unq_id)+5
color2 = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors2)]

plt.figure(11, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('GRWL Segments',  fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
for i in list(range(len(unq_id))):
    seg = np.where(grwl.finalID == unq_id[i])
    plt.scatter(grwl.x[seg], grwl.y[seg], c=color2[i], s = 5, edgecolors = 'None')
w = np.where(grwl.finaleps > 0)[0]
plt.scatter(grwl.x[w], grwl.y[w], c='black', s= 20, edgecolors = None)
plt.show()

z = np.where(grwl.tribs == 1)[0]
plt.scatter(grwl.x[z], grwl.y[z], c='yellow', s= 20, edgecolors = None)

