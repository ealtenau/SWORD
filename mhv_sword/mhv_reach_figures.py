from __future__ import division
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import random

###############################################################################
###############################################################################
###############################################################################

reaches_len = np.copy(subcls.rch_len6)
Type = np.copy(subcls.type6)

np.mean(reaches_len)/1000
np.median(reaches_len)/1000

np.mean(subcls.node_len)/1000
np.median(subcls.node_len)/1000

dams = np.where(Type == 4)[0]
lakes = np.where(Type == 3)[0]
coast = np.where(Type == 5)[0]
rivers = np.where(Type == 1)[0]

np.max(reaches_len[dams]/1000)
np.min(reaches_len[dams]/1000)
np.median(reaches_len[dams]/1000)

np.max(reaches_len[lakes]/1000)
np.min(reaches_len[lakes]/1000)
np.median(reaches_len[lakes]/1000)

np.max(reaches_len[coast])/1000
np.min(reaches_len[coast]/1000)
np.median(reaches_len[coast]/1000)

np.max(reaches_len[rivers])/1000
np.min(reaches_len[rivers]/1000)
np.median(reaches_len[rivers]/1000)

###############################################################################
###############################################################################
###############################################################################


### sub-basin reaches and endpoints
reaches = subcls.reach_id
x = subcls.x
y = subcls.y
eps = np.where(subcls.rch_eps6 == 1)

unq_rchs = np.unique(reaches)
number_of_colors = len(unq_rchs)
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]

plt.figure(1, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('North America Reaches',  fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
for i in list(range(len(unq_rchs))):
    seg = np.where(reaches == unq_rchs[i])
    plt.scatter(x[seg], y[seg], c=color[i], s = 5, edgecolors = 'None')
# plt.scatter(x[eps], y[eps], c='black', s = 10, edgecolors = 'None')
plt.show()

eps = np.where(new_rch_eps[rch] > 0)[0]
plt.figure(1, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Checking Reach Indexes',  fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.scatter(rch_x, rch_y, c=new_rch_ind[rch], s = 10, edgecolors = 'None')
plt.scatter(rch_x[eps], rch_y[eps], c='red', s = 20, edgecolors = 'None')
plt.show()


eps = np.where(subcls.type6 == 6)[0]
eps = np.where(subcls.ghost == 1)[0]
plt.figure(1, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Ghost Reaches',  fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.scatter(subcls.x, subcls.y, c='blue', s = 10, edgecolors = 'None')
plt.scatter(subcls.x[eps], subcls.y[eps], c='red', s = 20, edgecolors = 'None')
plt.show()


nodes = np.where(subcls.node_id == 76300000090871)[0]
rch = np.where(subcls.reach_id == 76300000091)[0]
plt.scatter(subcls.x[rch], subcls.y[rch], c=subcls.rch_ind6[rch], s = 10, edgecolors = 'None')
plt.show()


plt.scatter(rch_x, rch_y, c = subcls.seg[rch])
plt.show()


plt.scatter(rch_x, rch_y, c = new_rch_ind[rch])
plt.show()

eps = np.where(rch_eps_all == 1)[0]
plt.scatter(rch_x, rch_y, c = subcls.seg[rch])
plt.scatter(rch_x[eps], rch_y[eps], c = 'red')
plt.show()


r1 = np.where(subcls.rch_id1 == 56)[0]
r2 = np.where(subcls.rch_id1 == 61)[0]
r3 = np.where(subcls.rch_id1 == 62)[0]

plt.scatter(subcls.x[r1], subcls.y[r1], c = 'red')
plt.scatter(subcls.x[r2], subcls.y[r2], c = 'blue')
plt.scatter(subcls.x[r3], subcls.y[r3], c = 'cyan')
plt.show()


rch = np.where(subcls.rch_id2 == 25516)[0]
plt.scatter(subcls.x[rch], subcls.y[rch], c = subcls.ind[rch])
plt.show()


rch = np.where(subcls.reach_id == 74252100181)[0]
plt.plot(subcls.x[rch], subcls.y[rch])
plt.show()

##########################
unq_rchs = np.unique(new_segs)
number_of_colors = len(unq_rchs)
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]

# zeros = np.where(new_segs == 0)[0]
plt.figure(1, figsize=(11,8))
plt.rcParams['axes.linewidth'] = 1.5
plt.tick_params(width=1.5, direction='out', length=5, top = 'off', right = 'off')
plt.title('Segments',  fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
for i in list(range(len(unq_rchs))):
    seg = np.where(new_segs == unq_rchs[i])
    plt.scatter(subcls.lon[seg], subcls.lat[seg], c=color[i], s = 5, edgecolors = 'None')
# plt.scatter(subcls.lon[zeros], subcls.lat[zeros], c='black', s = 15, edgecolors = 'None')
plt.show()



rch1 = np.where(subcls.rch_id1 == 25498)[0]
rch2 = np.where(subcls.rch_id1 == 25516)[0]
rch3 = np.where(subcls.rch_id1 == 25526)[0]
rch4 = np.where(subcls.rch_id1 == 7874)[0]
plt.scatter(subcls.x[rch1], subcls.y[rch1], c = 'black')
plt.scatter(subcls.x[rch2], subcls.y[rch2], c = 'red')
plt.scatter(subcls.x[rch3], subcls.y[rch3], c = 'blue')
plt.scatter(subcls.x[rch4], subcls.y[rch4], c = 'cyan')
plt.show()


plt.scatter(subcls.x[rch1], subcls.y[rch1], c = 'black')
plt.scatter(subcls.x[rch2], subcls.y[rch2], c = 'red')
plt.scatter(subcls.x[rch3], subcls.y[rch3], c = 'blue')
plt.scatter(subcls.x[rch4], subcls.y[rch4], c = 'cyan')
plt.scatter(subcls.x[test], subcls.y[test], c = subcls.rch_ind2[test])
plt.show()

# plt.scatter(subcls.x[rch], subcls.y[rch], c = subcls.rch_ind2[rch])
plt.scatter(subcls.x[rch], subcls.y[rch], c = subcls.ind[rch])
plt.scatter(subcls.x[rch[eps_ind[final_eps]]], subcls.y[rch[eps_ind[final_eps]]], c = 'red')
plt.show()

rch = np.where(subcls.rch_id4 ==  2351)[0]
plt.scatter(subcls.x[rch], subcls.y[rch], c = subcls.rch_ind4[rch])
plt.show()

rch = np.where(basin_rch == 25500)[0]
rch2 = np.where(basin_rch == 25516)[0]
rch3 = np.where(basin_rch == 25526)[0]
rch4 = np.where(basin_rch == 37374)[0]

plt.scatter(basin_lon[rch], basin_lat[rch], c = 'black')
plt.scatter(basin_lon[rch2], basin_lat[rch2], c = 'red')
plt.scatter(basin_lon[rch3], basin_lat[rch3], c = 'blue')
plt.scatter(basin_lon[rch4], basin_lat[rch4], c = 'gold')
plt.scatter(basin_lon[rch3], basin_lat[rch3], c = subcls.ind[rch3])
plt.show()


#try removing duplicate rows/indexes of x-y before finding endpoints. There seem to be overlapping points a lot.


plt.scatter(subcls.x[rch], subcls.y[rch], c = rch_ind_temp)
plt.scatter(subcls.x[rch[divs]], subcls.y[rch[divs]], c = 'red')
plt.show()

r = np.where(subcls.reach_id == 74249400181)[0]

plt.scatter(subcls.x[r], subcls.y[r], c = subcls.rch_ind6[r])
plt.plot(subcls.x[r], subcls.y[r])
plt.show()

plt.scatter(subcls.x[r], subcls.y[r], c = subcls.seg[r])
plt.show()

plt.scatter(subcls.x[rch], subcls.y[rch], c = subcls.seg[rch])
plt.show()