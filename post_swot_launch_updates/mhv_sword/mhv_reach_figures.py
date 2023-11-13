from __future__ import division
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import random

###############################################################################
###############################################################################
###############################################################################

reaches_len = np.copy(subcls.rch_len5)
Type = np.copy(subcls.type5)

np.mean(reaches_len)/1000
np.median(reaches_len)/1000

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
reaches = subcls.rch_id2[0:1000]
x = subcls.x[0:1000]
y = subcls.y[0:1000]
eps = np.where(subcls.rch_eps2[0:1000] == 1)

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
plt.scatter(x[eps], y[eps], c='black', s = 10, edgecolors = 'None')
plt.show()