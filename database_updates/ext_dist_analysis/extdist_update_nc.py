from __future__ import division
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

### Reading in data. 
region = 'OC'
title = '3sig_max5_nchan3'
sword_fn = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16_glows/netcdf/'+region.lower()+'_sword_v16_glows_3sig_max5.nc'
outfig = '/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v16_glows/plots/'+region.lower()+'_extdist_'+title+'.png'

sword = nc.Dataset(sword_fn,'r+')
nx = np.array(sword['/nodes/x/'][:])
ny = np.array(sword['/nodes/y/'][:])
nid = np.array(sword['/nodes/node_id/'][:])
nchan_max = np.array(sword['/nodes/n_chan_max/'][:])
nchan_mod = np.array(sword['/nodes/n_chan_mod/'][:])
wth = np.array(sword['/nodes/width/'][:])
max_wth = np.array(sword['/nodes/max_width/'][:])
glows_sig = np.array(sword['/nodes/glows_wth_1sig/'][:])
rid = np.array(sword['/nodes/reach_id/'][:])
ext_dist = np.array(sword['/nodes/ext_dist_coef/'][:])
edit_flag = np.array(sword['/nodes/edit_flag/'][:])

#trying cassie's ratio with glow-s data. 
### add condition to not mess with pre-existing 1 values
### also try with node level sigma values. 
# max_val = 5
# unq_rchs = np.unique(rid)
# for r in list(range(len(unq_rchs))):
#     npts = np.where(rid == unq_rchs[r])[0]
#     max_sig = np.max(glows_sig[npts])
#     chan_med = np.median(nchan_max[npts])
#     if max_sig > 0:
#         ext_dist[npts] = np.ceil((wth[npts]+(3*max_sig))/wth[npts])
#         edit_val = edit_flag[npts][0]
#         if edit_val == 'NaN':
#             edit_flag[npts] = '8'
#         else:
#             edit_flag[npts] = edit_val + ',8'
#         if chan_med > 1:
#             update = np.where(ext_dist[npts] > max_val)[0]
#             ext_dist[npts[update]] = max_val
# ext_dist[np.where(ext_dist < 1)] = 1
# ext_dist[np.where(ext_dist > max_val)] = max_val

max_val = 5
unq_rchs = np.unique(rid)
for r in list(range(len(unq_rchs))):
    npts = np.where(rid == unq_rchs[r])[0]
    # max_sig = np.max(glows_sig[npts])
    # chan_med = np.median(nchan_max[npts])
    sigs = np.where((glows_sig[npts] > 0) & (ext_dist[npts] != 1))[0]
    if len(sigs) > 0:
        ext_dist[npts[sigs]] = np.ceil((wth[npts[sigs]]+(3*glows_sig[npts[sigs]]))/wth[npts[sigs]])
        edit_val = edit_flag[npts][0]
        if edit_val == 'NaN':
            edit_flag[npts] = '8'
        else:
            edit_flag[npts] = edit_val + ',8'
        # if chan_med > 1:
        #     update = np.where(ext_dist[npts] > max_val)[0]
        #     ext_dist[npts[update]] = max_val
ext_dist[np.where(ext_dist < 1)] = 1
ext_dist[np.where(ext_dist > max_val)] = max_val
ext_dist[np.where(nchan_mod >= 3)[0]] = max_val


#updating netcdf
sword['/nodes/ext_dist_coef/'][:] = ext_dist
sword.close()

print(np.unique(ext_dist))
print(np.unique(edit_flag))

#pie chart
one = np.where(ext_dist == 1)[0]
two = np.where(ext_dist == 2)[0]
three = np.where(ext_dist == 3)[0]
four = np.where(ext_dist == 4)[0]
five = np.where(ext_dist == 5)[0]
p = np.array([(len(one)/len(ext_dist))*100, (len(two)/len(ext_dist))*100 , 
              (len(three)/len(ext_dist))*100, (len(four)/len(ext_dist))*100,
              (len(five)/len(ext_dist))*100])

mylabels = ["1 ("+str(np.round((len(one)/len(ext_dist))*100,1))+"%)",
            "2 ("+str(np.round((len(two)/len(ext_dist))*100,1))+"%)", 
            "3 ("+str(np.round((len(three)/len(ext_dist))*100,1))+"%)", 
            "4 ("+str(np.round((len(four)/len(ext_dist))*100,1))+"%)", 
            "5 ("+str(np.round((len(five)/len(ext_dist))*100,1))+"%)"]
mycolors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
plt.pie(p,labels = mylabels, colors = mycolors, wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
plt.title(region+' '+title)
plt.savefig(outfig)
print('Done')


#pie chart
# one = np.where(ext_dist == 1)[0]
# two = np.where(ext_dist == 2)[0]
# three = np.where(ext_dist == 3)[0]
# p = np.array([(len(one)/len(ext_dist))*100, (len(two)/len(ext_dist))*100, 
#               (len(three)/len(ext_dist))*100])

# mylabels = ["1 ("+str(np.round((len(one)/len(ext_dist))*100,1))+"%)",
#             "2 ("+str(np.round((len(two)/len(ext_dist))*100,1))+"%)", 
#             "3 ("+str(np.round((len(three)/len(ext_dist))*100,1))+"%)"]
# mycolors = ["#ffffcc", "#41b6c4", "#253494"]
# plt.pie(p,labels = mylabels, colors = mycolors, wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
# plt.title(region+' '+title)
# plt.savefig(outfig)
# print('Done')
























##########################################################################
### adjusting max widths for outlier nodes. 

# def calc_outliers(vals):
#     q3, q1 = np.percentile(vals, [90, 10]) #typical - 75,25
#     iqr = q3 - q1
#     outliers = [_ for _ in vals if _ > q3 + (iqr * 1.5) or _ < q1 - (iqr * 1.5)]
#     outliers = np.unique(np.array(outliers))
#     if len(outliers) > 0:
#         out_ind = np.where(np.in1d(vals, outliers) == True)[0]
#     print(outliers)
#     # return outliers, out_ind

# unq_rchs = np.unique(rid)
# for r in list(range(len(unq_rchs))):
#     print(r)
#     rch = np.where(rid == unq_rchs[r])[0]
#     rw = wth[rch]
#     rmw = max_wth[rch]
#     calc_outliers(rw)
#     calc_outliers(rmw)
