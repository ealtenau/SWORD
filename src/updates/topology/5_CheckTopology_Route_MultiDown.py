#!/usr/bin/env python3
#*******************************************************************************
#CheckTopology_Route_MultiDown.py
#*******************************************************************************

#Purpose:
#This script is the same as CheckTopology_Route.py, except can be applied to areas 
#with multiple downstream segments
#Assign value of 1 to every reach and use lumped routing to accumulate downstream
#This will allow me to visualize if the connectivity information is correct


#*******************************************************************************
#Import Python modules
#*******************************************************************************
import os
main_dir = os.getcwd()
import sys
import shutil
import csv
import netCDF4
import numpy
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import scipy
import pandas as pd
import geopandas as gp
import argparse

#*******************************************************************************
#Command Line Variables / Instructions:
#*******************************************************************************
# 1 - SWORD Continent (i.e. AS)
# 2 - Level 2 Pfafstetter Basin (i.e. 36)
# Example Syntax: "python CheckTopology_Route_MultiDown.py AS 36"
#*******************************************************************************

#*******************************************************************************
#Get command line arguments
#*******************************************************************************
parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("basin", help="<Required> Level Two Pfafstetter Basin (i.e. 74)", type = str)
parser.add_argument("rerun", help="whether to rerun the routing from manual or not", type = str)
args = parser.parse_args()

b = args.basin
region = args.region
rerun = args.rerun

data_dir = main_dir+'/data/outputs/Topology/'+region+'/b'+b+'/'

rrr_con_csv= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_LSFix_MS_TopoFix_con.csv'
rrr_bas_csv= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_LSFix_MS_TopoFix_riv.csv'
rrr_acc_shp= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_LSFix_MS_TopoFix_Acc.shp'
rrr_rch_acc= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_LSFix_MS_TopoFix_acc.csv'
if rerun == 'True':
     rrr_top_shp= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_LSFix_MS_TopoFix_Acc.shp'
else:
     rrr_top_shp= data_dir + region.lower()+'_sword_reaches_hb' + b + '_v17_FG1_LSFix_MS_TopoFix.shp'
#*******************************************************************************
#Print input information
#*******************************************************************************
print('Command line inputs')
print('- '+rrr_con_csv)
print('- '+rrr_bas_csv)
# print('- '+rrr_m3r_ncf)
# print('- '+rrr_Qob_csv)
# print('- '+rrr_obs_csv)
# print('- '+rrr_use_csv)
print('- '+rrr_rch_acc)


#*******************************************************************************
#Check if files exist 
#*******************************************************************************
try:
     with open(rrr_con_csv) as file:
          pass
except IOError as e:
     print('ERROR - Unable to open '+rrr_con_csv)
     raise SystemExit(22) 

try:
     with open(rrr_bas_csv) as file:
          pass
except IOError as e:
     print('ERROR - Unable to open '+rrr_bas_csv)
     raise SystemExit(22) 


#*******************************************************************************
#Reading connectivity file
#*******************************************************************************
print('Reading connectivity file')

IV_riv_tot_id=[]
IV_riv_tot_dn=[]
IV_riv_tot_dn_multi=[]
with open(rrr_con_csv,'r') as csvfile:
    csvreader=csv.reader(csvfile)
    for row in csvreader:
        #If 'n_rch_dn' is 2, append list of both reaches to IV_riv_tot_dn
        #Also add both 'rch_id_dn_1' and 'rch_id_dn_2' to IV_riv_tot_dn_multi
        #list to store reach IDs that are part of a multi downstream junction
        if int(row[1]) == 2:
            IV_riv_tot_id.append(int(row[0]))
            IV_riv_tot_dn.append([int(row[2]), int(row[3])])

            IV_riv_tot_dn_multi.append(int(row[2]))
            IV_riv_tot_dn_multi.append(int(row[3]))  

        elif int(row[1]) == 3:
            IV_riv_tot_id.append(int(row[0]))
            IV_riv_tot_dn.append([int(row[2]), int(row[3]), int(row[4])])

            IV_riv_tot_dn_multi.append(int(row[2]))
            IV_riv_tot_dn_multi.append(int(row[3])) 
            IV_riv_tot_dn_multi.append(int(row[4])) 

        elif int(row[1]) == 4:
            IV_riv_tot_id.append(int(row[0]))
            IV_riv_tot_dn.append([int(row[2]), int(row[3]), int(row[4]), int(row[5])])

            IV_riv_tot_dn_multi.append(int(row[2]))
            IV_riv_tot_dn_multi.append(int(row[3])) 
            IV_riv_tot_dn_multi.append(int(row[4])) 
            IV_riv_tot_dn_multi.append(int(row[5])) 

        else:
            IV_riv_tot_id.append(int(row[0]))
            IV_riv_tot_dn.append([int(row[2])])



IS_riv_tot=len(IV_riv_tot_id)
print('- Number of river reach connections in rrr_con_csv: '+str(IS_riv_tot))

# IV_riv_tot_dn.index([81380100035, 81380100655])
# IV_riv_tot_dn[13]
# IV_riv_tot_id[13]
# IV_riv_tot_dn.index(81380100655)
# IV_riv_tot_dn.index(81380100371)

# IV_riv_tot_id.index(81181300101)
# IV_riv_tot_dn[3471]

        #If 'rch_id_dn_2' has a value, also append
        #Also add both 'rch_id_dn_1' and 'rch_id_dn_2' to IV_riv_tot_dn_multi
        #list to store reach IDs that are part of a multi downstream junction
        # if int(row[3]) != 0:
        #     IV_riv_tot_id.append(int(row[0]))
        #     IV_riv_tot_dn.append(int(row[3]))

        #     IV_riv_tot_dn_multi.append(int(row[2]))
        #     IV_riv_tot_dn_multi.append(int(row[3]))       


#*******************************************************************************
#Reading basin file
#*******************************************************************************
print('Reading basin file')

IV_riv_bas_id=[]
with open(rrr_bas_csv,'r') as csvfile:
     csvreader=csv.reader(csvfile)
     for row in csvreader:
          IV_riv_bas_id.append(int(row[0]))

IS_riv_bas=len(IV_riv_bas_id)
print('- Number of river reaches in rrr_bas_csv: '+str(IS_riv_bas))

if IS_riv_bas!=IS_riv_tot:
     print('ERROR - Different number of reaches in basin and network')
     raise SystemExit(22) 


#*******************************************************************************
#Creating hash tables
#*******************************************************************************
print('Creating hash tables')

IM_hsh_tot={}
for JS_riv_tot in range(IS_riv_tot):
     IM_hsh_tot[IV_riv_tot_id[JS_riv_tot]]=JS_riv_tot

IM_hsh_bas={}
for JS_riv_bas in range(IS_riv_bas):
     IM_hsh_bas[IV_riv_bas_id[JS_riv_bas]]=JS_riv_bas

IV_riv_ix1=[IM_hsh_bas[IS_riv_id] for IS_riv_id in IV_riv_tot_id]
IV_riv_ix2=[IM_hsh_tot[IS_riv_id] for IS_riv_id in IV_riv_bas_id]
#These arrays allow for index mapping such that IV_riv_tot_id[JS_riv_tot]
#                                              =IV_riv_bas_id[JS_riv_bas]
#IV_riv_ix1[JS_riv_tot]=JS_riv_bas
#IV_riv_ix2[JS_riv_bas]=JS_riv_tot

print('- Hash tables created')


#*******************************************************************************
#Creating network matrix
#*******************************************************************************
print('Creating network matrix')

IV_row=[]
IV_col=[]
IV_val=[]
# 3184
for JS_riv_bas in range(IS_riv_bas):
     # print(IV_riv_bas_id[JS_riv_bas])
     JS_riv_tot=IM_hsh_tot[IV_riv_bas_id[JS_riv_bas]]

     #If the IV_riv_tot_dn reach is in IV_riv_tot_dn_multi, then assign
     #a value of 0.5 because this is a multi-downstream junction
     # if len(IV_riv_tot_dn[JS_riv_tot]) > 1:
     #      for i in range(len(IV_riv_tot_dn[JS_riv_tot])):
     #        JS_riv_ba2=IM_hsh_bas[IV_riv_tot_dn[JS_riv_tot][i]]
     #        IV_row.append(JS_riv_ba2)
     #        IV_col.append(JS_riv_bas)
     #        IV_val.append(0.5)         

     # elif IV_riv_tot_dn[JS_riv_tot][0] != 0:
     #      JS_riv_ba2=IM_hsh_bas[IV_riv_tot_dn[JS_riv_tot][0]]
     #      IV_row.append(JS_riv_ba2)
     #      IV_col.append(JS_riv_bas)
     #      IV_val.append(1)

     if len(IV_riv_tot_dn[JS_riv_tot]) == 2:
          for i in range(len(IV_riv_tot_dn[JS_riv_tot])):
            JS_riv_ba2=IM_hsh_bas[IV_riv_tot_dn[JS_riv_tot][i]]
            IV_row.append(JS_riv_ba2)
            IV_col.append(JS_riv_bas)
            IV_val.append(0.5)         

     elif len(IV_riv_tot_dn[JS_riv_tot]) == 3:
          for i in range(len(IV_riv_tot_dn[JS_riv_tot])):
            JS_riv_ba2=IM_hsh_bas[IV_riv_tot_dn[JS_riv_tot][i]]
            IV_row.append(JS_riv_ba2)
            IV_col.append(JS_riv_bas)
            IV_val.append(1/3)   

     elif len(IV_riv_tot_dn[JS_riv_tot]) == 4:
          for i in range(len(IV_riv_tot_dn[JS_riv_tot])):
            JS_riv_ba2=IM_hsh_bas[IV_riv_tot_dn[JS_riv_tot][i]]
            IV_row.append(JS_riv_ba2)
            IV_col.append(JS_riv_bas)
            IV_val.append(0.25)   

     elif IV_riv_tot_dn[JS_riv_tot][0] != 0:
          JS_riv_ba2=IM_hsh_bas[IV_riv_tot_dn[JS_riv_tot][0]]
          IV_row.append(JS_riv_ba2)
          IV_col.append(JS_riv_bas)
          IV_val.append(1)


# IV_riv_bas_id.index(81380100411)
# IV_riv_bas_id[165]
# IM_hsh_tot[IV_riv_bas_id[165]]
# IV_riv_tot_dn[IM_hsh_tot[IV_riv_bas_id[165]]]

# [i for i in range(len(IV_val)) if IV_val[i] == 0.5] # should be 4 values


ZM_Net=csc_matrix((IV_val,(IV_row,IV_col)),shape=(IS_riv_bas,IS_riv_bas))

print('- Network matrix created')
print('  . Total number of connections: '+str(len(IV_val)))
# print('ZM_Net')
# print(ZM_Net)

#*******************************************************************************
#Creating identity matrix
#*******************************************************************************
print('Creating identity matrix')

IV_row=range(IS_riv_bas)
IV_col=range(IS_riv_bas)
IV_val=[1]*IS_riv_bas
ZM_I=csc_matrix((IV_val,(IV_row,IV_col)),shape=(IS_riv_bas,IS_riv_bas))

# print('ZM_I')
# print(ZM_I)

print('- Done')


#*******************************************************************************
#Computing (I-N)^-1
#*******************************************************************************
print('Computing (I-N)^-1')

# scipy.sparse.linalg.inv(ZM_I - ZM_Net).toarray()
ZM_inN = scipy.sparse.linalg.inv(ZM_I - ZM_Net)

# IV_bas_tmp_id=IV_riv_bas_id
# IV_bas_tmp_cr=IV_riv_bas_id

# IV_row=list(range(IS_riv_bas)) # had to add list bc in python 3 range returns a range obj rather than list
# IV_col=list(range(IS_riv_bas)) # see: https://stackoverflow.com/questions/22437297/python-3-range-append-returns-error-range-object-has-no-attribute-appen
# IV_val=[1]*IS_riv_bas

# for JS_riv_bas in range(IS_riv_bas):
#      #--------------------------------------------------------------------------
#      #Determine the IDs of all rivers downstream of the current rivers
#      #--------------------------------------------------------------------------
#      #  IV_bas_tmp_dn=[IV_riv_tot_dn[IM_hsh_tot[x]] for x in IV_bas_tmp_cr]
#      IV_bas_tmp_dn = []
#      #  for x in IV_bas_tmp_cr:
#      #       for y in range(len(IV_riv_tot_dn[IM_hsh_tot[x]])):
#      #            IV_bas_tmp_dn.append(IV_riv_tot_dn[IM_hsh_tot[x]][y])
#      for x in IV_bas_tmp_cr:
#         IV_bas_tmp_dn.append(IV_riv_tot_dn[IM_hsh_tot[x]])
#         # if type(x) == list:
#         #     for y in range(len(x)):
#         #         IV_bas_tmp_dn.append(IV_riv_tot_dn[IM_hsh_tot[x[y]]])


#      #--------------------------------------------------------------------------
#      #Only keep locations where downstream ID is not zero
#      #--------------------------------------------------------------------------
#     #  IV_idx=[i for i, x in enumerate(IV_bas_tmp_dn) if x != 0]
#      IV_idx=[i for i, x in enumerate(IV_bas_tmp_dn) if x != [0]]
#      IV_bas_tmp_id=[IV_bas_tmp_id[i] for i in IV_idx]
#      IV_bas_tmp_cr=[IV_bas_tmp_cr[i] for i in IV_idx]
#      IV_bas_tmp_dn=[IV_bas_tmp_dn[i] for i in IV_idx]

#      #--------------------------------------------------------------------------
#      #Add a value of one at corresponding location
#      #--------------------------------------------------------------------------
#      IS_bas_tmp=len(IV_bas_tmp_id)
#      for JS_bas_tmp in range(IS_bas_tmp):
#           for x in range(len(IV_bas_tmp_dn[JS_bas_tmp])):
#             IV_row.append(IM_hsh_bas[IV_bas_tmp_dn[JS_bas_tmp][x]])
#             IV_col.append(IM_hsh_bas[IV_bas_tmp_id[JS_bas_tmp]])
#             IV_val.append(1)
     
#      #--------------------------------------------------------------------------
#      #Update list of current rivers
#      #--------------------------------------------------------------------------
#      #  IV_bas_tmp_cr=IV_bas_tmp_dn
#      IV_bas_tmp_cr = [y for x in IV_bas_tmp_dn for y in x]
#      IV_bas_tmp_cr = list(dict.fromkeys(IV_bas_tmp_cr))


# ZM_inN=csc_matrix((IV_val,(IV_row,IV_col)),shape=(IS_riv_bas,IS_riv_bas))


# len(list(set(IV_bas_tmp_cr)))
# # print('ZM_inN')
# # print(ZM_inN)

print('- Done')


# #*******************************************************************************
# #Reading m3_riv_file
# #*******************************************************************************
# print('Reading m3_riv file')

# f=netCDF4.Dataset(rrr_m3r_ncf, 'r')

# #-------------------------------------------------------------------------------
# #Dimensions
# #-------------------------------------------------------------------------------
# if 'COMID' in f.dimensions:
#      YV_rivid='COMID'
# elif 'rivid' in f.dimensions:
#      YV_rivid='rivid'
# else:
#      print('ERROR - Neither COMID nor rivid exist in '+rrr_m3r_ncf)
#      raise SystemExit(22) 

# IS_m3r_tot=len(f.dimensions[YV_rivid])
# print('- The number of river reaches is: '+str(IS_m3r_tot))

# if 'Time' in f.dimensions:
#      YV_time='Time'
# elif 'time' in f.dimensions:
#      YV_time='time'
# else:
#      print('ERROR - Neither Time nor time exist in '+rrr_m3r_ncf)
#      raise SystemExit(22) 

# IS_m3r_tim=len(f.dimensions[YV_time])
# print('- The number of time steps is: '+str(IS_m3r_tim))

# #-------------------------------------------------------------------------------
# #Variables
# #-------------------------------------------------------------------------------
# if 'm3_riv' in f.variables:
#      YV_var='m3_riv'
# else:
#      print('ERROR - m3_riv does not exist in '+rrr_m3r_ncf)
#      raise SystemExit(22) 

# if YV_rivid in f.variables:
#      IV_m3r_tot_id=list(f.variables[YV_rivid])
#      if IV_m3r_tot_id==IV_riv_tot_id:
#           print('- The river IDs in rrr_m3r_ncf and rrr_con_csv are the same')
#      else:
#           print('ERROR - The river IDs in rrr_m3r_ncf and rrr_con_csv differ')
#           raise SystemExit(22) 

# if YV_time in f.variables:
#      # if IS_m3r_tim > 1:
#      #      ZS_TaR=f.variables[YV_time][1]-f.variables[YV_time][0]
#      #      # ZS_TaR=2678400
#      #      print('- The time step in rrr_m3r_ncf was determined as: '+str(ZS_TaR)+   \
#      #           ' seconds')
#      # else:
#      #      ZS_TaR=2629920 # average monthly time step between 1980 and 2009
#      ZS_TaR=2629920 # average monthly time step between 1980 and 2009
#      print('- The time step in rrr_m3r_ncf was determined as: '+str(ZS_TaR)+   \
#           ' seconds')
# else:
#      ZS_TaR=10800
#      print('- No time variables in rrr_m3r_ncf, using default of : '           \
#             +str(ZS_TaR)+' seconds')
     

# #*******************************************************************************
# #Reading observation file and computing average
# #*******************************************************************************
# print('Reading observation file')

# with open(rrr_Qob_csv,'r') as csvfile:
#      csvreader=csv.reader(csvfile)
#      row=next(csvreader)
#      IS_obs_tot=len(row)
# print('- Number of gauges in rrr_Qob_csv: '+str(IS_obs_tot))

# print('- Done')


# #*******************************************************************************
# #Reading available gauges file
# #*******************************************************************************
# print('Reading available gauges file')

# IV_obs_tot_id=[]
# with open(rrr_obs_csv,'r') as csvfile:
#      csvreader=csv.reader(csvfile)
#      for row in csvreader:
#           IV_obs_tot_id.append(int(row[0]))

# if len(IV_obs_tot_id)==IS_obs_tot:
#      print('- Number of river reaches in rrr_obs_csv: '+str(IS_obs_tot))
# else:
#      print('ERROR - the number of reaches in '+rrr_obs_csv+                    \
#            ' differs from the number of gauges in '+rrr_Qob_csv)
#      raise SystemExit(22) 

# IM_hsh_obs={}
# for JS_obs_tot in range(IS_obs_tot):
#      IM_hsh_obs[IV_obs_tot_id[JS_obs_tot]]=JS_obs_tot


# #*******************************************************************************
# #Reading used gauges file
# #*******************************************************************************
# print('Reading used gauges file')

# IV_obs_use_id=[]
# with open(rrr_use_csv,'r') as csvfile:
#      csvreader=csv.reader(csvfile)
#      for row in csvreader:
#           IV_obs_use_id.append(int(row[0]))

# IS_obs_use=len(IV_obs_use_id)
# print('- Number of river reaches in rrr_use_csv: '+str(IS_obs_use))


# #*******************************************************************************
# #Creating selection matrix
# #*******************************************************************************
# print('Creating selection matrix')

# IV_row=[]
# IV_col=[]
# IV_val=[]
# for JS_obs_use in range(IS_obs_use):
#      IV_row.append(JS_obs_use)
#      IV_col.append(IM_hsh_bas[IV_obs_use_id[JS_obs_use]])
#      IV_val.append(1)

# ZM_Sel=csc_matrix((IV_val,(IV_row,IV_col)),shape=(IS_obs_use,IS_riv_bas))

# print('ZM_Sel')
# print(ZM_Sel.shape)


#*******************************************************************************
#Setting reaches to 1
#*******************************************************************************
#### Set ZV_rch to 1 
ZV_rch = [0] * len(IV_riv_tot_id)
for JS_riv_tot_id in range(len(IV_riv_tot_id)):
    #  if IV_riv_tot_id[JS_riv_tot_id] in IV_obs_use_id:
        ZV_rch[JS_riv_tot_id] = 1
ZV_rch = numpy.array(ZV_rch)

ZV_rch=ZV_rch[IV_riv_ix2]


#*******************************************************************************
#Accumulating reaches
#*******************************************************************************
ZV_rch_acc=spsolve(ZM_I-ZM_Net,ZV_rch)
# ZV_num_up_gag=ZM_Sel*ZV_num_up_gag
# subtracting 1 because the result includes the gauge at the current reach
# and we're interested in how many gauges are upstream of that reach
# ZV_num_up_gag=ZV_num_up_gag-1 

ZV_rch_acc=ZV_rch_acc[IV_riv_ix1]

print('ZV_rch_acc')
print(ZV_rch_acc)
print('- Shape of ZV_rch_acc'+str(ZV_rch_acc.shape))
print('- Average ZV_rch_acc at stations: '+str(ZV_rch_acc.mean()))
print('- Minimum ZV_rch_acc at stations: '+str(ZV_rch_acc.min()))
print('- Maximum ZV_rch_acc at stations: '+str(ZV_rch_acc.max()))

print('Writing csv file')

with open(rrr_rch_acc, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, dialect='excel')
    for JS_riv_tot_id in range(len(IV_riv_tot_id)):
        IV_line=[IV_riv_tot_id[JS_riv_tot_id],
                 ZV_rch_acc[JS_riv_tot_id]]
    # for JS_obs_tot_id in range(len(IV_obs_tot_id)):
    #     IV_line=[IV_obs_tot_id[JS_obs_tot_id],
                #  ZV_num_up_gag[JS_obs_tot_id]]
        csvwriter.writerow(IV_line) 


print('Attaching Data to Shp')
data = pd.read_csv(rrr_rch_acc, header=None)
shp = gp.read_file(rrr_top_shp)

results = numpy.zeros(len(shp['reach_id']))
for r in list(range(len(data[0]))):
     rch_idx = numpy.where(shp['reach_id'] == data[0][r])
     results[rch_idx] = data[1][r]

shp['acc'] = results
shp.to_file(rrr_acc_shp)

print('DONE - Routing Results:', rrr_acc_shp)



