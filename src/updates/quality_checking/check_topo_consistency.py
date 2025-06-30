# -*- coding: utf-8 -*-
"""
Checking SWORD Topology (check_topo_consistency.py)
=====================
Script for checking the topological consistency of
the SWOT River Database (SWORD). 

The script is run at a regional/continental scale. 
Command line arguments required are the two-letter 
region identifier (i.e. NA) and SWORD version (i.e. v17).

Execution example (terminal):
    python check_topo_consistency.py NA v17 

"""

from __future__ import division
import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)
import numpy as np
import argparse
import time
from src.updates.sword import SWORD 

def check_topology(domain_reachids,domain_reach_data,Output):

    """ 
      check_topology

      Checks to run
      0. Didn't find fields or other basic things 
      1. For each reach, see if the number of reach ids provided matches the provided number of reaches in each direction
      2. For each reach, see if the reaches it says its connected to, check that those reaches say they are connected to it! 
      3. Check to see if any ghost reaches have both upstream and downstream neighors (note this is not an error, just unexpected)
      4. Check for reaches that are NOT ghost reaches but that have no upstream neighbors
      5. A reach has an upstream or downstream neighbor with the same reachid as itself
      
      by Mike Durand, October 2023
      
      Output = 0 : show no errors or warnings
               1 : show errors but not warnings
               2 : show errors and warnings

    """
    
    import numpy as np
    
    ntypes=5
    nerrors=np.zeros( (ntypes,) )
    
    reaches_with_issues=[]

    for reachid in domain_reachids:
        # print(str(reachid))
        reachidstr=str(reachid)
        reachtype=reachidstr[-1]        
       
        # make sure key fields exist  
        errortype=0
        try:
            nup=domain_reach_data[reachidstr]['n_rch_up']
            ndn=domain_reach_data[reachidstr]['n_rch_down']      
        except:
            if Output > 0:
                print('Type 0: Error: did not find one or both of nup or ndn for',reachidstr,'. exiting...')
            nerrors[errortype]+=1
            reaches_with_issues.append(reachidstr)
            continue

        # Check 1
        errortype=1
        ndn_actual=sum(domain_reach_data[reachidstr]['rch_id_dn'] != 0 )
        if ndn_actual != ndn:
            if Output > 0: 
                print('Type 1: Error: ',reachidstr,'claims to have ',ndn,'downstream neighbors, but there are ',ndn_actual,'values listed')
            nerrors[errortype]+=1
            reaches_with_issues.append(reachidstr)
        nup_actual=sum(domain_reach_data[reachidstr]['rch_id_up'] != 0 )    
        if nup_actual != nup:
            if Output > 0:
                print('Type 1: Error: ',reachidstr,'claims to have ',nup,'upstream neighbors, but there are ',nup_actual,'values listed')
            nerrors[errortype]+=1
            reaches_with_issues.append(reachidstr)

        # Check 2
        errortype=2
        for i in range(ndn):
            reachiddnstr=str(domain_reach_data[reachidstr]['rch_id_dn'][i])        

            try:
                rch_id_up=domain_reach_data[reachiddnstr]['rch_id_up']
                
                if not reachid in rch_id_up:
                    if Output > 0: 
                        print('Type 2: Error: ',reachidstr,'says',reachiddnstr,'is a downstream neighbor, but the neighbor designation is unrequited')  
                    nerrors[errortype]+=1
                    reaches_with_issues.append(reachidstr)
                    reaches_with_issues.append(reachiddnstr)

            except:
                if Output > 0:
                    print('Type 2: Error: Could not find reach:',reachiddnstr,'referenced by',reachidstr)
                nerrors[errortype]+=1
                reaches_with_issues.append(reachiddnstr)



        for i in range(nup):
            reachidupstr=str(domain_reach_data[reachidstr]['rch_id_up'][i])

            try:
                rch_id_dn=domain_reach_data[reachidupstr]['rch_id_dn']                
                if not reachid in rch_id_dn:
                    if Output > 0:
                        print('Type 2: Error:',reachidstr,'says',reachidupstr,'is an upstream neighbor, but the neighbor designation is unrequited')
                    nerrors[errortype]+=1               
                    reaches_with_issues.append(reachidstr)
                    reaches_with_issues.append(reachidupstr)
            except:
                if Output > 0:
                    print('Type 2: Error: Could not find reach:',reachidupstr,'referenced by',reachidstr)
                nerrors[errortype]+=1



        # Check 3        
        errortype=3
        if reachtype=='6':
            if domain_reach_data[reachidstr]['n_rch_up'] > 0 and domain_reach_data[reachidstr]['n_rch_down']:
                if Output > 1:
                    print('Type 3: Warning: Ghost reach',reachidstr, 'has both upstream and downstream neighbors')
                    nerrors[errortype]+=1
                
        # Check 4
        errortype=4
        if domain_reach_data[reachidstr]['n_rch_up'] == 0 and reachtype != '6':
            if Output > 1:
                print('Type 4: Warning: Non-ghost reach',reachidstr,'has no upstream neighbors')
                nerrors[errortype]+=1

        # Check 5
        errortype=5
        
    reaches_with_issues=list(set(reaches_with_issues))
        
    return nerrors,reaches_with_issues


###############################################################################################
###############################################################################################
###############################################################################################

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("region", help="<Required> Two-Letter Continental SWORD Region (i.e. NA)", type = str)
parser.add_argument("version", help="version", type = str)
parser.add_argument("subset", nargs="?", default="All", help="basin to subset", type = str)
args = parser.parse_args()

region = args.region
version = args.version
subset = args.subset

#reading data
sword = SWORD(main_dir, region, version)

#subset data to seleted basin. Default is to check entire continent. 
if subset != 'All':
    BasinLevel=len(subset)
    domain_reachids=[]
    for reachid in sword.reaches.id:
        reachidstr=str(reachid)
        if reachidstr[0:BasinLevel] == subset:
            domain_reachids.append(reachid)
    print('there are a total of ',len(domain_reachids),'reaches in SWORD for basin', subset)

else:
    domain_reachids = sword.reaches.id.tolist()

domain_reach_data={}

# create a dictionary keyed off reach data with centerline and reach properties for domain: 
# this takes a second
for reach in domain_reachids:
    reachstr=str(reach)
    # deal with points
    indxs=np.argwhere(sword.centerlines.reach_id[0,:]==int(reach))   
    indxs=indxs[:,0]
    points=[]
    for indx in indxs:           
        points.append(tuple([sword.centerlines.y[indx],sword.centerlines.x[indx]]))       
    # deal with reaches
    # indx = swordreachids.index(int(reach))
    indx = domain_reachids.index(int(reach))
    domain_reach_data[reachstr]={}
    domain_reach_data[reachstr]['clpoints']=points
    domain_reach_data[reachstr]['river_name']=sword.reaches.river_name[indx]
    domain_reach_data[reachstr]['drainage_area_km2']=sword.reaches.facc[indx]
    domain_reach_data[reachstr]['swot_orbits']=sword.reaches.orbits[:,indx]
    domain_reach_data[reachstr]['rch_id_up']=sword.reaches.rch_id_up[:,indx]
    domain_reach_data[reachstr]['rch_id_dn']=sword.reaches.rch_id_down[:,indx]    
    domain_reach_data[reachstr]['n_rch_up']=sword.reaches.n_rch_up[indx]
    domain_reach_data[reachstr]['n_rch_down']=sword.reaches.n_rch_down[indx]    

#run topology checker
Verbose=1
nerrors,reaches_with_issues=check_topology(domain_reachids,domain_reach_data,Verbose)
print('There are a total of ',nerrors[1],'type 1 errors')
print('There are a total of ',nerrors[2],'type 2 errors')

end = time.time()
print(str(np.round((end-start)/60,2))+' mins')
