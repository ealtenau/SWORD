'''
Try making it a while loop. Will need to keep track of indexes in unq_rch in order to attach
correct geometry to the right reach attributes. 

1) sort unq_rchs by n_rch_down. 
2) for reaches with more than one neighbor, attach current reach and all other
    reaches to first neighbor listed. 
3) flag all other neighbors complete. 
'''

def define_geometry(unq_rch, reach_id_input, cl_x, cl_y, cl_id, region):
    geom = []
    rm_ind = []
    reach_id = np.copy(reach_id_input)
    connections = np.zeros([reach_id.shape[0], reach_id.shape[1]], dtype=int)
    for ind in list(range(len(unq_rch))):
        print(ind, len(unq_rch)-1)
        in_rch = np.where(reach_id[0,:] == unq_rch[ind])[0]
        sort_ind = in_rch[np.argsort(cl_id[in_rch])]
        x_coords = cl_x[sort_ind]
        y_coords = cl_y[sort_ind]

        #appending neighboring reach endpoints to coordinates
        in_rch_up_dn = []
        for ngh in list(range(1,4)):
            neighbors = np.where(reach_id[ngh,:]==unq_rch[ind])[0]
            keep = np.where(connections[ngh,neighbors] == 0)[0]
            in_rch_up_dn.append(neighbors[keep])
        #formating into single list.
        in_rch_up_dn = np.unique([j for sub in in_rch_up_dn for j in sub]) #reach_id[0,in_rch_up_dn]
        #loop through and find what ends each point belong to.
        if len(in_rch_up_dn) > 0:
            end1_dist = []; end2_dist = []
            end1_pt = []; end2_pt = []
            end1_x = []; end2_x = []
            end1_y = []; end2_y = []
            for ct in list(range(len(in_rch_up_dn))):
                x_pt = cl_x[in_rch_up_dn[ct]]
                y_pt = cl_y[in_rch_up_dn[ct]]
                if region == 'AS' and x_pt < 0 and np.min([cl_x[sort_ind[0]], cl_x[sort_ind[-1]]]) > 0:
                    print(unq_rch[ind])
                    continue
                elif region == 'AS' and x_pt > 0 and np.min([cl_x[sort_ind[0]], cl_x[sort_ind[-1]]]) < 0:
                    print(unq_rch[ind])
                    continue
                else:
                    #distance to first and last point. 
                    coords_1 = (y_pt, x_pt)
                    coords_2 = (cl_y[sort_ind[0]], cl_x[sort_ind[0]])
                    coords_3 = (cl_y[sort_ind[-1]], cl_x[sort_ind[-1]])
                    d1 = geopy.distance.geodesic(coords_1, coords_2).m
                    d2 = geopy.distance.geodesic(coords_1, coords_3).m
                    
            ##### NEW START
                    if d1 < d2:
                        end1_pt.append(in_rch_up_dn[ct])
                        end1_dist.append(d1)
                        end1_x.append(x_pt)
                        end1_y.append(y_pt)
                    if d1 > d2:
                        end2_pt.append(in_rch_up_dn[ct])
                        end2_dist.append(d2)
                        end2_x.append(x_pt)
                        end2_y.append(y_pt)

            #append coords to ends
            if len(end1_pt) > 0: #reach_id[:,end1_pt]
                end1_pt = np.array(end1_pt)
                end1_dist = np.array(end1_dist)
                end1_x = np.array(end1_x)
                end1_y = np.array(end1_y)
                sort_ind1 = np.argsort(end1_dist)
                end1_pt = end1_pt[sort_ind1]
                end1_dist = end1_dist[sort_ind1]
                end1_x = end1_x[sort_ind1]
                end1_y = end1_y[sort_ind1]
                if np.min(end1_dist) <= 200:
                    x_coords = np.insert(x_coords, 0, end1_x[0], axis=0)
                    y_coords = np.insert(y_coords, 0, end1_y[0], axis=0)
                if np.min(end1_dist) > 200:
                    ngh_x = cl_x[np.where(reach_id[0,:] == reach_id[0,end1_pt[0]])]
                    ngh_y = cl_y[np.where(reach_id[0,:] == reach_id[0,end1_pt[0]])]
                    d=[]
                    for c in list(range(len(ngh_x))):
                        temp_coords = (ngh_y[c], ngh_x[c])
                        d.append(geopy.distance.geodesic(coords_2, temp_coords).m)
                    if np.min(d) <= 200:
                        append_x = ngh_x[np.where(d == np.min(d))]
                        append_y = ngh_y[np.where(d == np.min(d))]
                        x_coords = np.insert(x_coords, 0, append_x[0], axis=0)
                        y_coords = np.insert(y_coords, 0, append_y[0], axis=0)
                #flag current reach for neighbors.
                ngh1 = reach_id[0,end1_pt[0]]
                connections[:,end1_pt[0]] = 1
                if len(end1_pt) > 1:
                    others = end1_pt[1::]
                    for i in list(range(len(others))):
                        if ngh1 in reach_id[:,others[i]]:
                            continue
                        else:
                            reach_id[3,others[i]] = ngh1
                cols = []
                cols.append(np.where(reach_id[1,:]== ngh1)[0])
                cols.append(np.where(reach_id[2,:]== ngh1)[0])
                cols.append(np.where(reach_id[3,:]== ngh1)[0])
                cols = np.array([item for sublist in cols for item in sublist]) 
                for cs in list(range(len(cols))):
                    if unq_rch[ind] in reach_id[:,cols[cs]]:
                        c = np.where(reach_id[:,cols[cs]] != ngh1)[0]
                        connections[c,cols[cs]] = 1

            if len(end2_pt) > 0: #reach_id[:,end2_pt]
                end2_pt = np.array(end2_pt)
                end2_dist = np.array(end2_dist)
                end2_x = np.array(end2_x)
                end2_y = np.array(end2_y)
                sort_ind2 = np.argsort(end2_dist)
                end2_pt = end2_pt[sort_ind2]
                end2_dist = end2_dist[sort_ind2]
                end2_x = end2_x[sort_ind2]
                end2_y = end2_y[sort_ind2]
                if np.min(end2_dist) < 200:
                    x_coords = np.insert(x_coords, len(x_coords), end2_x[0], axis=0)
                    y_coords = np.insert(y_coords, len(y_coords), end2_y[0], axis=0)
                if np.min(end2_dist) > 200:
                    ngh_x = cl_x[np.where(reach_id[0,:] == reach_id[0,end2_pt[0]])]
                    ngh_y = cl_y[np.where(reach_id[0,:] == reach_id[0,end2_pt[0]])]
                    d=[]
                    for c in list(range(len(ngh_x))):
                        temp_coords = (ngh_y[c], ngh_x[c])
                        d.append(geopy.distance.geodesic(coords_3, temp_coords).m)
                    if np.min(d) <= 200:
                        append_x = ngh_x[np.where(d == np.min(d))]
                        append_y = ngh_y[np.where(d == np.min(d))]
                        x_coords = np.insert(x_coords, len(x_coords), append_x[0], axis=0)
                        y_coords = np.insert(y_coords, len(y_coords), append_y[0], axis=0)
                #flag current reach for neighbors.
                ngh2 = reach_id[0,end2_pt[0]]
                connections[:,end2_pt[0]] = 1
                if len(end2_pt) > 1:
                    others = end2_pt[1::]
                    for i in list(range(len(others))):
                        if ngh2 in reach_id[:,others[i]]:
                            continue
                        else:
                            reach_id[3,others[i]] = ngh2
                cols=[]
                cols.append(np.where(reach_id[1,:]== ngh2)[0])
                cols.append(np.where(reach_id[2,:]== ngh2)[0])
                cols.append(np.where(reach_id[3,:]== ngh2)[0])
                cols = np.array([item for sublist in cols for item in sublist]) 
                for cs in list(range(len(cols))):
                    if unq_rch[ind] in reach_id[:,cols[cs]]:
                        c = np.where(reach_id[:,cols[cs]] != ngh2)[0]
                        connections[c,cols[cs]] = 1

        pts = GeoSeries(map(Point, zip(x_coords, y_coords)))
        if len(pts) <= 1:
            rm_ind.append(ind)
            continue
        else:
            line = LineString(pts.tolist())
            geom.append(line) 

    return geom, rm_ind