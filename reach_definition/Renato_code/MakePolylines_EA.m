%clear variables.
clear

%set working directory.
cd C:/Users/ealtenau/Documents/Research/SWAG/For_Server/scripts/

%input/output files.
nc_file = '../outputs/Reaches_Nodes/netcdf/na_apriori_rivers_v01.nc';
outfilepath='../outputs/Reaches_Nodes/shp/NA/';
filename='na_apriori_rivers_reaches_hb';
version = '_v01';

%assigning variables to vectors.
cl_ind_all = ncread(nc_file,'/centerlines/cl_id');
reach_id_all = ncread(nc_file,'/centerlines/reach_id');
x_all = ncread(nc_file,'/centerlines/x');
y_all = ncread(nc_file,'/centerlines/y');
rch_id_all = ncread(nc_file,'/reaches/reach_id');
reach_len_all = ncread(nc_file,'/reaches/reach_length');
n_nodes_all = ncread(nc_file,'/reaches/n_nodes');
wse_all = ncread(nc_file,'/reaches/wse');
wse_var_all = ncread(nc_file,'/reaches/wse_var');
width_all = ncread(nc_file,'/reaches/width');
wth_var_all = ncread(nc_file,'/reaches/width_var');
n_chan_max_all = ncread(nc_file,'/reaches/n_chan_max');
n_chan_mod_all = ncread(nc_file,'/reaches/n_chan_mod');
grod_id_all = ncread(nc_file,'/reaches/grod_id');
slope_all = ncread(nc_file,'/reaches/slope');
dist_out_all = ncread(nc_file,'/reaches/dist_out');
n_rch_up_all = ncread(nc_file,'/reaches/n_rch_up');
n_rch_down_all = ncread(nc_file,'/reaches/n_rch_down');

mod_ids_cl = mod(reach_id_all(:,1), 1000000000);
level2_cl = (reach_id_all(:,1)-mod_ids_cl)./1000000000;
mod_ids_rch = mod(rch_id_all, 1000000000);
level2_rch = (rch_id_all-mod_ids_rch)./1000000000;
uniq_level2 = unique(level2_cl);
%loop through every level 2 basin and output shapefiles. 
for ind=1:length(uniq_level2)    
    
    tic
    disp(['Starting Basin: ' num2str(uniq_level2(ind))])
    
    l2_cl = find(level2_cl == uniq_level2(ind));
    l2_rch = find(level2_rch == uniq_level2(ind));

    cl_ind = cl_ind_all(l2_cl);
    reach_id = reach_id_all(l2_cl);
    x = x_all(l2_cl);
    y = y_all(l2_cl);
    rch_id = rch_id_all(l2_rch);
    reach_len = reach_len_all(l2_rch);
    n_nodes = n_nodes_all(l2_rch);
    wse = wse_all(l2_rch);
    wse_var = wse_var_all(l2_rch);
    width = width_all(l2_rch);
    wth_var = wth_var_all(l2_rch);
    n_chan_max = n_chan_max_all(l2_rch);
    n_chan_mod = n_chan_mod_all(l2_rch);
    grod_id = grod_id_all(l2_rch);
    slope = slope_all(l2_rch);
    dist_out = dist_out_all(l2_rch);
    n_rch_up = n_rch_up_all(l2_rch);
    n_rch_down = n_rch_down_all(l2_rch);

    %finding unique reaches to loop through
    reach_id_un=unique(reach_id(:,1));
    reach_id_un=reach_id_un(reach_id_un>0);
    centerline = struct;
    for ct=1:length(reach_id_un) %test index -> ct=5381, 1358, 1344
        in_reach=find(reach_id(:,1)==reach_id_un(ct)); %position in the vector of points that belong to a reach
        var_ct = find(rch_id(:,1)==reach_id_un(ct)); %subset for variables
        %cl_ind when sorted gives you the rigth order of points, so I'm sorting them and using that order to sort the lon and lat vectors
        [~,sort_ind]=sort(cl_ind(in_reach));
        Lon=x(in_reach);
        Lat=y(in_reach);     
        centerline(ct).Geometry='PolyLine';
        centerline(ct).BoundingBox=[[min(Lon) min(Lat)]; [max(Lon) max(Lat)]];
        centerline(ct).Lon=Lon(sort_ind);%save in the vector that will be used in the shapefiles
        centerline(ct).Lat=Lat(sort_ind);
        in_reach_up_dwn=[];
        for ct2=2:4
            %see if there are points that belong to this reach in other
            %positions of reach_id
            in_reach_up_dwn=[in_reach_up_dwn;find(reach_id(:,ct2)==reach_id_un(ct))];
        end
        if ~isempty(in_reach_up_dwn)
            for ct3=1:length(in_reach_up_dwn)
                %check if the point is closer to the first or last point in the
                %reach
                x_pt=x(in_reach_up_dwn(ct3));
                y_pt=y(in_reach_up_dwn(ct3));
                %distance to first point
                d1=deg2km(distance(centerline(ct).Lat(1), centerline(ct).Lon(1), y_pt, x_pt))*1000;
                d2=deg2km(distance(centerline(ct).Lat(end), centerline(ct).Lon(end), y_pt, x_pt))*1000;
                %don't connect if minimum distance greater than 200 meters.
                if min(d1, d2) > 200
                    continue
                end
                if d1<d2
                    centerline(ct).Lon=[x_pt;centerline(ct).Lon];
                    centerline(ct).Lat=[y_pt;centerline(ct).Lat];
                end    
                if d1>d2
                    centerline(ct).Lon=[centerline(ct).Lon;x_pt];
                    centerline(ct).Lat=[centerline(ct).Lat;y_pt];
                end    
            end
        end        
        r=reach_id_un(ct);
        centerline(ct).reach_id=num2str(r);
        centerline(ct).reach_len=reach_len(var_ct);
        centerline(ct).n_nodes=n_nodes(var_ct);
        centerline(ct).wse=wse(var_ct);
        centerline(ct).wse_var=wse_var(var_ct);
        centerline(ct).width=width(var_ct);
        centerline(ct).width_var=wth_var(var_ct);
        centerline(ct).n_chan_max=n_chan_max(var_ct);
        centerline(ct).n_chan_mod=n_chan_mod(var_ct);
        centerline(ct).grod_id=grod_id(var_ct);
        centerline(ct).slope=slope(var_ct);
        centerline(ct).dist_out=dist_out(var_ct);
        centerline(ct).n_rch_up=n_rch_up(var_ct);
        centerline(ct).n_rch_dn=n_rch_down(var_ct);
        %disp(ct) %print loop index
    end

    %write shapefiles
    outfilename=[outfilepath filename num2str(uniq_level2(ind)) version];
    shapewrite(centerline,outfilename)
    %write projection information
    string='GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]';
    fileID = fopen([outfilename '.prj'],'w');
    fprintf(fileID,string);
    fclose(fileID);

    toc
end