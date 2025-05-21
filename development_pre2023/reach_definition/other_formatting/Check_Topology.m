%clear variables.
clear

%set working directory.
cd C:/Users/ealtenau/Documents/Research/SWAG/For_Server/scripts/reach_definition/post_formatting/;

%input/output files.
nc_file = 'C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/Reaches_Nodes/netcdf/oc_sword_v11.nc';

%assigning variables to vectors.
reach_id = ncread(nc_file,'/reaches/reach_id');
n_rch_up = ncread(nc_file,'/reaches/n_rch_up');
n_rch_down = ncread(nc_file,'/reaches/n_rch_down');
rch_id_up = ncread(nc_file,'/reaches/rch_id_up');
rch_id_dn = ncread(nc_file,'/reaches/rch_id_dn');


nReach=length(reach_id);
m=0;
for i=1:nReach
    %check upstream
    for j=1:n_rch_up(i)
       k=find(reach_id==rch_id_up(i,j)); 
       if ~any(rch_id_dn(k,:)==reach_id(i))
           m=m+1;
           disp(['Issue #' num2str(m) ': reach ' num2str(reach_id(i)) ' lists reach ' num2str(rch_id_up(i,j)) ' as being upstream, but ' ...
                 'reach ' num2str(rch_id_up(i,j)) ' does not list reach '  num2str(reach_id(i)) ' as being downstream.' ])
       end
    end
    %check downstream
    for j=1:n_rch_down(i)
       k=find(reach_id==rch_id_dn(i,j)); 
       if ~any(rch_id_up(k,:)==reach_id(i))
           m=m+1;
           disp(['Issue #' num2str(m) ': reach ' num2str(reach_id(i)) ' lists reach ' num2str(rch_id_dn(i,j)) ' as being downstream, but ' ...
                 'reach ' num2str(rch_id_dn(i,j)) ' does not list reach '  num2str(reach_id(i)) ' as being upstream.' ])           
       end
    end
    
end
