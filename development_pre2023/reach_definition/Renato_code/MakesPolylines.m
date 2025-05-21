%points.mat is a points with the 10k first points in the prior database, read from the shapefiles
%sent by EA in october. This is just for testing
%points=shaperead('./na_apriori_rivers_reaches_highres_v01.shp','Attributes',{'cl_ind','reach_id'});
points=shaperead('./na_apriori_rivers_reaches_highres_v01.shp','Attributes',{'cl_ind'});
reach_id=ncread('./na_apriori_rivers_v02.nc','/centerlines/reach_id');
outfilepath='./shp/'
filename='NA';
numbentries=length(points);
x=zeros(numbentries,1);
y=zeros(numbentries,1);
cl_ind=zeros(numbentries,1);
%reach_id=zeros(numbentries,4);
for ct=1:numbentries
    x(ct)=points(ct).X;
    y(ct)=points(ct).Y;
    cl_ind(ct)=points(ct).cl_ind;
    %reach_id(ct,:)=str2num(points(ct).reach_id);
end
%rch_id_up=ncread('./na_apriori_rivers_v02.nc','/reaches/rch_id_up');
%rch_id_dn=ncread('./na_apriori_rivers_v02.nc','/reaches/rch_id_dn');
reach_id_un=unique(reach_id(:,1));
reach_id_un=reach_id_un(reach_id_un>0);
for ct=1:length(reach_id_un)
    in_reach=find(reach_id(:,1)==reach_id_un(ct)); %position in the vector of points that belong to a reach
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
        for ct2=1:length(in_reach_up_dwn)
            %check if the point is closer to the first or last point in the
            %reach
            x_pt=x(in_reach_up_dwn(ct2));
            y_pt=y(in_reach_up_dwn(ct2));
            %distance to first point
            d1=sqrt((centerline(ct).Lon(1)-x_pt)^2+(centerline(ct).Lat(1)-y_pt)^2);
            d2=sqrt((centerline(ct).Lon(end)-x_pt)^2+(centerline(ct).Lat(end)-y_pt)^2);
            if d1<d2
                centerline(ct).Lon=[x_pt;centerline(ct).Lon];
                centerline(ct).Lat=[y_pt;centerline(ct).Lat];
            else
                centerline(ct).Lon=[centerline(ct).Lon,;x_pt];
                centerline(ct).Lat=[centerline(ct).Lat;y_pt];
            end
        end
    end        
    r=reach_id_un(ct);
    centerline(ct).reach_id=r;
    %pick the last digit in the reach id to set as the lakeflag
%     Lon_dn=x(in_reach_down);
%     Lat_dn=y(in_reach_down);
    lakeflag=r-10*floor(r/10);
    centerline(ct).lakeflag=lakeflag;
end

%write shapefiles

outfilename=[outfilepath filename 'Reachfile'];
shapewrite(centerline,outfilename)
%write projection information
string='GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]';
fileID = fopen([outfilename '.prj'],'w');
fprintf(fileID,string);
fclose(fileID);