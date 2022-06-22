%this function reads the input database, sets up the calculation, calls the
%main function performing the calculations (SinuosityMinAreaVarMinReach) and
%saves the output
clear
inputfilelocation='C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/Reaches_Nodes/netcdf/';
outputlocation='C:/Users/ealtenau/Documents/Research/SWAG/For_Server/inputs/Sinuosity_Files/mat/';
debugPlot=0; %1 will make plots of each reach showing the locations of the meanders
Files=dir([inputfilelocation '*.nc']);
NumbFiles=size(Files,1);
%for countfiles=1:NumbFiles
for countfiles=3:3
    disp('count files=')
    disp(countfiles)
    filename=Files(countfiles).name;
    %distance between endpoints to calculate sinuosity in km
    Lat = ncread([inputfilelocation filename],'/centerlines/y');
    Lon= ncread([inputfilelocation filename],'/centerlines/x');
    Width=ncread([inputfilelocation filename],'/nodes/width');
    cl_node_id=ncread([inputfilelocation filename],'/centerlines/node_id'); %this field has the node id for each centerline point
    cl_node_id=cl_node_id(:,1);% some centerline vertices belong to more than 1 node, for the purpose of this calculation, I can ignore this
    node_id=ncread([inputfilelocation filename],'/nodes/node_id'); %this field has the node id of each node
    node_reach_id=ncread([inputfilelocation filename],'/nodes/reach_id'); %lets you retrieve which nodes belong to a given reach
    reach_id=ncread([inputfilelocation filename],'/centerlines/reach_id');
    reach_id=reach_id(:,1); %some centerline vertices belong to more than 1 reach. for the purpose of this calculation, I can ignore this
    %Sin=nan(size(Lat));
    Sin_nodes=nan(size(node_id));
    WaveLength_nodes=nan(size(node_id));
    %FlowDist=nan(size(Lat));
    %WaveLength=nan(size(Lat));
    AvailReaches=unique(reach_id);
    dummy=1:length(Lat);
    for count=1:length(AvailReaches) %Cycle throught he available segments
        seg=dummy(reach_id==AvailReaches(count));%current segment
        LatSeg=Lat(seg); 
        LonSeg=Lon(seg);
        Seg_node_id=cl_node_id(seg);
        WidthSeg=zeros(size(Seg_node_id));  
        uniquenodes=unique(Seg_node_id); %unique nodes inside a given reach
        for cn=1:length(uniquenodes)
            WidthSeg(Seg_node_id==uniquenodes(cn))=Width(node_id==uniquenodes(cn));
        end
        %Leopold and Wolman (1960) define sinuosity to be the ratio of arc
        %distance to half the meander length
        %Leopold and Wolman (1960) found meander wave length = 10 river widths 
        %Soar and Thorne (2001) found meander wave length = 10.23 * bankfull
        %width
        %lambda=10*median(WidthSeg(lakeSeg==0)); %Estimate of meander wavelength
        lambda=10*median(WidthSeg); %Estimate of meander wavelength
        %meander wave length = 10.23* bankfull (Soar and Thorne 2001)
        Dist=max(lambda/2,150); %distance used to evaluate sinuosity, measured along the centerline
        %using a mininum distance of 1km to allow at least ~5 points for the calculation of sinuosity
%         MinDist=max(median(WidthSeg(lakeSeg==0)),150);
        MinDist=max(median(WidthSeg),150);
        %[Sin(seg),FlowDist(seg),WaveLength(seg)]=SinuosityMinAreaVarMinReach(LatSeg,LonSeg,WidthSeg,Dist,MinDist,1,1);
        [Sin,~,WaveLength]=SinuosityMinAreaVarMinReach(LatSeg,LonSeg,WidthSeg,Dist,MinDist,1,debugPlot);
        %aggregate to node scale        
        for cn=1:length(uniquenodes)
            %this assumes that I commented the Sin(seg) part and the
            %creation of a variable Sin=nan(size(Lat));
            Sin_nodes(node_id==uniquenodes(cn))=nanmean(Sin(Seg_node_id==uniquenodes(cn)));
            WaveLength_nodes(node_id==uniquenodes(cn))=nanmean(WaveLength(Seg_node_id==uniquenodes(cn)));
        end
    end
    %save(['./output/' filename(1:end-3) 'output.mat'],'Sin', 'WaveLength')
    if ~exist(outputlocation, 'dir')
       mkdir(outputlocation)
    end
    save([outputlocation filename(1:end-3) 'output.mat'],'Sin_nodes', 'WaveLength_nodes','node_id')
end