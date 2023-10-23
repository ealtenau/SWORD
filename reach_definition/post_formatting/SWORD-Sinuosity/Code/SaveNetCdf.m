clear
inputfilelocation='/Users/ealteanau/Documents/SWORD_Dev/inputs/Sinuosity_Files/mat/';
outputlocation='/Users/ealteanau/Documents/SWORD_Dev/inputs/Sinuosity_Files/netcdf/';
debugPlot=0; %1 will make plots of each reach showing the locations of the meanders
Files=dir([inputfilelocation '*.mat']);
NumbFiles=size(Files,1);

for cf=1:6
    load([Files(cf).folder '/' Files(cf).name])
    if ~exist(outputlocation, 'dir')
       mkdir(outputlocation)
    end
    numbnodes=length(Sin_nodes);
    nccreate([outputlocation Files(cf).name(1:end-4) '.nc'],'/nodes/node_id', 'Dimensions', {'num_nodes', numbnodes'});
    ncwrite([outputlocation Files(cf).name(1:end-4) '.nc'],'/nodes/node_id', node_id);
    nccreate([outputlocation Files(cf).name(1:end-4) '.nc'],'/nodes/meanderwavelength', 'Dimensions', {'num_nodes', numbnodes'});
    ncwriteatt([outputlocation Files(cf).name(1:end-4) '.nc'],'/nodes/meanderwavelength','units','meters');
    ncwrite([outputlocation Files(cf).name(1:end-4) '.nc'],'/nodes/meanderwavelength', WaveLength_nodes);
    nccreate([outputlocation Files(cf).name(1:end-4) '.nc'],'/nodes/sinuosity', 'Dimensions', {'num_nodes', numbnodes'});
    ncwriteatt([outputlocation Files(cf).name(1:end-4) '.nc'],'/nodes/sinuosity','units','dimensionless (meters/meter)');
    ncwrite([outputlocation Files(cf).name(1:end-4) '.nc'],'/nodes/sinuosity', Sin_nodes);
    ncwriteatt([outputlocation Files(cf).name(1:end-4) '.nc'],'/','creation_date',datestr(now));
end
    
    