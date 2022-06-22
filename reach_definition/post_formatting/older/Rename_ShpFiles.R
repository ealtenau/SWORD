library('sp')
library('raster')
library('rgdal')
library('proj4')
library('maptools')

all_files = list.files('C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/Reaches_Nodes/shp/OC/', full.names = TRUE)
rch_files = all_files[grep('_reaches', all_files)]
node_files = all_files[grep('_nodes', all_files)]

### Reaches
for(i in 1:length(rch_files)){
  
  fle = rch_files[i]
  region = substr(fle,83,84)
  name = substr(fle,85,103)
  ext = substr(fle,108,112)
  pattern = paste(region, name, '_v12', ext, sep='')
  outpath = paste('C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/Reaches_Nodes/shp/OC/', pattern, sep = "")
  #outpath = paste('E:/Users/Elizabeth Humphries/Documents/SWORD/For_Server/outputs/Reaches_Nodes_v10/shp/OC/', pattern, sep = "")
  file.copy(fle, outpath)
}

### NODES
for(j in 1:length(node_files)){
  
  fle2 = node_files[j]
  region2 = substr(fle2,83,84)
  name2 = substr(fle2,85,101)
  ext2 = substr(fle2,106,110)
  pattern2 = paste(region2, name2, '_v12', ext2, sep='')
  outpath2 = paste('C:/Users/ealtenau/Documents/Research/SWAG/For_Server/outputs/Reaches_Nodes/shp/OC/', pattern2, sep = "")
  #outpath2 = paste('E:/Users/Elizabeth Humphries/Documents/SWORD/For_Server/outputs/Reaches_Nodes_v10/shp/OC/', pattern2, sep = "")
  file.copy(fle2, outpath2)
}

