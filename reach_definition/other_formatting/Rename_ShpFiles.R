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
  name = substr(fle,100,112)
  ext = substr(fle,117,120)
  pattern = paste(region, '_sword', name, '_v1', ext, sep='')
  outpath = paste('E:/Users/Elizabeth Humphries/Documents/SWORD/Manuscript/SWORD_Version1/shp/OC/', pattern, sep = "")
  #outpath = paste('E:/Users/Elizabeth Humphries/Documents/SWORD/For_Server/outputs/Reaches_Nodes_v10/shp/OC/', pattern, sep = "")
  file.copy(fle, outpath)
}

### NODES
for(j in 1:length(node_files)){
  
  fle2 = node_files[j]
  region2 = substr(fle2,83,84)
  name2 = substr(fle2,100,110)
  ext2 = substr(fle2,115,118)
  pattern2 = paste(region2, '_sword', name2, '_v1', ext2, sep='')
  outpath2 = paste('E:/Users/Elizabeth Humphries/Documents/SWORD/Manuscript/SWORD_Version1/shp/OC/', pattern2, sep = "")
  #outpath2 = paste('E:/Users/Elizabeth Humphries/Documents/SWORD/For_Server/outputs/Reaches_Nodes_v10/shp/OC/', pattern2, sep = "")
  file.copy(fle2, outpath2)
}

