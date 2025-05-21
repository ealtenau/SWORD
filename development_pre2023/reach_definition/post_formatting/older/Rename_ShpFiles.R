library('sp')
library('raster')
library('rgdal')
#library('proj4')
library('maptools')

all_files = list.files('/Users/ealteanau/Documents/SWORD_Dev/outputs/v12/shp/OC/', full.names = TRUE)
rch_files = all_files[grep('_reaches', all_files)]
node_files = all_files[grep('_nodes', all_files)]

### Reaches
for(i in 1:length(rch_files)){
  
  fle = rch_files[i]
  region = substr(fle,58,59)
  name = substr(fle,60,78)
  ext = substr(fle,83,86)
  pattern = paste(region, name, '_v13', ext, sep='')
  outpath = paste('/Users/ealteanau/Documents/SWORD_Dev/outputs/v13/shp/OC/', pattern, sep = "")
  file.copy(fle, outpath)
}

### NODES
for(j in 1:length(node_files)){
  
  fle2 = node_files[j]
  region2 = substr(fle2,58,59)
  name2 = substr(fle2,60,76)
  ext2 = substr(fle2,81,85)
  pattern2 = paste(region2, name2, '_v13', ext2, sep='')
  outpath2 = paste('/Users/ealteanau/Documents/SWORD_Dev/outputs/v13/shp/OC/', pattern2, sep = "")
  file.copy(fle2, outpath2)
}

