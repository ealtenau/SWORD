library('sp')
library('raster')
library('rgdal')
library('proj4')
library('maptools')

NA_Tiles = read.csv('C:/Users/ealtenau/Documents/Research/SWAG/GRWL/GRWL_NA_Tiles.csv', header = TRUE)
tiles = NA_Tiles$Tile_Name

grwl_files = list.files('C:/Users/ealtenau/Documents/Research/SWAG/GRWL/GRWL_vector_V01.01_LatLonNames/', full.names = TRUE)
#grwl_files = grwl_files[-grep('.xml', grwl_files)]

i = 30

### FOR LOOP
for(i in 1:length(tiles)){
  
  pattern = tiles[i]
  fn_grwl = grwl_files[grep(pattern, grwl_files)]
  
  for(j in 1:length(fn_grwl)){
    fle = fn_grwl[j]
    ext = substr(fle,86,89)
    grwl_outpath = paste('C:/Users/ealtenau/Documents/Research/SWAG/For_Server/inputs/GRWL/NorthAmerica/', pattern, ext, sep = "")
    file.copy(fle, grwl_outpath)
  }
  
  print(i)
  
}





################### editing stuff...
grwl_files = list.files('C:/Users/ealtenau/Documents/Research/SWAG/GRWL/GRWL_vector_V01.01/GRWL_vector_V01.01/', full.names = TRUE)
fn_grwl = grwl_files[grep('NH16', grwl_files)]
fn_grwl = fn_grwl[-grep('.xml', fn_grwl)]

pattern = 'n28w090'

for(j in 1:length(fn_grwl)){
  fle = fn_grwl[j]
  ext = substr(fle,90,93)
  grwl_outpath = paste('C:/Users/ealtenau/Documents/Research/SWAG/For_Server/inputs/GRWL/NorthAmerica/', pattern, ext, sep = "")
  file.copy(fle, grwl_outpath)
}