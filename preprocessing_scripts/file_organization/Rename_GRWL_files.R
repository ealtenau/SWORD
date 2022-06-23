library('sp')
library('raster')
library('rgdal')
library('proj4')
library('maptools')

#grwl_path = 'C:/Users/ealtenau/Documents/Research/SWAG/GRWL/GRWL_vector_V01.01/GRWL_vector_V01.01'

mask_files = list.files('C:/Users/ealtenau/Documents/Research/SWAG/GRWL/masks/GRWL_mask_V01.01/', '.tif', full.names = TRUE)
mask_files = mask_files[-grep('.tif.aux', mask_files)]
mask_files = mask_files[-grep('.tif.ovr', mask_files)]
mask_files = mask_files[-grep('.tif.xml', mask_files)] #14 is not in grwl shapefiles

grwl_files = list.files('C:/Users/ealtenau/Documents/Research/SWAG/GRWL/GRWL_vector_V01.01/GRWL_vector_V01.01/', full.names = TRUE)
grwl_files = grwl_files[-grep('.xml', grwl_files)]

### FOR LOOP
# initialize variables 
original_name = vector()
new_name = vector()
lat_min = vector()
lat_max = vector()
lon_min = vector()
lon_max = vector()

for(i in 1:length(mask_files)){
  fn_mask = mask_files[i]
  pattern = substr(fn_mask,71,74)
  fn_grwl = grwl_files[grep(pattern, grwl_files)]
  
  #grwl = readOGR(grwl_path, pattern)
  
  mask = raster(fn_mask)
  xmin = extent(mask)[1]
  xmax = extent(mask)[2]
  ymin = extent(mask)[3]
  ymax = extent(mask)[4]
  
  prj = crs(mask, asText=TRUE)
  
  pts_utm = data.frame(rbind(c(xmin,ymin), c(xmax, ymax)))  
  pts_prj = project(pts_utm, prj, inverse=TRUE)
  pts_latlon = data.frame(lat=pts_prj$y, lon=pts_prj$x)
  #print(pts_latlon)
  
  lat_min[i] = pts_latlon[1,1]
  lat_max[i] = pts_latlon[2,1]
  lon_min[i] = pts_latlon[1,2]
  lon_max[i] = pts_latlon[2,2]
  
  lat = round(as.numeric(pts_latlon[1,1]), digits = 0)
  lon = round(as.numeric(pts_latlon[1,2]), digits = 0)
  
  if(nchar(as.character(abs(lat))) == 1){
    if(lat > 0){
      lat_prefix = 'n0'
    }
    
    if(lat < 0){
      lat_prefix = 's0'
    }
    
    if(lat == 0){
      lat_prefix = 'n0'
    }
  }
  
  if(nchar(as.character(abs(lat))) == 2){
    if(lat > 0){
      lat_prefix = 'n'
    }
    
    if(lat < 0){
      lat_prefix = 's'
    }
  }
  
  
  if(nchar(as.character(abs(lon))) == 1){
    if(lon > 0){
      lon_prefix = 'e00'
    }
    
    if(lon < 0){
      lon_prefix = 'w00'
    }
    
    if(lon == 0){
      lon_prefix = 'e00'
    }
  }
  
  if(nchar(as.character(abs(lon))) == 2){
    if(lon > 0){
      lon_prefix = 'e0'
    }
    
    if(lon < 0){
      lon_prefix = 'w0'
    }
  }
  
  if(nchar(as.character(abs(lon))) == 3){
    if(lon > 0){
      lon_prefix = 'e'
    }
    
    if(lon < 0){
      lon_prefix = 'w'
    }
  }
  
  name = paste(lat_prefix, as.character(abs(lat)), lon_prefix, as.character(abs(lon)), sep = "")
  mask_outpath = paste('C:/Users/ealtenau/Documents/Research/SWAG/GRWL/masks/', 'GRWL_mask_V01.01_LatLonNames/', name, '.tif', sep = "")
  
  original_name[i] = pattern
  new_name[i] = name
  #print(i)
  
  for(j in 1:length(fn_grwl)){
    fle = fn_grwl[j]
    ext = substr(fle,90,93)
    grwl_outpath = paste('C:/Users/ealtenau/Documents/Research/SWAG/GRWL/GRWL_vector_V01.01_LatLonNames/', name, ext, sep = "")
    file.copy(fle, grwl_outpath)
  }
  
  #file.copy(fn_mask, mask_outpath)
  print(name)
  
}

#name_table = as.data.frame(cbind(original_name, new_name, lon_min, lon_max, lat_min, lat_max))
#colnames(name_table) = c('original_name', 'new_name', 'x_min', 'x_max', 'y_min', 'y_max')
#outpath = 'C:/Users/ealtenau/Documents/Research/SWAG/GRWL/name_table.csv'
#write.csv(name_table, outpath, row.names = FALSE)
