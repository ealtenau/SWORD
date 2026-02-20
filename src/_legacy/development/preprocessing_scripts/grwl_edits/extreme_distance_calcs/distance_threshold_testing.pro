PRO distance_threshold_testing

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; Testing the dist_thresh coeficient for SWORD.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;Read in GEOTIFF image, shapefile, and associated latitude and longitude arrays.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  ;!PATH = Expand_Path('path\to\coyote\') + ';' + !PATH
  
  ;tm=systime(1)

  mask_dir = '\inputs\GRWL\lakes_near_rivers\shp\2367_grwl.tif';n40w096_clip.tif' 
  lake_dir = '\inputs\GRWL\lakes_near_rivers\shp\2367_lakes.tif';2543_lakes_raster.tif'
  outpath = '\inputs\GRWL\lakes_near_rivers\shp\2367_dist_thresh_test.tif'
  
  rivers_org = Read_Tiff(mask_dir, GEOTIFF=geotag)
  rivers_org = Reverse(rivers_org, 2)
  rivers = rivers_org*0
  rivers(where(rivers_org gt 0)) = 1
  s1 = Size(rivers, /DIMENSIONS)
  ;rivers = rivers[0:2951,0:3178] ;scene 2543
  
  lakes_org = Read_Tiff(lake_dir, GEOTIFF=geotag2)
  lakes_org = Reverse(lakes_org, 2)
  lakes = lakes_org*0
  lakes(where(lakes_org gt 0)) = 1
  s2 = Size(lakes, /DIMENSIONS)
 
  ;remove lakes that already exist in GRWL.
  lakes[where(rivers eq 1 and lakes eq 1)] = 0
 
  ;Creating x-y UTM arrays. [Coyote Library]
;  mapCoord = cgGeoMap(lake_dir, Image= geoImage)
;  mapCoord -> GetProperty, XRANGE=xr, YRANGE=yr
;  xvec = cgScaleVector(Findgen(s2[0]), xr[0], xr[1]) ;xvec = cgScaleVector(Findgen(s[0]+1), xr[0], xr[1])
;  yvec = cgScaleVector(Findgen(s2[1]), yr[0], yr[1]) ;yvec = cgScaleVector(Findgen(s[1]+1), yr[0], yr[1])
;  xarr = x_array(xvec, s2)
;  yarr = y_array(yvec, s2)
; 
  ;dialating river mask.
  operator=[[1b,1,1],[1,1,1],[1,1,1]]
  rivers_dilate = dilate(rivers, operator)
  rivers_dilate = dilate(rivers_dilate, operator)
  rivers_dilate = dilate(rivers_dilate, operator)
  rivers_dilate = dilate(rivers_dilate, operator)
  rivers_dilate = dilate(rivers_dilate, operator)
  rivers_dilate = dilate(rivers_dilate, operator)
  rivers_dilate = dilate(rivers_dilate, operator)
  rivers_dilate = dilate(rivers_dilate, operator)

  ;checking out data
  fill_img = rivers*0
  fill_img[where(rivers eq 1)] = 1
  fill_img[where(rivers_dilate eq 1 and rivers eq 0)] = 2
  fill_img[where(lakes eq 1)] = 3
  fill_img[where(lakes eq 1 and rivers_dilate eq 1)] = 4
  
  region_img = lakes*0
  region_img[where(fill_img eq 4)] = 1
  region_labels = label_region(region_img)
  
  ;val = region_labels[where(fill_img eq 4)]
  
  ;xcol = xcol.toArray(DIMENSION=1)
  ;ycol = ycol.toArray(DIMENSION=1)
  ;val = val.toArray(DIMENSION=1)
  
  final_img = Reverse(region_labels, 2)
  write_tiff, outpath, final_img, /float, GEOTIFF = geotag
  
END