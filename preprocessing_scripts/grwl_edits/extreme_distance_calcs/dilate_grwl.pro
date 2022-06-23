PRO dilate_grwl

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;Read in GEOTIFF image and associated latitude and longitude arrays.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;!PATH = Expand_Path('+C:\Users\ealtenau\.idl\idl\coyote\') + ';' + !PATH

  tm=systime(1)

  region = 'OC'

  region_dir = string('C:\Users\ealtenau\Documents\Research\SWAG\For_Server\inputs\GRWL\Edits\', region,'\')
  region_list = file_search(region_dir, '*.shp')
  region_names = strmid(region_list, 74, 7)
  
  grwl_dir = 'C:\Users\ealtenau\Documents\Research\SWAG\GRWL\masks\GRWL_mask_V01.01_LatLonNames\' ;grwl mask directory.
  grwl_masks = file_search(grwl_dir, '*.tif')
  mask_names = strmid(grwl_masks, 82, 7)

  ;assign indexes to run through.
  start_ind = 0 ;to run all rows this should be set to: 0.
  end_ind = n_elements(region_names)-1 ; to run all rows this should be set to: n_elements(mask_names)-1.

  for ind=start_ind, end_ind do begin ; adjust start_ind and end_ind to debug when error occurs.
    print, ind
    
    ;creating empty variables to fill.
    xcol = List()
    ycol = List()
    
    ;Finding grwl mask to read in. 
    current_file = where(mask_names eq region_names[ind])
    
    ;Reading in and loading GRWL Mask.
    img_org = Read_Tiff(grwl_masks[current_file], GEOTIFF=geotag)
    img_org = Reverse(img_org, 2)
    
    ;Creating binary water mask. 
    img = img_org*0
    img(where(img_org gt 0)) = 1
    s = Size(img, /DIMENSIONS)

    ;Creating x-y UTM arrays. [Coyote Library]
    mapCoord = cgGeoMap(grwl_masks[current_file], Image=geoImage)
    mapCoord -> GetProperty, XRANGE=xr, YRANGE=yr
    xvec = cgScaleVector(dindgen(s[0]), xr[0], xr[1]) ;xvec = cgScaleVector(Findgen(s[0]+1), xr[0], xr[1])
    yvec = cgScaleVector(dindgen(s[1]), yr[0], yr[1]) ;yvec = cgScaleVector(Findgen(s[1]+1), yr[0], yr[1])
    xarr = x_array(xvec, s)
    yarr = y_array(yvec, s)
    
    ;convert utm arrays to lat lon arrays.
    latlon =  mapCoord -> Inverse(xarr, yarr)
    arr_lon = latlon[0,*]
    arr_lat = latlon[1,*]
    arr_lon = reform(arr_lon,s[0],s[1])
    arr_lat = reform(arr_lat,s[0],s[1])
    
    ;dialating river mask.
    operator=[[1b,1,1],[1,1,1],[1,1,1]]
    rivers_dilate1 = dilate(img, operator)
    rivers_dilate2 = dilate(rivers_dilate1, operator)
    rivers_dilate3 = dilate(rivers_dilate2, operator)
    rivers_dilate4 = dilate(rivers_dilate3, operator)
    rivers_dilate5 = dilate(rivers_dilate4, operator)
    rivers_dilate6 = dilate(rivers_dilate5, operator)
    rivers_dilate7 = dilate(rivers_dilate6, operator)
    rivers_dilate_final = dilate(rivers_dilate7, operator)
    
    fill_img = img*0
    fill_img[where(img eq 0 and rivers_dilate_final gt 0)] = 1
    
    locs = where(fill_img gt 0)
    xcol.add, arr_lon[locs]
    ycol.add, arr_lat[locs]  
    
    ;Turn lists into array to write out csv.
    xcol = xcol.toArray(DIMENSION=1)
    ycol = ycol.toArray(DIMENSION=1)
    
    outpath = string('E:\Users\Elizabeth Humphries\Documents\SWORD\mask_buffer\',region,'\',region_names[ind],'_250m.csv')
    NewHeader = ['x','y']
    data = dindgen(2,n_elements(xcol))
    data[0,*] = xcol
    data[1,*] = ycol
    write_csv, outpath, data, HEADER = NewHeader

  endfor
 
  print, 'Finsihed dilating grwl masks in: ', systime(1)-tm, ' sec'


END

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

function x_array, xvec, s
  xarr = double(replicate(0,s[0], s[1]))
  for y=0,(s[1]-1) do begin
    xarr[*,y] = xvec[*]
  endfor
  return, xarr
end

function y_array, yvec, s
  yarr = double(replicate(0,s[0], s[1]))
  for x=0,(s[0]-1) do begin
    yarr[x,*] = yvec[*]
  endfor
  return, yarr
end
