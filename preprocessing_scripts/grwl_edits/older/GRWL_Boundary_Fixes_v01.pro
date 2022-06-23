PRO GRWL_Boundary_Fixes_v01

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; Some of the functions used in this code are from the Coyote Library available
  ;; for download at:  http://www.idlcoyote.com/documents/programs.php
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;Read in GEOTIFF image and associated latitude and longitude arrays.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;!PATH = Expand_Path('+C:\Users\ealtenau\.idl\idl\coyote\') + ';' + !PATH

  tm=systime(1)

  openw, 1, 'C:\Users\ealtenau\Documents\Research\SWAG\GRWL\GRWL_BoundaryFix_Messages_OC_071019.txt'

  mosaic_path = 'C:\Users\ealtenau\Documents\Research\SWAG\GRWL\masks\temp_mos.tif'
  prj_outpath = 'C:\Users\ealtenau\Documents\Research\SWAG\GRWL\masks\temp_img.tif'
  warp_path = 'C:\Users\ealtenau\Anaconda2\Library\bin\gdalwarp.exe'
  
  shp_dir = 'C:\Users\ealtenau\Documents\Research\SWAG\GRWL\GRWL_vector_V01.01_LatLonNames\'
  shp_files = file_search(shp_dir, '*.shp')
  shp_names = strmid(shp_files, 78, 7)

  grwl_dir = 'C:\Users\ealtenau\Documents\Research\SWAG\GRWL\masks\GRWL_mask_V01.01_LatLonNames\'
  grwl_masks = file_search(grwl_dir, '*.tif')
  mask_names = strmid(grwl_masks, 82, 7)

  fix_path = 'C:\Users\ealtenau\Documents\Research\SWAG\GRWL\GRWL_TileFixes_NA.csv'
  fixes = read_csv(fix_path, HEADER=FixHeader)
  ;help, fixes, /STRUCTURES ; shows information on the file structure
  grwl_names = fixes.FIELD6
  grwl_names2 = fixes.FIElD7
  longitude1 = fixes.FIELD2
  latitude1 = fixes.FIELD3
  longitude2 = fixes.FIELD4
  latitude2 = fixes.FIELD5
  flags = fixes.FIELD1

  ;for ind=0, n_elements(mask_names)-1 do begin ;to run all tiles
  for ind=162, 162 do begin ; to adjust range when error occurs
    tm2=systime(1)

    rows = where(grwl_names eq mask_names[ind]) ; identifies the rows in the csv file that correspond to the grwl mask file.

    if min(rows) lt 0 then begin
      printf, 1, FORMAT='("No Fixes Required: ",A7)', mask_names[ind]
      continue
    endif
    
    xcol = List()
    ycol = List()
    val = List()
    lakeFlag = List()
    
    count = 1
    for rw=0, n_elements(rows)-1 do begin

      ;print, 'Row Number = ', rw, systime(1)-tm
      
      ;Skip rows that are inner-tile fixes. 
      if flags[rows[rw]] lt 5 or flags[rows[rw]] gt 6 then begin
        printf, 1, FORMAT='("Not a Tile Boundary Fix: ",A7)', rw
        continue
      endif
      
      ;Find utm zones and mosaic tiles together.
      f1 = grwl_masks[where(mask_names eq grwl_names[rows[rw]])] 
      f2 = grwl_masks[where(mask_names eq grwl_names2[rows[rw]])]
      img1 = read_tiff(f1, GEOTIFF = locs1)
      img2 = read_tiff(f2, GEOTIFF = locs2)
      utm_zone1 = strmid(locs1.GTCITATIONGEOKEY, 20, 3)
      utm_zone2 = strmid(locs2.GTCITATIONGEOKEY, 20, 3)
      
      if utm_zone1 ne utm_zone2 then begin ;mosaicing is easier in same utm zones
        prj1 = string('"+proj=utm +zone=', utm_zone1 +' +ellips=WGS84 +datum=WGS84"')
        prj2 = string('"+proj=utm +zone=', utm_zone2 +' +ellips=WGS84 +datum=WGS84"')
        command = string(warp_path, " -s_srs ", prj2, " -t_srs ", prj1, " -srcnodata ", 256, " -dstnodata ", 99, " -overwrite ", f2, prj_outpath)
        spawn, command
        prj_img = prj_outpath
      endif

      if utm_zone1 eq utm_zone2 then prj_img = f2
      mosaic = cgGeoMosaic(f1, prj_img, filename = mosaic_path)
      filename = mosaic_path

      ;Reading in and loading GRWL Mask.
      img_org = Read_Tiff(filename, GEOTIFF=geotag)
      img_org = Reverse(img_org, 2)
      
      ;Need to correct wonky values in mosaic from 2 different utm zones. 
      img_org[where(img_org eq 99)] = 0
      img_org[where(img_org eq 139)] = 180
      img_org[where(img_org eq 177)] = 255
      img_org[where(img_org eq 112)] = 126
      img_org[where(img_org eq 92)] = 86
      
      img = img_org*0
      img(where(img_org gt 0)) = 1
      s = Size(img, /DIMENSIONS)

      ;Creating x-y UTM arrays. [Coyote Library]
      mapCoord = cgGeoMap(filename, Image= geoImage)
      mapCoord -> GetProperty, XRANGE=xr, YRANGE=yr
      xvec = cgScaleVector(Findgen(s[0]), xr[0], xr[1]) ;xvec = cgScaleVector(Findgen(s[0]+1), xr[0], xr[1])
      yvec = cgScaleVector(Findgen(s[1]), yr[0], yr[1]) ;yvec = cgScaleVector(Findgen(s[1]+1), yr[0], yr[1])
      xarr = x_array(xvec, s)
      yarr = y_array(yvec, s)

      ;Reading in current GRWL Locations.
      fn_shp = shp_files[where(shp_names eq grwl_names[rows[rw]])]
      fn_shp2 = shp_files[where(shp_names eq grwl_names2[rows[rw]])]
      grwl_img = img*0
      grwl = extract_shape(fn_shp)
      grwl_xvals = Value_Locate(xvec, grwl)
      grwl_yvals = Value_Locate(yvec, grwl)
      grwl_x = grwl_xvals[0,*]
      grwl_y = grwl_yvals[1,*]
      grwl_img[grwl_x, grwl_y] = 1
      grwl2 = extract_shape(fn_shp)
      grwl_xvals2 = Value_Locate(xvec, grwl2)
      grwl_yvals2 = Value_Locate(yvec, grwl2)
      grwl_x2 = grwl_xvals2[0,*]
      grwl_y2 = grwl_yvals2[1,*]
      grwl_img[grwl_x2, grwl_y2] = 1
      
      ;Creating new image to fill with fixed points.
      fix_pts = img*0
      filler = img*0
      ;print, 'read in tiff', systime(1)-tm
      
      ; Assign points based of future spreadsheet...
      lon1 = longitude1[rows[rw]] ;-81.82312
      lat1 = latitude1[rows[rw]] ;26.69145
      lon2 = longitude2[rows[rw]] ;-81.82453
      lat2 = latitude2[rows[rw]] ;26.68916 ;30.37379

      ;;; Converting lat lon to utm and finding assosiated pixels. [Coyote Library]
      xy1 = mapCoord -> Forward(lon1, lat1)
      xy2 = mapCoord -> Forward(lon2, lat2)
      x1 = Value_Locate(xvec, xy1[0])
      y1 = Value_Locate(yvec, xy1[1])
      x2 = Value_Locate(xvec, xy2[0])
      y2 = Value_Locate(yvec, xy2[1])

      pt_mask = img*0
      pt_mask[x1,y1] = 1
      pt_mask[x2,y2] = 1

      ;;; If flag indicates linear interpolation...
      ;if flags[rows[rw]] gt 3 then begin
        ;printf, 1, FORMAT='("Tile Boundary Problem [skipped for now]: ",A7, " Row Number = ", I03)', mask_names[ind], rw
        ;continue
      ;endif

      if flags[rows[rw]] eq 6 then begin
        add_pts = straight_line(x1,x2,y1,y2,img)
        add_pts[where(pt_mask eq 1)] = 1

        grwl_edit = grwl_img
        grwl_edit[where(add_pts eq 1)] = 2
        grwl_edit[where(grwl_img eq 1 and grwl_edit eq 2)] = 1
        grwl_edit[where(fix_pts gt 0 and grwl_edit eq 2)] = 1

        new_pts = grwl_edit*0
        new_pts[where(grwl_edit eq 2)] = 1
        lab = label_region(new_pts, /all_neighbors, /ulong)

        if max(lab) gt 1 then begin
          lab_hist = histogram(lab, binsize=1)
          lab_max = max(lab_hist[1:n_elements(lab_hist)-1])
          keep = where(lab_hist eq lab_max)
          new_pts[where(lab ne max(keep))] = 0
          ;grwl_edit[where(new_pts eq 5)] = 0
        endif

        fix_pts[where(new_pts eq 1)] = count
        ;fix_pts[where(add_pts eq 1)] = count
        ;fix_pts[where(pt_mask eq 1)] = count
        
        locs = where(fix_pts gt 0)
        xcol.add, xarr[locs]
        ycol.add, yarr[locs]
        val.add, fix_pts[locs]
        lakeFlag.add, img_org[locs]
        
        count = count+1
        ;iimage, fix_pts
      endif else begin

        ;print, 'Flag = ', flags[rows[rw]], systime(1)-tm

        ;;; Clip image, calculate centerline, and find centerline between two points.
        ;radius_img = img*0
        width = s[0]
        height = s[1]
        w = 100
        x_pts = [x1,x2]
        y_pts = [y1,y2]
        x_min = min(x_pts) - w
        x_max = max(x_pts) + w
        y_min = min(y_pts) - w
        y_max = max(y_pts) + w

        if x_min ge w and x_max le width-1-w and y_min ge w and y_max le height-1-w then dims = [x_min,x_max,y_min,y_max] ; middle
        if x_min lt w and y_min lt w then dims = [x_min-x_min,x_max,y_min-y_min,y_max] ; c1
        if x_max gt width-1-w and y_min lt w then dims = [x_min,width-1,y_min-y_min,y_max] ; c2
        if x_min lt w and y_max gt height-1-w then dims = [x_min-x_min,x_max,y_min,height-1] ; c3
        if x_max gt width-1-w and y_max gt height-1-w then dims = [x_min,width-1,y_min,height-1] ; c4
        if x_min gt w and x_max lt width-1-w and y_min lt w then dims = [x_min,x_max,y_min-y_min,y_max] ; e1
        if x_max gt width-1-w and y_min gt w and y_max lt height-1-w then dims = [x_min,width-1,y_min,y_max] ; e2
        if x_min gt w and x_max lt width-1-w and y_max gt height-1-w then dims = [x_min,x_max,y_min,height-1] ; e3
        if x_min lt w and y_min gt w and y_max lt height-1-w then dims = [x_min-x_min,x_max,y_min,y_max] ; e4

        clip_img = img[dims[0]:dims[1], dims[2]:dims[3]]

        operator=[[1b,1,1],[1,1,1],[1,1,1]]
        if flags[rows[rw]] eq 5 then clip_img = dilate(clip_img, operator)

        if n_elements(where(clip_img gt 0)) eq -1 then begin
          printf, 1, FORMAT='("No Mask Cells - Check Problem [skipped for now]: ",A7, " Row Number = ", I03)', mask_names[ind], rw
          print, FORMAT='("No Mask Cells - Check Problem [skipped for now]: ",A7, " Row Number = ", I03)', mask_names[ind], rw
          continue
        endif

        grwl_clip = grwl_img[dims[0]:dims[1], dims[2]:dims[3]]

        pt_mask_clip = pt_mask[dims[0]:dims[1], dims[2]:dims[3]]
        pts = where(pt_mask_clip eq 1)

        ;Calculate Initial Centerline
        clip_dims = Size(clip_img, /DIMENSIONS)
        wth = clip_dims[0,0]
        hgt = clip_dims[1,0]
        Cline=CL_Calc(clip_img, wth, hgt, tm)

        ; print, 'Initial Cline ', systime(1)-tm

        ;These four lines remove any initial centerline pixels within 12 pixels of the image boundary, because RivWidth produces errors close to the edge.
        Cline[0:wth-1,0:11]=0
        Cline[0:wth-1,hgt-12:hgt-1]=0
        Cline[0:11,0:hgt-1]=0
        Cline[wth-12:wth-1,0:hgt-1]=0

        ;;Selects every pixel in the initial centerline
        input_locs=where(Cline eq 1)
        input_size=size(input_locs, /dimensions)

        y=0l
        r = 20
        for x=0l,input_size[0]-1 do begin
          if (input_locs(x) mod wth)-r ge 0 and (input_locs(x) mod wth)+r lt wth and input_locs(x)/wth-r ge 0 and input_locs(x)/wth+r lt hgt then begin
            input_reg=max(label_region(Cline[(input_locs(x) mod wth)-r:(input_locs(x) mod wth)+r,input_locs(x)/wth-r:input_locs(x)/wth+r], /all_neighbors))
            Cline[input_locs(x)]=0
            interim_reg=max(label_region(Cline[(input_locs(x) mod wth)-r:(input_locs(x) mod wth)+r,input_locs(x)/wth-r:input_locs(x)/wth+r], /all_neighbors))
            if interim_reg eq input_reg then begin
              Cline(input_locs(x))=0
            endif else Cline(input_locs(x))=1
          endif
          y=y+1
        endfor

        ;print, 'Cleaned Cline ', systime(1)-tm

        singlepix=Cline*0
        singlepix[pts[0]]=1
        singlepix[pts[1]]=1
        x_clip = xarr[dims[0]:dims[1], dims[2]:dims[3]]
        y_clip = yarr[dims[0]:dims[1], dims[2]:dims[3]]
        cline_locs = where(Cline eq 1)
        pt_locs = where(singlepix eq 1)
        cl_pts = closest_cline_pts(pt_locs, cline_locs, x_clip, y_clip)

        ;print, 'CL_Points calculated in = ', systime(1)-tm

        ;;; Finding centerline points between
        fixes = Cline*0
        pixels=where(cl_pts eq 1) ;extracts the locations of these single-neighbor pixels
        pix_x=pixels mod wth & pix_y=pixels/wth ;creates x and y coordinates from the image coordinates
        pix_1=transpose(pix_y)
        pix_1(where(pix_1 ge 0))=1 ;pix1 is an array of the same size as pix_x and pix_y, but transposed
        pix_xs=pix_1 ## pix_x  ;this operator creates an image where the x coordinates of each pixel are stored in successive image rows.
        pix_ys=pix_1 ## pix_y
        pix_dist=sqrt((pix_xs-transpose(pix_xs))^2+(pix_ys-transpose(pix_ys))^2) ;calculates distance from each 1-neighbor centerline pixel to every other 1-neighbor centerline pixel.
        max_dist=where(pix_dist eq max(pix_dist)) ;determines the maximum distance between two pixels
        start_end=[pixels[max_dist[0] mod size(pix_x,/dimensions)],pixels[max_dist[0]/size(pix_x,/dimensions)]] ;finds the locations of the starting (sp) and ending (ep) points for the longest centerline segment
        sp=[start_end[1] mod wth, start_end[1]/wth]
        ep=[start_end[0] mod wth, start_end[0]/wth]
        add_to_output=dynamic_vector_v3(Cline, wth, hgt, sp[0], sp[1], ep[0], ep[1]) ;this is a function otherwise discussed below which extracts the coordinates of the shortest path between sp and ep.
        fixes(add_to_output) = 1
        fixes[where(cl_pts eq 1)] = 1
        fixes[where(fixes gt 0 and grwl_clip gt 0)] = 0
        fixes[where(fixes gt 0 and fix_pts[dims[0]:dims[1], dims[2]:dims[3]] gt 0)] = 0
        lab = label_region(fixes, /all_neighbors, /ulong)

        if max(lab) gt 1 then begin
          lab_hist = histogram(lab, binsize=1)
          lab_max = max(lab_hist[1:n_elements(lab_hist)-1])
          keep = where(lab_hist eq lab_max)
          fixes[where(lab ne max(keep))] = 0
        endif

        fixes[where(fixes eq 1)] = count
        filler[dims[0]:dims[1], dims[2]:dims[3]] = fixes
        ;filler[where(filler[dims[0]:dims[1], dims[2]:dims[3]] lt 2)] = 0
        ;filler[where(filler[dims[0]:dims[1], dims[2]:dims[3]] gt 0)] = count
        fix_pts[where(filler gt 0)] = filler[where(filler gt 0)]
        ;fix_pts[where(pt_mask gt 0)] = count
        ;iimage, fix_pts[dims[0]:dims[1], dims[2]:dims[3]], window_title = string(mask_names[ind], ' row = ', rw)
        
        locs = where(fix_pts gt 0)
        xcol.add, xarr[locs]
        ycol.add, yarr[locs]
        val.add, fix_pts[locs]
        lakeFlag.add, img_org[locs]
        
        count = count+1
        ;print, 'fix done in ', systime(1)-tm

      endelse
    endfor

    ;iimage, fix_pts, window_title = string(mask_names[ind])
    ;outpath = string('C:\Users\ealtenau\Documents\Research\SWAG\GRWL\masks\Mask_Edits\',mask_names[ind],'_edit.tif')
    ;write_tiff, outpath, fix_pts, GEOTIFF=geotag
    
    xcol = xcol.toArray(DIMENSION=1)
    ycol = ycol.toArray(DIMENSION=1)
    val = val.toArray(DIMENSION=1)
    lakeFlag = lakeFlag.toArray(DIMENSION=1)
    
    
    if n_elements(val) eq 0 then begin
      printf, 1, FORMAT='("All Inner-Boundary Fixes: ",A7)', mask_names[ind]
      continue
    endif

    outpath = string('C:\Users\ealtenau\Documents\Research\SWAG\GRWL\masks\Mask_Edits\',mask_names[ind],'_bnds.csv')
    NewHeader = ['x','y','seg', 'lakeFlag']
    data = fltarr(4,n_elements(xcol))
    data[0,*] = xcol
    data[1,*] = ycol
    data[2,*] = val
    data[3,*] = lakeFlag

    write_csv, outpath, data, HEADER = NewHeader

    printf, 1, FORMAT='("Tile Edits Finished in: ",A7, F)', mask_names[ind], systime(1)-tm2
    print, FORMAT='("Tile Edits Finished in: ",A7, F)', mask_names[ind], systime(1)-tm2

  endfor

  printf, 1, FORMAT='("All GRWL Tile Edits Finished in: ", F)', systime(1)-tm
  print, FORMAT='("All GRWL Tile Edits Finished in: ", F)', systime(1)-tm
  close, 1

END

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;The following functions are called by the control function above.  Each performs one task in RivWidth:
;CL_Calc:  Takes in the River Mask and calculates an initial centerline
;Dynamic_Vector_v3:  Takes in the initial centerline and user-defined start and endpoints and outputs a one-pixel, directional (start to end) centerline vector.
;Return_Orthogonal_v3:  Takes in the final centerline and determines the line segment orthogonal to the centerline at each centerline pixel
;z_values_v3:  Takes in the orthogonal endpoints and the centerline and calculates the total flow width along each orthogonal
;
;There are also a variety of minor supporting functions below, including eight_vector, getpoints, distance, and bound.  The function read_inputs takes in the parameter file and parses it to provide
;input to the control function above.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;
;CL_Calc is a function, called above, that calculates the initial centerline for the river channel.  This is based on edge-detection methods detailed in Pavelsky and Smith (2008, IEEE GRSL).
;;;;;;;;;;;;;;;;;;

function CL_Calc, img, width, height, tm

  ;1) Create an image (imgbnd) where the nonriver pixels that are adjacent to river pixels have a value of 2, and all other pixels have a value of 0.
  operator=[[1,1,1],[1,0,1],[1,1,1]]
  imgbnd=img
  imgbnd(where(img eq 0 and convol(img,operator) gt 0))=2
  imgbnd(where(imgbnd lt 2))=0

  ;2) Calculate an initial, noneuclidean distance map using the IDL function morph_distance
  MD=morph_distance(img,neighbor_sampling=3)

  ;3) Calculate some initial inputs for the loop to follow that will refine MD into a true euclidean distance map
  pix=long(where(MD gt 2))
  maxpix=size(pix, /dimensions)
  tst=0
  x=0l

  ;4) Calculates a generic, square euclidean distance map that shows the distance from the image center
  MDm=fix(max(MD))+3
  distmap=shift(dist(MDm*2),fix(MDm),fix(MDm))

  ;5) The following loop goes through each river pixel in the image.  For each pixel, it subsets the generic euclidean distance map to be slightly larger than the noneuclidean distance map value.
  ;   This distance map subset is centered on top of the river pixel in question.  Then, the minimum distance from the set of boundary pixels (in imgbnd) that fall within the distance map window is
  ;   determined and assigned to the river pixel under consideration.  This provides the correct, euclidean distance without the necessity of searching through many individual pixel distances.
  while tst eq 0 do begin
    MDv=fix(MD[pix[x]])+2
    distmap2=distmap[MDm-MDv:MDm+MDv,MDm-MDv:MDm+MDv]
    pixx=pix[x] mod width
    pixy=pix[x]/width
    if MDv le pixx and MDv le pixy and pixx+MDv lt width and pixy+MDv lt height then begin
      flipsqr=imgbnd[pixx-MDv:pixx+MDv,pixy-MDv:pixy+MDv]
      if max(flipsqr) lt 2 then print, "error"
      distmap2(where(flipsqr ne 2))=0
    endif
    MD[pixx,pixy]=min(distmap2(where(distmap2 ne 0)))

    if x eq maxpix[0]-1 then tst=1
    x++
  endwhile

  ;6) With the true euclidean distance map computed, we can now calculate the centerline.

  ;6.1) The horizontal and vertical laplacian operators
  operator_hor=[-0.5,0,0.5]
  operator_vert=[[0.5],[0],[-0.5]]

  ;6.2) The two operators above are convolved with the distance map to produce a laplacian map.  River values will be close to 1 except near the river centerline, where they will be near 0.
  MD_Deriv=convol(MD, operator_hor)^2 + convol(MD, operator_vert)^2
  ;print, "MD_Deriv ", MD_Deriv[5088,19565], MD_Deriv[9436,26991]
  MD=0

  ;6.3) The following steps extract the centerline from the raw laplacian image and refine it by removing any centerline pixels not attached to the main centerline.  Note that you can change the
  ;dividing value of MD_Deriv (currently 0.92) to any value you wish.  The higher it is, the more pixels will be classified as "centerline" pixels.  The lower it is, the fewer pixels will be included.
  ;Note that if you lower it too much, then you're liable to get a centerline with many gaps, which is not ideal.

  ;6.3.1) Extracts the initial raw centerline.
  MD_Deriv(where(MD_Deriv gt 0.92 or img eq 0))=2
  MD_Deriv(where(MD_Deriv le 0.92))=1.
  MD_Deriv(where(MD_Deriv eq 2))=0

  ;6.3.2) Labels each contiguous region of centerline pixels.
  Cline_region=label_region(MD_Deriv, /all_neighbors, /ulong)

  ;6.3.3) Removes small centerline regions that clutter up the image (anything smaller than 100 pixels).  Nearly always, the true centerline will be
  ;longer than 100 pixels.
  labelhist=histogram(Cline_region, binsize=1)
  lhlength=size(labelhist, /dimensions)
  maxhist=where(labelhist gt 100 and labelhist ne 0, count)
  for x=0,count-1 do Cline_region(where(Cline_region eq maxhist[x] and img ne 0))=1
  Cline_region(where(Cline_region gt 1))=0
  Cline_region(where(Cline_region eq 1))=1
  Cline_region=byte(Cline_region*MD_Deriv)

  Rough_cl=rivmask_linker(Cline_region,width,height)

  ;7) Returns the finished centerline
  return, Rough_cl

end

function rivmask_linker, w_mask, w, h
  ;This function added in Summer 2012 connects individual centerline segments when, due to strange geometries that come up very occoasionally,
  ;there is a 1-pixel gap that appears in the initial centerline.  It solves the very annoying problem of having to find these minute gaps manually.
  w_bnd=bytarr(w,h)
  operator=[[1,1,1],[1,0,1],[1,1,1]]
  w_bnd(where(w_mask eq 0 and convol(w_mask,operator) gt 0))=1

  label=label_region(w_mask, /all_neighbors)

  fix_pts=reg_test(w,h,w_bnd,label)

  w_mask(where(fix_pts ne 0))=1

  return, w_mask

end

function reg_test, w, h, w_bnd, label
  ;This function is a subfunction of rivmask_linker, above.  It identifies those pixels where there are two or more adjacent centerline pixels
  ;that fall within different centerline regions. In other words, it finds the pixels that link up different centerline regions, thus solving
  ;the annoying problem identified above.

  w_bnd_1=where(w_bnd eq 1)
  w_bnd_size=size(where(w_bnd eq 1),/dimensions)
  w_bnd_len=w_bnd_size[0]

  output=w_bnd-w_bnd

  county=0

  for m=0,w_bnd_len-1 do begin

    if w_bnd_1(m) mod w gt 1 and w_bnd_1(m)/w gt 1 and w_bnd_1(m) mod w lt w-2 and w_bnd_1(m)/w lt h-2 then begin
      fix_hist=histogram(label((w_bnd_1(m) mod w)-1:(w_bnd_1(m) mod w)+1, (w_bnd_1(m)/w)-1:(w_bnd_1(m)/w)+1))
    endif else begin
      fix_hist=bytarr(3,3)
    endelse
    if size(where(fix_hist gt 0), /dimensions) gt 2 then begin
      output(w_bnd_1(m))=1
      county++
    endif

  endfor

  return, output

end



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;Begin Dynamic_Vector Code:  Dynamic vector uses a minimum cost search algorithm to determine the shortest path from
;the start point (spx,spy) to the end point (epx,epy) contained within the initial centerline.
;This should not be edited except in very unusual circumstances
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

function dynamic_vector_v3, input_image, width, height, spx, spy, epx, epy

  ;These lines are no longer needed.  They were required to reconcile ENVI's base-1 dimensions with IDL's base=0 dimensions.
  ;epx=epx-1
  ;epy=epy-1

  ;1) Variables and arrays initialized that will be used in the loop below:
  count=0L ;initializes number of values in queue
  cost=0L ;initializes the cost value
  signal=0 ;a signal variable that is switched to 1 when the destination pixel is reached
  step=bytarr(width,height) ;an array that stores all centerline pixels that have already been searched
  cim=lonarr(width,height,2) ;an array thot holds the x/y coordinates of the adjacent centerline pixel from which the pixel in question was reached.
  ;out_image=bytarr(width,height) ;the image that will contain the final, 1-pixel centerline
  queue=lonarr(height*2,5) ;the queue that contains the pixels to be searched for (epx,epy)

  ;2) Initialize the queueu by adding the start pixel
  queue[count,0:4]=[spx,spy,cost,0,0]  ;initializes the first pixel in the queue with the start pixel
  step[spx,spy]=1
  count=count+1

  ;cur=[0,0,0]

  ;3) Each iteration of the following loop extracts a pixel from the queue and determines if it is the end pixel (epx, epy)
  ;   If it is, then we can output our final centerline.  If not, then we find all of the pixels in an eight-connect neighborhood
  ;   that have not already been added to the queue and add them to the queue.

  while signal ne 1 AND count ne 0 do begin

    cur=reform(queue[0,0:4]) ;extracts the first pixel in the queue)
    cim[cur[0],cur[1],0]=cur[3]  ;keeps track of the prior pixel coordinates so we can determine our path back to (spx,spy)
    cim[cur[0],cur[1],1]=cur[4]
    queue[0:count-1,0:4]=queue[1:count,0:4] ;resets the queue by removing the pixel we are about to consider and shifting all other pixels up one slot
    count=count-1 ;keeps track of the queue length
    ;if count le 0 then print, "Error in centerline.  Check for gaps at ", cur[0], cur[1]
    if cur[0] eq epx AND cur[1] eq epy then begin  ;If we have reached our end pixel (epx,epy), then this runs
      signal=1  ;sets signal variable to stop our while loop
      ;out_image[cur[0],cur[1]]=1  ;assigns a value of 1 to our end pixel (epx,epy) in the output image
      out_array=lonarr(cur[2],2)  ;initializes our output array, which just contains the x,y coordinates of the centerline pixels
      out_array2=lonarr(cur[2])
      larray=[cur[3],cur[4]]  ;initializes larray, which allows us to go back through our search and determine the fastest route back to (spx,spy) from (epx,epy)
      for x=0,cur[2]-1 do begin
        out_array[cur[2]-x-1,0]=larray[0]
        out_array[cur[2]-x-1,1]=larray[1]
        out_array2[cur[2]-x-1]=larray[0]+width*larray[1]
        ;out_image[larray[0],larray[1]]=1
        larray=[cim[larray[0],larray[1],0],cim[larray[0],larray[1],1]]
      endfor
    endif else begin  ;If we have not yet reached our end pixel (epx,epy), then this runs
      ec=eight_vector(cur[0],cur[1],input_image, width, height)  ;determines the number of initial centerline pixels in an 8-connect neighborhood around our current pixel
      if ec ge 1 AND cur[0] ne 0 AND cur[1] ne 0 AND cur[0] ne width-1 ANd cur[1] ne height-1 then begin
        neigh=getpoints(input_image,ec,cur[0],cur[1], cur[2]) ;gets the coordinates of all pixels in an 8-connect neighborhood with values of 1
        for x=0,ec-1 do begin
          if step[neigh[x,0],neigh[x,1]] eq 0 then begin  ;if it hasn't already been added, add each pixel obtained using getpoints to the queue
            queue[count,0:4]=[neigh[x,0],neigh[x,1],neigh[x,2],neigh[x,3],neigh[x,4]]
            step[neigh[x,0],neigh[x,1]]=1
            ;print, queue[0,0], queue[0,1], queue[0,2], queue[0,3], queue[0,4]
            count=count+1
          endif
        endfor
      endif
    endelse
  endwhile

  ;;this outputs an image showing the final one pixel wide center line
  ;openw, 2, FILEPATH('binary_line',ROOT_DIR='e:\MODIS\Lena\Karen\gina\')
  ;
  ;writeu, 2, out_image
  ;
  ;close, 2

  ;4) Return the final centerline array
  return, out_array2

end


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;subfunctions of function dynamic_vector_v3 above
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


function eight_vector, spx, spy, in, width, height

  if spx eq 0 OR spx eq width-1 OR spy eq 0 or SPY eq height-1 then begin
    return, fix(3)
  endif else begin
    eight_conn=in[spx,spy+1]+in[spx,spy-1]+in[spx-1,spy]+in[spx+1,spy]+in[spx-1,spy+1]+in[spx+1,spy+1]+in[spx+1,spy-1]+in[spx-1,spy-1]
    return, eight_conn
  endelse

end

function getpoints, in, ec, cx, cy, cost

  outpoints=lonarr(ec+1,5)
  n=0

  if in[cx+1,cy] eq 1 then begin
    outpoints[n,0:4]=[cx+1,cy,cost+1,cx,cy]
    n++
  endif
  if in[cx-1,cy] eq 1 then begin
    outpoints[n,0:4]=[cx-1,cy,cost+1,cx,cy]
    n++
  endif
  if in[cx,cy+1] eq 1 then begin
    outpoints[n,0:4]=[cx,cy+1,cost+1,cx,cy]
    n++
  endif
  if in[cx,cy-1] eq 1 then begin
    outpoints[n,0:4]=[cx,cy-1,cost+1,cx,cy]
    n++
  endif
  if in[cx+1,cy-1] eq 1 then begin
    outpoints[n,0:4]=[cx+1,cy-1,cost+1,cx,cy]
    n++
  endif
  if in[cx+1,cy+1] eq 1 then begin
    outpoints[n,0:4]=[cx+1,cy+1,cost+1,cx,cy]
    n++
  endif
  if in[cx-1,cy-1] eq 1 then begin
    outpoints[n,0:4]=[cx-1,cy-1,cost+1,cx,cy]
    n++
  endif
  if in[cx-1,cy+1] eq 1 then begin
    outpoints[n,0:4]=[cx-1,cy+1,cost+1,cx,cy]
    n++
  endif

  return, outpoints

end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; OTHER FUNCTIONS
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

function straight_line, x1,x2,y1,y2,img
  s = Size(img, /DIMENSIONS)
  nx=s[0]
  ny=s[1]
  dx=x1-x2
  dy=y1-y2
  n = abs(dx) > abs(dy)

  if abs(dx) gt abs(dy) then begin
    if y1 ge y2 and x1 ge x2 then s=1
    if y1 ge y2 and x1 le x2 then s=-1
    if y1 le y2 and x1 ge x2 then s=1
    if y1 le y2 and x1 le x2 then s=-1
    sy = float( float(y1-y2)/float(abs(dx)) )
  endif else begin
    if x1 ge x2 and y1 ge y2 then sy=1
    if x1 ge x2 and y1 le y2 then sy=-1
    if x1 le x2 and y1 ge y2 then sy=1
    if x1 le x2 and y1 le y2 then sy=-1
    s = float( float(x1-x2)/float(abs(dy)) )
  endelse
  xx = long(lindgen(n+1l)*s+x2)
  yy = long(lindgen(n+1l)*sy+y2)
  if min(xx) lt 0 or max(xx) ge nx or min(yy) lt 0 or max(yy) ge ny then begin
    wherevals=where(xx ge 0 and xx lt nx and yy ge 0 and yy lt ny)
    xx=xx(wherevals)
    yy=yy(wherevals)
  endif

  new_img = img*0
  new_img[xx,yy] = 1

  return, new_img
end

function closest_cline_pts, pt_locs, cline_locs, x_clip, y_clip
  pt1_dist = x_clip*0
  for i=0, (n_elements(cline_locs)-1) do begin
    d = distance(x_clip[pt_locs[0]], y_clip[pt_locs[0]], x_clip[cline_locs[i]], y_clip[cline_locs[i]])
    pt1_dist[cline_locs[i]] = d
  endfor

  pt2_dist = x_clip*0
  for i=0, (n_elements(cline_locs)-1) do begin
    d = distance(x_clip[pt_locs[1]], y_clip[pt_locs[1]], x_clip[cline_locs[i]], y_clip[cline_locs[i]])
    pt2_dist[cline_locs[i]] = d
  endfor

  cp1 = cline_locs[where(pt1_dist(cline_locs) eq min(pt1_dist(cline_locs)))]
  cp2 = cline_locs[where(pt2_dist(cline_locs) eq min(pt2_dist(cline_locs)))]

  if n_elements(cp1) gt 1 then begin
    cp1 = cp1[0]
    ;cp1 = cline_locs[where(pt_dist2[cp1] eq min(pt_dist2[cp1])]
  endif else cp1=cp1

  if n_elements(cp2) gt 1 then begin
    cp2 = cp2[0]
    ;cp2 = cline_locs[where(pt_dist1[cp2] eq min(pt_dist1[cp2])]
  endif else cp2=cp2

  cl_pts = x_clip*0
  cl_pts[cp1] = 1
  cl_pts[cp2] = 1

  return, cl_pts
end

function x_array, xvec, s
  xarr = float(replicate(0,s[0], s[1]))
  for y=0,(s[1]-1) do begin
    xarr[*,y] = xvec[*]
  endfor
  return, xarr
end

function y_array, yvec, s
  yarr = float(replicate(0,s[0], s[1]))
  for x=0,(s[0]-1) do begin
    yarr[x,*] = yvec[*]
  endfor
  return, yarr
end

function distance, ax,ay,bx,by
  dstance=sqrt(long(ax-bx)^2 + long(ay-by)^2)
  return, dstance
end

function extract_shape, fn_shp

  shp = obj_new('IDLffShape', fn_shp)
  shp ->GetProperty, N_ENTITIES=num_ent
  grwl_x = fltarr(num_ent)
  grwl_y = fltarr(num_ent)
  grwl = transpose([[grwl_x], [grwl_y]])

  for x=0L, (num_ent-1) do begin
    attr = shp->GetAttributes(x)
    grwl(0,x) = attr.attribute_0
    grwl(1,x) = attr.attribute_1
  endfor

  return, grwl

end
