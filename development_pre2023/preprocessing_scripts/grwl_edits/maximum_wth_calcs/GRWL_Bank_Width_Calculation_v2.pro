PRO GRWL_Bank_Width_Calculation_v2

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;; Some of the functions used in this code are from the Coyote Library available
  ;; for download at:  http://www.idlcoyote.com/documents/programs.php
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;Read in GEOTIFF image, shapefile, and associated latitude and longitude arrays.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;!PATH = Expand_Path('+C:\Users\ealtenau\.idl\idl\coyote\') + ';' + !PATH

  tm=systime(1)

  ;region = 'NA'

  shp_dir = 'C:\Users\ealtenau\Documents\Research\SWAG\GRWL\shps\GRWL_vector_V01.01_LatLonNames\' ;grwl shapefile directory.
  shp_files = file_search(shp_dir, '*.shp')
  shp_names = strmid(shp_files, 83, 7)

  grwl_dir = 'C:\Users\ealtenau\Documents\Research\SWAG\GRWL\masks\GRWL_mask_V01.01_LatLonNames\' ;grwl mask directory.
  grwl_masks = file_search(grwl_dir, '*.tif')
  mask_names = strmid(grwl_masks, 82, 7)

  ;assign indexes to run through.
  start_ind = 100 ;151 ;to run all rows this should be set to: 0.
  end_ind = 100 ;n_elements(shp_names)-1 ;n_elements(shp_names)-1 ;to run all rows this should be set to: n_elements(mask_names)-1.

  for ind=start_ind, end_ind do begin ; adjust start_ind and end_ind to debug when error occurs.
    tm2=systime(1)
    print, "starting index: ", ind, " of ", n_elements(shp_names)-1
    
    ;Reading in and loading GRWL Mask.
    filename = grwl_masks[where(mask_names eq shp_names[ind])]
    img_org = Read_Tiff(filename, GEOTIFF=geotag)
    img_org = Reverse(img_org, 2)
    img = img_org*0
    img(where(img_org gt 0)) = 1
    ;chan_mask = img
    s = Size(img, /DIMENSIONS)
    height = s[1]
    width = s[0]
    resolution = 30 
    ;iimage, img
    
    ;Creating x-y UTM arrays. [Coyote Library]
    mapCoord = cgGeoMap(filename, Image=geoImage)
    mapCoord -> GetProperty, XRANGE=xr, YRANGE=yr
    xvec = cgScaleVector(Findgen(s[0]), xr[0], xr[1]) ;xvec = cgScaleVector(Findgen(s[0]+1), xr[0], xr[1])
    yvec = cgScaleVector(Findgen(s[1]), yr[0], yr[1]) ;yvec = cgScaleVector(Findgen(s[1]+1), yr[0], yr[1])
    xarr = x_array(xvec, s)
    yarr = y_array(yvec, s)
    ule = min(xvec)
    uln = max(yvec)
    
;    ;Reading in current GRWL Locations.
;    fn_shp = shp_files[ind]
;    grwl = extract_shape(fn_shp)
;    grwl_locs = grwl(0:1,*)
;    grwl_xvals = Value_Locate(xvec, grwl_locs)
;    grwl_yvals = Value_Locate(yvec, grwl_locs)
;    grwl_x = grwl_xvals[0,*]
;    grwl_y = grwl_yvals[1,*]
;    grwl_img = img*0
;    grwl_img[grwl_x, grwl_y] = 1
;    
;    ;Creating new image to fill with fixed points.
;    Cline = img*0
;    Cline[where(grwl_img eq 1)] = 1
;    
;    Segs = img*0
;    update = where(grwl(2,*) eq 0)
;    grwl(2,update) = max(grwl(2,*))+1
;    Segs[grwl_x, grwl_y] = grwl(2,*)
;    ;iimage, Segs
    
;    ;1.4)  Removes all islands from within the mask and uses a series of erosions and dilations to develop the river mask, from which the river center line will be computed
;    operator=[[0b,1,0],[1,0,1],[0,1,0]] ;this is a 4-connect operator that is used for dilations and erosions
;    
    ;flipim is a version of the binary mask in which water=0 and land=1.  It is used to label and remove islands in the river.
    operator=[[0b,1,0],[1,0,1],[0,1,0]]
    if where(img ne 0) ne [-1] then flipim=abs(img-1) else print, "big error"
    labelflip=uint(label_region(flipim, /ulong))

    label=0
    img=0
    p=1
    dcnt=1
   
    ;1.4.2)  Loop that first detects all land areas that intersect the edge of the image.  These are retained as "land" in the river mask.
    ;All other land polygons are considered to be islands in the river and their values are changed accordingly.  With each loop iteration,
    ;we dilate the river area using a 4-connect operator in order to fully bound islands that are incompletely bounded by water in the original
    ;channel mask.  At the moment, this process ony happens once, but this number is arbitrary and can be modified in the last line of the loop (dcnt gt x), below.

    while p eq 1 do begin
      labelhist=histogram(labelflip, binsize=1, /L64)
      lhlength=size(labelhist, /dimensions)

      side1hist=histogram(labelflip[1,0:height-1], binsize=1)
      s1s=size(side1hist, /dimensions)
      side2hist=histogram(labelflip[width-2,0:height-1], binsize=1)
      s2s=size(side2hist, /dimensions)
      side3hist=histogram(labelflip[0:width-1,1], binsize=1)
      s3s=size(side3hist, /dimensions)
      side4hist=histogram(labelflip[0:width-1,height-2], binsize=1)
      s4s=size(side4hist, /dimensions)

      sidespoly=uint([where(side1hist ne 0),where(side2hist ne 0),where(side3hist ne 0),where(side4hist ne 0)])
      sizepoly=size(sidespoly, /dimensions)

      for y=0, sizepoly[0]-1 do begin
        if where(labelflip eq sidespoly[y]) ne [-1L] and sidespoly[y] ne 0 then begin
          labelflip(where(labelflip eq sidespoly[y]))=65535
        endif
      endfor

      if where(labelflip gt 0) ne [-1] then begin
        labelflip(where(labelflip gt 0 and labelflip ne 65535))=0
        labelflip(where(labelflip eq 65535))=1

        riv_maskfl=labelflip
        riv_mask=byte(abs(labelflip-1))

        riv_maskd=dilate(riv_mask,operator)
        flipim=abs(riv_maskd-1)
        labelflip=uint(label_region(flipim, /ulong))
        dcnt++
      endif else p=0

      if dcnt gt 0 then p=0  ;Modify the numerical value in "dcnt gt x" to do either more or fewer dilations
    endwhile

    riv_mask=byte(abs(riv_maskfl-1))
    chan_mask = riv_mask*0
    chan_mask[where(riv_mask gt 0)] = 1 
    ;iimage, chan_mask
    ;iimage, riv_mask
  
    ;calculate initial centerline.  
    Cline=CL_Calc(riv_mask, width, height, tm)    
    agval=1 ;set this value to the number of pixels that you want to aggregate.  In other words, if you want mean widths over a length of 10 centerline pixels, then this value should be 10.
    
    ;This calculation of a distance image is no longer used for centerline calculations--instead, it's just used to determine the length of the orthogonal for each channel cross-section.
    dist_im=morph_distance(riv_mask, neighbor_sampling=3)
    ;iimage, dist_im
    
    ;These four lines remove any initial centerline pixels within 12 pixels of the image boundary, because RivWidth produces errors close to the edge.
    Cline[0:width-1,0:11]=0
    Cline[0:width-1,height-12:height-1]=0
    Cline[0:11,0:height-1]=0
    Cline[width-12:width-1,0:height-1]=0

    ;Selects every pixel in the initial centerline
    input_locs=where(Cline eq 1)
    input_size=size(input_locs, /dimensions)

    y=0l

    ;Calculates the total number of regions in a 21x21 pixel region around each centerline pixel. Then, determines whether removing that pixel increases the number of regions.  If it doesn't, then that pixel is superfluous
    ;to the calculation and is removed.  The end result is an almost entirely 1 pixel wide centerline.
    for x=0l,input_size[0]-1 do begin
      if (input_locs(x) mod width)-10 ge 0 and (input_locs(x) mod width)+10 lt width and input_locs(x)/width-10 ge 0 and input_locs(x)/width+10 lt height then begin
        input_reg=max(label_region(Cline[(input_locs(x) mod width)-10:(input_locs(x) mod width)+10,input_locs(x)/width-10:input_locs(x)/width+10], /all_neighbors))
        Cline[input_locs(x)]=0
        interim_reg=max(label_region(Cline[(input_locs(x) mod width)-10:(input_locs(x) mod width)+10,input_locs(x)/width-10:input_locs(x)/width+10], /all_neighbors))
        if interim_reg eq input_reg then begin
          Cline(input_locs(x))=0
        endif else Cline(input_locs(x))=1
      endif
      y=y+1
    endfor

    ;The following code (through line 278) takes in the thinned initial centerline, finds the largest region of centerline, determines all pixels in it with only one neighbor, and extracts the 1-pixel centerline between
    ;the two pixels with only one neighbor that are the furthest apart by Euclidean distance.  The vector of centerline pixels is added to a master vector, output, with values of -1 separating individual centerline
    ;segments.  This process is then repeated until a threshold is reached in which there are no more regions large enough. Currently this threshold is set to 333 pixels, but that is arbitrary
    ;(333 pixels is equal to ~10km in 30m pixels).

    regions=label_region(Cline, /all_neighbors)
    lrhist=histogram(regions, binsize=1)
    lrhmax=where(lrhist eq max(lrhist(where(lrhist ne max(lrhist)))))
    lrhmax1=lrhmax[0]
    ;outimg=bytarr(width,height)
    output=[0]
    counter=0  ;counter is a variable that is present just to make sure we don't end up in an infinite loop below

    while lrhist[lrhmax1] gt 33 and counter le 200 do begin
      working=bytarr(width,height)
      working(where(regions eq lrhmax1))=1 ;extracts pixels in largest region and adds them to new image called working
      eightreg=convol(working,[[1,1,1],[1,0,1],[1,1,1]], /edge_zero) ;convolves working with a kernel that returns an imagine showing the number of neighboring pixels
      singlepix=bytarr(width,height)
      singlepix(where(eightreg eq 1 and Cline eq 1))=1 ;creates an image where only the centerline pixels with one neighbor (ie potential endpoints) equal 1.

      pixels=where(singlepix eq 1) ;extracts the locations of these single-neighbor pixels
      pix_x=pixels mod width & pix_y=pixels/width ;creates x and y coordinates from the image coordinates
      pix_1=transpose(pix_y)
      pix_1(where(pix_1 ge 0))=1 ;pix1 is an array of the same size as pix_x and pix_y, but transposed
      pix_xs=pix_1 ## pix_x  ;this operator creates an image where the x coordinates of each pixel are stored in successive image rows.
      pix_ys=pix_1 ## pix_y
      pix_dist=sqrt((pix_xs-transpose(pix_xs))^2+(pix_ys-transpose(pix_ys))^2) ;calculates distance from each 1-neighbor centerline pixel to every other 1-neighbor centerline pixel.
      max_dist=where(pix_dist eq max(pix_dist)) ;determines the maximum distance between two pixels
      start_end=[pixels[max_dist[0] mod size(pix_x,/dimensions)],pixels[max_dist[0]/size(pix_x,/dimensions)]] ;finds the locations of the starting (sp) and ending (ep) points for the longest centerline segment

      if start_end[1] eq start_end[0] then print, "Error! You only have one centerline pixel with one neighbor"
      sp=[start_end[1] mod width, start_end[1]/width]
      ep=[start_end[0] mod width, start_end[0]/width]
      add_to_output=dynamic_vector_v3(Cline, width, height, sp[0], sp[1], ep[0], ep[1]) ;this is a function otherwise discussed below which extracts the coordinates of the shortest path between sp and ep.
      output=[output,add_to_output,[-1]] ;adds the locations of all centerline pixels to the vector "output" and adds a value of -1 after it, to show that the end of a segment has been reached.
      ;outimg(add_to_output)=1
      Cline(add_to_output)=0 ;Removes the centerline that has just been calculated from the original centerline image.
      ;The following five lines reset everything for the next iteration of the loop
      regions=label_region(Cline, /all_neighbors)
      lrhist=histogram(regions, binsize=1)
      lrhmax=where(lrhist eq max(lrhist(where(lrhist ne max(lrhist)))))
      lrhmax1=lrhmax[0]
      counter++
    endwhile

    print, "Finished calculating final centerlines in (seconds): ", systime(1)-tm
    iimage, Cline
    
    break
    
    output_size=size(output, /dimensions)
    neg1=where(output eq -1)  ;finds the locations of everywhere in the output vector with a value of -1
    neg1_max=size(neg1, /dimensions)
    z=0l

    openw, 3, rundata.f7  ;opens up a text file to write the final width values into
    printf, 3, "xcoord, ycoord, width, num_channels, easting, northing"  ;prints column headings for the final .csv width output file
    openw, 5, rundata.f6  ;opens up an image file to store the final width image
    out_image=fltarr(width,height)
    easting=(lindgen(width,height) mod width)*resolution+ule
    toggle=1

    while z lt output_size-1 do begin

      ;Divides up the final centerlines into lengths of ~3000 pixels, in cases where they are longer.  This is done because the RivWidth run time increases with increasing centerline segment length.
      ;In other words, RivWidth runs faster with several moderate-length centerline segments than with one very long centerline segment.
      if z+3000 lt min(neg1) then begin
        if toggle eq 1 then vector_one=output[z+1:z+3000] else vector_one=output[z-20:z+3000]
        z=z+3000
        toggle=0
      endif else begin
        vector_one=output[z+1:min(neg1)-1]
        z=min(neg1)
        if size(neg1,/dimensions) gt 1 then neg1=neg1[1:neg1_max-1]
        neg1_max--
        toggle=1
      endelse

      vector_one_size=size(vector_one, /dimensions)
      vector_out=lonarr(vector_one_size,2)
      vector_out[*,0]=vector_one mod width
      vector_out[*,1]=vector_one/width

      ;3.3) Compute the line segment (or, more apprpriately the endpoints of the line segment) orthogonal to the final centerline at each centerline pixel.
      ;Note that it is possible you might want to modify the values "dist" and "seglen" in the function return_orthogonal_v3.
      orth_array=return_orthogonal_v3(vector_out, width, height, agval, dist_im)
      ;print, "Finished calculating orthogonals to centerline in (seconds): ", systime(1)-tm

      ;3.4) Zoutput determines the total flow width along the orthogonal segment for each centerline pixel.
      zoutput=z_values_v3(chan_mask, orth_array, agval, resolution, riv_mask)

      zoutput_size=size(zoutput, /Dimensions)
      zlength=zoutput_size[0]

      agcounter=0
      agaverage=0
      agdist=0
      startpoint=intarr(2)

      ;Very occasionally, on very thin rivers RivWidth will output a width of 0.  In these cases, we assign a width value equal to the pixel resolution.
      zwidths=zoutput[*,6]
      zx=zoutput[*,4]
      zy=zoutput[*,5]
      zwidths(where(zwidths eq 0 and zx ne 0 and zy ne 0))=resolution
      zoutput[*,6]=zwidths

      ;3.5) The following code computes average widths for defined river segment length.  The default length is 1 pixel (ie compute the width at every pixel), but the value of "agval" (Line 192) can be modified to any value
      ;up to the total centerline length.
      if agval gt 1 then begin
        for f=0,zlength-1 do begin
          if zoutput[f,4] eq 0 and zoutput[f,5] eq 0 AND f lt zlength-1 then begin
            startpoint=[zoutput[f+1,4],zoutput[f+1,5]]
          endif
          if zoutput[f,4] gt 0 and zoutput[f,5] gt 0 then begin
            agdist=distance(startpoint[0],startpoint[1],zoutput[f,4],zoutput[f,5])
            if fix(agdist) le agval then begin
              agaverage=agaverage+zoutput[f,6]
              agcounter=agcounter+1
            endif
            if fix(agdist) ge agval then begin
              out_image(zoutput[(f-(fix(agcounter)-1)/2),4],zoutput[(f-(fix(agcounter)-1)/2),5])=agaverage/agcounter
              printf,3,zoutput[(f-(fix(agcounter)-1)/2),4],zoutput[(f-(fix(agcounter)-1)/2),5],agaverage/agcounter,zoutput[f,7], easting[xout,yout], northing[xout,yout],$
                format='((I6),",",(I6),",",(I6),",",(I6),",",(I6),",",(I10))'
              agdist=0
              agcounter=0
              agaverage=0
              startpoint=[zoutput[f,4],zoutput[f,5]]
            endif
          endif
        endfor
      endif else begin
        ;This code gets run if we are computing the width for every centerline pixel and not doing any aggregating.  The code below simply outputs the final data to a text file and assigns the appropriate width
        ;values to the output width image (out_image).
        for f=0,zlength-1 do begin
          if zoutput[f,4] ne 0 or zoutput[f,5] ne 0 then printf,3,zoutput[f,4],zoutput[f,5],zoutput[f,6],zoutput[f,7],easting[zoutput[f,4],zoutput[f,5]],uln-zoutput[f,5]*resolution, $
            format='((I6),",",(I6),",",(I6),",",(I6),",",(I10),",",(I10))'
          out_image[zoutput[f,4],zoutput[f,5]]=zoutput[f,6]
        endfor
      endelse

      print, "Finished calculating river widths for centerline pixels up to", z," out of", output_size[0], " in (seconds): ", fix(systime(1)-tm)

    endwhile
    
    ;;;;;; 
    outpath = string('E:\Users\Elizabeth Humphries\Documents\SWORD\GRWL\GRWL_Bank_Widths\',shp_names[ind],'_bw.csv') ;creating final outpath.
    locs = where(out_image gt 0)

    if min(locs) lt 0 then begin
      print, mask_names[ind], 'No Centerline Pixels'
      continue
    endif

    xcol = xarr[locs]
    ycol = yarr[locs]
    val = out_image[locs]
    NewHeader = ['x','y','bank_wth']
    data = fltarr(3,n_elements(xcol))
    data[0,*] = xcol
    data[1,*] = ycol
    data[2,*] = val

    ;check1 = size(where(Cline gt 0), /dimensions)
    ;check2 = size(locs, /dimensions)
    ;print, check1, check2

    write_csv, outpath, data, HEADER = NewHeader

    print, mask_names[ind], " :Finished Tile Bank Width Calculations in: ", systime(1)-tm2
    print, "max bank width: ", max(out_image[UNIQ(out_image, SORT(out_image))])
  endfor
;
;  printf, 1, FORMAT='("All GRWL Tile Edits Finished in: ", F)', systime(1)-tm
  print, 'all done'
;  close, 1

END

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;The following functions are called by the control function above.  Each performs one task in RivWidth:
;Return_Orthogonal_v3:  Takes in the final centerline and determines the line segment orthogonal to the centerline at each centerline pixel
;z_values_v3:  Takes in the orthogonal endpoints and the centerline and calculates the total flow width along each orthogonal

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;Begin Return Orthogonal Code
;The variabls seglen and dist may be edited.  Everything else should be left as is except under unusual circumstances
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;
;The following functions are called by the control function above.  Each performs one task in RivWidth:
;CL_Calc:  Takes in the River Mask and calculates an initial centerline
;Dynamic_Vector_v3:  Takes in the initial centerline and user-defined start and endpoints and outputs a one-pixel, directional (start to end) centerline vector.
;Return_Orthogonal_v3:  Takes in the final centerline and determines the line segment orthogonal to the centerline at each centerline pixel
;z_values_v3:  Takes in the orthogonal endpoints and the centerline and calculates the total flow width along each orthogonal
;
;There are also a variety of minor supporting functions below, including eight_vector, getpoints, distance, and bound.  The function read_inputs takes in the parameter file and parses it to provide
;input to the control function above.
;;;;;;;;;;;;;;;;;;;

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
MD_Deriv(where(MD_Deriv gt 0.80 or img eq 0))=2
MD_Deriv(where(MD_Deriv le 0.80))=1.
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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;Begin Return Orthogonal Code
;The variabls seglen and dist may be edited.  Everything else should be left as is except under unusual circumstances
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

function return_orthogonal_v3, input, width, height, agval, distim

;1) Initialize variables to be used further down in the function

orth_image=bytarr(width,height) ;intializes an image that contains the centerline pixel and the corresponding orthogonal endpoint pixels

seglen=11L  ;This is the length of the centerline segment against which the orthogonal is be computed.  It should probably be equal to approximately the river width. Rivwidth is sensitive to this number
            ;only in rivers that are highly sinuous.  Also, see lines 592-596 below.  When working in rivers narrower than seglen, RivWidth will automatically adjust seglen down to reduce potential error
            ;The impact of seglen on width calculations was investigated extensively by Zach Miller in Fall 2012 in the Mississippi Basin.

seglenh=long((seglen-1)/2)     ;segment length / 2
n=long(seglenh)
input_size=size(input, /Dimensions)
cline_length=input_size[0]
ortharr=lonarr(cline_length,8) ;the final output array listing each centerline pixel coordinates and the coordinates of its two orthogonal endpoints
;distfull=shift(dist(1000,1000),500,500) ;This is a generic euclidean distance map showing distance from the image center.  Note that it must be smaller than the image under consideration.
if width gt 5000 and height gt 5000 then begin
  dfval=5000
  distfull=shift(dist(5000,5000),2500,2500) ;This is a generic euclidean distance map showing distance from the image center.  Note that it must be smaller than the image under consideration.
endif else begin  ;If the image is smaller than the dimensions above, then the smaller dimension of the input image is used to calculate the generic distance map.
  dfval=min([width,height])
  distfull=shift(dist(dfval,dfval),dfval/2,dfval/2)
endelse

;2) The following code goes through each pixel in the centerline and calculates the coordinates of the orthogonal endpoints using simple trigonometry.
;   Since we know the length of the centerline segment against which we are calculating the orthogonal and we know the length of the orthogonal, we can
;   Compute the length of the hypotenuse formed by these two segments.  We can then find the two points (one on each side of the river) that have the
;   Correct distances from both the centerline pixel and an end of the centerline segment against which the orthogonal will be computed.

while n lt cline_length-seglenh do begin
  
    ;The following section of code adjusts seglen and seglenh downwards if width calculations are being performed in narrow rivers.  A threshold of 11 pixels for seglen was chosen
    ;based on direct experimentation across a wide range of rivers in the Mississippi Basin.  It can be modified as necessary, however.
    if input[n,0] lt width-seglenh and input[n,0] ge seglenh and input[n,1] lt height -seglenh and input[n,1] ge seglenh then $
      maxwval=fix(fix(max(distim[input[n,0]-seglenh:input[n,0]+seglenh,input[n,1]-seglenh:input[n,1]+seglenh])*2)) else maxwval=seglen
    if maxwval lt seglen then begin 
      seglen=long(maxwval) & seglenh=long(maxwval/2)
      if seglen lt 7l then seglen=7l & seglenh=3l
    endif

    ;computes the length of the orthogonal based on the maximum distance value from a window of  the river mask distance map including the centerline pixel
    ; +/- seglenh in each direction.  The orthogonal length is never allowed to be less than 16 pixels to preserve accuracy.
    dist=max([fix(fix(max(distim[input[n,0]-seglenh:input[n,0]+seglenh,input[n,1]-seglenh:input[n,1]+seglenh]))*2*1.25),16])

    ;The hypotenuse of our triangle, set up by our centerline length and our 
    hyp=sqrt((float(seglenh)^2)+(float(dist)^2))

    distdif=5.0
    pt1=intarr(2)
    pt2=intarr(2)
    begin_dist=10
    
    ;2.1) This loop finds the two correct endpoints of the centerline segment against which the orthogonal will be computed 
    for d=0,seglenh do begin
       ;Determines the centerline segment endpoint in the upstream direction
       if distance(input[n-d,0],input[n-d,1],input[n,0],input[n,1]) gt seglenh then begin
         ;Basically iterates downstream until a pixel with a euclidean distance of gt seglenh from the center pixel is found.  
         ;This takes care of the problem that not all pixels are 1 pixel length from one another in the centerline (ie some are diagonal and thus sqrt(2) distant)
         old_begin_dist=begin_dist
         begin_dist=distance(input[n-d,0],input[n-d,1],input[n,0],input[n,1])-seglenh
         if begin_dist lt old_begin_dist and begin_dist ge 0 then begin
          spx=input[n-d,0]
          spy=input[n-d,1]
         endif
       endif else begin
         spx=input[n-seglenh,0]
         spy=input[n-seglenh,1]
       endelse
       
       ;Determines the centerline segment endpoint in the downstream direction
       if distance(input[n+d,0],input[n+d,1],input[n,0],input[n,1]) gt seglenh then begin
          old_begin_dist=begin_dist
          begin_dist=distance(input[n+d,0],input[n+d,1],input[n,0],input[n,1])-seglenh
          if begin_dist lt old_begin_dist and begin_dist ge 0 then begin
           epx=input[n+d,0]
           epy=input[n+d,1]
          endif
       endif else begin
         epx=input[n+seglenh,0]
         epy=input[n+seglenh,1]
       endelse
    endfor
    
    cpx=long(spx+(epx-spx)/2)
    cpy=long(spy+(epy-spy)/2)
    if abs(spx-epx) ge abs(spy-epy) then divd=1
    if abs(spx-epx) lt abs(spy-epy) then divd=0
    orthd=dist
    orthd2=dist+1
    hyp=sqrt((float(seglenh)^2)+(float(orthd)^2))  ; D. Mayer changed hyp to return a floating point number

    ;2.2) The following segment of code determines the x/y coordinates of the orthogonal endpoints

    ;2.2.1) If the centerline vector is moving more in the x direction, then we split the image into two halves horizontally (parallel to the x axis)

    if divd eq 1 then begin
      ;2.2.1.1) Bottom Half of the Image
      diste=distfull[dfval/2-orthd2:dfval/2+orthd2,dfval/2-orthd2:dfval/2]
      distc=distfull[dfval/2-orthd2-(epx-cpx):dfval/2+orthd2-(epx-cpx),dfval/2-orthd2-(epy-cpy):dfval/2-(epy-cpy)]
      disteh=diste-hyp
      distcd=distc-orthd
      
      outpoint=where(abs(disteh) le 0.5 and abs(distcd) le 0.5)
      if outpoint[0] ne long(-1) then outvals=disteh(where(abs(disteh) le 0.5 and abs(distcd) le 0.5)) else outvals=0 ;Added by D.Mayer to match section 2.2.2 below
      finalpts=outpoint(where(outvals eq min(outvals)))
      finalpt=finalpts[0]
      array_size=size(disteh, /dimensions)
      arrayx=array_size[0]
      finx=cpx+long((finalpt mod arrayx)-arrayx/2)
      finy=cpy+long(finalpt/arrayx)-arrayx/2      
      pt1=[finx,finy]

      ;2.2.1.2) Top Half of the Image
      diste=distfull[dfval/2-orthd2:dfval/2+orthd2,dfval/2+1:dfval/2+1+orthd2]
      distc=distfull[dfval/2-orthd2-(epx-cpx):dfval/2+orthd2-(epx-cpx),dfval/2+1-(epy-cpy):dfval/2+1+orthd2-(epy-cpy)]      
      disteh=diste-hyp
      distcd=distc-orthd
      
      outpoint=where(abs(disteh) le 0.5 and abs(distcd) le 0.5)
      if outpoint[0] ne long(-1) then outvals=disteh(where(abs(disteh) le 0.5 and abs(distcd) le 0.5)) else outvals=0 ;Added by D.Mayer to match section 2.2.2 below
      finalpts=outpoint(where(outvals eq min(outvals)))
      finalpt=finalpts[0]
      array_size=size(disteh, /dimensions)
      arrayx=array_size[0]
      arrayy=array_size[1]
      finx=cpx+long((finalpt mod arrayx)-arrayx/2)
      finy=cpy+long(finalpt/arrayx)          
      pt2=[finx,finy]
    endif
    
    ;2.2.2) If the vector is moving more in the y direction, then we split the image into two halves vertically (parallel to the y axis)

    if divd eq 0 then begin
      ;2.2.2.1) Left half of the image
      diste=distfull[dfval/2-orthd2:dfval/2,dfval/2-orthd2:dfval/2+orthd2]
      distc=distfull[dfval/2-orthd2-(epx-cpx):dfval/2-(epx-cpx),dfval/2-orthd2-(epy-cpy):dfval/2+orthd2-(epy-cpy)]      
      disteh=diste-hyp
      distcd=distc-orthd
      
      outpoint=where(abs(disteh) le 0.5 and abs(distcd) le 0.5)
      if outpoint[0] ne long(-1) then outvals=disteh(where(abs(disteh) le 0.5 and abs(distcd) le 0.5)) else outvals=0
      finalpts=outpoint(where(outvals eq min(outvals)))
      finalpt=finalpts[0]
      array_size=size(disteh, /dimensions)
      arrayx=array_size[0]
      arrayy=array_size[1]
      finx=cpx+long(finalpt mod arrayx)-long(arrayx)
      finy=cpy+long(finalpt/arrayx)-long(arrayy/2)  
      pt1=[finx,finy]

      ;2.2.2.2) Right half of the image
      diste=distfull[dfval/2+1:dfval/2+1+orthd2,dfval/2-orthd2:dfval/2+orthd2]
      distc=distfull[dfval/2+1-(epx-cpx):dfval/2+1+orthd2-(epx-cpx),dfval/2-orthd2-(epy-cpy):dfval/2+orthd2-(epy-cpy)]      
      disteh=diste-hyp
      distcd=distc-orthd
      
      outpoint=where(abs(disteh) le 0.5 and abs(distcd) le 0.5)
      if outpoint[0] ne long(-1) then outvals=disteh(where(abs(disteh) le 0.5 and abs(distcd) le 0.5)) else outvals=0 ;Added by D.Mayer to match section 2.2.2.1
      finalpts=outpoint(where(outvals eq min(outvals)))
      finalpt=finalpts[0]
      array_size=size(disteh, /dimensions)
      arrayx=array_size[0]
      arrayy=array_size[1]
      finx=cpx+long(finalpt mod arrayx)
      finy=cpy+long(finalpt/arrayx)-long(arrayy/2)          
      pt2=[finx,finy]
 
    endif
    
    ;The following code only matters if you are computing average width across more than 1 centerline pixel (see the control function for more explanation)
    if (n MOD agval) eq agval/2 AND n ge agval AND bound(pt1[0],pt1[1],width,height) eq 1 and bound(pt2[0],pt2[1],width,height) eq 1 then begin
       orth_image[pt1[0],pt1[1]]=3
       orth_image[pt2[0],pt2[1]]=2
       plotptsx=[pt1[0],pt2[0]]
       plotptsy=[pt1[1],pt2[1]]
       orth_image[input[n,0],input[n,1]]=1
    endif
    
    ortharr[n,0:7]=[pt1[0],pt1[1],pt2[0],pt2[1],input[n,0],input[n,1],cpx,cpy]
    n=n+1
    
    ;if n mod 100 eq 0 then print, "Finished calcuating ", n, " of ", final-seglenh, " orthogonals"
endwhile

;;shows the center line along with the points used to compute the orthogonal
;openw, 1, FILEPATH('cline_plus_orthpoints_newtest2',ROOT_DIR='/Volumes/Data/files/Alaska/Yukon_width/')
;
;writeu, 1, orth_image
;
;close, 1

return, ortharr  ;Returns an array containing the x,y coordinates of the centerline pixels along with the endpoints of the orthogonals.  
                 ;This is the primary input for z_values_v3, below.  It is assigned to the variable inarr in z_values_v3.
end

;;;;;;;;;
;Subfunctions of function Return_orthogonal_v3 above
;;;;;;;;;

;This funciton calculates and returns the euclidean distance between two points.
function distance, ax,ay,bx,by
    dstance=sqrt(long(ax-bx)^2 + long(ay-by)^2)
    return, dstance
end

;This function determines whether a pixel lies on a boundary of the image.
function bound, a,b,width,height
  if a lt 0 OR a gt width-1 then return,0
  if b lt 0 OR b gt height-1 then return,0
  return, 1
end

;;;;;;;;;;;;;;;;;;;
;Begin Z_values section
;This section should be edited only under very unusual circumstances.
;;;;;;;;;;;;;;;;;;;

function z_values_v3, inim, inarr, agval, pixel_size, rivmask

;1)  Initializes a whole bunch of variables used further down
inarr_size=size(inarr, /Dimensions)
tot_n=inarr_size[0]
inim_size=size(inim, /Dimensions)
nx=inim_size[0]
ny=inim_size[1]
finalout=lonarr(tot_n,8)
finalimage=lonarr(nx,ny)

xout = lonarr(3022,3022)
yout = lonarr(3022,3022)
zout = fltarr(3022,3022)
rivzout=fltarr(3022,3022)

u=1

;2) The following for loop goes through each centerline pixel and extracts the line of pixels from the channel mask that falls along the
;   orthogonal to that pixel.  So what you end up with are three arrays (xout, yout, and zout) which contain, respectively, the x coordinate
;   y coordinate, and value (either 0 or 1) of each pixel in the orthogonal.  Note on attribution:  this loop is heavily modified from some
;   code originally by Doug Alsdorf.

for i=0, tot_n-1 do begin
    if (inarr[i,0] eq 0 AND inarr[i,1] eq 0) OR (inarr[i,2] eq 0 AND inarr[i,3] eq 0) AND i lt tot_n/2 then u=i
    if (inarr[i,0] gt 0 OR inarr[i,1] gt 0) AND (inarr[i,2] gt 0 OR inarr[i,3] gt 0) then begin
         dx=inarr[i,0]-inarr[i,2]
         dy=inarr[i,1]-inarr[i,3]
         n = abs(dx) > abs(dy)
         if abs(dx) gt abs(dy) then begin
            if inarr[i,0] ge inarr[i,2] then s=1 else s=-1
            sy = float( float(inarr[i,1]-inarr[i,3])/float(abs(dx)) )
         endif else begin
            if inarr[i,1] ge inarr[i,3] then sy=1 else sy=-1
            s = float( float(inarr[i,0]-inarr[i,2])/float(abs(dy)) )
         endelse
         xx = long(lindgen(n+1l)*s+inarr[i,2])
         yy = long(lindgen(n+1l)*sy+inarr[i,3])
         if min(xx) lt 0 or max(xx) ge nx or min(yy) lt 0 or max(yy) ge ny then begin
            wherevals=where(xx ge 0 and xx lt nx and yy ge 0 and yy lt ny)
            xx=xx(wherevals)
            yy=yy(wherevals)
         endif
         xysize=size(yy, /dimensions)
         xylen=xysize[0]
         zz = inim[long(yy)*nx + xx]
         rivzz = rivmask[long(yy)*nx + xx]
         
         xout[i,0:xylen-1]=xx & yout[i,0:xylen-1]=yy & zout[i,0:xylen-1]=zz & rivzout[i,0:xylen-1]=rivzz

     endif
endfor

;3) The following while loop goes through each pixel in the centerline and adds up the euclidean distance across each channel intersected by each orthogonal cross section.
;   What we get in the end is an array, finalout, which contains the x/y coordinates of the two orthogonal pixels and the centerline pixel (positions 0-5 in the second dimension
;   of finalout), the total flow width in whatever units the resolution of the images is provided in (position 6), and the total number of channels crossed (position 7).

while u lt tot_n-agval-1 do begin
    counter=0
    s=0
    switch_var=zout[u,0] ;switch variable
    switch_count=0 ;switch variable counter
    sv2=rivzout[u,0] ;switch variable 2
    if sv2 gt 1 then sv2=1
    svc=0 ;switch value 2 counter
    start_pixel=lonarr(2)    
    
    for t=0,s do begin
       counter=0
       tot_dist=double(0.0)
       start_pixel=[xout[u+t,0],yout[u+t,0]]
       rivlabel=fltarr(tot_n/2)
       rivdist=fltarr(tot_n/2)
       for f=0,tot_n/2-1 do begin
        if rivzout[u+t,f] gt 1 then rivzout[u+t,f]=1
        if rivzout[u+t,f] eq sv2 then begin
          sv2=abs(sv2-1)
          svc++
        endif
        if rivzout[u+t,f] eq 1 then begin
          rivlabel[f]=svc
          rivdist[f]=distance(xout[u+t,f],yout[u+t,f],inarr[u+t,6],inarr[u+t,7])
          if xout[u+t,f] eq inarr[u+t,6] and yout[u+t,f] eq inarr[u+t,7] then rivdist[f]=-1
        endif
       endfor
       
       ;if there is more than one section of the river mask crossed by the orthogonal, then this section makes sure that only the section that includes the centerline pixel is measured.  This
       ;helps to deal with very sinuous rivers in which it's easy for the orthogonal to cross multiple meanders.
       if svc gt 3 then begin
        rightlabel=rivlabel(where(rivdist eq min(rivdist(where(rivdist ne 0)))))
        rightloc=where(rivdist eq min(rivdist(where(rivdist ne 0))))
        zvals=reform(zout[u+t,*])
        zvalssafe=zvals
        rivzvals=reform(rivzout[u+t,*])
        if rivzvals(rightloc[0]) eq 1 then zvals(where(rivlabel ne rightlabel[0]))=0 else print, "Error in correcting width.  Value for point ", inarr[u+t,4], inarr[u+t,5], " is unreliable."
        zout[u+t,*]=zvals
       endif else zvalssafe=reform(zout[u+t,*])
                  
       for z=0,tot_n/2-1 do begin
         if zout[u+t,z] eq switch_var then begin
          switch_var=abs(switch_var-1)
          switch_count++
          if switch_var eq 1 AND xout[u+t,z] gt 0 then begin
              tot_dist=tot_dist+distance(start_pixel[0],start_pixel[1],xout[u+t,z],yout[u+t,z])             
          endif
          start_pixel[0]=xout[u+t,z]
          start_pixel[1]=yout[u+t,z]         
         endif        
       endfor
       zout[u+t,*]=zvalssafe
       n_channel=switch_count/2
       finalout[u+t,0]=inarr[u+t,0]
       finalout[u+t,1]=inarr[u+t,1]
       finalout[u+t,2]=inarr[u+t,2]
       finalout[u+t,3]=inarr[u+t,3]
       finalout[u+t,4]=inarr[u+t,4]
       finalout[u+t,5]=inarr[u+t,5]
       finalout[u+t,6]=tot_dist*pixel_size
       finalout[u+t,7]=n_channel
    endfor

    u=u+s+1
    if u ge tot_n then u=tot_n-1

endwhile

;;this image shows the boundaries of the cross-sectional windows used to compute width
;openw, 6, FILEPATH('image_crossections_ohio_40',ROOT_DIR='/Files/SWOT/Rivwidth/')
;
;writeu, 6, finalimage
;
;close, 6

return, finalout

end


function read_inputs,inputparams

;Function to read the input data from the parameter file
;  by Mike Durand, April 29, 2008

;1) Open the input file
openr,1, inputparams

;2) Read the input file
while eof(1) eq 0 do begin
  ;2.1) Declare buffer as an empty string 
  buffer=''
  ;2.2) Read one line of input from the file
  readf,1,buffer
  ;2.3) Split the line into the name/value pair
  args=strsplit(buffer,/extract)
  arg_name=args(0)
  arg_value=args(1)
  
  ;2.4) Query each of the names, and copy values to correct varaibles
  ;2.4.1) Image details
  if(strmatch(arg_name,'width') eq 1) then width=long(arg_value)
  if(strmatch(arg_name,'height') eq 1) then height=long(arg_value)
  if(strmatch(arg_name,'resolution') eq 1) then resolution=uint(arg_value)
  if(strmatch(arg_name,'braidnarrowflag') eq 1) then braidnarrowflag=uint(arg_value)
  if(strmatch(arg_name,'ul_easting') eq 1) then ul_easting=long(arg_value)
  if(strmatch(arg_name,'ul_northing') eq 1) then ul_northing=long(arg_value)
  if(strmatch(arg_name,'UTM_zone') eq 1) then UTM_zone=uint(arg_value)
  ;2.4.2) Input file name - river classification
  if(strmatch(arg_name,'water_class') eq 1) then water_class_name=arg_value
  if(strmatch(arg_name,'riv_class') eq 1) then  riv_class_name=arg_value
  if(strmatch(arg_name,'chan_class') eq 1) then  chan_class_name=arg_value   
  ;2.4.3) Output file names
  ;if(strmatch(arg_name,'dist_image') eq 1) then dist_img_name=arg_value
  if(strmatch(arg_name,'cl_image') eq 1) then cl_img_name=arg_value
  ;if(strmatch(arg_name,'cl1_image') eq 1) then cl_img_name1=arg_value
  if(strmatch(arg_name,'w_image') eq 1) then w_img_name=arg_value
  if(strmatch(arg_name,'w_csv') eq 1) then w_csv_name=arg_value
  ;2.4.4) Switches to control program flow    
  if(strmatch(arg_name,'calc_masks') eq 1) then dist_sw=uint(arg_value)
  if(strmatch(arg_name,'calc_centerline') eq 1) then cl_sw=uint(arg_value)
  if(strmatch(arg_name,'calc_width') eq 1) then w_sw=uint(arg_value)
  ;2.4.5) Start and stop image pixels from ENVI
  ;if(strmatch(arg_name,'spx') eq 1) then spx=uint(arg_value)
  ;if(strmatch(arg_name,'spy') eq 1) then spy=uint(arg_value)
  ;if(strmatch(arg_name,'epx') eq 1) then epx=uint(arg_value)
  ;if(strmatch(arg_name,'epy') eq 1) then epy=uint(arg_value)  
    
endwhile

;3) Close input file
close, 1

;4) Copy input variables to a structure, and return it
;rundata={w:width,h:height,r:resolution,b:braidnarrowflag,$
;          f0:water_class_name,f1:riv_class_name,f2:chan_class_name,f3:dist_img_name,f4:cl_img_name,f5:cl_img_name1,$
;          f6:w_img_name,f7:w_csv_name,s1:dist_sw,s2:cl_sw,s3:w_sw,spx:spx,spy:spy,epx:epx,epy:epy}
rundata={w:width,h:height,r:resolution,b:braidnarrowflag,ule:ul_easting, uln:ul_northing, zone:UTM_zone,$
          f0:water_class_name,f1:riv_class_name,f2:chan_class_name,f4:cl_img_name,$
          f6:w_img_name,f7:w_csv_name,s1:dist_sw,s2:cl_sw,s3:w_sw}
return, rundata

end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; OTHER FUNCTIONS
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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

function extract_shape, fn_shp

  shp = obj_new('IDLffShape', fn_shp)
  shp ->GetProperty, N_ENTITIES=num_ent
  grwl_x = fltarr(num_ent)
  grwl_y = fltarr(num_ent)
  segs = fltarr(num_ent)
  grwl = transpose([[grwl_x], [grwl_y], [segs]])

  for x=0L, (num_ent-1) do begin
    attr = shp->GetAttributes(x)
    grwl(0,x) = attr.attribute_0
    grwl(1,x) = attr.attribute_1
    grwl(2,x) = attr.attribute_4
  endfor

  return, grwl

end

function cut_segments, segments, vals

  new_segs = segments
  
  for idx=0L, (n_elements(vals)-1) do begin
    max_seg = max(new_segs)+1
    seg = where(segments eq vals[idx])
    if n_elements(seg) gt 3000 then begin
      divisions = round(float(n_elements(seg))/3000)
      for d=0L, (max(divisions)-1) do begin
        if d eq 0 then start_ind = 0 else start_ind = 3000*d
        if d eq (max(divisions)-1) then end_ind = (n_elements(seg)-1) else end_ind = 3000*(d+1)
        new_segs[seg[start_ind:end_ind]] = max_seg
        max_seg = max(new_segs)+1 
      endfor
    endif
  endfor

  return, new_segs

end
