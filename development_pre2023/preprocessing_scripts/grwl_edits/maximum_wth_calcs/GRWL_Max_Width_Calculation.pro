PRO GRWL_Max_Width_Calculation

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
  shp_names = strmid(shp_files, 83, 7) ;Identifies the filename (i.e. n40w150). Beginning character (i.e. 82) may need to be changed based on path length. 

  grwl_dir = 'C:\Users\ealtenau\Documents\Research\SWAG\GRWL\masks\GRWL_mask_V01.01_LatLonNames\' ;grwl mask directory.
  grwl_masks = file_search(grwl_dir, '*.tif')
  mask_names = strmid(grwl_masks, 82, 7) ;Identifies the filename (i.e. n40w150). Beginning character (i.e. 82) may need to be changed based on path length. 

  ;assign indexes to run through.
  start_ind = 0 ;to run all rows this should be set to: 0.
  end_ind = n_elements(shp_names)-1 ;n_elements(shp_names)-1 ;to run all rows this should be set to: n_elements(mask_names)-1.

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
    
    ;Reading in current GRWL Locations.
    fn_shp = shp_files[ind]
    grwl = extract_shape(fn_shp)
    grwl_locs = grwl(0:1,*)
    grwl_xvals = Value_Locate(xvec, grwl_locs)
    grwl_yvals = Value_Locate(yvec, grwl_locs)
    grwl_x = grwl_xvals[0,*]
    grwl_y = grwl_yvals[1,*]
    grwl_img = img*0
    grwl_img[grwl_x, grwl_y] = 1
    
    ;Creating new image to fill with fixed points.
    Cline = img*0
    Cline[where(grwl_img eq 1)] = 1
    
    Segs = img*0
    update = where(grwl(2,*) eq 0)
    grwl(2,update) = max(grwl(2,*))+1
    Segs[grwl_x, grwl_y] = grwl(2,*)
    ;iimage, Segs
    
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
  
        
    agval=1 ;set this value to the number of pixels that you want to aggregate.  In other words, if you want mean widths over a length of 10 centerline pixels, then this value should be 10.
    
    ;This calculation of a distance image is no longer used for centerline calculations--instead, it's just used to determine the length of the orthogonal for each channel cross-section.
    dist_im=morph_distance(riv_mask, neighbor_sampling=3)
    ;iimage, dist_im
    
    output = where(Cline eq 1)
    segments = Segs[output]
    vals = segments[UNIQ(segments, SORT(segments))]
    ;print, vals
    num_vals = size(vals, /dimensions)
    ;print, num_vals
    new_segments = cut_segments(segments, vals)
    seg_vals = new_segments[UNIQ(new_segments, SORT(new_segments))]
    num_segs = size(seg_vals, /dimensions)
    
    out_image=fltarr(width,height)
    easting=(lindgen(width,height) mod width)*resolution+ule
    toggle=1
    
    for idx=0, num_segs[0]-1 do begin ;num_vals[0]-1

      subset = where(new_segments eq seg_vals[idx])
      segment_len = n_elements(subset)
      vector_one = output[subset]
      vector_one_size=size(vector_one, /dimensions)
      vector_out=lonarr(vector_one_size,2)
      vector_out[*,0]=vector_one mod width
      vector_out[*,1]=vector_one/width

      ;3.3) Compute the line segment (or, more apprpriately the endpoints of the line segment) orthogonal to the final centerline at each centerline pixel.
      ;Note that it is possible you might want to modify the values "dist" and "seglen" in the function return_orthogonal_v3.
      orth_array=return_orthogonal_v3(vector_out, width, height, agval, dist_im, segment_len)
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
      
      ;This code gets run if we are computing the width for every centerline pixel and not doing any aggregating.  The code below simply outputs the final data to a text file and assigns the appropriate width
      ;values to the output width image (out_image).
      for f=0,zlength-1 do begin
        out_image[zoutput[f,4],zoutput[f,5]]=zoutput[f,6]
      endfor
      
      ;print, "Finished calculating river widths for segment", vals[idx]," out of", max(vals), " in (seconds): ", fix(systime(1)-tm)
      
    endfor

    ;iimage, out_image
    
    ;;;;;; 
    outpath = string('E:\Users\Elizabeth Humphries\Documents\SWORD\GRWL\bank_widths\',shp_names[ind],'_bw.csv') ;creating final outpath.
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

    print, shp_names[ind], " :Finished Tile Bank Width Calculations in: ", systime(1)-tm2
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

function return_orthogonal_v3, input, width, height, agval, distim, segment_len

  ;1) Initialize variables to be used further down in the function

  orth_image=bytarr(width,height) ;intializes an image that contains the centerline pixel and the corresponding orthogonal endpoint pixels

  seglen=11L  ;This is the length of the centerline segment against which the orthogonal is be computed.  It should probably be equal to approximately the river width. Rivwidth is sensitive to this number
  ;only in rivers that are highly sinuous.  Also, see lines 592-596 below.  When working in rivers narrower than seglen, RivWidth will automatically adjust seglen down to reduce potential error
  ;The impact of seglen on width calculations was investigated extensively by Zach Miller in Fall 2012 in the Mississippi Basin.

  seglenh=long((seglen-1)/2)     ;segment length / 2
  n=long(seglenh)
  input_size=size(input, /Dimensions)
  img_size=size(distim, /dimensions)
  cline_length=input_size[0]
  ortharr=lonarr(cline_length,8) ;the final output array listing each centerline pixel coordinates and the coordinates of its two orthogonal endpoints
  if cline_length gt 2500 then clip = (img_size[0]/2)+3000 else clip = 10000 ; added this line to adjust for long segments in grwl... ADDED 5/12/2021.
  ;distfull=shift(dist(1000,1000),500,500) ;This is a generic euclidean distance map showing distance from the image center.  Note that it must be smaller than the image under consideration.
  if width gt clip and height gt clip/2 then begin ; Changed this from 5000 ... need to investigate why... 
    dfval=clip
    distfull=shift(dist(clip,clip),clip/2,clip/2) ;This is a generic euclidean distance map showing distance from the image center.  Note that it must be smaller than the image under consideration.
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
    
    
    x_start = input[n,0]-seglenh
    x_end = input[n,0]+seglenh
    y_start = input[n,1]-seglenh
    y_end = input[n,1]+seglenh
    
    if x_start lt 0 then x_start = 0
    if y_start lt 0 then y_start = 0
    if x_end ge width then x_end = width-1
    if y_end ge height then y_end = height-1
    
    dist=max([fix(fix(max(distim[x_start:x_end,y_start:y_end]))*2*1.25),16])

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
      
      if (epx-cpx) ge dfval/2-orthd2 then xc1 = 0 else xc1 = dfval/2-orthd2-(epx-cpx)
      if xc1 gt dfval-1 then xc1 = dfval-1
      if dfval/2+orthd2-(epx-cpx) ge dfval then xc2 = dfval-1 else xc2 = dfval/2+orthd2-(epx-cpx)
      if dfval/2-orthd2-(epy-cpy) lt 0 then yc1 = 0 else yc1 = dfval/2-orthd2-(epy-cpy)
      if dfval/2-(epy-cpy) ge (dfval/2) then yc2 = (dfval/2)-1 else yc2 = dfval/2-(epy-cpy)
      
      diste=distfull[dfval/2-orthd2:dfval/2+orthd2,dfval/2-orthd2:dfval/2]
      distc=distfull[xc1:xc2,yc1:yc2]
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
      
      if (epx-cpx) ge dfval/2-orthd2 then xc3 = 0 else xc3 = dfval/2-orthd2-(epx-cpx)
      if xc3 gt dfval-1 then xc3 = dfval-1
      if dfval/2+orthd2-(epx-cpx) ge dfval then xc4 = dfval-1 else xc4 = dfval/2+orthd2-(epx-cpx)
      if dfval/2+1-(epy-cpy) lt 0 then yc3 = 0 else yc3 = dfval/2+1-(epy-cpy)
      ;if dfval/2+1+orthd2-(epy-cpy) ge (dfval/2) then yc4 = (dfval/2)-1 else yc4 = dfval/2+1+orthd2-(epy-cpy)
      
      diste=distfull[dfval/2-orthd2:dfval/2+orthd2,dfval/2+1:dfval/2+1+orthd2]
      distc=distfull[xc3:xc4,dfval/2+1-(epy-cpy):dfval/2+1+orthd2-(epy-cpy)]      
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
  
  xout = lonarr(10000,10000) ; need to make these the size of "tot_n"? was lonarr(3022,3022) for all arrays.
  yout = lonarr(10000,10000)
  zout = fltarr(10000,10000)
  rivzout=fltarr(10000,10000)
 
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
      if abs(dx) le 0 AND abs(dy) le 0 then begin
        ;print, 'case 1: only one ortho img'
        s = 1
        sy = 1
      endif else if abs(dx) gt abs(dy) then begin
        ;print, 'case 2: x difference larger'
        if inarr[i,0] ge inarr[i,2] then s=1 else s=-1
        sy = float( float(inarr[i,1]-inarr[i,3])/float(abs(dx)) )
      endif else begin
        ;print, 'case 3: y difference larger'
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

;This funciton calculates and returns the euclidean distance between two points.
function distance, ax,ay,bx,by
  dstance=sqrt(long(ax-bx)^2 + long(ay-by)^2)
  return, dstance
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
