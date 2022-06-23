f1 = 'C:\Users\ealtenau\Documents\Research\SWAG\GRWL\masks\GRWL_mask_V01.01_LatLonNames\n64w102.tif'
f2 = 'C:\Users\ealtenau\Documents\Research\SWAG\GRWL\masks\GRWL_mask_V01.01_LatLonNames\n64w096.tif'
mosaic_path = 'C:\Users\ealtenau\Documents\Research\SWAG\GRWL\masks\temp_mos.tif'
prj_outpath = 'C:\Users\ealtenau\Documents\Research\SWAG\GRWL\masks\temp_img.tif'
warp_path = 'C:\Users\ealtenau\Anaconda2\Library\bin\gdalwarp.exe'

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

;; Condition that mosaics multiple tiles together if there is a tile boundary flag (Flags = 5 or 6).
if utm_zone1 eq utm_zone2 then prj_img = f2
mosaic = cgGeoMosaic(f1, prj_img, filename = mosaic_path)
filename = mosaic_path

img = Read_Tiff(filename, GEOTIFF=geotag)
img = Reverse(img, 2)
;iimage, img

img[where(img eq 99)] = 0
img[where(img eq 139)] = 180
img[where(img eq 177)] = 255
img[where(img eq 112)] = 126
img[where(img eq 92)] = 86
iimage, img


