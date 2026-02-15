SWOT River Database (SWORD)

Product Description Document
Release v17b
March 2025

Elizabeth H. Altenau (ealtenau@unc.edu)
University of North Carolina at Chapel Hill, Chapel Hill, NC

Tamlin M. Pavelsky
University of North Carolina at Chapel Hill, Chapel Hill, NC

Michael T. Durand
The Ohio State University, Columbus, OH

Elyssa Collins
University of North Carolina at Chapel Hill, Chapel Hill, NC

SWORD Product Description Document v17
____________________________________________________________________________________

Table of Contents

1. Versions
2. Overview
3. Data Sources
4. Data Formats
4.1 NetCDF
4.2 Geopackage
4.3 Shapefiles

5. Summary Figures and Statistics
6. References

2
5
5
6
6
21
21
27
31

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

1

SWORD Product Description Document v17
____________________________________________________________________________________

1. Versions

●  Beta v0.1

o  Produced: September 2019
o  First beta version of SWORD

●  Beta v0.2

o  Produced: December 2019
o  Added ghost nodes and reaches to the database (Type = 6).

●  Beta v0.3

o  Produced: March 2020
o  Adjusted the “wth_coef” values from 0.5 to 1.

●  Beta v0.4

o  Produced: April 2020
o  Fixed formatting issues with multi-dimensional variables (/centerlines/rch_id,

/centerlines/node_id, /nodes/cl_ids, /reaches/cl_ids/, reaches/rch_id_up, /reaches/rch_id_down)
that occurred between v02 and v01.

●  Beta v0.5

o  Produced: April 2020
o  Added discharge groups (reaches/MetroMan, reaches/BAM, reaches/HiVDI, reaches/MOMMA,

reaches/SADS) and associated variables.

●  Beta v0.6

o  Produced: June 2020
o  Improved topology algorithm.
o  Improved ghost node identification.
o  Added unconstrained/constrained discharge variables to netCDFs.
o  Updated GROD dataset used in reach definition.
o  Added “ext_dist_coef” attribute to dataset for improving errors caused by lakes-near-rivers in

SWOT pixel cloud.

●  Beta v0.7

o  Produced: September 2020
o  Added flow accumulation values to prior reach and node products.
o  Added low permeable dam and waterfalls to reach definition.

●  Beta v0.8

o  Produced: November 2020
o  Improved local topology algorithm.
o  Improved distance-from-outlet algorithm.

●  Beta v0.9 (Public v0)

o  Produced: January 2021
o  Adjusted reach definition to produce slightly longer reaches on average.
o  Added GROD and HydroFALLS ids to products.
o  Included SWOT pass and number of observations in shapefile products.
o  Added climatological ice flag values to netcdf.

●  Release v10 (Public v1)

o  Produced: June 2021
o  Corrected missing GROD locations around the equator.
o  First round of manual edits to fix incorrect reach geometries.
o  Added “max_width” to the netCDF attributes.

●  Release v11

o  Produced: July 2021

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

2

SWORD Product Description Document v17
____________________________________________________________________________________

o  Included the Prior Lake Database (PLD) information in reach definition.
o  Added “river_name” and “sbQ_rel” discharge parameter to netCDF.
o  Improved centerline representations in coastal/estuary areas.
o  Filled in width=1 node values with reach width values.
o  Added “manual_add” node attribute for identifying the nodes that were manually added to the

original GRWL centerlines.

●  Release v12 (Public v2)

o  Produced: October 2021
o  Added “meander_length” and “sinuosity” attributes to the nodes.
o  Improved centerline representations in Peace-Athabasca Delta.

●  Release v13

o  Produced: July 2022
o  Added SIC4DVar discharge group and associated variables.
o  Added “low_slope_flag” to reaches group.
o  Adjusted “ext_dist_coef” values for several reaches on the Yukon River.

●  Release v14

o  Produced: November 2022
o  Improved topology in reaches covered by the 1-day CalVal SWOT orbit.
o  Custom reach definition for several Tier1 CalVal sites: Willamette River, Tanana River,

Connecticut River, Peace Athabasca Delta, North Saskatchewan River, Sagavanirktok River,
Waimak River.

o  Removed duplicated reaches around EU / AS border.
o  Added geopackage file format.

●  Release v15

o  Produced: February 2023
o  Corrected topology in the world’s large river systems (i.e. Mississippi, Yukon, Congo, Amazon,

etc.).

o  Centerline adjustments to several rivers:

▪  Yukon, Kuskokwim, Slave, Mississippi, Atchafalaya, Ob, and Amur Rivers: Update

requests by Bo Wang and Laurence Smith

▪  Weser, Rhine, and Elbe Rivers: Update requests by Luciana Fenoglio

▪  Centerline additions and connectivity improvements in large PLD lakes: Update requests

by Jida Wang

▪  Amazon River delta additions: Updates by Elizabeth Altenau

o  Added a tributary flag (“trib_flag”) to the reach and node attributes. The tributary flag indicates
whether a larger river identified in MERIT Hydro-Vector, but not in SWORD, is entering a
reach or node.

●  Release v15b

o  Produced: April 2023
o  Reduced default (max) “ext_dist_coef” value from 20 to 5.
o  Corrected “edit_flag” values in custom calval basins.

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

3

SWORD Product Description Document v17
____________________________________________________________________________________

●  Release v16

o  Produced: August 2023
o  Continued topology updates in the high priority level 2 basins.
o  Reach ID corrections at the level 6 basin level. ~5% of Reach IDs changed between v15 and

v17.

o  Removed reaches consisting of a single 30 m centerline point.
o  Updated ext_dist_coef values to be static for all nodes within reaches that have many water

bodies near the river edge.

●  Release v17

o  October 2024
o  Topological updates to ensure consistency.
o  Distance from outlet recalculation based on shortest paths between outlets and headwaters.
o  New variables for nodes and reaches: “path_freq”, “path_order”, “path_segs”, “main_side”,

“stream_order”, “end_reach”,” network”.
o  Improved geometry for reach shapefiles.
o  Additional channels added for improved network connectivity.
o  New reach and node ids that better reflect the improved topology.
o  Corrected node lengths to match reach lengths when summed.

●  Release v17b

o  March 2025
o  Type change for 1662 reaches and associated nodes globally.
o  Updates to reach and node lengths and distance from outlet to correct bug in node length

calculation in select reaches (<2% globally).

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

4

SWORD Product Description Document v17
____________________________________________________________________________________

2. Overview

The upcoming Surface Water and Ocean Topography (SWOT) satellite mission, planned to
launch in 2022, will vastly expand observations of river water surface elevations (WSE),
inundation extent, and slope [Biancamaria et al., 2016]. For practical interpretation and
application of SWOT measurements, a global prior database of river networks and reaches is
required. The SWOT River Database (SWORD) was built to support the development of
RiverObs, the central algorithm that will process SWOT pixel cloud data into vector products. A
major purpose for SWORD is to provide fixed node locations, reach boundaries, and
high-resolution reach centerlines in a way that facilitates the generation of SWOT vector
products. SWORD provides high-resolution river nodes (200 m) and reaches (~10 km) in vector
and netCDF formats with attached hydrologic variables (WSE, width, slope, etc.) as well as a
consistent topological system for global rivers 30 m wide and greater.

3. Data Sources

SWORD is generated by combining multiple global hydrography databases into one congruent
product. This section briefly describes the main data sources that are used in the development of
SWORD. Table 1 provides a summary of data sets and the attributes they contribute to the final
product. For detailed information regarding the development of SWORD see [Altenau et al.,
2021].

Table 1: Summary of data sets used in the development of SWORD.

Dataset

Attribute Contribution

Global River Widths from Landsat (GRWL)
[Allen & Pavelsky, 2018]

MERIT Hydro
[Yamazaki et al., 2019]

HydroBASINS
[Lehner & Grill, 2013]

Provides river centerline locations at 30 m
resolution and associated width, water body type,
and number of channels attributes.

Provides elevation and flow accumulation at 3
arc-second resolution (~90 m at the equator).

Provides Pfafstetter nested basin codes up to level
6.

Global River Obstruction Database (GROD)
[Whittemore et al., 2020]

Provides global locations of anthropogenic river
obstructions along the GRWL river network.

Global Delta Maps
[Tessler et al., 2015]

Provides the spatial extent of 48 of the world’s
largest river deltas.

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

5

SWORD Product Description Document v17
____________________________________________________________________________________

SWOT Orbits
[https://www.aviso.altimetry.fr/en/missions/future-missions/s
wot/orbit.html]

Provides polygons containing SWOT track
coverage for each pass throughout the 21-day
cycle orbit.

HydroFALLS
[http://wp.geog.mcgill.ca/hydrolab/hydrofalls/]

Provides global locations of waterfalls and natural
river obstructions.

4. Data Formats

SWORD data is provided in netCDF, geopackage, and shapefile formats. NetCDF and
geopackage files are distributed at continental scales, while shapefiles are split into level 2 basins
within each continent. HydroBASINS Pfafstetter codes have a total of 9 continental regions,
however, for ease of distribution some HydroBASINS regions are grouped under a single
continent in the SWORD database. In these cases, the original HydroBASINS Pfafstetter codes
remain the same. For example, HydroBASINS Pfafstetter levels representing North America (7),
the Arctic (8) and Greenland (9) are all grouped under the North American identifier (‘na’).
Additionally, HydroBASINS Pfafstetter levels representing Asia (4) and Siberia (3) are grouped
under the Asia identifier (‘as’) in the SWORD database. File syntax denotes the regional
information for each file and is described for each format below.

4.1 NetCDF

A comprehensive version of SWORD is available in netCDF format. Each netCDF file contains
a set of global attributes and three groups of variables for the different spatial scales of the
database (/centerlines, /nodes, /reaches). Table 3 provides descriptions of the groups and
variables in the netCDF files. NetCDF file names are distributed at continental scales and are
defined by a two-digit identifier (Table 2): [continent]_sword_v1.nc (i.e. na_sword_v17.nc).

Table 2: Continent identifiers in the SWORD database.

Identifier

Continent

na

eu

as

sa

af

oc

North America

Europe/Middle East

Asia

South America

Africa

Oceania

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

6

SWORD Product Description Document v17
____________________________________________________________________________________

Table 3: NetCDF variable and attribute descriptions. (*These variables contain fill values).

Group

Variable

Description

Units

Dimensions

global attributes

Name

none

N/A

2 letters identifying the
continent (NA – North
America, SA – South
America, AF – Africa, EU –
Europe/Middle East, AS –
Asia, OC –
Australia/Oceania).

x_min, x_max,
ymin, y_max

Bounding box of longitudes
and latitudes included in a file.
Note that files may have
overlapping boxes.

decimal
degrees

N/A

production date

Date when the files were
generated.

none

N/A

/centerlines

cl_id

high-resolution centerline
point id

none

[number of points]

x

y

reach_id

longitude of the point ranging
from 180°E to 180°W

decimal
degrees

[number of points]

latitude of the point ranging
from 90°S to 90°N

decimal
degrees

[number of points]

none

[4, number of points]

id of each reach the
high-resolution centerline
point is associated with. The
format of the id is as follows:
CBBBBBRRRRT where C =
Continent (the first number of
the Pfafstetter basin code), B =
Remaining Pfafstetter basin
codes up to level 6, R = Reach
id (assigned sequentially
within a level 6 basin starting
at the downstream end
working upstream, T = Type
(1 – river, 3 – lake on river, 4 –
dam or waterfall, 5 –

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

7

SWORD Product Description Document v17
____________________________________________________________________________________

none

[4, number of points]

node_id

unreliable topology, 6 – ghost
reach)

id of each node the
high-resolution centerline
point is associated with. The
format of the id is as follows:
CBBBBBRRRRNNNT where
C = Continent (the first
number of the Pfafstetter basin
code), B = Remaining
Pfafstetter basin codes up to
level 6, R = Reach id
(assigned sequentially within a
level 6 basin starting at the
downstream end working
upstream), N = Node id
(assigned sequentially within a
reach starting at the
downstream end working
upstream), T = Type (1 – river,
3 – lake on river, 4 – dam or
waterfall, 5 – unreliable
topology, 6 – ghost node)

/nodes

cl_ids

x

y

node_id

minimum and maximum
high-resolution centerline
point ids along each node.

none

[2, number of nodes]

longitude of each node ranging
from 180°E to 180°W

decimal
degrees

[number of nodes]

latitude of each node, ranging
from
90°S to 90°N

decimal
degrees

[number of nodes]

none

[number of nodes]

id of each node. The format of
the id is as follows:
CBBBBBRRRRNNNT where
C = Continent (the first
number of the Pfafstetter basin
code), B = Remaining
Pfafstetter basin code up to

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

8

SWORD Product Description Document v17
____________________________________________________________________________________

level 6, R = Reach id
(assigned sequentially within a
level 6 basin starting at the
downstream end working
upstream), N = Node id
(assigned sequentially within a
reach starting at the
downstream end working
upstream), T = Type (1 – river,
3 – lake on river, 4 – dam or
waterfall, 5 – unreliable
topology, 6 – ghost node)

node length measured along
the high-resolution centerline
points

id of the reach each node is
associated with. The format of
the id is as follows:
CBBBBBRRRRT where C =
Continent (the first number of
the Pfafstetter basin code), B =
Remaining Pfafstetter basin
codes up to level 6, R = Reach
id (assigned sequentially
within a level 6 basin starting
at the downstream end
working upstream), T = Type
(1 – river, 3 – lake on river, 4 –
dam or waterfall, 5 –
unreliable topology, 6 – ghost
reach)

node average water surface
elevation

water surface elevation
variance along the
high-resolution centerline
points used to calculate the
average water surface
elevation for each node

node_length

reach_id

wse

wse_var

meters

[number of nodes]

none

[number of nodes]

meters

[number of nodes]

meters^2

[number of nodes]

width

node average width

meters

[number of nodes]

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

9

SWORD Product Description Document v17
____________________________________________________________________________________

width_var

width variance along the
high-resolution centerline
points used to calculate the
average width for each node

meters^2

[number of nodes]

n_chan_max

maximum number of channels
for each node

none

[number of nodes]

n_chan_mod

mode of the number of
channels for each node

none

[number of nodes]

obstr_type

grod_id

hfalls_id

dist_out

facc

lakeflag

max_width

wth_coef

Type of obstruction for each
node based on GROD and
HydroFALLS databases.
Obstr_type values: 0 - No
Dam, 1 - Dam, 2 - Lock, 3 -
Low Permeable Dam, 4 -
Waterfall.

The unique GROD ID for each
node with obstr_type values
1-3.

The unique HydroFALLS ID
for each node with obstr_type
value 4.

distance from the river outlet
for each node

maximum flow accumulation
value for each node

GRWL water body identifier
for each node:  0 – river, 1 –
lake/reservoir, 2 – canal ,  3  –
tidally influenced river.

maximum width value across
the channel for each node that
includes island and bar areas.

coefficient that is multiplied
by the width variable to
inform the RiverObs search
window for pixel cloud points.

none

[number of nodes]

none

[number of nodes]

none

[number of nodes]

meters

[number of nodes]

km^2

[number of nodes]

none

[number of nodes]

meters

[number of nodes]

none

[number of nodes]

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

10

SWORD Product Description Document v17
____________________________________________________________________________________

none

[number of nodes]

meters

[number of nodes]

none

[number of nodes]

none

[number of nodes]

none

[number of nodes]

none

[number of nodes]

ext_dist_coef

meander_length

sinuosity

river_name

manual_add

edit_flag

coefficient that informs the
maximum RiverObs search
window for pixel cloud points.

length of the meander that a
node belongs to, measured
from beginning of the meander
to its end in meters. For nodes
longer than one meander, the
meander length will represent
the average length of all
meanders belonging to the
node.

the total reach length the node
belongs to divided by the
Euclidean distance between
the reach end points.

all river names associated with
a node. If there are multiple
names for a node they are
listed in alphabetical order and
separated by a semicolon.

binary flag indicating whether
the node was manually added
to the public GRWL
centerlines. These nodes were
originally given a width = 1,
but have since been updated to
have the reach width values.

numerical flag indicating the
type of update applied to
SWORD nodes from the
previous version. Flag
descriptions:

1 - reach type change;
2 - node order change;
3 - reach neighbor change;
41 - flow accumulation
update;
42 - elevation update;

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

11

SWORD Product Description Document v17
____________________________________________________________________________________

43 - width update;
44 - slope update;
45 - river name update;
5 - reach id change;
6 - reach boundary change
7 - reach/node addition

Nodes where multiple updates
have been applied will have
each flag number separated by
commas, i.e: “41,2”.

binary flag indicating if a large
tributary not represented in
SWORD is entering a node. 0
- no tributary, 1 - tributary.

the number of times a node is
traveled to get to any given
headwater from the primary
outlet.

unique values representing
continuous paths from the
river outlet to the headwaters.
Values are unique within level
two Pfafstetter basins. The
lowest value is always the
longest path from outlet to
farthest headwater point in a
connected river network.
Higher path values branch off
from the longest path value to
other headwater points.

unique values indicating
continuous river segments
between river junctions.
Values are unique within level
two Pfafstetter basins.

value indicating whether a
node is on the main network
(0), side network (1), or is a

trib_flag

path_freq

path_order

path_segs

main_side

none

[number of nodes]

none

[number of nodes]

none

[number of nodes]

none

[number of nodes]

none

[number of nodes]

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

12

SWORD Product Description Document v17
____________________________________________________________________________________

stream_order

end_reach

network

secondary outlet on the main
network (2).

stream order based on the log
scale of the path frequency.
stream order is calculated for
the main network only (see
“main_side” description).
stream order is not included
for side channels which are
given a no data value of -9999.

value indicating whether a
reach is a headwater (1), outlet
(2), or junction (3) reach. A
value of 0 means it is a normal
main stem river reach.

unique value for each
connected river network.
Values are unique within level
two Pfafstetter basins.

none

[number of nodes]

none

[number of nodes]

none

[number of nodes]

/reaches

cl_ids

x

y

x_max, x_min,
y_max, y_min

reach_id

minimum and maximum
high-resolution centerline
point ids along each reach.

none

[2, number of reaches]

longitude of the reach center
ranging from 180°E to 180°W

decimal
degrees

[number of reaches]

decimal
degrees

decimal
degrees

[number of reaches]

[number of reaches]

none

[number of reaches]

latitude of the reach center
ranging from 90°S to 90°N

Bounding box of longitudes
and latitudes for a reach. Note
that reaches may have
overlapping boxes.

id of each reach. The format of
the id is as follows:
CBBBBBRRRRT where C =
Continent (the first number of
the Pfafstetter basin code), B =

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

13

SWORD Product Description Document v17
____________________________________________________________________________________

Remaining Pfafstetter basin
codes up to level 6, R = Reach
id (assigned sequentially
within a level 6 basin starting
at the downstream end
working upstream, T = Type
(1 – river, 3 – lake on river, 4 –
dam or waterfall, 5 –
unreliable topology, 6 – ghost
reach)

reach length measured along
the high-resolution centerline
points

reach average water surface
elevation

water surface elevation
variance along the
high-resolution centerline
points used to calculate the
average water surface
elevation for each reach

reach_length

wse

wse_var

meters

[number of reaches]

meters

[number of reaches]

meters^2

[number of reaches]

width

reach average width

meters

[number of reaches]

width_var

max_width

width variance along the
high-resolution centerline
points used to calculate the
average width for each reach

maximum width value across
the channel for each reach that
includes island and bar areas.

meters^2

[number of reaches]

meters

[number of reaches]

n_nodes

number of nodes associated
with each reach

none

[number of reaches]

n_chan_max

maximum number of channels
for each reach

none

[number of reaches]

n_chan_mod

mode of the number of
channels for each reach

none

[number of reaches]

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

14

SWORD Product Description Document v17
____________________________________________________________________________________

obstr_type

grod_id

hfalls_id

slope

lakeflag

dist_out

facc

n_rch_up

Type of obstruction for each
reach based on GROD and
HydroFALLS databases.
Obstr_type values: 0 - No
Dam, 1 - Dam, 2 - Lock, 3 -
Low Permeable Dam, 4 -
Waterfall.

The unique GROD ID for each
reach with obstr_type values
1-3.

The unique HydroFALLS ID
for each reach with obstr_type
value 4.

reach average slope calculated
along the high-resolution
centerline points

GRWL water body identifier
for each reach:  0 – river, 1 –
lake/reservoir, 2 – canal ,  3  –
tidally influenced river.

distance from the river outlet
for each reach

maximum flow accumulation
value for each reach

number of upstream reaches
for each reach

none

[number of reaches]

none

[number of reaches]

none

[number of reaches]

m/km

[number of reaches]

none

[number of reaches]

meters

[number of reaches]

km^2

[number of reaches]

none

[number of reaches]

n_rch_down

number of downstream
reaches for each reach

none

[number of reaches]

rch_id_up

rch_id_dn

ice_flag

reach ids of the upstream
reaches

reach ids of the downstream
reaches

none

[4, number of reaches]

none

[4, number of reaches]

Ice flag values for each SWOT
reach are modeled river ice

none

[366, number of reaches]

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

15

SWORD Product Description Document v17
____________________________________________________________________________________

swot_obs

swot_orbits

river_name

low_slope_flag

edit_flag

conditions based on an
empirical river ice model
[Yang et al., 2019], that takes
surface air temperature (SAT)
data from ERA5 Land (9 km
resolution) as model input.

Values include 0 – ice free, 1 –
mixed, 2 – ice cover.

The maximum number of
SWOT passes to intersect each
reach during the 21 day orbit
cycle.

A list of the SWOT orbit
tracks that intersect each reach
during the 21 day orbit cycle.

all river names associated with
a reach. If there are multiple
names for a reach they are
listed in alphabetical order and
separated by a semicolon.

binary flag where a value of 1
indicates the reach slope is too
low for effective discharge
estimation.

numerical flag indicating the
type of update applied to a
SWORD reach from the
previous version. Flag
descriptions:

1 - reach type change;
2 - node order change;
3 - reach neighbor change;
41 - flow accumulation
update;
42 - elevation update;
43 - width update;
44 - slope update;

none

[number of reaches]

none

[75, number of reaches]

none

[number of reaches]

none

[number of reaches]

none

[number of reaches]

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

16

SWORD Product Description Document v17
____________________________________________________________________________________

45 - river name update;
5 - reach id change;
6 - reach boundary change
7 - reach/node addition

Reaches where multiple
updates have been applied will
have each flag number
separated by commas, i.e:
“41,2”.

binary flag indicating if a large
tributary not represented in
SWORD is entering a reach. 0
- no tributary, 1 - tributary.

the number of times a reach is
traveled to get to any given
headwater from the primary
outlet.

unique values representing
continuous paths from the
river outlet to the headwaters.
Values are unique within level
two Pfafstetter basins. The
lowest value is always the
longest path from outlet to
farthest headwater point in a
connected river network.
Higher path values branch off
from the longest path value to
other headwater points.

unique values indicating
continuous river segments
between river junctions.
Values are unique within level
two Pfafstetter basins.

value indicating whether a
node is on the main network
(0), side network (1), or is a
secondary outlet on the main
network (2).

trib_flag

path_freq

path_order

path_segs

main_side

none

[number of reaches]

none

[number of reaches]

none

[number of reaches]

none

[number of reaches]

none

[number of reaches]

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

17

SWORD Product Description Document v17
____________________________________________________________________________________

/reaches/area_fits

strm_order

end_rch

network

*h_break

*w_break

*h_variance

*w_variance

stream order based on the log
scale of the path frequency.
stream order is not included
for side channels which are
given a no data value of -9999.

value indicating whether a
reach is a headwater (1), outlet
(2), or junction (3) reach. A
value of 0 means it is a normal
main stem river reach.

unique value for each
connected river network.
Values are unique within level
two Pfafstetter basins.

the sub-domain boundary
values for water surface
elevation, calculated by the
fitting of curves to
height-width data

the sub-domain width
boundary values for water
surface elevation

height variance for the
calculation of the w,h
covariance matrix

width variance for the
calculation of the w,h
covariance matrix

none

[number of reaches]

none

[number of reaches]

none

[number of reaches]

meters

[4, number of reaches]

meters

[4, number of reaches]

meters^2

[number of reaches]

meters^2

[number of reaches]

*hw_covariance

covariance between height and
width the calculation of the
w,h covariance matrix

meters^2

[number of reaches]

*fit_coeffs

fit coefficients for the
computation of A prime

meters^2

[2, 3, number of reaches]

*med_flow_area

the cross-sectional area at
median flow

meters^2

[number of reaches]

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

18

SWORD Product Description Document v17
____________________________________________________________________________________

*h_err_stdev

height error standard deviation  meters

[number of reaches]

*w_err_stdev

width error standard deviation  meters

[number of reaches]

*h_w_nobs

number of observations for
w,h covariance matrix

none

[number of reaches]

/reaches/discharge_models/
[unconstrained][constraine
d]/MetroMan

*Abar

*ninf

*p

wetted cross-sectional area
during median discharge

meters^2

[number of reaches]

friction relationship coefficient  none

[number of reaches]

friction relationship exponent

none

[number of reaches]

*Abar_stdev

standard deviation of Abar

meters^2

[number of reaches]

*ninf_stdev

standard deviation of ninf

none

[number of reaches]

*p_stdev

standard deviation of p

none

[number of reaches]

*ninf_p_cor

correlation between ninf and p

none

[number of reaches]

*ninf_Abar_cor

correlation between Abar and
ninf

none

[number of reaches]

*p_Abar_cor

correlation between Abar and
p

none

[number of reaches]

*sbQ_rel

relative uncertainty of
timeseries mean discharge at
each reach, specified as a
standard deviation

none

[number of reaches]

/reaches/discharge_models/
[unconstrained][constraine
d]/BAM

*Abar

wetted cross-sectional area
during median discharge

meters^2

[number of reaches]

*n

manning coefficient

none

[number of reaches]

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

19

SWORD Product Description Document v17
____________________________________________________________________________________

*sbQ_rel

relative uncertainty of
timeseries mean discharge at
each reach, specified as a
standard deviation

none

[number of reaches]

/reaches/discharge_models/
[unconstrained][constraine
d]/HiVDI

/reaches/discharge_models/
[unconstrained][constraine
d]/MOMMA

/reaches/discharge_models/
[unconstrained][constraine
d]/SADS

*Abar

*alpha

*beta

*sbQ_rel

wetted cross-sectional area
during median discharge

meters^2

[number of reaches]

reciprocal friction relationship
coefficient

none

[number of reaches]

negative friction relationship
exponent

relative uncertainty of
timeseries mean discharge at
each reach, specified as a
standard deviation

none

[number of reaches]

none

[number of reaches]

*B

*H

the elevation of zero flow

meters

[number of reaches]

bankfull water surface
elevation

meters

[number of reaches]

*Save

averaged slope

m/km

[number of reaches]

*sbQ_rel

relative uncertainty of
timeseries mean discharge at
each reach, specified as a
standard deviation

none

[number of reaches]

*Abar

wetted cross-sectional area
during median discharge

meters^2

[number of reaches]

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

20

SWORD Product Description Document v17
____________________________________________________________________________________

*n

manning coefficient

none

[number of reaches]

*sbQ_rel

relative uncertainty of
timeseries mean discharge at
each reach, specified as a
standard deviation

none

[number of reaches]

/reaches/discharge_models/
[unconstrained][constraine
d]/SIC4DVar

*Abar

wetted cross-sectional area
during median discharge

meters^2

[number of reaches]

*n

manning coefficient

none

[number of reaches]

*sbQ_rel

relative uncertainty of
timeseries mean discharge at
each reach, specified as a
standard deviation

none

[number of reaches]

4.2 Geopackage

SWORD geopackage files are split into two files for nodes and reaches per continental region,
where nodes are represented as 200 m spaced points and reaches are represented as polylines. All
geopackage files are in geographic (latitude/longitude) projection, referenced to datum WGS84.
Attributes included in the node and reach files for both geopackage and shapefile formats are
listed in Tables 4 and 5. Geopackage file names are distributed at continental scales and are
defined by a two-digit identifier (Table 2): [continent]_sword_[nodes/reaches]_v1.gpkg (i.e.
na_sword_nodes_v17.gpkg; na_sword_reaches_v17.gpkg).

4.3 Shapefiles

SWORD shapefiles consist of four main files (.dbf, .prj, .shp, .shx). There are separate shapefiles
for nodes and reaches, where nodes are represented as 200 m spaced points and reaches are
represented as polylines. All shapefiles are in geographic (latitude/longitude) projection,
referenced to datum WGS84. Attributes included in the node and reach files for both geopackage
and shapefile formats are listed in Tables 4 and 5. Shapefiles are split into HydroBASINS
Pfafstetter level 2 basins (hbXX) within each continent (Table 2) with a naming convention as
follows: [continent]_sword_[nodes/reaches]_hb[XX]_v1.shp (i.e.
na_sword_nodes_hb74_v17.shp; na_sword_reaches_hb74_v17.shp).

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

21

SWORD Product Description Document v17
____________________________________________________________________________________

Table 4: Node shapefile and geopackage  attribute descriptions..

Attribute

Description

x

y

node_id

longitude of each node ranging from 180°E to 180°W

latitude of each node, ranging from
90°S to 90°N

id of each node. The format of the id is as follows: CBBBBBRRRRNNNT
where C = Continent (the first number of the Pfafstetter basin code), B =
Remaining Pfafstetter basin code up to level 6, R = Reach id (assigned
sequentially within a level 6 basin starting at the downstream end working
upstream), N = Node id (assigned sequentially within a reach starting at the
downstream end working upstream), T = Type (1 – river, 3 – lake on river, 4 –
dam or waterfall, 5 – unreliable topology, 6 – ghost node)

Units

decimal degrees

decimal degrees

none

node_length

node length measured along the high-resolution centerline points

reach_id

id of the reach each node is associated with. The format of the id is as follows:
CBBBBBRRRRT where C = Continent (the first number of the Pfafstetter
basin code), B = Remaining Pfafstetter basin codes up to level 6, R = Reach id
(assigned sequentially within a level 6 basin starting at the downstream end
working upstream), T = Type (1 – river, 3 – lake on river, 4 – dam or waterfall,
5 – unreliable topology, 6 – ghost reach)

wse

node average water surface elevation

wse_var

water surface elevation variance along the high-resolution centerline points
used to calculate the average water surface elevation for each node

width

node average width

width_var

width variance along the high-resolution centerline points used to calculate the
average width for each node

n_chan_max

maximum number of channels for each node

n_chan_mod

mode of the number of channels for each node

obstr_type

Type of obstruction for each node based on GROD and HydroFALLS
databases. Obstr_type values: 0 - No Dam, 1 - Dam, 2 - Lock, 3 - Low
Permeable Dam, 4 - Waterfall.

grod_id

The unique GROD ID for each node with obstr_type values 1-3.

hfalls_id

The unique HydroFALLS ID for each node with obstr_type value 4.

meters

none

meters

meters^2

meters

meters^2

none

none

none

none

none

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

22

SWORD Product Description Document v17
____________________________________________________________________________________

dist_out

distance from the river outlet for each node

type

facc

lakeflag

max_width

river_name

sinuosity

meand_len

manual_add

trib_flag

path_freq

path_order

path_segs

main_side

meters

none

Node type identifier: 1 – river, 3 – lake on river, 4 – dam or waterfall, 5 –
unreliable topology, 6 – ghost reach.

Maximum flow accumulation value for each node.

kilometers^2

GRWL water body identifier for each node:  0 – river, 1 – lake/reservoir, 2 –
canal,  3 – tidally influenced river.

none

maximum width value across the channel for each node that includes island
and bar areas.

meters

all river names associated with a node. If there are multiple names for a node
they are listed in alphabetical order and separated by a semicolon.

the total reach length the node belongs to divided by the Euclidean distance
between the reach end points.

length of the meander that a node belongs to, measured from beginning of the
meander to its end in meters. For nodes longer than one meander, the meander
length will represent the average length of all meanders belonging to the node.

binary flag indicating whether the nodes was manually added to the public
GRWL centerlines. These nodes were originally given a width = 1, but have
since been updated to have the reach width values.

none

none

meters

none

binary flag indicating if a large tributary not represented in SWORD is entering
a node. 0 - no tributary, 1 - tributary.

none

the number of times a node is traveled to get to any given headwater from the
primary outlet.

unique values representing continuous paths from the river outlet to the
headwaters. Values are unique within level two Pfafstetter basins. The lowest
value is always the longest path from outlet to farthest headwater point in a
connected river network. Higher path values branch off from the longest path
value to other headwater points.

unique values indicating continuous river segments between river junctions.
Values are unique within level two Pfafstetter basins.

value indicating whether a node is on the main network (0), side network (1),
or is a secondary outlet on the main network (2).

none

none

none

none

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

23

SWORD Product Description Document v17
____________________________________________________________________________________

strm_order

stream order based on the log scale of the path frequency. stream order is not
included for side channels which are given a no data value of -9999.

end_rch

network

value indicating whether a reach is a headwater (1), outlet (2), or junction (3)
reach. A value of 0 means it is a normal main stem river reach.

unique value for each connected river network. Values are unique within level
two Pfafstetter basins.

none

none

none

Table 5: Reach shapefile and geopackage attribute descriptions.

Attribute

Description

Units

x

y

reach_id

longitude of the reach center ranging from 180°E to 180°W

decimal degrees

latitude of the reach center ranging from 90°S to 90°N

decimal degrees

id of each reach. The format of the id is as follows: CBBBBBRRRRT where C
= Continent (the first number of the Pfafstetter basin code), B = Remaining
Pfafstetter basin codes up to level 6, R = Reach id (assigned sequentially
within a level 6 basin starting at the downstream end working upstream, T =
Type (1 – river, 3 – lake on river, 4 – dam or waterfall, 5 – unreliable topology,
6 – ghost reach)

none

reach_length

reach length measured along the high-resolution centerline points

wse

reach average water surface elevation

wse_var

water surface elevation variance along the high-resolution centerline points
used to calculate the average water surface elevation for each reach

width

reach average width

width_var

width variance along the high-resolution centerline points used to calculate the
average width for each reach

n_nodes

number of nodes associated with each reach

n_chan_max

maximum number of channels for each reach

n_chan_mod

mode of the number of channels for each reach

obstr_type

type of obstruction for each reach based on GROD and HydroFALLS
databases. Obstr_type values: 0 - No Dam, 1 - Dam, 2 - Lock, 3 - Low
Permeable Dam, 4 - Waterfall.

meters

meters

meters^2

meters

meters^2

none

none

none

none

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

24

meters

none

none

none

none

none

meters

none

SWORD Product Description Document v17
____________________________________________________________________________________

grod_id

the unique GROD ID for each reach with obstr_type values 1-3.

hfalls_id

the unique HydroFALLS ID for each reach with obstr_type value 4.

none

none

slope

reach average slope calculated along the high-resolution centerline points

m/km

dist_out

distance from the river outlet for each reach

n_rch_up

number of upstream reaches for each reach

n_rch_down

number of downstream reaches for each reach

rch_id_up

reach ids of the upstream reaches

rch_id_dn

reach ids of the downstream reaches

lakeflag

max_width

type

facc

swot_obs

swot_orbits

river_name

trib_flag

path_freq

path_order

GRWL water body identifier for each reach:  0 – river, 1 – lake/reservoir, 2 –
canal,  3 – tidally influenced river.

maximum width value across the channel for each reach that includes island
and bar areas.

Reach type identifier: 1 – river, 3 – lake on river, 4 – dam or waterfall, 5 –
unreliable topology, 6 – ghost reach.

Maximum flow accumulation value for each reach.

kilometers^2

The maximum number of SWOT passes to intersect each reach during the 21
day orbit cycle.

A list of the SWOT orbit tracks that intersect each reach during the 21 day
orbit cycle.

all river names associated with a reach. If there are multiple names for a reach
they are listed in alphabetical order and separated by a semicolon.

none

none

none

binary flag indicating if a large tributary not represented in SWORD is entering
a reach. 0 - no tributary, 1 - tributary.

none

the number of times a reach is traveled to get to any given headwater from the
primary outlet.

unique values representing continuous paths from the river outlet to the
headwaters. Values are unique within level two Pfafstetter basins. The lowest
value is always the longest path from outlet to farthest headwater point in a
connected river network. Higher path values branch off from the longest path
value to other headwater points.

none

none

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

25

SWORD Product Description Document v17
____________________________________________________________________________________

path_segs

main_side

strm_order

end_rch

network

unique values indicating continuous river segments between river junctions.
Values are unique within level two Pfafstetter basins.

value indicating whether a node is on the main network (0), side network (1),
or is a secondary outlet on the main network (2).

stream order based on the log scale of the path frequency. stream order is not
included for side channels which are given a no data value of -9999.

value indicating whether a reach is a headwater (1), outlet (2), or junction (3)
reach. A value of 0 means it is a normal main stem river reach.

unique value for each connected river network. Values are unique within level
two Pfafstetter basins.

none

none

none

none

none

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

26

SWORD Product Description Document v17
____________________________________________________________________________________

5. Summary Figures and Statistics

Figure 1: SWORD reach numbers per continent (not including ghost reaches). Colors display the updated distance from outlet
based on shortest paths between outlets and headwaters.

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

27

SWORD Product Description Document v17
____________________________________________________________________________________

Figure 2: Modified lumped routing results of accumulated reaches based on updated topology.

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

28

SWORD Product Description Document v17
____________________________________________________________________________________

Figure 3: New SWORD variables starting in version 17. Variable descriptions in tables 3-5.

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

29

SWORD Product Description Document v17
____________________________________________________________________________________

Figure 4: Pie chart of global reach type identifiers (1 – river, 3 – lake on river, 4 – dam (or waterfall), 5 – unreliable topology).
Percentages do not include ghost reaches.

Table 6: Reach lengths per continent (excluding ghost reaches). Numbers in parentheses are calculated for reaches with river
type identifiers only.

Reach Length (L)

NA

SA

AS

EU

AF

OC

Global

L < 5 km

23.9%
(14.2%)

13.1%
(7.4%)

27.4%
(14.7%)

37.1%
(19.5%)

12.5%
(7.1%)

13.6%
(7.4%)

23.6%
(12.6%)

5 km ≤ L < 10 km

17.8%
(13.5%)

16.5%
(14.8%)

15.5%
(15.9%)

15.7%
(17.0%)

15.6%
(13.9%)

20.9%
(16.4%)

16.3%
(15.3%)

10 km ≤ L ≤ 20 km

58.2%
(72.3%)

70.4%
(77.8%)

57.8%
(69.4%)

47.1%
(63.4%)

71.9%
(79.0%)

65.5%
(76.2%)

60.0%
(72.1%)

L > 20 km

0.02%
(0.03%)

0.02%
(0.02%)

0.02%
(0.03%)

0.01%
(0.02%)

0.01%
(0.01%)

0%
(0%)

0.02%
(0.02%)

Mean

Median

9.7 km
(11.1 km)

11.0 km
(11.9 km)

9.3 km
(11.0 km)

8.1 km
(10.4 km)

11.2 km
(12.1 km)

10.7 km
(11.8 km)

9.8 km
(11.3 km)

10.3 km
(10.9 km)

10.5 km
(10.7 km)

10.3 km
(10.8 km)

9.5 km
(10.6 km)

10.6 km
(10.8 km)

10.4 km
(10.7 km)

10.3 km
(10.8 km)

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

30

SWORD Product Description Document v17
____________________________________________________________________________________

6. References

Allen, G. H., & Pavelsky, T. M. (2018). Global extent of rivers and streams. Science, 361(6402), 585-588.

Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang X., Frasson, R. P. d. M., & Bendezu, L. (2021). The
Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for
satellite data products”. Water Resources Research.

Biancamaria, S., Lettenmaier, D. P., & Pavelsky, T. M. (2016). The SWOT mission and its capabilities for land
hydrology. In Remote Sensing and Water Resources (pp. 117-147). Springer, Cham.

Lehner, B., Grill G. (2013): Global river hydrography and network routing: baseline data and new approaches to
study the world’s large river systems. Hydrological Processes, 27(15): 2171–2186. Data is available at
www.hydrosheds.org.

Tessler, Z. D., Vörösmarty, C. J., Grossberg, M., Gladkova, I., Aizenman, H., Syvitski, J. P. M., &
Foufoula-Georgiou, E. (2015). Profiling risk and sustainability in coastal deltas of the world. Science,
349(6248), 638-643.

Whittemore, A., Ross, M. R., Dolan, W., Langhorst, T., Yang, X., Pawar, S., Jorissen, M., Lawton, E.,
Januchowski-Hartley, S., & Pavelsky, T. (2020). A Participatory Science Approach to Expanding Instream
Infrastructure Inventories. Earth's Future, 8(11), e2020EF001558.

Yamazaki, D., Ikeshima, D., Sosa, J., Bates, P. D., Allen, G., & Pavelsky, T. (2019). MERIT Hydro: A
high-resolution global hydrography map based on latest topography datasets. Water Resources Research.
https://doi.org/10.1029/2019WR024873.

Yang, X., Pavelsky, T. M., Allen, G. H. (2019). The past and future of global river ice. Nature.

SWOT Orbits: https://www.aviso.altimetry.fr/en/missions/future-missions/swot/orbit.html

HydroFALLS: http://wp.geog.mcgill.ca/hydrolab/hydrofalls/

______________________________________________________________________________

If you use the SWORD Database in your work, please cite: Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., &
Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite
data products. Water Resources Research, 57, e2021WR030054. https://doi. org/10.1029/2021WR030054

31


