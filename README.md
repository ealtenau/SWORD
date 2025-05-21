<p align="center">
    <img src="https://github.com/ealtenau/SWORD/blob/main/docs/figures%20/SWORD_Logo.png" width="100">
</p>
# SWOT River Database (SWORD)
Source code for developing, updating, and maintaining the SWOT River Database (SWORD).

### SWORD Background
The [Surface Water and Ocean Topography (SWOT) satellite mission](https://swot.jpl.nasa.gov/), launched in December 2022, vastly expands observations of river water surface elevation (WSE), width, and slope [(Biancamaria et al., 2016)](https://link.springer.com/chapter/10.1007/978-3-319-32449-4_6). In order to facilitate a wide range of new analyses with flexibility, the SWOT mission provides a range of relevant data products. One product the SWOT mission provides are river vector products stored in shapefile format for each SWOT overpass. The SWOT vector data products are most broadly useful if they allow multitemporal analysis of river nodes and reaches covering the same river areas. Doing so requires defining SWOT reaches and nodes a priori, so that SWOT data can be assigned to them. The **SWO**T **R**iver **D**atabase (**SWORD**) combines multiple global river- and satellite-related datasets to define the nodes and reaches that constitute SWOT river vector data products. SWORD provides high-resolution river nodes (200 m) and reaches (~10 km) in shapefile, geopackage, and netCDF formats with attached hydrologic variables (WSE, width, slope, etc.) as well as a consistent topological system for global rivers 30 m wide and greater. Information regarding the development of SWORD are detailed by [Altenau et al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021WR030054).

### Current Version: v17b
Before using SWORD, please first read through the [SWORD Product Description Document](https://drive.google.com/file/d/1_1qmuJhL_Yd6ThW2QE4gW0G1eHH_XAer/view?usp=sharing). If you have additional questions, feel free to email **_sword.riverdb@gmail.com_**. SWORD v17b is the official version for SWOT **Version D** [**RiverSP Vector Products**](https://podaac.jpl.nasa.gov/SWOT?tab=datasets-information&sections=about). 

**Version Notes:**
Versions 17 and 17b include significant updates and improvements to SWORD topology. Version 17b is structurally similar to Version 17 but has updates to some Reach and Node IDs and a minor bug fix. 

Version 17
- Produced: October 2024
- Topological updates to ensure consistency.
- Distance from outlet recalculation based on shortest paths between outlets and headwaters. o New variables for nodes and reaches: “path_freq”, “path_order”, “path_segs”, “main_side”,
“stream_order”, “end_reach”,” network”.
- Improved geometry for reach shapefiles.
- Additional channels added for improved network connectivity.
- New reach and node ids that better reflect the improved topology. o Corrected node lengths to match reach lengths when summed.

Version 17b
- Produced: March 2025
- "Type" change for 1662 reaches and associated nodes globally. This change updates the Reach and Node IDs for impacted reaches and nodes. 
- Updates to reach and node lengths and distance-from-outlet variable to correct a bug in the node length calculation for select reaches (impacted <2% of reaches globally).

### How to Download
SWORD can be downloaded from two sources:
- [SWORD Explorer](https://www.swordexplorer.com/) allows users to explore and download the most current version. 
- [Zenodo](https://zenodo.org/records/15299138) provides downloads for the current and previous versions of SWORD, as well as a DOI for citing the database. 

### Citations
- **Development publication:** Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. D. M., & Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite data products. Water Resources Research, 57(7), e2021WR030054.
- **Database DOI:** Elizabeth H. Altenau, Tamlin M. Pavelsky, Michael T. Durand, Xiao Yang, Renato P. d. M. Frasson, & Liam Bendezu. (2025). SWOT River Database (SWORD) (Version v17b) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15299138

![Fig1](https://github.com/ealtenau/SWORD/blob/main/docs/figures%20/global_map_dist_out_legend_basins_rch_numbers.png)
**_Figure 1:_** SWORD reach numbers per continent. Colors display the distance from outlet calculated from shortest paths between outlets and headwaters.

![Fig2](https://github.com/ealtenau/SWORD/blob/main/docs/figures%20/global_map_routing_legend.png)
**_Figure 2:_** Modified lumped routing results of accumulated reaches based on updated topology.