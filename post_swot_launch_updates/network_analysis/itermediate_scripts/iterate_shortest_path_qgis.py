'''
proceed with: Plugins > Python Console > Show Editor > New Editor and paste the script below.
'''

# Here names of input layers must be specified 
start_points = QgsProject.instance().mapLayersByName('hb81_outlets_v17a')[0]
end_points = QgsProject.instance().mapLayersByName('hb81_headwaters_v17a')[0]
network = QgsProject.instance().mapLayersByName('hb81_sword_dissolved')[0]

# Looping through all start point features
for feat in start_points.getFeatures():
    start_point_id = feat["reach_id"]
    start_point_geom = feat.geometry()

    parameters = {
        'DEFAULT_DIRECTION' : 2,
        'DEFAULT_SPEED' : 50,
        'DIRECTION_FIELD' : '',
        'END_POINTS' : end_points,
        'INPUT' : network,
        'OUTPUT' : 'TEMPORARY_OUTPUT',
        'SPEED_FIELD' : '',
        'START_POINT' : start_point_geom,
        'STRATEGY' : 0,
        'TOLERANCE' : 0,
        'VALUE_BACKWARD' : '',
        'VALUE_BOTH' : '',
        'VALUE_FORWARD' : ''
        }

    result = processing.run("qgis:shortestpathpointtolayer", parameters)['OUTPUT']
    # result.setName(f'/Users/ealtenau/Documents/SWORD_Dev/outputs/Reaches_Nodes/v17a/network_testing/hb74_shortest_path_{start_point_id}') # changing the output name
    result.setName(f'Shortest path from point {start_point_id}') # changing the output name
    QgsProject.instance().addMapLayer(result) # adding output to the map
    QgsProject.instance().addMapLayer(result) # adding output to the map
    
