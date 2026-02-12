// Load the GAUL dataset
var gaul = ee.FeatureCollection("FAO/GAUL/2015/level1");

// Filter the GAUL dataset for Massachusetts (USA)
var massachusetts = gaul.filter(ee.Filter.eq('ADM1_NAME', 'Massachusetts'));

// Load the SRTM DEM dataset
// The DEM values in the SRTM dataset (USGS/SRTMGL1_003) are in meters, representing the elevation above sea level.
var srtm = ee.Image("USGS/SRTMGL1_003");

// Clip the DEM to the Massachusetts boundary
var demMassachusetts = srtm.clip(massachusetts);

// Display the DEM on the map
Map.centerObject(massachusetts, 7);
Map.addLayer(demMassachusetts, {min: 0, max: 3000, palette: ['blue', 'green', 'yellow', 'brown', 'white']}, 'SRTM DEM Massachusetts');

// Calculate the slope from the DEM
/*The slope data calculated from a terrain DEM is expressed in degrees.
It measures the rate of elevation change, with values ranging from 0° (flat terrain) to 90° (vertical). 
The local gradient is computed using the 4-connected neighbors of each pixel.
https://developers.google.com/earth-engine/apidocs/ee-terrain-slope
*/
var slopeMassachusetts = ee.Terrain.slope(demMassachusetts);

// Display the slope on the map
Map.addLayer(slopeMassachusetts, {min: 0, max: 60, palette: ['00FFFF', '008000', 'FFFF00', 'FFA500', 'FF0000']}, 'Slope Massachusetts');

//  CRS: 'EPSG:26986'

// Export the DEM layer to Google Drive with higher maxPixels
Export.image.toDrive({
  image: demMassachusetts,
  description: 'SRTM_DEM_Massachusetts',
  scale: 30,
  region: massachusetts.geometry(),
  fileFormat: 'GeoTIFF',
  maxPixels: 1e13
});

// Export the Slope layer to Google Drive with higher maxPixels
Export.image.toDrive({
  image: slopeMassachusetts,
  description: 'SRTM_Slope_Massachusetts',
  scale: 1,
  region: massachusetts.geometry(),
  fileFormat: 'GeoTIFF',
  maxPixels: 1e13
});
