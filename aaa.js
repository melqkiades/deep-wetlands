var geometry = ee.Geometry.Polygon(
        [[[30.0014347527015, 37.57084315925745],
          [30.0014347527015, 37.50986387270507],
          [30.121597716568687, 37.50986387270507],
          [30.121597716568687, 37.57084315925745]]], null, false)
var visualization = {"bands":["B4","B3","B2"],"min":0,"max":0.3};

// Applies scaling factors.
function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBands, null, true);
}

// Landat 8 surface reflection data
var L8Coll = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(geometry).sort('DATE_ACQUIRED')
    .map(applyScaleFactors)
    .filterMetadata('CLOUD_COVER', 'less_than', 1)
    .map(function(image){return image.clip(geometry)});


print(L8Coll)

var L8Coll=L8Coll.select(['SR_B2', 'SR_B3', 'SR_B4','SR_B5', 'SR_B6', 'SR_B7','ST_B10'] , ['B2', 'B3', 'B4','B5', 'B6', 'B7','B10'])

print(L8Coll)
Map.addLayer(L8Coll.first(), visualization, 'L8 2013 ');

Map.addLayer(L8Coll.sort('CLOUD_COVER').first(), visualization, 'L8 Best ');
Map.centerObject(L8Coll)

print('L8Coll Least Cloud Cover',L8Coll.sort('CLOUD_COVER').first())
print('Landsat Collection Date Acquired',L8Coll.aggregate_array('DATE_ACQUIRED'))
print('Landsat Collection Cloud Cover',L8Coll.aggregate_array('CLOUD_COVER'))
print('Landsat Collection Path/Row',L8Coll.aggregate_array('WRS_PATH'),
      L8Coll.aggregate_array('WRS_ROW'))

var L8LeastCC=L8Coll.sort('CLOUD_COVER').first()

print('L8 Least Cloud Cover',L8LeastCC)
print(ee.Date(L8Coll.first().get('system:time_start')).format("yyyy-MM-dd"))

//+++++++++++++++++++++
/*
// Load a Landsat 8 collection for a single path-row.
var collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filter(ee.Filter.eq('WRS_PATH', 179))
  .filter(ee.Filter.eq('WRS_ROW', 34));
*/

// This function adds a band representing the image timestamp.
var addTime = function(image) {
  return image.addBands(image.metadata('system:time_start'));
};

// Map the function over the collection and display the result.
var L8CollT=L8Coll.map(addTime)
print('L8coll system Time',L8CollT);


print(ee.Date(L8Coll.first().get('system:time_start')).format("yyyy-MM-dd"))

var doy = ee.Date(L8Coll.first().get('system:time_start')).getRelative('day', 'year');

print('doy',doy)
//++++++++++++++++++++++


// Create RGB visualization images for use as animation frames.
var rgbVis = L8Coll.map(function(img) {
  return img.visualize(visualization).clip(geometry)
});
print('rgbVis',rgbVis)

// Define GIF visualization parameters.
var gifParams = {
  'region': geometry,
  'dimensions': 600,
  'crs': 'EPSG:3857',
  'framesPerSecond': 1//,  'format': 'gif'
};

// Print the GIF URL to the console.
print(rgbVis.getVideoThumbURL(gifParams));

// Render the GIF animation in the console.
print(ui.Thumbnail(rgbVis, gifParams));

////+++++++++++++++








var aaa = /* color: #d63000 */ee.Geometry.Point([30.0787, 37.5219]);

// var geometry=geometry2

print(ee.Image(L8Coll.first()).date().get('year'))
print(ee.Image(L8Coll.first()).date().get('month'))
print(ee.Image(L8Coll.first()).date().get('day'))
print(ee.Image(L8Coll.first()).date())

//print(L8Coll.first().get('system:time_start').format('YYYY-MM-DD').toString())

print(ee.Date(L8Coll.first().get('system:time_start')).format("yyyy-MM-dd"))

var doy = ee.Date(L8Coll.first().get('system:time_start')).getRelative('day', 'year');

print('doy',doy)
//++++++++++++++++++++++


// Create RGB visualization images for use as animation frames.
var rgbVis = L8Coll.map(function(img) {
  return img.visualize(visualization).clip(geometry)
});
print('rgbVis',rgbVis)

// Define GIF visualization parameters.
var gifParams = {
  'region': geometry,
  'dimensions': 300,
  'crs': 'EPSG:3857',
  'framesPerSecond': 2//,  'format': 'gif'
};

// Print the GIF URL to the console.
// print(rgbVis.getVideoThumbURL(gifParams));

// Render the GIF animation in the console.
//print(ui.Thumbnail(rgbVis, gifParams));

////+++++++++++++++

// Define arguments for animation function parameters.
var videoArgs = {
  dimensions: 300,
  region: geometry,
  framesPerSecond: 2,
  crs: 'EPSG:3857',
};

var text = require('users/gena/packages:text'); // Import gena's package which allows text overlay on image

var annotations = [
  {position: 'right', offset: '1%', margin: '1%', property: 'label', scale: 30} //large scale because image if of the whole world. Use smaller scale otherwise
  ]
//print(L8Coll.first().get('CLOUD_COVER'))
function addText(image){

  var timeStamp = image.get('DATE_ACQUIRED'); // get the time stamp of each frame. This can be any string. Date, Years, Hours, etc.
  var timeStamp = ee.String(timeStamp); //convert time stamp to string
  var image = image.visualize({ //convert each frame to RGB image explicitly since it is a 1 band image
      forceRgbOutput: true,
      bands: ['B4','B3','B2'],
      min: 0,
      max: 0.3,
//      palette: ['blue', 'purple', 'cyan', 'green', 'yellow', 'red']
    }).set({'label':timeStamp}); // set a property called label for each image

  var annotated = text.annotateImage(image, {}, aaa, annotations); // create a new image with the label overlayed using gena's package

  return annotated
}

var tempCol = L8Coll.map(addText) //add time stamp to all images

print(ui.Thumbnail(tempCol, videoArgs)); //print gif
