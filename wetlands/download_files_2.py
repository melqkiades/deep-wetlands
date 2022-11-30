import os
import time

from dotenv import load_dotenv
import geopandas as gpd
import ee
import eeconvert as eec

from wetlands import utils


def download_ndwi():
    geojson_file = os.getenv("GEOJSON_FILE")

    # Read data using GeoPandas
    geoboundary = gpd.read_file(geojson_file)
    print("Data dimensions: {}".format(geoboundary.shape))

    shape_name = os.getenv('REGION_NAME')

    ee.Authenticate()
    ee.Initialize()

    product = 'COPERNICUS/S2'

    # Dates for Sala kommun
    # min_date = '2018-01-01'
    # max_date = '2020-01-01'

    # Dates for Lindesberg kommun
    min_date = '2018-01-01'
    max_date = '2020-01-01'

    # min_date = os.getenv('START_DATE')
    # max_date = os.getenv('END_DATE')
    cloud_pct = 10

    # Get the shape geometry for Sala kommun
    region = geoboundary.loc[geoboundary.shapeName == shape_name]
    centroid = region.iloc[0].geometry.centroid.coords[0]
    region = eec.gdfToFc(region)

    image = ee.ImageCollection(product) \
        .filterBounds(region) \
        .filterDate(str(min_date), str(max_date)) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct)) \
        .median() \
        .clip(region)

    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    # Map = emap.Map(center=[centroid[1], centroid[0]], zoom=10)
    # Map.addLayer(ndwi, {'palette': ['red', 'yellow', 'green', 'cyan', 'blue']}, 'NDWI')
    # Map.addLayerControl()
    # Map

    # Create NDWI mask
    ndwiThreshold = ndwi.gte(0.0)
    ndwiMask = ndwiThreshold.updateMask(ndwiThreshold)
    semiNdwiImage = ndwiThreshold.neq(0.0)
    semiNdwiMask = ndwiThreshold.eq(0.0)
    new_image = semiNdwiMask.multiply(0.5).add(semiNdwiImage.multiply(semiNdwiMask.neq(0.0)))
    new_image = new_image.add(semiNdwiImage)
    semiNdwiMask = semiNdwiMask.multiply(0.5).add(ndwiThreshold.multiply(ndwiThreshold.eq(0.0)))

    folder = 'new_geo_exports'  # Change this to your file destination folder in Google drive
    newImageTask = export_image(new_image, shape_name + '_new_image_2018-07', region, folder)


def export_image(image, filename, region, folder):
    """Export Image to Google Drive.

    Args:
      image (ee.image.Image): Generated Sentinel-2 image
      filename (str): Name of image, without the file extension
      geometry (ee.geometry.Geometry): The geometry of the area of
        interest to filter to.
      folder (str): The destination folder in your Google Drive.

    Returns:
      ee.batch.Task: A task instance
    """

    print('Exporting to {}.tif ...'.format(filename))

    task = ee.batch.Export.image.toDrive(
        image=image,
        driveFolder=folder,
        scale=10,
        region=region.geometry(),
        description=filename,
        fileFormat='GeoTIFF',
        crs='EPSG:4326',
        maxPixels=900000000
    )
    task.start()

    return task


def download_sar():

    geojson_file = os.getenv("GEOJSON_FILE")

    # Read data using GeoPandas
    geoboundary = gpd.read_file(geojson_file)
    print("Data dimensions: {}".format(geoboundary.shape))

    shape_name = os.getenv('REGION_NAME')

    ee.Authenticate()
    ee.Initialize()

    product = 'COPERNICUS/S1_GRD'

    # Dates for Sala kommun
    # min_date = '2018-01-01'
    # max_date = '2020-01-01'

    # Dates for Lindesberg kommun
    min_date = '2018-01-01'
    max_date = '2019-01-01'

    # min_date = os.getenv('START_DATE')
    # max_date = os.getenv('END_DATE')

    # Get the shape geometry for Sala kommun
    region = geoboundary.loc[geoboundary.shapeName == shape_name]
    centroid = region.iloc[0].geometry.centroid.coords[0]
    region = eec.gdfToFc(region)

    # sarImage = ee.ImageCollection(product)\
    #         .filterBounds(region)\
    #         .filterDate(str(min_date), str(max_date))\
    #         .median()\
    #         .clip(region)

    sarImage = ee.ImageCollection(product) \
        .filterBounds(region) \
        .filterDate(str(min_date), str(max_date)) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
        .median() \
        .clip(region)

    bandNames = sarImage.bandNames()
    print(bandNames.getInfo())

    sarImage = sarImage.select(['VV'])

    percentiles = sarImage.reduceRegion(
        reducer=ee.Reducer.percentile([0, 1, 5, 50, 95, 99, 100]),
        geometry=region,
        scale=10
    )

    minValue = percentiles.get("VV_p1").getInfo()
    maxValue = percentiles.get("VV_p99").getInfo()

    print(minValue)
    print(maxValue)

    sarImage = sarImage.float()

    folder = 'new_geo_exports'  # Change this to your file destination folder in Google drive
    task = export_image(sarImage, shape_name + '_sar_vv_single_5', region, folder)


def main():
    load_dotenv()

    country_code = os.getenv("COUNTRY_CODE")
    file_name = os.getenv("GEOJSON_FILE")
    utils.download_country_boundaries(country_code, 'ADM2', file_name)
    download_ndwi()
    download_sar()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))


