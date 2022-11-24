import time

# Standard imports
from tqdm.notebook import tqdm
import requests
import json

import pandas as pd
import numpy as np
from PIL import Image

# Geospatial processing packages
import geopandas as gpd
import geojson

import shapely
import rasterio as rio
from rasterio.plot import show
import rasterio.mask
from shapely.geometry import box

# Mapping and plotting libraries
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import ee
import eeconvert as eec
import geemap
import geemap.eefolium as emap
import folium

# Deep learning libraries
import torch
from torchvision import datasets, models, transforms


def download_ndwi():
    ISO = 'SWE'  # "DEU" is the ISO code for Germany
    ADM = 'ADM2'  # Equivalent to administrative districts

    # Query geoBoundaries
    r = requests.get("https://www.geoboundaries.org/gbRequest.html?ISO={}&ADM={}".format(ISO, ADM))
    dl_path = r.json()[0]['gjDownloadURL']

    # Save the result as a GeoJSON
    filename = 'geoboundary.geojson'
    geoboundary = requests.get(dl_path).json()
    with open(filename, 'w') as file:
        geojson.dump(geoboundary, file)

    # Read data using GeoPandas
    geoboundary = gpd.read_file(filename)
    print("Data dimensions: {}".format(geoboundary.shape))

    shape_name = 'Sala kommun'

    ee.Authenticate()
    ee.Initialize()

    product = 'COPERNICUS/S2'
    min_date = '2018-01-01'
    max_date = '2020-01-01'
    range_min = 0
    range_max = 2000
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

    # var mask = image.gt(2)
    # var new_image = mask.multiply(value).add(image.multiply(mask.not()))

    # semiNdwiMask = semiNdwiMask.multiply(0.5).add(semiNdwiMask.multiply(ndwiThreshold.eq(0.0)))
    # semiNdwiMask = semiNdwiMask.multiply(0.5)
    semiNdwiMask = semiNdwiMask.multiply(0.5).add(ndwiThreshold.multiply(ndwiThreshold.eq(0.0)))

    folder = 'new_geo_exports'  # Change this to your file destination folder in Google drive
    # task = export_image(image, shape_name + '_all_bands_2', region, folder)
    newImageTask = export_image(new_image, shape_name + '_new_image_4', region, folder)


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
    ISO = 'SWE'  # "DEU" is the ISO code for Germany
    ADM = 'ADM2'  # Equivalent to administrative districts

    # Query geoBoundaries
    r = requests.get("https://www.geoboundaries.org/gbRequest.html?ISO={}&ADM={}".format(ISO, ADM))
    dl_path = r.json()[0]['gjDownloadURL']

    # Save the result as a GeoJSON
    filename = 'geoboundary.geojson'
    geoboundary = requests.get(dl_path).json()
    with open(filename, 'w') as file:
        geojson.dump(geoboundary, file)

    # Read data using GeoPandas
    geoboundary = gpd.read_file(filename)
    print("Data dimensions: {}".format(geoboundary.shape))

    shape_name = 'Sala kommun'

    ee.Authenticate()
    ee.Initialize()

    product = 'COPERNICUS/S1_GRD'
    min_date = '2018-01-01'
    max_date = '2020-01-01'
    range_min = 0
    range_max = 2000
    cloud_pct = 10

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
        scale=20
    )

    minValue = percentiles.get("VV_p1").getInfo()
    maxValue = percentiles.get("VV_p99").getInfo()

    print(minValue)
    print(maxValue)

    sarImage = sarImage.float()

    folder = 'new_geo_exports'  # Change this to your file destination folder in Google drive
    task = export_image(sarImage, shape_name + '_sar_vv_single_4', region, folder)


def main():

    # download_ndwi()
    download_sar()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))


