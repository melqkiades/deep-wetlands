import os
import time
from datetime import datetime, date

import ee
import eeconvert as eec
import geopandas as gpd
import numpy as np
from dotenv import load_dotenv
import rasterio as rio
from matplotlib import pyplot as plt

from wetlands import utils


def download_train_region():
    file_name = os.getenv('GEOJSON_FILE')
    country_code = os.getenv('COUNTRY_CODE')

    utils.download_country_boundaries(country_code, 'ADM2', file_name)


def download_sar_train_region():
    ee.Authenticate()
    ee.Initialize()

    product = 'COPERNICUS/S1_GRD'

    # Get the shape geometry for Sala kommun
    geoboundary = gpd.read_file(os.getenv('GEOJSON_FILE'))
    shape_name = os.getenv('REGION_NAME')
    polarization = os.getenv('SAR_POLARIZATION')

    region = geoboundary.loc[geoboundary.shapeName == shape_name]
    region = eec.gdfToFc(region)
    train_date = ee.Date(os.getenv('TRAIN_DATE'))

    sar_image_collection = ee.ImageCollection(product) \
        .filterBounds(region) \
        .filterDate(train_date, train_date.advance(1, 'day')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization)) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))

    num_images = int(sar_image_collection.size().getInfo())
    print("Number of SAR images = ", num_images)

    if num_images != 1:
        raise RuntimeError(f'Expected 1 image for the given date, instead found {num_images} images')

    sar_image = sar_image_collection.first().clip(region)
    band_names = sar_image.bandNames()
    print('All SAR band names', band_names.getInfo())
    # sar_image = sar_image.select('V.+')
    sar_image = sar_image.select(polarization)
    band_names = sar_image.bandNames()
    print('Selected SAR band names', band_names.getInfo())
    date_miliseconds = sar_image.date().getInfo()['value']
    image_date = str(date.fromtimestamp(date_miliseconds / 1000.0))
    print('SAR image date', image_date)

    sar_projection = sar_image.select(['VV']).projection().getInfo()
    print('SAR projection', sar_projection)
    print('CRS', sar_projection['crs'])
    print(create_file_name(sar_image, 'SAR', image_date, True))

    export_image(sar_image, region, 'SAR', 'geo_exports', image_date, True)


def download_optical_train_region():
    ee.Authenticate()
    ee.Initialize()

    product = 'COPERNICUS/S2'
    cloud_pct = 10

    # Get the shape geometry for Sala kommun
    geoboundary = gpd.read_file(os.getenv('GEOJSON_FILE'))
    shape_name = os.getenv('REGION_NAME')

    region = geoboundary.loc[geoboundary.shapeName == shape_name]
    region = eec.gdfToFc(region)
    train_date = ee.Date(os.getenv('TRAIN_DATE'))

    optical_image_collection = ee.ImageCollection(product) \
        .filterBounds(region) \
        .filterDate(train_date, train_date.advance(1, 'day'))
        # .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))

    num_images = int(optical_image_collection.size().getInfo())
    print("Number of Optical images = ", num_images)

    # optical_image = optical_image_collection.first().clip(region)
    optical_image = optical_image_collection.mosaic().clip(region)
    band_names = optical_image.bandNames()
    print('All optical band names', band_names.getInfo())
    optical_image = optical_image.select('B.+')
    # optical_image = optical_image.select(polarization)
    band_names = optical_image.bandNames()
    print('Selected optical band names', band_names.getInfo())
    date_miliseconds = optical_image_collection.first().date().getInfo()['value']
    image_date = str(date.fromtimestamp(date_miliseconds / 1000.0))
    print('Optical image date', image_date)

    optical_projection = optical_image_collection.first().select('B2').projection().getInfo()
    print('Optical projection', optical_projection)
    print('CRS', optical_projection['crs'])
    print(create_file_name(optical_image, 'optical', image_date, False))

    export_image(optical_image, region, 'Optical', 'geo_exports', image_date)


def download_ndwi_train_region():
    ee.Authenticate()
    ee.Initialize()

    product = 'COPERNICUS/S2'
    cloud_pct = 10

    # Get the shape geometry for Sala kommun
    geoboundary = gpd.read_file(os.getenv('GEOJSON_FILE'))
    shape_name = os.getenv('REGION_NAME')

    region = geoboundary.loc[geoboundary.shapeName == shape_name]
    region = eec.gdfToFc(region)
    train_date = ee.Date(os.getenv('TRAIN_DATE'))

    optical_image_collection = ee.ImageCollection(product) \
        .filterBounds(region) \
        .filterDate(train_date, train_date.advance(1, 'day')) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))

    num_images = int(optical_image_collection.size().getInfo())
    print("Number of Optical images = ", num_images)

    optical_image = optical_image_collection.mosaic().clip(region)

    date_miliseconds = optical_image_collection.first().date().getInfo()['value']
    image_date = str(date.fromtimestamp(date_miliseconds / 1000.0))
    print('Optical image date', image_date)

    ndwi_image = optical_image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndwi_projection = ndwi_image.select(['NDWI']).projection().getInfo()
    band_names = ndwi_image.bandNames()
    print('Selected optical band names', band_names.getInfo())
    print('NDWI projection', ndwi_projection)
    print('CRS', ndwi_projection['crs'])
    transform = ndwi_projection['transform']
    print('Transform', transform)
    print(create_file_name(ndwi_image, 'ndwi', image_date, True))

    export_image(ndwi_image, region, 'NDWI', 'geo_exports', image_date, True)


def create_ndwi_mask():
    tiff_file = os.getenv('OPTICAL_TIFF_FILE')
    tiff_image = rio.open(tiff_file)
    print(tiff_image.read().shape)

    green_band = tiff_image.read([3]).astype(float)
    nir_band = tiff_image.read([8]).astype(float)
    ndwi_mask = (green_band - nir_band) / (green_band + nir_band)
    ndwi_mask[ndwi_mask >= 0.0] = 1
    ndwi_mask[ndwi_mask < 0.0] = 0
    ndwi_mask = ndwi_mask[0]

    ndwi_file = '/tmp/ndwi.tiff'

    with rio.open(
            ndwi_file,
            mode='w',
            driver='GTiff',
            height=ndwi_mask.shape[0],
            width=ndwi_mask.shape[1],
            count=1,
            dtype=ndwi_mask.dtype,
            crs=tiff_image.crs,
            nodata=None,  # change if data has nodata value
            transform=tiff_image.transform
    ) as ndwi_dataset:
        ndwi_dataset.write(ndwi_mask, 1)


    # Export NDWI mask to TIFF file



def export_image(image, region, sensor, folder, image_date, single_band=False):
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

    file_name = create_file_name(image, sensor, image_date, single_band)

    print('Exporting to {}.tif ...'.format(file_name))

    band = image.bandNames().getInfo()[0]
    projection = image.select([band]).projection().getInfo()
    # transform = projection['transform']
    transform = [10, 0, 499980, 0, -10, 6700020]

    task = ee.batch.Export.image.toDrive(
        image=image,
        driveFolder=folder,
        # scale=10,
        region=region.geometry(),
        description=file_name,
        fileFormat='GeoTIFF',
        crs=projection['crs'],
        crsTransform=transform
    )
    task.start()

    return task


def create_file_name(image, sensor, image_date, single_band=True):
    country_code = os.getenv('COUNTRY_CODE')
    shape_name = os.getenv('REGION_NAME')

    band = image.bandNames().getInfo()[0]
    crs = image.select([band]).projection().getInfo()['crs']
    # image_date = date.fromtimestamp(image.date().getInfo()['value'] / 1000.0)

    if single_band:
        file_name = f'{sensor}_{band}_{country_code}_{shape_name}_{crs}_{image_date}'
    else:
        file_name = f'{sensor}_{country_code}_{shape_name}_{crs}_{image_date}'
    file_name = file_name.replace(' ', '-').replace(':', '-').lower()

    return file_name


def align_tiff():
    sar_file = '/tmp/sar_swe_sala-kommun_epsg-32633_2018-07-04_2.tif'
    ndwi_file = '/tmp/ndwi_swe_sala-kommun_epsg-32633_2018-07-04_2.tif'

    sar_image = rio.open(sar_file)
    print(sar_image.read().shape)
    # numpy_image = tiff_image.read([4, 3, 2])
    sar_numpy = sar_image.read([1])[0]

    ndwi_image = rio.open(ndwi_file)
    print(ndwi_image.read().shape)
    # numpy_image = tiff_image.read([4, 3, 2])
    ndwi_numpy = ndwi_image.read([1])[0]

    sar_numpy = np.where(np.isnan(ndwi_numpy), ndwi_numpy, sar_numpy)
    ndwi_numpy = np.where(np.isnan(sar_numpy), sar_numpy, ndwi_numpy)

    aligned_sar_file = '/tmp/sar_swe_sala-kommun_epsg-32633_2018-07-04_aligned.tif'

    with rio.open(
            aligned_sar_file,
            mode='w',
            driver='GTiff',
            height=sar_numpy.shape[0],
            width=sar_numpy.shape[1],
            count=1,
            dtype=sar_numpy.dtype,
            crs=sar_image.crs,
            nodata=None,  # change if data has nodata value
            transform=sar_image.transform
    ) as sar_dataset:
        sar_dataset.write(sar_numpy, 1)

    aligned_ndwi_file = '/tmp/ndwi_swe_sala-kommun_epsg-32633_2018-07-04_aligned.tif'

    with rio.open(
            aligned_ndwi_file,
            mode='w',
            driver='GTiff',
            height=ndwi_numpy.shape[0],
            width=ndwi_numpy.shape[1],
            count=1,
            dtype=ndwi_numpy.dtype,
            crs=ndwi_image.crs,
            nodata=None,  # change if data has nodata value
            transform=ndwi_image.transform
    ) as ndwi_dataset:
        ndwi_dataset.write(ndwi_numpy, 1)


def create_ndwi_mask_2():
    ndwi_file = '/tmp/ndwi_swe_sala-kommun_epsg-32633_2018-07-04_aligned.tif'
    ndwi_image = rio.open(ndwi_file)
    print(ndwi_image.read().shape)
    # numpy_image = tiff_image.read([4, 3, 2])
    ndwi_numpy = ndwi_image.read([1])[0]

    ndwi_numpy[ndwi_numpy >= 0.0] = 1
    ndwi_numpy[ndwi_numpy < 0.0] = 0

    # plt.imshow(ndwi_numpy, cmap='gray')
    # plt.show()

    ndwi_mask_file = '/tmp/ndwi_mask_swe_sala-kommun_epsg-32633_2018-07-04_aligned.tif'

    with rio.open(
            ndwi_mask_file,
            mode='w',
            driver='GTiff',
            height=ndwi_numpy.shape[0],
            width=ndwi_numpy.shape[1],
            count=1,
            dtype=ndwi_numpy.dtype,
            crs=ndwi_image.crs,
            nodata=None,  # change if data has nodata value
            transform=ndwi_image.transform
    ) as ndwi_dataset:
        ndwi_dataset.write(ndwi_numpy, 1)


def download_optical_train_region_2():
    ee.Authenticate()
    ee.Initialize()

    product = 'COPERNICUS/S2'
    cloud_pct = 10

    # Get the shape geometry for Sala kommun
    geoboundary = gpd.read_file(os.getenv('GEOJSON_FILE'))
    shape_name = os.getenv('REGION_NAME')

    region = geoboundary.loc[geoboundary.shapeName == shape_name]
    region = eec.gdfToFc(region)
    train_date = ee.Date(os.getenv('TRAIN_DATE'))


    # // Returns all the images between the start date and the end date
    # // taken on the area of interest
    opticalImageCollection = ee.ImageCollection('COPERNICUS/S2')\
        .filterBounds(region)\
        .filterDate(train_date, train_date.advance(1, 'day'))

    print("Number of optical images = ", opticalImageCollection.size().getInfo())

    opticalImage = opticalImageCollection.first().clip(region)
    # opticalImage = opticalImageCollection.mosaic().clip(region)
    opticalImage = opticalImage.select(['B2', 'B3', 'B4', 'B8'])

    date_miliseconds = opticalImageCollection.first().date().getInfo()['value']
    image_date = str(date.fromtimestamp(date_miliseconds / 1000.0))
    print('Optical image date', image_date)

    print("Opical image taken at = ", opticalImageCollection.first().date())

    # export_image(opticalImage, region, 'Optical', 'geo_exports', image_date, True)

    projection = opticalImageCollection.first().select('B2').projection().getInfo()
    print(projection)
    # Export.image.toDrive({
    #   image: opticalImage,
    #   description: 'mosaic_test_3',
    #   # // crs: 'EPSG:4326',
    #   # // scale: 10,
    #   crs: projection.crs,
    #   crsTransform: projection.transform,
    #   region: areaOfInterest
    # });







def main():
    load_dotenv()

    # download_train_region()
    # download_sar_train_region()
    download_optical_train_region_2()
    # download_ndwi_train_region()
    # create_ndwi_mask()
    # align_tiff()
    # create_ndwi_mask_2()



start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))