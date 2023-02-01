import os
import time

import geetools
from dotenv import load_dotenv
import geopandas as gpd
import ee
import eeconvert as eec
from unidecode import unidecode

from wetlands import utils


def download_ndwi(region):
    product = 'COPERNICUS/S2'
    shape_name = os.getenv("REGION_NAME")
    cloud_pct = 10

    image_collection = get_image_collection(product, region) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))

    image = aggregate_and_clip(image_collection, region)

    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')

    # Create NDWI mask
    ndwiThreshold = ndwi.gte(0.0)
    ndwiMask = ndwiThreshold.updateMask(ndwiThreshold)
    semiNdwiImage = ndwiThreshold.neq(0.0)
    semiNdwiMask = ndwiThreshold.eq(0.0)
    new_image = semiNdwiMask.multiply(0.5).add(semiNdwiImage.multiply(semiNdwiMask.neq(0.0)))
    new_image = new_image.add(semiNdwiImage)
    semiNdwiMask = semiNdwiMask.multiply(0.5).add(ndwiThreshold.multiply(ndwiThreshold.eq(0.0)))

    folder = 'new_geo_exports'  # Change this to your file destination folder in Google drive
    start_date = os.getenv("START_DATE")
    aggregate_function = os.getenv("AGGREGATE_FUNCTION")
    file_name = f'{shape_name}_{aggregate_function}_{start_date}_ndwi_mask'
    newImageTask = export_image(new_image, file_name, region, folder)


def get_region():
    geojson_file = os.getenv("GEOJSON_FILE")

    # Read data using GeoPandas
    geoboundary = gpd.read_file(geojson_file)
    print("Data dimensions: {}".format(geoboundary.shape))

    shape_name = os.getenv('REGION_NAME')

    # Get the shape geometry
    region = geoboundary.loc[geoboundary.shapeName == shape_name]
    region = eec.gdfToFc(region)

    return region


def get_image_collection(product, region):

    start_date = os.getenv("START_DATE")
    end_date = os.getenv("END_DATE")

    image_collection = ee.ImageCollection(product) \
        .filterBounds(region) \
        .filterDate(str(start_date), str(end_date))

    return image_collection


def aggregate_and_clip(image_collection, region):
    aggregate_function = os.getenv("AGGREGATE_FUNCTION")

    if aggregate_function == "median":
        image = image_collection.median().clip(region)
    elif aggregate_function == "mosaic":
        image = image_collection.mosaic().clip(region)
    else:
        raise ValueError(f"Unknown aggregate function: {aggregate_function}")

    return image


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
        description=unidecode(filename),
        fileFormat='GeoTIFF',
        crs='EPSG:4326',
        maxPixels=1e10
    )
    task.start()

    return task


def download_sar(region):
    product = 'COPERNICUS/S1_GRD'
    shape_name = os.getenv("REGION_NAME")
    polarization = os.getenv("SAR_POLARIZATION")
    orbit_pass = os.getenv("ORBIT_PASS")

    image_collection = get_image_collection(product, region)

    image_collection = image_collection \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization)) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))

    sar_image = aggregate_and_clip(image_collection, region)

    band_names = sar_image.bandNames()
    print(band_names.getInfo())

    sar_image = sar_image.select([polarization])

    percentiles = sar_image.reduceRegion(
        reducer=ee.Reducer.percentile([0, 1, 5, 50, 95, 99, 100]),
        geometry=region,
        scale=10,
        maxPixels=1e10
    )

    min_value = percentiles.get(f"{polarization}_p1").getInfo()
    max_value = percentiles.get(f"{polarization}_p99").getInfo()

    print(min_value)
    print(max_value)

    sar_image = sar_image.float()

    folder = 'new_geo_exports'  # Change this to your file destination folder in Google drive
    start_date = os.getenv("START_DATE")
    aggregate_function = os.getenv("AGGREGATE_FUNCTION")
    file_name = f'{shape_name}_{aggregate_function}_{start_date}_sar_{polarization}'
    task = export_image(sar_image, file_name, region, folder)


def download_sar_vv_plus_vh(region):
    product = 'COPERNICUS/S1_GRD'
    shape_name = os.getenv("REGION_NAME")
    polarization = os.getenv("SAR_POLARIZATION")
    orbit_pass = os.getenv("ORBIT_PASS")

    image_collection = get_image_collection(product, region)

    image_collection = image_collection \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization)) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))

    sar_image = aggregate_and_clip(image_collection, region)
    sar_image = sar_image.float()

    vv_image = sar_image.select(['VV'])
    vh_image = sar_image.select(['VH'])
    vv_plus_vh_image = vv_image.add(vh_image).rename('VV_pls_VH')
    vv_minus_vh_image = vv_image.subtract(vh_image).rename('VV-VH')
    vh_minus_vv_image = vh_image.subtract(vv_image).rename('VH-VV')
    vv_over_vh_image = vv_image.divide(vh_image).rename('VH/VV')
    vh_over_vv_image = vh_image.divide(vv_image).rename('VV/VH')
    nd_vv_vh_image = sar_image.normalizedDifference(['VV', 'VH']).rename('ND_VV-VH')
    nd_vh_vv_image = sar_image.normalizedDifference(['VH', 'VV']).rename('ND_VH-VV')

    folder = 'new_geo_exports'  # Change this to your file destination folder in Google drive
    start_date = os.getenv("START_DATE")
    aggregate_function = os.getenv("AGGREGATE_FUNCTION")

    # file_name = f'{shape_name}_sar_vv_plus_vh_{aggregate_function}_{start_date}'
    # export_image(vv_plus_vh_image, file_name, region, folder)
    #
    # file_name = f'{shape_name}_sar_vv_minus_vh_{aggregate_function}_{start_date}'
    # export_image(vv_minus_vh_image, file_name, region, folder)
    #
    # file_name = f'{shape_name}_sar_vh_minus_vv_{aggregate_function}_{start_date}'
    # export_image(vh_minus_vv_image, file_name, region, folder)
    #
    # file_name = f'{shape_name}_sar_vv_over_vh_{aggregate_function}_{start_date}'
    # export_image(vv_over_vh_image, file_name, region, folder)
    #
    # file_name = f'{shape_name}_sar_vh_over_vv_{aggregate_function}_{start_date}'
    # export_image(vh_over_vv_image, file_name, region, folder)

    file_name = f'{shape_name}_sar_nd_vv-vh_{aggregate_function}_{start_date}'
    export_image(nd_vv_vh_image, file_name, region, folder)

    file_name = f'{shape_name}_sar_nd_vh-vv_{aggregate_function}_{start_date}'
    export_image(nd_vh_vv_image, file_name, region, folder)


def bulk_export_sar_flacksjon():
    startDate = '2023-01-01'
    endDate = '2023-12-31'
    orbit_pass = os.getenv("ORBIT_PASS")

    roi = ee.Geometry.Polygon(
        [[[16.278247539412213, 59.84820707394825],
          [16.363563243757916, 59.84820707394825],
          [16.363563243757916, 59.88922451187625],
          [16.278247539412213, 59.88922451187625]]])

    collection = ee.ImageCollection('COPERNICUS/S1_GRD')\
        .filterDate(startDate, endDate)\
        .filterBounds(roi)\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))\
        .filter(ee.Filter.eq('resolution', 'H'))\
        .filter(ee.Filter.eq('resolution_meters', 10))

    print('Collection size:', collection.size().getInfo())

    # batch export to Google Drive
    geetools.batch.Export.imagecollection.toDrive(
        collection,
        'bulk_export_flacksjon_2014',
        namePattern='{id}',
        scale=10,
        dataType="float",
        region=roi,
        crs='EPSG:4326',
        datePattern=None,
        extra=None,
        verbose=False
    )


def bulk_export_ndwi_flacksjon():
    start_date = '2019-01-01'
    end_date = '2022-12-31'

    roi = ee.Geometry.Polygon(
        [[[16.278247539412213, 59.84820707394825],
          [16.363563243757916, 59.84820707394825],
          [16.363563243757916, 59.88922451187625],
          [16.278247539412213, 59.88922451187625]]])

    def clip_image(image):
        return image.clip(roi)

    def create_ndwi_mask(image):
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI-collection')
        ndwiThreshold = ndwi.gte(0.0)
        semiNdwiImage = ndwiThreshold.neq(0.0)
        semiNdwiMask = ndwiThreshold.eq(0.0)
        new_image = semiNdwiMask.multiply(0.5).add(semiNdwiImage.multiply(semiNdwiMask.neq(0.0)))
        new_image = new_image.add(semiNdwiImage).rename('NDWI-mask')

        return new_image
        # return ee.Image(image).addBands(ndwi).addBands(new_image).copyProperties(image)

    collection = ee.ImageCollection('COPERNICUS/S2')\
        .filterDate(start_date, end_date)\
        .filterBounds(roi)\
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 1))\
        .map(lambda image: clip_image(image))\
        .map(lambda image: create_ndwi_mask(image))

    print('NDWI collection size:', collection.size().getInfo())

    # batch export to Google Drive
    geetools.batch.Export.imagecollection.toDrive(
        collection,
        'bulk_export_flacksjon_ndwi',
        namePattern='{id}',
        scale=10,
        dataType="float",
        region=roi,
        crs='EPSG:4326',
        datePattern=None,
        extra=None,
        verbose=False
    )


def bulk_export_rgb_flacksjon():
    start_date = '2019-01-01'
    end_date = '2022-12-31'

    roi = ee.Geometry.Polygon(
        [[[16.278247539412213, 59.84820707394825],
          [16.363563243757916, 59.84820707394825],
          [16.363563243757916, 59.88922451187625],
          [16.278247539412213, 59.88922451187625]]])

    collection = ee.ImageCollection('COPERNICUS/S2')\
        .filterDate(start_date, end_date)\
        .filterBounds(roi)\
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 1))
    collection = collection.select(['B4', 'B3', 'B2'])

    print('RGB collection size:', collection.size().getInfo())

    # batch export to Google Drive
    geetools.batch.Export.imagecollection.toDrive(
        collection,
        'bulk_export_flacksjon_rgb',
        namePattern='{id}',
        scale=10,
        dataType="float",
        region=roi,
        crs='EPSG:4326',
        datePattern=None,
        extra=None,
        verbose=False
    )


def main():
    load_dotenv()
    ee.Authenticate()
    ee.Initialize()

    country_code = os.getenv("COUNTRY_CODE")
    file_name = os.getenv("GEOJSON_FILE")
    region_admin_level = os.getenv("REGION_ADMIN_LEVEL")
    utils.download_country_boundaries(country_code, region_admin_level, file_name)
    region = get_region()
    download_ndwi(region)
    download_sar(region)
    # download_sar_vv_plus_vh(region)
    # bulk_export_sar_flacksjon()
    # bulk_export_ndwi_flacksjon()
    # bulk_export_rgb_flacksjon()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))


