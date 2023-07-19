import os
import time

import geetools
from dotenv import load_dotenv
import geopandas as gpd
import ee
import eeconvert as eec
from unidecode import unidecode

from wetlands import utils


def get_area_of_interest(area_name):

    areas_of_interest = {
        'flacksjon': ee.Geometry.Polygon(
            [[[16.278247539412213, 59.84820707394825],
              [16.363563243757916, 59.84820707394825],
              [16.363563243757916, 59.88922451187625],
              [16.278247539412213, 59.88922451187625]]]),
        'ojesjon': ee.Geometry.Polygon(
            [[[16.223059119542746, 59.86626918485007],
              [16.223059119542746, 59.84864122850999],
              [16.25893634732595, 59.84864122850999],
              [16.25893634732595, 59.86626918485007]]]),
        'kavsjon': ee.Geometry.Polygon(
            [[[13.926995174156906, 57.319539025703946],
              [13.926995174156906, 57.28968100640124],
              [13.978493587242843, 57.28968100640124],
              [13.978493587242843, 57.319539025703946]]]),
        'hornborgasjon': ee.Geometry.Polygon(
            [[[13.483334875467396, 58.368121823117356],
              [13.483334875467396, 58.2620882538591],
              [13.630963659647083, 58.2620882538591],
              [13.630963659647083, 58.368121823117356]]]),
        'eman': ee.Geometry.Polygon(
            [[[16.363172675766727, 57.15409709560314],
              [16.363172675766727, 57.13873142366187],
              [16.380167152085086, 57.13873142366187],
              [16.380167152085086, 57.15409709560314]]]),
        'karlhulteson': ee.Geometry.Polygon(
            [[[16.343431617417117, 57.15591261704456],
              [16.343431617417117, 57.14776534925009],
              [16.36137023130872, 57.14776534925009],
              [16.36137023130872, 57.15591261704456]]]),
        'huge_sweden': ee.Geometry.Polygon(
            [[[13.847571015930042, 62.112298691800014],
              [12.052864172203748, 58.768875396751994],
              [13.78167204881429, 56.13588308174233],
              [16.558946861713117, 60.44050284414736]]]),
        'medium_sweden': ee.Geometry.Polygon(
            [[[12.855710204458406, 60.22361199467255],
              [12.855710204458406, 58.60278344449708],
              [15.272702391958406, 58.60278344449708],
              [15.272702391958406, 60.22361199467255]]]),
        'small_sweden': ee.Geometry.Polygon(
            [[[12.855710204458406, 59.478737352511644],
              [12.855710204458406, 58.60278344449708],
              [15.272702391958406, 58.60278344449708],
              [15.272702391958406, 59.478737352511644]]]),
        'angarn': ee.Geometry.Polygon(
            [[[18.136216336105946, 59.540488418602365],
              [18.18448168655358, 59.54061727336778],
              [18.18441472424892, 59.55704443997286],
              [18.13634570498334, 59.55691564499851],
              [18.136216336105946, 59.540488418602365]]]),
    }

    return areas_of_interest[area_name]


def download_ndwi_mask(region):
    product = 'COPERNICUS/S2'
    shape_name = os.getenv("REGION_NAME")
    cloud_pct = 10

    image_collection = get_image_collection(product, region) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))

    image = aggregate_and_clip(image_collection, region)

    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')

    # Create NDWI mask
    ndwi_threshold = ndwi.gte(0.0)
    semi_ndwi_image = ndwi_threshold.neq(0.0)
    semi_ndwi_mask = ndwi_threshold.eq(0.0)
    new_image = semi_ndwi_mask.multiply(0.5).add(semi_ndwi_image.multiply(semi_ndwi_mask.neq(0.0)))
    new_image = new_image.add(semi_ndwi_image)

    folder = 'new_geo_exports'  # Change this to your file destination folder in Google drive
    start_date = os.getenv("START_DATE")
    aggregate_function = os.getenv("AGGREGATE_FUNCTION")
    file_name = f'{shape_name}_{aggregate_function}_{start_date}_ndwi_mask'
    task = export_image(new_image, file_name, region, folder)


def download_mndwi_mask(region):
    product = 'COPERNICUS/S2'
    shape_name = os.getenv("REGION_NAME")
    cloud_pct = 10

    image_collection = get_image_collection(product, region) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))

    image = aggregate_and_clip(image_collection, region)

    mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')

    # Create MNDWI mask
    mndwi_threshold = mndwi.gte(0.0)
    semi_mndwi_image = mndwi_threshold.neq(0.0)
    semi_mndwi_mask = mndwi_threshold.eq(0.0)
    new_image = semi_mndwi_mask.multiply(0.5).add(semi_mndwi_image.multiply(semi_mndwi_mask.neq(0.0)))
    new_image = new_image.add(semi_mndwi_image)

    folder = 'new_geo_exports'  # Change this to your file destination folder in Google drive
    start_date = os.getenv("START_DATE")
    aggregate_function = os.getenv("AGGREGATE_FUNCTION")
    file_name = f'{shape_name}_{aggregate_function}_{start_date}_mndwi_mask'
    task = export_image(mndwi, file_name, region, folder)


def download_ndwi_range(region):
    product = 'COPERNICUS/S2'
    shape_name = os.getenv("REGION_NAME")
    cloud_pct = 10

    image_collection = get_image_collection(product, region) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))

    image = aggregate_and_clip(image_collection, region)

    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI_RANGE')

    folder = 'new_geo_exports'  # Change this to your file destination folder in Google drive
    start_date = os.getenv("START_DATE")
    aggregate_function = os.getenv("AGGREGATE_FUNCTION")
    file_name = f'{shape_name}_{aggregate_function}_{start_date}_ndwi_range'
    task = export_image(ndwi, file_name, region, folder)


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
        # region=region,
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
    # file_name = f'small_sweden_sar_{polarization}'
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


def bulk_export_sar(area_name):
    start_date = '2014-01-01'
    end_date = '2023-12-31'
    orbit_pass = os.getenv("ORBIT_PASS")

    roi = get_area_of_interest(area_name)

    collection = ee.ImageCollection('COPERNICUS/S1_GRD')\
        .filterDate(start_date, end_date)\
        .filterBounds(roi)\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))\
        .filter(ee.Filter.eq('resolution', 'H'))\
        .filter(ee.Filter.eq('resolution_meters', 10))

    print('SAR Collection size:', collection.size().getInfo())

    # batch export to Google Drive
    geetools.batch.Export.imagecollection.toDrive(
        collection,
        f'bulk_export_sar_{area_name}',
        namePattern='{id}',
        scale=10,
        dataType="float",
        region=roi,
        crs='EPSG:4326',
        datePattern=None,
        extra=None,
        verbose=False
    )


def bulk_export_ndwi(area_name):
    start_date = '2014-01-01'
    end_date = '2023-12-31'

    roi = get_area_of_interest(area_name)

    def clip_image(image):
        return image.clip(roi)

    def create_ndwi_mask(image):
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI-collection')
        ndwi_threshold = ndwi.gte(0.0)
        semi_ndwi_image = ndwi_threshold.neq(0.0)
        semi_ndwi_mask = ndwi_threshold.eq(0.0)
        new_image = semi_ndwi_mask.multiply(0.5).add(semi_ndwi_image.multiply(semi_ndwi_mask.neq(0.0)))
        new_image = new_image.add(semi_ndwi_image).rename('NDWI-mask')

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
        f'bulk_export_{area_name}_ndwi',
        namePattern='{id}',
        scale=10,
        dataType="float",
        region=roi,
        crs='EPSG:4326',
        datePattern=None,
        extra=None,
        verbose=False
    )


def bulk_export_rgb(area_name):
    start_date = '2014-01-01'
    end_date = '2023-12-31'

    roi = get_area_of_interest(area_name)

    collection = ee.ImageCollection('COPERNICUS/S2')\
        .filterDate(start_date, end_date)\
        .filterBounds(roi)\
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 1))
    collection = collection.select(['B4', 'B3', 'B2'])

    print('RGB collection size:', collection.size().getInfo())

    # batch export to Google Drive
    geetools.batch.Export.imagecollection.toDrive(
        collection,
        f'bulk_export_{area_name}_rgb',
        namePattern='{id}',
        scale=10,
        dataType="float",
        region=roi,
        crs='EPSG:4326',
        datePattern=None,
        extra=None,
        verbose=False
    )


def get_image_date(image):
    return ee.Feature(None, {'date': image.date().format('YYYY-MM-dd')})


def get_sentinel1_dates(area_of_interest):

    sar_image_collection = ee.ImageCollection('COPERNICUS/S1_GRD')\
        .filterBounds(area_of_interest)\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
        .filter(ee.Filter.eq('instrumentMode', 'IW'))\
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))

    print("Number of SAR images = ", sar_image_collection.size().getInfo())

    dates = sar_image_collection.map(lambda image: get_image_date(image)) \
        .distinct('date') \
        .aggregate_array('date') \
        .getInfo()

    # print(dates)

    return dates


def get_sentinel2_dates(area_of_interest):

    optical_image_collection = ee.ImageCollection('COPERNICUS/S2')\
        .filterBounds(area_of_interest)\
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 1)

    print("Number of optical images = ", optical_image_collection.size().getInfo())

    dates = optical_image_collection.map(lambda image: get_image_date(image))\
        .distinct('date')\
        .aggregate_array('date') \
        .getInfo()

    # print(dates)

    return dates


def get_matching_dates(area_of_interest):

    sentinel1_dates = get_sentinel1_dates(area_of_interest)
    sentinel2_dates = get_sentinel2_dates(area_of_interest)

    matching_dates = list(set(sentinel1_dates) & set(sentinel2_dates))

    print(f'There are {len(matching_dates)} matching dates')
    print(matching_dates)

    return matching_dates


def main():
    load_dotenv()
    ee.Authenticate()
    ee.Initialize()

    country_code = os.getenv("COUNTRY_CODE")
    file_name = os.getenv("GEOJSON_FILE")
    region_admin_level = os.getenv("REGION_ADMIN_LEVEL")
    study_area = os.getenv("STUDY_AREA")
    utils.download_country_boundaries(country_code, region_admin_level, file_name)
    region = get_region()
    # region = get_area_of_interest('small_sweden')
    download_ndwi_mask(region)
    download_mndwi_mask(region)
    download_sar(region)
    # download_ndwi_range(region)
    # download_sar_vv_plus_vh(region)
    # bulk_export_sar(study_area)
    # bulk_export_ndwi(study_area)
    # bulk_export_rgb(study_area)
    # get_sentinel2_dates()
    # get_sentinel1_dates(get_flacksjon_geometry())
    # get_matching_dates(get_flacksjon_geometry())


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))


