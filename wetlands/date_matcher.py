import time

from dotenv import load_dotenv
import ee


def get_flacksjon_geometry():
    area_of_interest = ee.Geometry.Polygon(
        [[[16.278247539412213, 59.84820707394825],
          [16.363563243757916, 59.84820707394825],
          [16.363563243757916, 59.88922451187625],
          [16.278247539412213, 59.88922451187625]]])

    return area_of_interest


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

    return dates


def get_matching_dates(area_of_interest):

    sentinel1_dates = get_sentinel1_dates(area_of_interest)
    sentinel2_dates = get_sentinel2_dates(area_of_interest)

    matching_dates = sorted(list(set(sentinel1_dates) & set(sentinel2_dates)))

    print(f'There are {len(matching_dates)} matching dates')
    print(matching_dates)

    return matching_dates


def main():
    load_dotenv()
    ee.Authenticate()
    ee.Initialize()

    get_matching_dates(get_flacksjon_geometry())


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
