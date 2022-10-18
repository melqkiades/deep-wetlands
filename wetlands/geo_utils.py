
import pandas as pd
import rasterio as rio
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import box


def generate_tiles(image_file, output_file, area_str, size=64):
    """Generates 64 x 64 polygon tiles.

    Args:
      image_file (str): Image file path (.tif)
      output_file (str): Output file path (.geojson)
      area_str (str): Name of the region
      size(int): Window size

    Returns:
      GeoPandas DataFrame: Contains 64 x 64 polygon tiles
    """

    # Open the raster image using rasterio
    raster = rio.open(image_file)
    width, height = raster.shape

    # Create a dictionary which will contain our 64 x 64 px polygon tiles
    # Later we'll convert this dict into a GeoPandas DataFrame.
    geo_dict = {'id': [], 'geometry': [], 'area': []}
    index = 0

    # Do a sliding window across the raster image
    with tqdm(total=width * height) as pbar:
        for w in range(0, width, size):
            for h in range(0, height, size):
                # Create a Window of your desired size
                window = rio.windows.Window(h, w, size, size)
                # Get the georeferenced window bounds
                bbox = rio.windows.bounds(window, raster.transform)
                # Create a shapely geometry from the bounding box
                bbox = box(*bbox)

                # Create a unique id for each geometry
                uid = '{}-{}'.format(area_str.lower().replace(' ', '_'), index)

                # Update dictionary
                geo_dict['id'].append(uid)
                geo_dict['area'].append(area_str)
                geo_dict['geometry'].append(bbox)

                index += 1
                pbar.update(size * size)

    # Cast dictionary as a GeoPandas DataFrame
    results = gpd.GeoDataFrame(pd.DataFrame(geo_dict))
    # Set CRS to EPSG:4326
    results.crs = {'init': 'epsg:4326'}
    # Save file as GeoJSON
    results.to_file(output_file, driver="GeoJSON")

    raster.close()

    return results


def get_tiles(shape_name, tif_file, geoboundary):
    cwd = '/Users/frape/Projects/DeepWetlands/src/deep-wetlands/external/data/Land Use and Land Cover Classification'
    output_file = cwd + '{}.geojson'.format(shape_name)
    tiles = generate_tiles(tif_file, output_file, shape_name, size=64)

    boundary = geoboundary[geoboundary.shapeName == shape_name]

    # Geopandas sjoin function
    tiles = gpd.sjoin(tiles, boundary, op='within')

    return tiles