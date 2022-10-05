import time

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from matplotlib import pyplot as plt
import rasterio as rio
from matplotlib.colors import ListedColormap
from rasterio.plot import show
import rasterio.mask
from shapely.geometry import box
from tqdm import tqdm
import geopandas as gpd

from wetlands import utils


def visualize_sentinel2_image(geoboundary, shape_name, tif_file):

    # Open image file using Rasterio
    image = rio.open(tif_file)
    boundary = geoboundary[geoboundary.shapeName == shape_name]

    # Plot image and corresponding boundary
    fig, ax = plt.subplots(figsize=(15, 15))
    boundary.plot(facecolor="none", edgecolor='red', ax=ax)
    show(image, ax=ax)
    print(type(image))
    plt.show()


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


def visualize_tiles(geoboundary, shape_name, tif_file):
    # cwd = './drive/My Drive/Colab Notebooks/Land Use and Land Cover Classification/'
    cwd = '/Users/frape/Projects/DeepWetlands/src/deep-wetlands/external/data/Land Use and Land Cover Classification'
    output_file = cwd + '{}.geojson'.format(shape_name)
    tiles = generate_tiles(tif_file, output_file, shape_name, size=64)
    print('Data dimensions: {}'.format(tiles.shape))
    tiles.head(3)

    image = rio.open(tif_file)
    fig, ax = plt.subplots(figsize=(15, 15))
    tiles.plot(facecolor="none", edgecolor='red', ax=ax)
    show(image, ax=ax)
    plt.show()

    image = rio.open(tif_file)
    boundary = geoboundary[geoboundary.shapeName == shape_name]

    # Geopandas sjoin function
    tiles = gpd.sjoin(tiles, boundary, op='within')

    fig, ax = plt.subplots(figsize=(15, 15))
    tiles.plot(facecolor="none", edgecolor='red', ax=ax)
    show(image, ax=ax)
    plt.show()

    return tiles


def get_tiles(shape_name, tif_file, geoboundary):
    cwd = '/Users/frape/Projects/DeepWetlands/src/deep-wetlands/external/data/Land Use and Land Cover Classification'
    output_file = cwd + '{}.geojson'.format(shape_name)
    tiles = generate_tiles(tif_file, output_file, shape_name, size=64)

    boundary = geoboundary[geoboundary.shapeName == shape_name]

    # Geopandas sjoin function
    tiles = gpd.sjoin(tiles, boundary, op='within')

    return tiles


def show_crop(image, shape, title=''):
    """Crops an image based on the polygon shape.
    Reference: https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html#rasterio.mask.mask

    Args:
        image (str): Image file path (.tif)
        shape (geometry): The tile with which to crop the image
        title(str): Image title
    """

    with rio.open(image) as src:
        out_image, out_transform = rio.mask.mask(src, shape, crop=True)
        # Crop out black (zero) border
        # _, x_nonzero, y_nonzero = np.nonzero(out_image)
        # out_image = out_image[
        #     :,
        #     np.min(x_nonzero):np.max(x_nonzero),
        #     np.min(y_nonzero):np.max(y_nonzero)
        # ]
        if out_image.shape[1] == 65:
            out_image = out_image[:, :-1, :]
        if out_image.shape[2] == 65:
            out_image = out_image[:, :, 1:]

        # Min-max scale the data to range [0, 1]
        # out_image[out_image > maxValue] = maxValue
        # out_image[out_image < minValue] = minValue
        # out_image = (out_image - minValue)/(maxValue - minValue)

        # Visualize image
        show(out_image, title=title)
        print(out_image.shape, type(out_image))
        print(out_image)
        plt.show()


def export_ndwi_mask_data(tiles, tif_file):

    minValue = 0.5
    maxValue = 1.0

    for index in tqdm(range(len(tiles)), total=len(tiles)):

        with rio.open(tif_file) as src:

            shape = [tiles.iloc[index]['geometry']]
            name = tiles.iloc[index]['id']
            # print('id', name)
            # print(type(src), type(shape))
            out_image, out_transform = rio.mask.mask(src, shape, crop=True)
            # Crop out black (zero) border
            # _, x_nonzero, y_nonzero = np.nonzero(out_image)
            # out_image = out_image[
            #     :,
            #     np.min(x_nonzero):np.max(x_nonzero),
            #     np.min(y_nonzero):np.max(y_nonzero)
            # ]
            if out_image.shape[1] == 65:
                out_image = out_image[:, :-1, :]
            if out_image.shape[2] == 65:
                out_image = out_image[:, :, 1:]

            # Min-max scale the data to range [0, 1]
            out_image[out_image > maxValue] = maxValue
            out_image[out_image < minValue] = minValue
            out_image = (out_image - minValue) / (maxValue - minValue)

            # Get the metadata of the source image and update it
            # with the width, height, and transform of the cropped image
            out_meta = src.meta
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            # Save the cropped image as a temporary TIFF file.
            temp_tif = '/tmp/ndwi_mask/{}-ndwi_mask.tif'.format(name)
            with rasterio.open(temp_tif, "w", **out_meta) as dest:
                dest.write(out_image)

            # Save the cropped image as a temporary PNG file.
            temp_png = '/tmp/ndwi_mask/{}-ndwi_mask.png'.format(name)

            # Get the color map by name:
            # cm = plt.get_cmap('viridis')
            cm = plt.get_cmap(ListedColormap(["black", "blue"]))

            # Apply the colormap like a function to any array:
            colored_image = cm(out_image[0])

            # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
            # But we want to convert to RGB in uint8 and save it:
            Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(temp_png)


def visualize_image_from_file(tif_file):
    # Open image file using Rasterio
    image = rio.open(tif_file)

    # Plot image and corresponding boundary
    fig, ax = plt.subplots(figsize=(15, 15))
    show(image, ax=ax)
    print(image.count)
    print(image.read(1).shape)
    plt.show()


def full_cycle():
    file_name = '/tmp/sweden.geojson'
    shape_name = 'Sala kommun'
    tif_file = '/Users/frape/Projects/DeepWetlands/src/deep-wetlands/external/data/{}_new_image.tif'.format(shape_name)

    utils.download_country_boundaries('SWE', 'ADM2', file_name)
    geoboundary = utils.get_region_boundaries(shape_name, file_name)

    tiles = get_tiles(shape_name, tif_file, geoboundary)
    export_ndwi_mask_data(tiles, tif_file)


def full_cycle_with_visualization():

    file_name = '/tmp/sweden.geojson'
    shape_name = 'Sala kommun'
    utils.download_country_boundaries('SWE', 'ADM2', file_name)
    geoboundary = utils.get_region_boundaries(shape_name, file_name)
    utils.show_region_boundaries(geoboundary, shape_name)
    tif_file = '/Users/frape/Projects/DeepWetlands/src/deep-wetlands/external/data/{}_new_image.tif'.format(shape_name)
    visualize_sentinel2_image(geoboundary, shape_name, tif_file)
    visualize_tiles(geoboundary, shape_name, tif_file)
    tiles = get_tiles(shape_name, tif_file, geoboundary)
    show_crop(tif_file, [tiles.iloc[10]['geometry']])

    export_ndwi_mask_data(tiles, tif_file)
    example_file = '/tmp/ndwi_mask/sala_kommun-1533-ndwi_mask.tif'
    visualize_image_from_file(example_file)


def main():
    full_cycle()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
