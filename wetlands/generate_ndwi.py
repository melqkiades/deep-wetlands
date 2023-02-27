import json
import os
import time

import numpy as np
import rasterio
from PIL import Image
from matplotlib import pyplot as plt
import rasterio as rio
from matplotlib.colors import ListedColormap
import rasterio.mask
from tqdm import tqdm

from wetlands import utils, viz_utils, geo_utils

from dotenv import load_dotenv, dotenv_values


def export_ndwi_mask_data(tiles, tif_file):

    export_folder = os.getenv('NDWI_MASK_DIR')
    patch_size = int(os.getenv('PATCH_SIZE'))

    with rio.open(tif_file) as src:
        dataset_array = src.read()
        minValue = np.nanpercentile(dataset_array, 1)
        maxValue = np.nanpercentile(dataset_array, 99)

    for index in tqdm(range(len(tiles)), total=len(tiles)):

        with rio.open(tif_file) as src:

            shape = [tiles.iloc[index]['geometry']]
            name = tiles.iloc[index]['id']
            # print('id', name)
            # print(type(src), type(shape))
            out_image, out_transform = rio.mask.mask(src, shape, crop=True)
            if np.isnan(out_image).any():
                raise ValueError(f'An image contains NaN values: {name}')
            # Crop out black (zero) border
            # _, x_nonzero, y_nonzero = np.nonzero(out_image)
            # out_image = out_image[
            #     :,
            #     np.min(x_nonzero):np.max(x_nonzero),
            #     np.min(y_nonzero):np.max(y_nonzero)
            # ]
            if out_image.shape[1] == patch_size + 1:
                out_image = out_image[:, :-1, :]
            if out_image.shape[2] == patch_size + 1:
                out_image = out_image[:, :, 1:]

            if out_image.shape[1] != patch_size or out_image.shape[2] != patch_size:
                continue

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
            temp_tif = export_folder + '/{}-ndwi_mask.tif'.format(name)
            with rasterio.open(temp_tif, "w", **out_meta) as dest:
                dest.write(out_image)

            # Save the cropped image as a temporary PNG file.
            temp_png = export_folder + '/{}-ndwi_mask.png'.format(name)

            # Get the color map by name:
            # cm = plt.get_cmap('viridis')
            cm = plt.get_cmap(ListedColormap(["black", "cyan"]))

            # Apply the colormap like a function to any array:
            colored_image = cm(out_image[0])

            # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
            # But we want to convert to RGB in uint8 and save it:
            Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(temp_png)


def full_cycle():
    file_name = os.getenv('GEOJSON_FILE')
    region_name = os.getenv('REGION_NAME')
    tif_file = os.getenv('NDWI_TIFF_FILE')
    country_code = os.getenv('COUNTRY_CODE')
    region_admin_level = os.getenv("REGION_ADMIN_LEVEL")
    patch_size = int(os.getenv("PATCH_SIZE"))

    utils.download_country_boundaries(country_code, region_admin_level, file_name)
    geoboundary = utils.get_region_boundaries(region_name, file_name)

    tiles = geo_utils.get_tiles(region_name, tif_file, geoboundary, patch_size)
    # print(tiles.columns.values)
    export_ndwi_mask_data(tiles, tif_file)
    # tiles['split'] = 'train'
    # num_rows = len(tiles)
    # test_rows = int(num_rows * 0.2)
    # tiles.loc[tiles.tail(test_rows).index, 'split'] = 'test'
    # tiles.to_csv('/tmp/my_tiles.csv', columns=['id', 'split'], index_label='index')


def full_cycle_with_visualization():

    file_name = os.getenv('GEOJSON_FILE')
    region_name = os.getenv('REGION_NAME')
    country_code = os.getenv('COUNTRY_CODE')
    region_admin_level = os.getenv("REGION_ADMIN_LEVEL")
    export_folder = os.getenv('NDWI_MASK_DIR')
    patch_size = int(os.getenv('PATCH_SIZE'))
    cwd = os.getenv('CWD_DIR')
    print('Shape name', region_name)

    utils.download_country_boundaries(country_code, region_admin_level, file_name)
    geoboundary = utils.get_region_boundaries(region_name, file_name)
    utils.show_region_boundaries(geoboundary, region_name)
    tif_file = os.getenv('NDWI_TIFF_FILE')
    viz_utils.visualize_sentinel2_image(geoboundary, region_name, tif_file)

    output_file = cwd + '{}.geojson'.format(region_name)
    tiles = geo_utils.generate_tiles(tif_file, output_file, region_name, size=patch_size)
    viz_utils.visualize_tiles(geoboundary, region_name, tif_file, tiles)
    tiles = geo_utils.get_tiles(region_name, tif_file, geoboundary, patch_size)
    # viz_utils.show_crop(tif_file, [tiles.iloc[10]['geometry']])
    # viz_utils.show_crop(tif_file, [tiles.iloc[20]['geometry']])
    # viz_utils.show_crop(tif_file, [tiles.iloc[30]['geometry']])
    # viz_utils.show_crop(tif_file, [tiles.iloc[40]['geometry']])
    # viz_utils.show_crop(tif_file, [tiles.iloc[50]['geometry']])
    # viz_utils.show_crop(tif_file, [tiles.iloc[60]['geometry']])
    # viz_utils.show_crop(tif_file, [tiles.iloc[70]['geometry']])
    # viz_utils.show_crop(tif_file, [tiles.iloc[80]['geometry']])
    viz_utils.show_crop(tif_file, [tiles.iloc[0]['geometry']])
    viz_utils.show_crop(tif_file, [tiles.iloc[1]['geometry']])
    viz_utils.show_crop(tif_file, [tiles.iloc[2]['geometry']])
    viz_utils.show_crop(tif_file, [tiles.iloc[3]['geometry']])
    viz_utils.show_crop(tif_file, [tiles.iloc[4]['geometry']])
    viz_utils.show_crop(tif_file, [tiles.iloc[5]['geometry']])
    viz_utils.show_crop(tif_file, [tiles.iloc[6]['geometry']])
    viz_utils.show_crop(tif_file, [tiles.iloc[7]['geometry']])
    viz_utils.show_crop(tif_file, [tiles.iloc[8]['geometry']])
    viz_utils.show_crop(tif_file, [tiles.iloc[9]['geometry']])

    print(tiles.count())

    # export_ndwi_mask_data(tiles, tif_file)
    # example_file = export_folder + '/sala_kommun-1533-ndwi_mask.tif'
    # viz_utils.visualize_image_from_file(example_file)


def main():
    load_dotenv()
    config = dotenv_values()
    print(json.dumps(config, indent=4))

    full_cycle()
    # full_cycle_with_visualization()


# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
