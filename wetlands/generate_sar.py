import json
import os
import time

import numpy
import numpy as np
import pandas
import rasterio
import rasterio as rio
import rasterio.mask
from PIL import Image
from dotenv import load_dotenv, dotenv_values
from matplotlib import pyplot as plt
from tqdm import tqdm

from wetlands import utils, viz_utils, geo_utils


def export_sar_data(tiles, tif_file):
    export_folder = os.getenv('SAR_DIR')

    with rio.open(tif_file) as src:
        dataset_array = src.read()
        minValue = numpy.nanpercentile(dataset_array, 1)
        maxValue = numpy.nanpercentile(dataset_array, 99)

    exported_files = []

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
            if out_image.shape[1] == 65:
                out_image = out_image[:, :-1, :]
            if out_image.shape[2] == 65:
                out_image = out_image[:, :, 1:]

            if out_image.shape[1] != 64 or out_image.shape[2] != 64:
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
            temp_tif = export_folder + '/{}-sar.tif'.format(name)
            with rasterio.open(temp_tif, "w", **out_meta) as dest:
                dest.write(out_image)

            # Save the cropped image as a temporary PNG file.
            temp_png = export_folder + '/{}-sar.png'.format(name)

            # Get the color map by name:
            cm = plt.get_cmap('gray')

            # Apply the colormap like a function to any array:
            colored_image = cm(out_image[0])

            # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
            # But we want to convert to RGB in uint8 and save it:
            Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(temp_png)

            exported_files.append({'index': tiles.index.values[index], 'id': name})

    return exported_files


def full_cycle():
    file_name = os.getenv('GEOJSON_FILE')
    region_name = os.getenv('REGION_NAME')
    tif_file = os.getenv('SAR_TIFF_FILE')
    country_code = os.getenv('COUNTRY_CODE')
    region_admin_level = os.getenv("REGION_ADMIN_LEVEL")

    utils.download_country_boundaries(country_code, region_admin_level, file_name)
    geoboundary = utils.get_region_boundaries(region_name, file_name)

    tiles = geo_utils.get_tiles(region_name, tif_file, geoboundary)
    # export_sar_data(tiles, tif_file)
    tiles_list = export_sar_data(tiles, tif_file)

    tiles_dataframe = pandas.DataFrame(tiles_list)
    tiles_dataframe.set_index('index', inplace=True)
    tiles_dataframe['split'] = 'test'
    num_rows = len(tiles_dataframe)
    test_rows = int(num_rows * 0.8)
    tiles_dataframe.loc[tiles_dataframe.head(test_rows).index, 'split'] = 'train'
    tiles_file = os.getenv("TILES_FILE")
    tiles_dataframe.to_csv(tiles_file, columns=['id', 'split'], index_label='index')


def full_cycle_with_visualization():
    file_name = os.getenv('GEOJSON_FILE')
    region_name = os.getenv('REGION_NAME')
    sar_file = os.getenv('SAR_TIFF_FILE')
    country_code = os.getenv('COUNTRY_CODE')
    export_folder = os.getenv('SAR_DIR')
    cwd = os.getenv('CWD_DIR')
    region_admin_level = os.getenv("REGION_ADMIN_LEVEL")

    utils.download_country_boundaries(country_code, region_admin_level, file_name)
    geoboundary = utils.get_region_boundaries(region_name, file_name)
    utils.show_region_boundaries(geoboundary, region_name)
    viz_utils.visualize_sentinel2_image(geoboundary, region_name, sar_file)
    output_file = cwd + '{}.geojson'.format(region_name)
    tiles = geo_utils.generate_tiles(sar_file, output_file, region_name, size=64)
    viz_utils.visualize_tiles(geoboundary, region_name, sar_file, tiles)
    tiles = geo_utils.get_tiles(region_name, sar_file, geoboundary)
    viz_utils.show_crop(sar_file, [tiles.iloc[10]['geometry']])

    export_sar_data(tiles, sar_file)
    example_file = export_folder + '/sala_kommun-1533-sar.tif'
    viz_utils.visualize_image_from_file(example_file)


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
