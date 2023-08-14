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

from wetlands import utils, geo_utils

from dotenv import load_dotenv, dotenv_values


def export_ndwi_mask_data(tiles, tif_file):

    export_folder = os.getenv('NDWI_MASK_DIR')
    patch_size = int(os.getenv('PATCH_SIZE'))

    with rio.open(tif_file) as src:
        dataset_array = src.read()
        minValue = np.nanpercentile(dataset_array, 1)
        maxValue = np.nanpercentile(dataset_array, 99)

    nan_tiles = 0

    for index in tqdm(range(len(tiles)), total=len(tiles)):

        with rio.open(tif_file) as src:

            shape = [tiles.iloc[index]['geometry']]
            name = tiles.iloc[index]['id']
            out_image, out_transform = rio.mask.mask(src, shape, crop=True)
            if np.isnan(out_image).any():
                nan_tiles += 1
                continue

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

    if nan_tiles > 0:
        print(f'Warning: There were {nan_tiles} tiles with NaN values.')


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
    export_ndwi_mask_data(tiles, tif_file)


def main():
    load_dotenv()
    config = dotenv_values()
    print(json.dumps(config, indent=4))

    full_cycle()


# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
