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
from pathlib import Path
from wetlands import utils, geo_utils

from dotenv import load_dotenv, dotenv_values
import glob


def export_ndwi_mask_data(tiles, tif_file):
    patch_size = int(os.getenv('PATCH_SIZE'))
    export_folder = os.getenv('NDWI_MASK_DIR')
    Path(export_folder).mkdir(parents=True, exist_ok=True)

    # with rio.open(tif_file) as src:
    #     dataset_array = src.read()
    #     plt.imshow(np.nan_to_num(dataset_array[0]))
    #     plt.show()
    #     minValue = np.nanpercentile(dataset_array, 1)
    #     maxValue = np.nanpercentile(dataset_array, 99)

    nan_tiles = 0

    for index in tqdm(range(len(tiles)), total=len(tiles)):

        with rio.open(tif_file) as src:

            shape = [tiles.iloc[index]['geometry']]
            name = tiles.iloc[index]['id']
            out_image, out_transform = rio.mask.mask(src, shape, crop=True)
            # if np.any(out_image==0):
            #     a=0
            if np.isnan(out_image).any():
                nan_tiles += 1
                continue

            if out_image.shape[1] == patch_size + 1:
                if np.all(out_image==0, axis=2)[0][0]:
                    out_image = out_image[:, 1:, :]
                else:
                    out_image = out_image[:, :-1, :]
            if out_image.shape[2] == patch_size + 1:
                if np.all(out_image == 0, axis=1)[0][0]:
                    out_image = out_image[:, :, 1:]
                else:
                    out_image = out_image[:, :, :-1]

            if out_image.shape[1] != patch_size or out_image.shape[2] != patch_size:
                continue

            # Min-max scale the data to range [0, 1]
            # out_image[out_image > maxValue] = maxValue
            # out_image[out_image < minValue] = minValue
            # out_image = (out_image - minValue) / (maxValue - minValue)

            out_image[out_image==0.5] = 0

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


def export_ndwi_mask_data_new(tiles, tif_file, date, patch_size, area_name='Örebro län'):
    export_folder = 'C:/Users/ioia4268/data/ndpi_masks_tiles/' + area_name + '_'+date + '_' + str(patch_size) + 'x' + str(patch_size)
    Path(export_folder).mkdir(parents=True, exist_ok=True)

    # with rio.open(tif_file) as src:
    #     dataset_array = src.read()
    #     plt.imshow(np.nan_to_num(dataset_array[0]))
    #     plt.show()
    #     minValue = np.nanpercentile(dataset_array, 1)
    #     maxValue = np.nanpercentile(dataset_array, 99)

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
                if np.all(out_image==0, axis=2)[0][0]:
                    out_image = out_image[:, 1:, :]
                else:
                    out_image = out_image[:, :-1, :]
            if out_image.shape[2] == patch_size + 1:
                if np.all(out_image == 0, axis=1)[0][0]:
                    out_image = out_image[:, :, 1:]
                else:
                    out_image = out_image[:, :, :-1]

            if out_image.shape[1] != patch_size or out_image.shape[2] != patch_size:
                continue
            # if out_image.shape[1] != patch_size or out_image.shape[2] != patch_size:
            #     out_image = np.pad(out_image, ((0,0), (0, patch_size - out_image.shape[1]), (0, patch_size - out_image.shape[2])), 'constant', constant_values=-1)


            # Min-max scale the data to range [0, 1]
            # out_image[out_image > maxValue] = maxValue
            # out_image[out_image < minValue] = minValue
            # out_image = (out_image - minValue) / (maxValue - minValue)

            ####### out_image[out_image == 0.5] = 0


            # Get the metadata of the source image and update it
            # with the width, height, and transform of the cropped image
            out_meta = src.meta
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "dtype":'float64'
            })

            # Save the cropped image as a temporary TIFF file.
            # temp_tif = export_folder + '/{}-ndwi_mask.tif'.format(name)
            temp_tif = export_folder + '/{}-ndpi.tif'.format(name)
            with rasterio.open(temp_tif, "w", **out_meta) as dest:
                # dest.write((out_image/255.0).astype(np.float64))
                dest.write((out_image).astype(np.float64))

            # Save the cropped image as a temporary PNG file.
            # temp_png = export_folder + '/{}-ndwi_mask.png'.format(name)
            #
            # # Get the color map by name:
            # # cm = plt.get_cmap('viridis')
            # cm = plt.get_cmap(ListedColormap(["black", "cyan"]))
            #
            # # Apply the colormap like a function to any array:
            # colored_image = cm(out_image[0])
            #
            # # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
            # # But we want to convert to RGB in uint8 and save it:
            # Image.fromarray(colored_image[:, :, :3].astype(np.uint8)).save(temp_png)

    if nan_tiles > 0:
        print(f'Warning: There were {nan_tiles} tiles with NaN values.')


def full_cycle():
    patch_size = 64
    tif_files = glob.glob("C:/Users/ioia4268/data/sar_ndpi/Orebro_lan/*")
    # tif_files = ["C:/Users/ioia4268/data/ndwi_masks/Örebro län/Orebro lan_mosaic_2018-07-04_ndwi_mask.tif",
    #              "C:/Users/ioia4268/data/ndwi_masks/Örebro län/Orebro lan_mosaic_2020-06-23_ndwi_mask.tif"]
    area_name = "Orebro_lan"
    for tif_file in tif_files:
        tiles = geo_utils.get_tiles_batch(area_name, tif_file, patch_size)
        export_ndwi_mask_data_new(tiles, tif_file, tif_file.split("_")[-2], patch_size, area_name)



def main():
    load_dotenv()
    config = dotenv_values()
    print(json.dumps(config, indent=4))

    full_cycle()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
