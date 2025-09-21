import collections
import json
import os
import time
from pathlib import Path
import numpy as np
import rasterio
import rasterio as rio
import rasterio.mask
from PIL import Image
from dotenv import load_dotenv, dotenv_values
from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import datetime
from wetlands import utils, geo_utils
import glob
from skimage import io


def export_sar_data(config, tiles, tif_file, patch_size, date, area_name='Orebro lan'):
    export_folder = config['DATA_DIR'] + 'sar_tiles/' + area_name + '_'+date + '_' + str(patch_size) + 'x' + str(patch_size)
    Path(export_folder).mkdir(parents=True, exist_ok=True)
    print(tif_file, export_folder)
    # patch_size = int(os.getenv('PATCH_SIZE'))

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
            temp_tif = export_folder + '/{}-sar.tif'.format(name)
            with rasterio.open(temp_tif, "w", **out_meta) as dest:
                dest.write(out_image)

            # # Save the cropped image as a temporary PNG file.
            # temp_png = export_folder + '/{}-sar.png'.format(name)
            #
            # # Get the color map by name:
            # cm = plt.get_cmap('gray')
            #
            # # Apply the colormap like a function to any array:
            # colored_image = cm(out_image[0])
            #
            # # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
            # # But we want to convert to RGB in uint8 and save it:
            # Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(temp_png)

    if nan_tiles > 0:
        print(f'Warning: There were {nan_tiles} tiles with NaN values.')


# def export_sar_data_new(tiles, tif_file, minValue, maxValue, date, patch_size, area_name='Örebro län', indexes=None):
#     # patch_size = int(os.getenv('PATCH_SIZE'))
#     export_folder = 'C:\\Users\\ioia4268\\data\\sar_tiles_test\\' +area_name + '_' + date + '_' + str(patch_size) + 'x' + str(patch_size)
#     Path(export_folder).mkdir(parents=True, exist_ok=True)
#     # minValue = np.inf
#     # maxValue = -np.inf
#     # for tif_file in tif_files:
#     # with rio.open('C:\\Users\\ioia4268\\PycharmProjects\\deep-wetlands\\data\\sar\\Orebro lan_mosaic_2018-07-04_sar_VH.tif') as src:
#     #     dataset_array = src.read()
#     #     minValue = numpy.nanpercentile(dataset_array, 1)
#     #     maxValue = numpy.nanpercentile(dataset_array, 99)
#
#     nan_tiles = 0
#
#     for index in tqdm(range(len(tiles)), total=len(tiles)):
#
#         with rio.open(tif_file) as src:
#
#             shape = [tiles.iloc[index]['geometry']]
#             name = tiles.iloc[index]['id']
#             if indexes is None:
#                 out_image, out_transform = rio.mask.mask(src, shape, crop=True)
#             else:
#                 out_image, out_transform = rio.mask.mask(src, shape, crop=True, indexes=indexes)
#             if np.isnan(out_image).any():
#                 nan_tiles += 1
#                 continue
#
#             if out_image.shape[1] == patch_size + 1:
#                 if np.all(out_image==0, axis=2)[0][0]:
#                     out_image = out_image[:, 1:, :]
#                 else:
#                     out_image = out_image[:, :-1, :]
#             if out_image.shape[2] == patch_size + 1:
#                 if np.all(out_image == 0, axis=1)[0][0]:
#                     out_image = out_image[:, :, 1:]
#                 else:
#                     out_image = out_image[:, :, :-1]
#
#             if out_image.shape[1] != patch_size or out_image.shape[2] != patch_size:
#                 continue
#
#             # Min-max scale the data to range [0, 1]
#             out_image[out_image > maxValue] = maxValue
#             out_image[out_image < minValue] = minValue
#             out_image = (out_image - minValue) / (maxValue - minValue)
#
# #            if out_image.shape[1] != patch_size or out_image.shape[2] != patch_size:
# #                out_image = np.pad(out_image, ((0,0), (0, patch_size - out_image.shape[1]), (0, patch_size - out_image.shape[2])), 'constant', constant_values=-1)
#
#             # Get the metadata of the source image and update it
#             # with the width, height, and transform of the cropped image
#             out_meta = src.meta
#             out_meta.update({
#                 "driver": "GTiff",
#                 "height": out_image.shape[1],
#                 "width": out_image.shape[2],
#                 "transform": out_transform
#             })
#             if indexes is not None:
#                 out_meta["count"] = len(indexes)
#
#             # Save the cropped image as a temporary TIFF file.
#             temp_tif = export_folder + '/{}-sar.tif'.format(name)
#             with rasterio.open(temp_tif, "w", **out_meta) as dest:
#                 dest.write(out_image)
#
#             # Save the cropped image as a temporary PNG file.
#             temp_png = export_folder + '/{}-sar.png'.format(name)
#
#             # Get the color map by name:
#             cm = plt.get_cmap('gray')
#
#             # Apply the colormap like a function to any array:
#             colored_image = cm(out_image[0])
#
#             # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
#             # But we want to convert to RGB in uint8 and save it:
#             Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(temp_png)
#
#     if nan_tiles > 0:
#         print(f'Warning: There were {nan_tiles} tiles with NaN values.')


def full_cycle(config):
    patch_size = 64

    tif_files = ["C:\\Users\\ioia4268\\data\\sar\\Orebro lan\\Orebro lan_mosaic_2018-07-04_sar_VH.tif",
                 "C:\\Users\\ioia4268\\data\\sar\\Orebro lan\\Orebro lan_mosaic_2020-06-23_sar_VH.tif"]
    for tif_file in tif_files:
        date = tif_file.split('_')[-3]
        area_name = tif_file.split('\\')[-2]
        tiles = geo_utils.get_tiles(area_name, tif_file, patch_size)
        export_sar_data(config, tiles, tif_file, patch_size, date, 'Orebro lan')


def main():
    load_dotenv()
    config = dotenv_values()
    print(json.dumps(config, indent=4))

    full_cycle(config)



print('GENERATING SAR')
start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
