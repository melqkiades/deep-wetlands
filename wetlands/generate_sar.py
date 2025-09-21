import collections
import json
import os
import time
from pathlib import Path

import numpy
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


def export_sar_data(tiles, tif_file):
    patch_size = int(os.getenv('PATCH_SIZE'))
    export_folder = os.getenv('SAR_DIR')
    Path(export_folder).mkdir(parents=True, exist_ok=True)

    with rio.open(tif_file) as src:
        dataset_array = src.read()
        minValue = numpy.nanpercentile(dataset_array, 1)
        maxValue = numpy.nanpercentile(dataset_array, 99)

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

    if nan_tiles > 0:
        print(f'Warning: There were {nan_tiles} tiles with NaN values.')


def export_sar_data_new(tiles, tif_file, minValue, maxValue, date, patch_size, area_name='Örebro län', indexes=None):
    # patch_size = int(os.getenv('PATCH_SIZE'))
    export_folder = 'C:\\Users\\ioia4268\\data\\sar_tiles_test\\' +area_name + '_' + date + '_' + str(patch_size) + 'x' + str(patch_size)
    Path(export_folder).mkdir(parents=True, exist_ok=True)
    # minValue = np.inf
    # maxValue = -np.inf
    # for tif_file in tif_files:
    # with rio.open('C:\\Users\\ioia4268\\PycharmProjects\\deep-wetlands\\data\\sar\\Orebro lan_mosaic_2018-07-04_sar_VH.tif') as src:
    #     dataset_array = src.read()
    #     minValue = numpy.nanpercentile(dataset_array, 1)
    #     maxValue = numpy.nanpercentile(dataset_array, 99)

    nan_tiles = 0

    for index in tqdm(range(len(tiles)), total=len(tiles)):

        with rio.open(tif_file) as src:

            shape = [tiles.iloc[index]['geometry']]
            name = tiles.iloc[index]['id']
            if indexes is None:
                out_image, out_transform = rio.mask.mask(src, shape, crop=True)
            else:
                out_image, out_transform = rio.mask.mask(src, shape, crop=True, indexes=indexes)
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
            out_image[out_image > maxValue] = maxValue
            out_image[out_image < minValue] = minValue
            out_image = (out_image - minValue) / (maxValue - minValue)

#            if out_image.shape[1] != patch_size or out_image.shape[2] != patch_size:
#                out_image = np.pad(out_image, ((0,0), (0, patch_size - out_image.shape[1]), (0, patch_size - out_image.shape[2])), 'constant', constant_values=-1)

            # Get the metadata of the source image and update it
            # with the width, height, and transform of the cropped image
            out_meta = src.meta
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            if indexes is not None:
                out_meta["count"] = len(indexes)

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

    if nan_tiles > 0:
        print(f'Warning: There were {nan_tiles} tiles with NaN values.')


def full_cycle():
    file_name = os.getenv('GEOJSON_FILE')
    region_name = os.getenv('REGION_NAME')
    tif_file = os.getenv('SAR_TIFF_FILE')

    country_code = os.getenv('COUNTRY_CODE')
    region_admin_level = os.getenv("REGION_ADMIN_LEVEL")
    # patch_size = int(os.getenv("PATCH_SIZE"))
    patch_size = 64

    # utils.download_country_boundaries(country_code, region_admin_level, file_name)
    # geoboundary = utils.get_region_boundaries(region_name, file_name)

    # tiles = geo_utils.get_tiles(region_name, tif_file, geoboundary, patch_size)
    # export_sar_data(tiles, tif_file)
    fig, ax = plt.subplots(2, 4)
    # with rio.open('C:\\Users\\ioia4268\\data\\sar\\Örebro län\\Orebro lan_mosaic_2018-07-04_sar_VH.tif') as src:
    #     dataset_array = src.read()
    #     minValue_2018 = numpy.nanpercentile(dataset_array, 0.5)
    #     maxValue_2018 = numpy.nanpercentile(dataset_array, 99.5)
    #     minValue_20181 = numpy.nanpercentile(dataset_array, 1)
    #     maxValue_20181 = numpy.nanpercentile(dataset_array, 99)
    #     minminValue_2018 = numpy.nanmin(dataset_array)
    #     maxmaxValue_2018 = numpy.nanmax(dataset_array)
    #     # ax[0, 0].hist(dataset_array.flatten(), bins=200, density=True)
    #     # ax[0, 1].hist(numpy.clip(dataset_array.flatten(), minValue_20181, maxValue_20181), bins=200, density=True)
    #     # ax[0, 2].hist(numpy.clip(dataset_array.flatten(), minValue_2018, maxValue_2018), bins=200, density=True)
    # with rio.open('C:\\Users\\ioia4268\\data\\sar\\Örebro län\\Orebro lan_mosaic_2020-06-23_sar_VH.tif') as src:
    #     dataset_array2 = src.read()
    #     minValue_2020 = numpy.nanpercentile(dataset_array2, 0.5)
    #     maxValue_2020 = numpy.nanpercentile(dataset_array2, 99.5)
    #     minValue_20201 = numpy.nanpercentile(dataset_array2, 1)
    #     maxValue_20201 = numpy.nanpercentile(dataset_array2, 99)
    #     minminValue_2020 = numpy.nanmin(dataset_array2)
    #     maxmaxValue_2020 = numpy.nanmax(dataset_array2)
    #     ax[1, 0].hist(dataset_array2.flatten(), bins=200, density=True)
    #     ax[1, 1].hist(numpy.clip(dataset_array2.flatten(), minValue_20201, maxValue_20201), bins=200, density=True)
    #     ax[1, 2].hist(numpy.clip(dataset_array2.flatten(), minValue_2020, maxValue_2020), bins=200, density=True)
    #     ax[1, 3].hist(numpy.clip(dataset_array2.flatten(), minValue_2018, maxValue_2018), bins=200, density=True)
    # plt.show()
    # tif_files = glob.glob('C:\\Users\\ioia4268\\data\\sar\\orebroiancopy\\*')
    # sar_files = [filename for filename in glob.glob('C:\\Users\\ioia4268\\data\\sar\\orebroiancopy\\*') if
    #              os.path.getsize(filename) > 600000000]
    # sar_dates = [filename.split('_')[-4] for filename in sar_files]
    # print([item for item, count in collections.Counter(sar_dates).items() if count > 1])
    # ndwi_files = glob.glob('C:\\Users\\ioia4268\\data\\ndwi_masks - Copy\\Örebro län\\*.tif')
    # for filename in ndwi_files:
    #     if '_test.tif' in filename and filename.replace('_test.tif', '.tif') not in ndwi_files:
    #         os.rename(filename, filename.replace('_test.tif', '.tif'))
    # ndwi_files = [filename for filename in ndwi_files if os.path.getsize(filename) > 19000000]
    # ndwi_dates = [filename.split('_')[-2] for filename in ndwi_files]
    # print([item for item, count in collections.Counter(ndwi_dates).items() if count > 1])
    # matching_dates = [date for date in sar_dates if date in ndwi_dates]
    # matching_dates = [matching_dates[i] for i in range(len(matching_dates)) if i not in [1,10,12,14,16,18,20,21,24,25,27,29,30,31,32,33,34,36,38,39,40,45,46,48]]


    # closest_dates = []
    # tc_sar_files = [filename for filename in glob.glob('C:\\Users\\ioia4268\\data\\sar\\orebroiancopy\\*') if filename[-6:] not in ['_A.tif', '_B.tif']]
    # tif_files = [tc_sar_files[36], tc_sar_files[37], tc_sar_files[42], tc_sar_files[43], tc_sar_files[48], tc_sar_files[49],
    #              sar_files[70], sar_files[84], sar_files[97]]
    # fig, axs = plt.subplots(6, 10)
    # for i in range(len(tc_sar_files)):
    #     # date = matching_dates[i + 49]
    #     # sar_file = [filename for filename in sar_files if date in filename]
    #     # ndwi_file = [filename for filename in ndwi_files if date in filename]
    #     dataset_array = np.nan_to_num(io.imread(tc_sar_files[i]))
    #     axs[i//10, i%10].imshow(dataset_array, cmap='Greys')
    #     axs[i//10, i%10].set_title(tc_sar_files[i].split('_')[-3])
    #     axs[i//10, i%10].axis("off")
    # plt.show()
    tif_files = ["C:\\Users\\ioia4268\\data\\sar\\Örebro län\\Orebro lan_mosaic_2018-07-04_sar_VH.tif",
                 "C:\\Users\\ioia4268\\data\\sar\\Örebro län\\Orebro lan_mosaic_2020-06-23_sar_VH.tif"]
    area_name = "örebro_län"
    for tif_file in tif_files:
        tiles = geo_utils.get_tiles_batch(area_name, tif_file, patch_size)
        if '2018' in tif_file or '2019' in tif_file:
            minValue = 0
            maxValue = 1
        else:
            minValue = 0
            maxValue = 1
        export_sar_data_new(tiles, tif_file, minValue, maxValue, tif_file.split("_")[-3], patch_size)


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
