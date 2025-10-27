from pathlib import Path
import numpy as np
import rasterio
import rasterio as rio
import rasterio.mask
from tqdm import tqdm
from wetlands import geo_utils
import glob


def export_sar_data(config, tiles, tif_file, patch_size, date, area_name='Orebro lan'):
    export_folder = config['DATA_DIR'] + config['SAR_TILES_DIR'] + area_name + '_'+date + '_' + str(patch_size) + 'x' + str(patch_size)
    Path(export_folder).mkdir(parents=True, exist_ok=True)

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


    if nan_tiles > 0:
        print(f'Warning: There were {nan_tiles} tiles with NaN values.')


def full_cycle(config):
    patch_size = config['PATCH_SIZE']

    tif_files = glob.glob(config['DATA_DIR']+config['SAR_DIR'] + config['AREA_NAME'] + '/*')
    for tif_file in tif_files:
        date = tif_file.split('_')[-3]
        area_name = tif_file.split('\\')[-2]
        tiles = geo_utils.get_tiles(area_name, tif_file, patch_size)
        export_sar_data(config, tiles, tif_file, patch_size, date, area_name)