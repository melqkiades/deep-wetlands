import numpy as np
import rasterio
import rasterio as rio
import rasterio.mask
from tqdm import tqdm
from pathlib import Path
from wetlands import geo_utils
import glob


def export_ndwi_mask_data(config, tiles, tif_file, patch_size, date, area_name='Orebro lan', test_images=False):
    export_folder = config['DATA_DIR'] + 'ndwi_masks_tiles/' + area_name + '_'+date + '_' + str(patch_size) + 'x' + str(patch_size)
    Path(export_folder).mkdir(parents=True, exist_ok=True)

    nan_tiles = 0

    for index in tqdm(range(len(tiles)), total=len(tiles)):

        with rio.open(tif_file) as src:

            shape = [tiles.iloc[index]['geometry']]
            name = tiles.iloc[index]['id']
            out_image, out_transform = rio.mask.mask(src, shape, crop=True)
            if np.isnan(out_image).any():
                nan_tiles += 1
                continue
            if not test_images:
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
            else:
                if out_image.shape[1] < 64 or out_image.shape[2] < 64:
                    continue


            if not test_images and (out_image.shape[1] != patch_size or out_image.shape[2] != patch_size):
                continue

            if test_images and (out_image.shape[1] < patch_size or out_image.shape[2] < patch_size):
                out_image = np.pad(out_image, ((0,0),(0, max(patch_size-out_image.shape[1], 0)), (0, max(patch_size-out_image.shape[2], 0))),'constant', constant_values=-1.)



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
            temp_tif = export_folder + '/{}-ndwi_mask.tif'.format(name)
            with rasterio.open(temp_tif, "w", **out_meta) as dest:
                dest.write((out_image).astype(np.float64))
    if nan_tiles > 0:
        print(f'Warning: There were {nan_tiles} tiles with NaN values.')


def full_cycle(config, area_name, test_images=False):
    patch_size = config['PATCH_SIZE']
    tif_files = glob.glob(config['DATA_DIR'] + config['NDWI_DIR'] + area_name + '/*')
    for tif_file in tif_files:
        date = tif_file.split('_')[-3]
        tiles = geo_utils.get_tiles(area_name, tif_file, patch_size)
        export_ndwi_mask_data(config, tiles, tif_file, patch_size, date, area_name, test_images=test_images)
