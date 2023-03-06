import json
import os
import time

import numpy as np
import rasterio as rio
import torch
from dotenv import load_dotenv, dotenv_values
from matplotlib import pyplot as plt
from rasterio.windows import Window
import geopandas as gpd
from osgeo import gdal
from osgeo import ogr

from wetlands import train_model, utils


def load_image(dir_path):

    tiff_image = rio.open(dir_path)
    numpy_image = tiff_image.read(1)

    min_value = np.nanpercentile(numpy_image, 1)
    max_value = np.nanpercentile(numpy_image, 99)

    numpy_image[numpy_image > max_value] = max_value
    numpy_image[numpy_image < min_value] = min_value

    array_min, array_max = np.nanmin(numpy_image), np.nanmax(numpy_image)
    normalized_array = (numpy_image - array_min) / (array_max - array_min)
    normalized_array[np.isnan(normalized_array)] = 0

    return normalized_array


def visualize_sentinel1(cwd, shape_name, start_date):
    # shape_name = 'Sala kommun'
    tif_file = cwd + '{}-sar-{}.tif'.format(shape_name, start_date)
    image = load_image(tif_file)

    fig, ax = plt.subplots(figsize=(15,15))
    plt.imshow(image, cmap='gray')
    print(type(image), image.shape)
    print(image[2000, 2000:])


def visualize_predicted_image(image, model, device):

    patch_size = int(os.getenv('PATCH_SIZE'))
    width = image.shape[0] - image.shape[0] % patch_size
    height = image.shape[1] - image.shape[1] % patch_size
    pred_mask = np.zeros(tuple((width, height)))

    for h in range(0, height, patch_size):
        for w in range(0, width, patch_size):
            image_crop = image[w:w + patch_size, h:h + patch_size]
            image_crop = image_crop[None, :]
            binary_image = np.where(image_crop.sum(2) > 0, 1, 0)
            image_crop = torch.from_numpy(image_crop.astype(np.float32)).to(device)[None, :]

            pred = model(image_crop).cpu().detach().numpy()
            pred = (pred).squeeze() * binary_image
            pred = np.where(pred < 0.5, 0, 1)
            pred_mask[w:w + patch_size, h:h + patch_size] = pred

    fig, ax = plt.subplots(figsize=(10, 10))
    pred_mask = 1 - pred_mask
    plt.imshow(pred_mask)
    plt.show()
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(image[:width, :height], cmap='gray')
    plt.show()
    plt.clf()

    return pred_mask


def generate_raster(image, src_tif, dest_file, step_size):
    with rio.open(src_tif) as src:
        # Create a Window and calculate the transform from the source dataset
        width = src.width - src.width % step_size
        height = src.height - src.height % step_size
        window = Window(0, 0, width, height)
        transform = src.window_transform(window)

        out_meta = src.meta
        out_meta.update({
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            # "transform": src.transform
            "transform": transform
        })
        with rio.open(dest_file, "w", **out_meta) as dest:
            dest.write(image.astype(rio.uint8), 1)


def generate_raster_image(pred_mask, pred_file, tif_file, step_size):
    generate_raster(pred_mask, tif_file, pred_file, step_size)
    mask = rio.open(pred_file)

    # Plot image and corresponding boundary
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(label=os.getenv("MODEL_NAME"))
    plt.imshow(mask.read(1))
    plt.show()
    plt.clf()


def polygonize_raster(src_raster_file, dest_file, tolerance=0.00005):
    src_raster = gdal.Open(src_raster_file)
    band = src_raster.GetRasterBand(1)
    band_array = band.ReadAsArray()
    driver = ogr.GetDriverByName("ESRI Shapefile")
    out_data_src = driver.CreateDataSource(dest_file)
    out_layer = out_data_src.CreateLayer("polygonized", srs=None)
    gdal.Polygonize(band, band, out_layer, -1, [], callback=None)
    out_data_src.Destroy()
    polygons = gpd.read_file(dest_file).simplify(tolerance)
    polygons.to_file(dest_file)


def polygonize_raster_full(cwd, pred_file, shape_name, start_date):
    out_shape_file = cwd + "{}_polygonized_{}.shp".format(shape_name, start_date)
    out_shape_file = out_shape_file.replace(' ', '_')
    polygonize_raster(pred_file, out_shape_file)
    print(f'Exported shape file to: {out_shape_file}')
    polygons = gpd.read_file(out_shape_file)
    ax = polygons.plot(figsize=(10, 10))
    ax.set_title(label=os.getenv("MODEL_NAME"))
    plt.show()
    plt.clf()


def full_cycle():

    cwd = os.getenv('TRAIN_CWD_DIR') + '/'
    start_date = os.getenv('START_DATE')
    shape_name = os.getenv('REGION_NAME')
    patch_size = int(os.getenv('PATCH_SIZE'))
    tif_file = os.getenv('SAR_TIFF_FILE')
    image = load_image(tif_file)
    device = utils.get_device()
    # model_file = os.getenv('MODEL_FILE')
    model_file = '/tmp/fresh-water-204_Orebro lan_mosaic_2018-07-04_sar_VH_20-epochs_0.00005-lr_42-rand.pth'
    pred_file = os.getenv('PREDICTIONS_FILE')
    model = train_model.load_model(model_file, device)
    pred_mask = visualize_predicted_image(image, model, device)
    generate_raster_image(pred_mask, pred_file, tif_file, patch_size)
    polygonize_raster_full(cwd, pred_file, shape_name, start_date)


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
