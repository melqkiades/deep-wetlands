import os
import time

import numpy as np
import rasterio as rio
import torch
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from rasterio.windows import Window
import geopandas as gpd
from osgeo import gdal
from osgeo import ogr

from wetlands import train_model, utils


def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))


def load_image(dir_path):
#     image_bands = []
#     for band in range(1, 4):
#         array = rio.open(dir_path).read(band)
#         array[np.isnan(array)] = 0
#         image_bands.append(normalize(array))

#     image = np.dstack(image_bands)
    image = rio.open(dir_path).read(1)
    image = normalize(image)
    return image


def visualize_sentinel1(cwd, shape_name, start_date):
    # shape_name = 'Sala kommun'
    tif_file = cwd + '{}-sar-{}.tif'.format(shape_name, start_date)
    image = load_image(tif_file)

    fig, ax = plt.subplots(figsize=(15,15))
    plt.imshow(image, cmap='gray')
    print(type(image), image.shape)
    print(image[2000, 2000:])


def visualize_predicted_image(image, model, device):
    n, step_size = 70, 64
    width, height = step_size * n, step_size * n
    pred_mask = np.zeros(tuple((width, height)))

    for h in range(0, height, step_size):
        for w in range(0, width, step_size):
            image_crop = image[w:w + step_size, h:h + step_size]
            image_crop = image_crop[None, :]
            binary_image = np.where(image_crop.sum(2) > 0, 1, 0)

            # image_crop = torch.from_numpy(
            #     (image_crop * 255.0).astype("uint8").transpose((2, 1, 0)).astype(np.float32)
            # ).to(device)[None, :]
            image_crop = torch.from_numpy(image_crop.astype(np.float32)).to(device)[None, :]

            pred = model(image_crop).cpu().detach().numpy()
            pred = (pred).squeeze() * binary_image
            pred = np.where(pred < 0.5, 0, 1)
            pred_mask[w:w + step_size, h:h + step_size] = pred

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


def generate_raster(image, src_tif, dest_file, width, height):
    with rio.open(src_tif) as src:
        # Create a Window and calculate the transform from the source dataset
        window = Window(0, 0, width, height)
        transform = src.window_transform(window)

        out_meta = src.meta
        out_meta.update({
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "transform": src.transform
        })
        with rio.open(dest_file, "w", **out_meta) as dest:
            dest.write(image.astype(rio.uint8), 1)


def generate_raster_image(pred_mask, pred_file, cwd, tif_file, width, height):
    generate_raster(pred_mask, tif_file, pred_file, width, height)
    mask = rio.open(pred_file)

    # Plot image and corresponding boundary
    fig, ax = plt.subplots(figsize=(10, 10))
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
    polygons.plot(figsize=(10, 10))
    plt.show()
    plt.clf()


def full_cycle():
    load_dotenv()

    cwd = os.getenv('TRAIN_CWD_DIR') + '/'
    start_date = os.getenv('START_DATE')
    end_date = os.getenv('END_DATE')
    shape_name = os.getenv('REGION_NAME')
    n, step_size = 70, 64
    width, height = step_size * n, step_size * n
    tif_file = cwd + '{}-sar-{}.tif'.format(shape_name, start_date)
    image = load_image(tif_file)
    device = utils.get_device()
    model_dir = os.getenv('MODELS_DIR') + '/'
    model_file = model_dir + 'best_model_20221014.pth'
    # model_file = os.getenv('MODEL_FILE')
    pred_file = cwd + 'temp.tif'
    model = train_model.load_model(model_file, device)
    pred_mask = visualize_predicted_image(image, model, device)
    generate_raster_image(pred_mask, pred_file, cwd, tif_file, width, height)
    polygonize_raster_full(cwd, pred_file, shape_name, start_date)


def main():
    full_cycle()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
