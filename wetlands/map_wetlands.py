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

from wetlands import train_model, utils, viz_utils


def visualize_predicted_image(image, model, device):

    patch_size = int(os.getenv('PATCH_SIZE'))
    width = image.shape[0] - image.shape[0] % patch_size
    height = image.shape[1] - image.shape[1] % patch_size
    pred_mask = predict_water_mask(image, model, device)

    fig, ax = plt.subplots(figsize=(10, 10))
    pred_mask = 1 - pred_mask
    plt.imshow(pred_mask)
    plt.show()
    plt.clf()
    # plt.imsave('/tmp/water_estimation/20211016_map_wetlands_pred.png', 1 - pred_mask)
    # img = Image.fromarray(np.uint8((1 - pred_mask) * 255), 'L')
    # img.save('/tmp/water_estimation/20211016_map_wetlands_pred_bw.png')
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(image[:width, :height], cmap='gray')
    plt.show()
    plt.clf()
    # plt.imsave('/tmp/water_estimation/20211016_map_wetlands_sar.png', image)

    return pred_mask


def predict_water_mask(sar_image, model, device):
    patch_size = int(os.getenv('PATCH_SIZE'))
    width = sar_image.shape[0] - sar_image.shape[0] % patch_size
    height = sar_image.shape[1] - sar_image.shape[1] % patch_size
    pred_mask = np.zeros(tuple((width, height)))

    for h in range(0, height, patch_size):
        for w in range(0, width, patch_size):
            sar_image_crop = sar_image[w:w + patch_size, h:h + patch_size]
            sar_image_crop = sar_image_crop[None, :]
            binary_image = np.where(sar_image_crop.sum(2) > 0, 1, 0)
            sar_image_crop = torch.from_numpy(sar_image_crop.astype(np.float32)).to(device)[None, :]

            pred = model(sar_image_crop).cpu().detach().numpy()
            pred = (pred).squeeze() * binary_image
            pred = np.where(pred < 0.5, 0, 1)
            pred_mask[w:w + patch_size, h:h + patch_size] = pred

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
    # cwd = '/tmp/water_estimation/'
    start_date = os.getenv('START_DATE')
    shape_name = os.getenv('REGION_NAME')
    # shape_name = 'flacksjon_2018-07-04'
    patch_size = int(os.getenv('PATCH_SIZE'))
    sar_polarization = os.getenv('SAR_POLARIZATION')
    tif_file = os.getenv('SAR_TIFF_FILE')
    # tif_file = '/tmp/bulk_export_sar_flacksjon/S1A_IW_GRDH_1SDV_20180505T052314_20180505T052339_021765_0258EB_0EB0.tif'
    # tif_file = '/tmp/bulk_export_sar_flacksjon/S1A_IW_GRDH_1SDV_20180704T052317_20180704T052342_022640_0273F3_FD0A.tif'
    # tif_file = '/tmp/bulk_export_sar_flacksjon/S1A_IW_GRDH_1SDV_20211016T052340_20211016T052405_040140_04C0F0_3BEE.tif'
    # tif_file = '/tmp/water_estimation/S1A_IW_GRDH_1SDV_20180505T052314_20180505T052339_021765_0258EB_0EB0.tif'
    image = viz_utils.load_image(tif_file, sar_polarization, ignore_nan=False)
    device = utils.get_device()
    model_file = os.getenv('MODEL_FILE')
    # model_file = '/tmp/fresh-water-204_Orebro lan_mosaic_2018-07-04_sar_VH_20-epochs_0.00005-lr_42-rand.pth'
    pred_file = os.getenv('PREDICTIONS_FILE')
    # pred_file = '/tmp/water_estimation/20211016_predictions_flacksjon.tif'
    model = train_model.load_model(model_file, device)
    pred_mask = visualize_predicted_image(image, model, device)
    generate_raster_image(1 - pred_mask, pred_file, tif_file, patch_size)
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
