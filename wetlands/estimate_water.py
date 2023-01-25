
import os
import time

import numpy as np
import pandas
import rasterio as rio
import geopandas as gpd
import torch
import tqdm
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from wetlands import train_model, utils


def load_image(dir_path, band):

    tiff_image = rio.open(dir_path)
    band_index = tiff_image.descriptions.index(band)

    numpy_image = tiff_image.read(band_index+1)

    min_value = np.nanpercentile(numpy_image, 1)
    max_value = np.nanpercentile(numpy_image, 99)

    numpy_image[numpy_image > max_value] = max_value
    numpy_image[numpy_image < min_value] = min_value

    array_min, array_max = np.nanmin(numpy_image), np.nanmax(numpy_image)
    normalized_array = (numpy_image - array_min) / (array_max - array_min)
    normalized_array[np.isnan(normalized_array)] = 0

    return normalized_array


def visualize_predicted_image(image, model, device, file_name):

    # n, step_size = 7, 64
    # width, height = step_size * n, step_size * n
    step_size = 64
    width = image.shape[0] - image.shape[0] % step_size
    height = image.shape[1] - image.shape[1] % step_size
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

    unique, counts = np.unique(pred_mask, return_counts=True)
    results = dict(zip(unique, counts))
    # print('Date', image_date, 'Counts = ', results)
    image_date = file_name[17:25]
    satellite = file_name[0:3]
    results['Date'] = image_date
    results['Satellite'] = satellite
    results['File_name'] = file_name


    # Plotting SAR
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.set_title(satellite + ' ' + image_date, fontdict = {'fontsize' : 60})
    plt.imshow(image[:width, :height], cmap='gray')
    plt.imsave('/tmp/new_exported_images/' + image_date + '_' + file_name + '_sar.png', image)
    # plt.show()
    # plt.clf()

    # Plotting prediction
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.set_title(satellite + ' ' + image_date, fontdict={'fontsize': 60})
    plt.imshow(pred_mask)
    plt.imsave('/tmp/new_exported_images/' + image_date + '_' + file_name + '_pred.png', pred_mask)
    # plt.show()
    # plt.clf()

    return results


def get_prediction_image(tiff_file, band, model, device):
    # tif_file = os.getenv('SAR_TIFF_FILE')
    image = load_image(tiff_file, band)
    file_name = os.path.basename(tiff_file)
    results = visualize_predicted_image(image, model, device, file_name)
    return results


def plot_results():
    # results_file = '/tmp/water_estimates_flacksjon_2018-07.csv'
    results_file = '/tmp/water_estimates.csv'
    data_frame = pandas.read_csv(results_file, usecols=['1.0', 'Date'], index_col=["Date"],  parse_dates=["Date"])

    data_frame.plot()
    plt.show()


def remove_cropped_files():
    tiff_dir = '/tmp/bulk_export_flacksjon'
    flist = open('/Users/frape/tmp/cropped_images/cropped_images.txt')
    for f in flist:
        # fname = f.rstrip()  # or depending on situation: f.rstrip('\n')
        # or, if you get rid of os.chdir(path) above,
        fname = os.path.join(tiff_dir, f.rstrip())
        if os.path.isfile(fname):  # this makes the code more robust
            os.remove(fname)


def update_water_estimates():
    with open('/Users/frape/tmp/cropped_images/cropped_images.txt') as file:
        lines = [line.rstrip() for line in file]
        results_file = '/tmp/water_estimates.csv'
        data_frame = pandas.read_csv(results_file, usecols=['1.0', 'Date', 'File_name'],  parse_dates=["Date"])
        print(data_frame.size)
        print(data_frame.columns.values)
        data_frame = data_frame[~data_frame['File_name'].isin(lines)]
        data_frame.drop(['File_name'], axis=1, inplace=True)
        data_frame = data_frame[data_frame['Date'].dt.month.isin([4, 5, 6, 7, 8, 9, 10, 11])]
        print(data_frame.size)
        print(data_frame.columns.values)

        data_frame.plot(x='Date', y='1.0', kind='scatter')
        plt.show()

        data_frame.to_csv('/tmp/new_water_estimates_filtered.csv')


def full_cycle():
    load_dotenv()

    # tiff_dir = '/tmp/bulk_export_flacksjon_2018-07'
    tiff_dir = '/tmp/bulk_export_flacksjon'
    # tiff_dir = '/tmp/new_geo_exports'
    filenames = next(os.walk(tiff_dir), (None, None, []))[2]  # [] if no file
    print(filenames)

    device = utils.get_device()
    model_file = os.getenv('MODEL_FILE')
    model = train_model.load_model(model_file, device)

    results_list = []

    for tiff_file in tqdm.tqdm(sorted(filenames)):
        results = get_prediction_image(tiff_dir + '/' + tiff_file, 'VH', model, device)
        results_list.append(results)

    data_frame = pandas.DataFrame(results_list)
    data_frame['Date'] = data_frame['Date'].apply(pandas.to_datetime).dt.date
    print(data_frame.head())
    data_frame.to_csv('/tmp/water_estimates.csv')


    # tiff_file = '/tmp/bulk_export_flacksjon/S1A_IW_GRDH_1SDV_20180704T052317_20180704T052342_022640_0273F3_FD0A.tif'
    # get_prediction_image(tiff_file, 'VH')


def main():
    remove_cropped_files()
    full_cycle()
    plot_results()
    update_water_estimates()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
