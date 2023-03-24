
import os
import time

import numpy as np
import pandas
import tqdm
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from wetlands import train_model, utils, viz_utils, map_wetlands


def visualize_predicted_image(image, model, device, file_name):
    model_name = os.getenv('MODEL_NAME')
    study_area = os.getenv('STUDY_AREA')

    images_dir = f'/tmp/descending_{model_name}_{study_area}_exported_images/'

    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)

    patch_size = int(os.getenv('PATCH_SIZE'))
    width = image.shape[0] - image.shape[0] % patch_size
    height = image.shape[1] - image.shape[1] % patch_size
    pred_mask = map_wetlands.predict_water_mask(image, model, device)

    unique, counts = np.unique(pred_mask, return_counts=True)
    results = dict(zip(unique, counts))
    image_date = file_name[17:25]
    satellite = file_name[0:3]
    results['Date'] = image_date
    results['Satellite'] = satellite
    results['File_name'] = file_name


    # Plotting SAR
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.set_title(satellite + ' ' + image_date, fontdict = {'fontsize' : 60})
    plt.imshow(image[:width, :height], cmap='gray')
    plt.imsave(images_dir + image_date + '_' + file_name + '_sar.png', image)
    # plt.show()
    # plt.clf()

    # Plotting prediction
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.set_title(satellite + ' ' + image_date, fontdict={'fontsize': 60})
    plt.imshow(pred_mask)
    plt.imsave(images_dir + image_date + '_' + file_name + '_pred.png', pred_mask)
    # plt.show()
    # plt.clf()

    return results


def get_prediction_image(tiff_file, band, model, device):
    # tif_file = os.getenv('SAR_TIFF_FILE')
    image = viz_utils.load_image(tiff_file, band, ignore_nan=True)

    if image is None:
        return None

    file_name = os.path.basename(tiff_file)
    results = visualize_predicted_image(image, model, device, file_name)
    return results


def plot_results():
    model_name = os.getenv('MODEL_NAME')
    study_area = os.getenv('STUDY_AREA')
    # results_file = '/tmp/water_estimates_flacksjon_2018-07.csv'
    results_file = f'/tmp/descending_{model_name}_{study_area}_water_estimates.csv'
    data_frame = pandas.read_csv(results_file, usecols=['1.0', 'Date'], index_col=["Date"],  parse_dates=["Date"])

    data_frame.plot(title=model_name)
    plt.savefig(f'/tmp/charts/descending_{model_name}_{study_area}_water_estimates.png')
    plt.show()


def update_water_estimates():
    model_name = os.getenv('MODEL_NAME')
    study_area = os.getenv('STUDY_AREA')

    with open('/Users/frape/tmp/cropped_images/cropped_images.txt') as file:
        lines = [line.rstrip() for line in file]
        results_file = f'/tmp/descending_{model_name}_{study_area}_water_estimates.csv'
        data_frame = pandas.read_csv(results_file, usecols=['1.0', 'Date', 'File_name'],  parse_dates=["Date"])
        print(data_frame.size)
        print(data_frame.columns.values)
        data_frame = data_frame[~data_frame['File_name'].isin(lines)]
        data_frame.drop(['File_name'], axis=1, inplace=True)
        data_frame = data_frame[data_frame['Date'].dt.month.isin([4, 5, 6, 7, 8, 9, 10, 11])]
        print(data_frame.size)
        print(data_frame.columns.values)

        data_frame.plot(x='Date', y='1.0', kind='scatter', title=model_name)
        plt.savefig(f'/tmp/charts/scatter_descending_{model_name}_{study_area}_new_water_estimates_filtered.png')
        plt.show()

        data_frame.to_csv(f'/tmp/descending_{model_name}_{study_area}_new_water_estimates_filtered.csv')


def full_cycle():
    load_dotenv()

    # tiff_dir = '/tmp/bulk_export_flacksjon'
    # tiff_dir = '/tmp/bulk_export_flacksjon_2018-07'
    tiff_dir = os.getenv('BULK_EXPORT_DIR')

    if not os.path.exists(tiff_dir):
        raise FileNotFoundError(f'The folder containing the TIFF files does not exist: {tiff_dir}')

    filenames = next(os.walk(tiff_dir), (None, None, []))[2]  # [] if no file
    print(filenames)

    device = utils.get_device()
    model_file = os.getenv('MODEL_FILE')
    # model_file = '/tmp/fresh-water-204_Orebro lan_mosaic_2018-07-04_sar_VH_20-epochs_0.00005-lr_42-rand.pth'
    model_name = os.getenv('MODEL_NAME')
    study_area = os.getenv('STUDY_AREA')
    sar_polarization = os.getenv('SAR_POLARIZATION')
    model = train_model.load_model(model_file, device)

    results_list = []
    incomplete_images = 0

    for tiff_file in tqdm.tqdm(sorted(filenames)):
        results = get_prediction_image(tiff_dir + '/' + tiff_file, sar_polarization, model, device)

        if results is None:
            incomplete_images += 1
        else:
            results_list.append(results)

    print(f'There were a total of {incomplete_images} incomplete images')

    data_frame = pandas.DataFrame(results_list)
    data_frame['Date'] = data_frame['Date'].apply(pandas.to_datetime).dt.date
    print(data_frame.head())
    data_frame.to_csv(f'/tmp/descending_{model_name}_{study_area}_water_estimates.csv')

    # tiff_file = '/tmp/bulk_export_flacksjon/S1A_IW_GRDH_1SDV_20180704T052317_20180704T052342_022640_0273F3_FD0A.tif'
    # get_prediction_image(tiff_file, 'VH')


def main():
    load_dotenv()

    full_cycle()
    plot_results()
    update_water_estimates()
    # transform_ndwi_tiff_to_grayscale_png()
    # transform_rgb_tiff_to_png()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
