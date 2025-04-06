import os
import random
import time
from pathlib import Path
import argparse
#import geojson
#import geopandas
import numpy as np
import pandas
#import requests
import torch
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=ascii)

def plant_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    # Check is GPU is enabled
    device = torch.device(
        # "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Device: {}".format(device))

    # Get specific GPU model
    if str(device) == "cuda:0":
        print("GPU: {}".format(torch.cuda.get_device_name(0)))

    return device


def download_country_boundaries(iso, adm, file_name):
    # ISO = 'SWE'  # "DEU" is the ISO code for Germany
    # ADM = 'ADM2'  # Equivalent to administrative districts

    my_file = Path(file_name)
    if my_file.is_file():
        # file exists
        print(f'GeoJSON file already exists: {file_name}')
        return

    # Query geoBoundaries
    r = requests.get("https://www.geoboundaries.org/gbRequest.html?ISO={}&ADM={}".format(iso, adm))
    dl_path = r.json()[0]['gjDownloadURL']

    # Save the result as a GeoJSON
    # filename = 'geoboundary.geojson'
    geoboundary = requests.get(dl_path).json()
    with open(file_name, 'w') as file:
        geojson.dump(geoboundary, file)

    print(f'Downloaded GeoJSON boundaries file: {file_name}')


def get_region_boundaries(shape_name, file_name):

    # Read data using GeoPandas
    geoboundary = geopandas.read_file(file_name)
    print("Data dimensions: {}".format(geoboundary.shape))

    return geoboundary


def show_region_boundaries(geoboundary, shape_name):
    geoboundary.sample(3)
    # shape_name = 'Sala kommun'
    fig, ax = plt.subplots(1, figsize=(10, 10))
    geoboundary[geoboundary.shapeName == shape_name].plot('shapeName', legend=True, ax=ax)
    plt.show()


def generate_model_file_name(epochs=None):
    load_dotenv()

    base_file_name = os.getenv('BASE_FILE_NAME')
    polarization = os.getenv('SAR_POLARIZATION')
    if epochs is None:
        epochs = os.getenv('EPOCHS')
    learning_rate = os.getenv('LEARNING_RATE')
    random_seed = os.getenv('RANDOM_SEED')
    model_name = f'{base_file_name}_sar_{polarization}_epochs-{epochs}_lr-{learning_rate}_rand-{random_seed}'

    # print(model_name)

    return model_name


def create_tiles_file_pipeline(pre_2020):
    args = parser.parse_args()
    data_dir = 'C:/Users/anubi/PycharmProjects/deep-wetlands-work/images/'
    patch_size = os.getenv('PATCH_SIZE')

    if pre_2020:
        images_dir = 'C:/Users/anubi/PycharmProjects/deep-wetlands-work/images/Orebro lan_mosaic_2018-07-04_64x64_sar/'
        masks_dir = 'C:/Users/anubi/PycharmProjects/deep-wetlands-work/images/Orebro lan_mosaic_2018-07-04_64x64_ndwi_mask/'
        tiles_data_file = data_dir + os.getenv('PRE_20_TILES_FILE')
    else:
        images_dir = data_dir + os.getenv('POST_20_SAR_DIR') + '/'
        masks_dir = data_dir + os.getenv('POST_20_MASK_DIR') + '/'
        tiles_data_file = data_dir + os.getenv('POST_20_TILES_FILE')
    print(images_dir, masks_dir)

    training_method = os.getenv('TRAINING_METHOD')
    if training_method == 'temporal_consistency':
        num_dates = int(os.getenv('TEMPORAL_CONSISTENCY_NUM_DATES'))
        if pre_2020:
            past_images_dir = data_dir + os.getenv('PRE_20_PAST_SAR_DIR') + '/'
            future_images_dir = data_dir + os.getenv('PRE_20_FUTURE_SAR_DIR') + '/'
        else:
            past_images_dir = data_dir + os.getenv('POST_20_PAST_SAR_DIR') + '/'
            future_images_dir = data_dir + os.getenv('POST_20_FUTURE_SAR_DIR') + '/'
        if num_dates > 1:
            if pre_2020:
                past_images2_dir = data_dir + os.getenv('PRE_20_PAST_SAR2_DIR') + '/'
                future_images2_dir = data_dir + os.getenv('PRE_20_FUTURE_SAR2_DIR') + '/'
            else:
                past_images2_dir = data_dir + os.getenv('POST_20_PAST_SAR2_DIR') + '/'
                future_images2_dir = data_dir + os.getenv('POST_20_FUTURE_SAR2_DIR') + '/'
    # if '/ndwi_masks_tiles/' in masks_dir:
    mask_type = 'ndwi'
    # elif '/otsu_masks_tiles/' in masks_dir:
    #     mask_type = 'otsu'
    print(images_dir, masks_dir)
    sar_files = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.tif')]
    sar_files.sort()
    mask_files.sort()
    if training_method == 'temporal_consistency':
        past_sar_files = [f for f in os.listdir(past_images_dir) if f.endswith('.tif')]
        future_sar_files = [f for f in os.listdir(future_images_dir) if f.endswith('.tif')]
        past_sar_files.sort()
        future_sar_files.sort()
        if num_dates > 1:
            past_sar2_files = [f for f in os.listdir(past_images2_dir) if f.endswith('.tif')]
            future_sar2_files = [f for f in os.listdir(future_images2_dir) if f.endswith('.tif')]
            past_sar2_files.sort()
            future_sar2_files.sort()

    # Remove the -sar.tif suffix from the file name
    sar_files = [f.replace('-sar.tif', '') for f in sar_files]
    mask_files = [f.replace(f'-{mask_type}_mask.tif', '') for f in mask_files]
    if training_method == 'temporal_consistency':
        past_sar_files = [f.replace('-sar.tif', '') for f in past_sar_files]
        future_sar_files = [f.replace('-sar.tif', '') for f in future_sar_files]
        if num_dates > 1:
            past_sar2_files = [f.replace('-sar.tif', '') for f in past_sar2_files]
            future_sar2_files = [f.replace('-sar.tif', '') for f in future_sar2_files]

    # Find the common files in both folders
    if training_method == 'standard':
        common_files = list(set(sar_files) & set(mask_files))
    elif training_method == 'temporal_consistency':
        if num_dates == 1:
            common_files = list(set(sar_files) & set(mask_files) & set(past_sar_files) & set(future_sar_files))
        elif num_dates == 2:
            common_files = list(set(sar_files) & set(mask_files) & set(past_sar_files) & set(future_sar_files) & set(past_sar2_files) & set(future_sar2_files))
    area_name = os.getenv('STUDY_AREA').lower().replace(' ', '_')
    print(area_name)
    # common_indexes = [int(f.replace(area_name + '-', '')) for f in common_files if area_name in f]
    common_indexes = [int(f.split('-')[1]) for f in common_files]
    print(common_files[:5])
    print(len(common_indexes))
    print(area_name == common_files[0][:11], area_name, common_files[0][:11])
    # common_indexes = [int(f.split('-')[-1]) for f in common_files if area_name in f]

    # Create a dataframe from common_files and common_indexes and sort it by id
    tiles_data_frame = pandas.DataFrame({'index': common_indexes, 'id': common_files})
    tiles_data_frame.set_index('index', inplace=True)
    tiles_data_frame = tiles_data_frame.sort_values(by=['index'])

    tiles_data_frame['split'] = 'test'
    num_rows = len(tiles_data_frame)
    test_rows = int(num_rows * 0.8)
    tiles_data_frame.loc[tiles_data_frame.head(test_rows).index, 'split'] = 'train'
    Path(os.path.dirname(tiles_data_file)).mkdir(parents=True, exist_ok=True)
    tiles_data_frame.to_csv(tiles_data_file, columns=['id', 'split'], index_label='index')

    print('There are a total of {} tiles'.format(num_rows))
    return tiles_data_frame


def main():

    file_name = '/tmp/sweden.geojson'
    download_country_boundaries('SWE', 'ADM2', file_name)
    get_region_boundaries('Sala kommun', file_name)


# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
