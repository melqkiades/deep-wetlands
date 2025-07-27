import os
import random
# import time
from pathlib import Path
import argparse
#import geojson
#import geopandas
import numpy as np
import pandas
#import requests
import torch
# from dotenv import load_dotenv
from matplotlib import pyplot as plt
# import sys
from dotenv import load_dotenv, dotenv_values
import datetime
import h5py
from unidecode import unidecode


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


# def create_tiles_file_pipeline(pre_2020):
#     args = parser.parse_args()
#     data_dir = args.data_dir[1:-1]
#     patch_size = os.getenv('PATCH_SIZE')
#     training_method = os.getenv('TRAINING_METHOD')
#     if pre_2020:
#         prefix = 'PRE'
#     else:
#         prefix = 'POST'
#     dates = [int(os.getenv(f'{prefix}_20_TRAIN_DATE'))]
#     if training_method == 'temporal_consistency':
#         num_dates = int(os.getenv('TEMPORAL_CONSISTENCY_NUM_DATES'))
#         dates += [os.getenv(f'{prefix}_20_PAST_DATE'), os.getenv(f'{prefix}_20_FUTURE_DATE')]
#         if num_dates > 1:
#             dates += [os.getenv(f'{prefix}_20_PAST_DATE2'), os.getenv(f'{prefix}_20_FUTURE_DATE2')]
#     dates = sorted(dates, key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
#     file_name = dates[0]
#     for date in dates[1:]:
#         file_name += '_' + date
#     file_name += '.h5py'
#     matching_indices_dir = data_dir + os.getenv('MATCHING_INDICES_DIR')
#     Path(matching_indices_dir).mkdir(parents=True, exist_ok=True)
#     if os.path.isfile(matching_indices_dir + '/' + file_name):
#         common_indices = h5py.File(matching_indices_dir + '/' + file_name, 'r')['indices'][()]
#     else:
#         sar_file = data_dir + os.getenv(f'{prefix}_20_SAR_FILE')
#         sar_indices = h5py.File(sar_file, 'r')['indices'][()]
#         mask_file = data_dir + os.getenv(f'{prefix}_20_MASK_FILE')
#         mask_indices = h5py.File(mask_file, 'r')['indices'][()]
#         if training_method == 'temporal_consistency':
#             num_dates = int(os.getenv('TEMPORAL_CONSISTENCY_NUM_DATES'))
#             past_sar_file = data_dir + os.getenv(f'{prefix}_20_PAST_SAR_FILE')
#             past_sar_indices = h5py.File(past_sar_file, 'r')['indices'][()]
#             future_sar_file = data_dir + os.getenv(f'{prefix}_20_FUTURE_SAR_FILE')
#             future_sar_indices = h5py.File(future_sar_file, 'r')['indices'][()]
#             if num_dates > 1:
#                 past_sar_file2 = data_dir + os.getenv(f'{prefix}_20_PAST_SAR2_FILE')
#                 past_sar_indices2 = h5py.File(past_sar_file2, 'r')['indices'][()]
#                 future_sar_file2 = data_dir + os.getenv(f'{prefix}_20_FUTURE_SAR2_FILE')
#                 future_sar_indices2 = h5py.File(future_sar_file2, 'r')['indices'][()]
#
#         # Find the common files in both folders
#         if training_method == 'standard':
#             common_indices = list(set(sar_indices) & set(mask_indices))
#         elif training_method == 'temporal_consistency':
#             if num_dates == 1:
#                 common_indices = list(set(sar_indices) & set(mask_indices) & set(past_sar_indices) & set(future_sar_indices))
#             elif num_dates == 2:
#                 common_indices = list(set(sar_indices) & set(mask_indices) & set(past_sar_indices) & set(future_sar_indices) & set(past_sar_indices2) & set(future_sar_indices2))
#         with h5py.File(matching_indices_dir + '/' + file_name, "w") as f:
#             f_indices = f.create_dataset('indices', data=common_indices)
#     print(common_indices[:5])
#     print(len(common_indices))
#
#     # Create a dataframe from common_files and common_indexes and sort it by id
#     tiles_data_frame = pandas.DataFrame({'index': common_indices})
#     tiles_data_frame.set_index('index', inplace=True)
#     tiles_data_frame = tiles_data_frame.sort_values(by=['index'])
#
#     tiles_data_frame['split'] = 'test'
#     num_rows = len(tiles_data_frame)
#     train_rows = int(num_rows * 0.8)
#     tiles_data_frame.loc[tiles_data_frame.head(train_rows).index, 'split'] = 'train'
#
#     print('There are a total of {} tiles'.format(num_rows))
#     return tiles_data_frame


def create_tiles_file_pipeline(config, pre_2020):
    data_dir = config['TEMP_DATA_DIR']
    print('data dir:', data_dir)
    if pre_2020:
        prefix = 'PRE'
    else:
        prefix = 'POST'
    images_dir = data_dir + config[f'{prefix}_20_SAR_DIR']
    masks_dir = data_dir + config[f'{prefix}_20_MASK_DIR']

    training_method = config['TRAINING_METHOD']
    if training_method == 'temporal_consistency':
        num_dates = int(config['TEMPORAL_CONSISTENCY_NUM_DATES'])
        past_images_dir = data_dir + config[f'{prefix}_20_PAST_SAR_DIR']
        future_images_dir = data_dir + config[f'{prefix}_20_FUTURE_SAR_DIR']
        if num_dates > 1:
            past_images2_dir = data_dir + config[f'{prefix}_20_PAST_SAR2_DIR']
            future_images2_dir = data_dir + config[f'{prefix}_20_FUTURE_SAR2_DIR']
    mask_type = config['MASK_TYPE']
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
    sar_files = [unidecode(f.replace('-sar.tif', '')) for f in sar_files]
    mask_files = [unidecode(f.replace(f'-{mask_type}_mask.tif', '')) for f in mask_files]
    if training_method == 'temporal_consistency':
        past_sar_files = [unidecode(f.replace('-sar.tif', '')) for f in past_sar_files]
        future_sar_files = [unidecode(f.replace('-sar.tif', '')) for f in future_sar_files]
        if num_dates > 1:
            past_sar2_files = [unidecode(f.replace('-sar.tif', '')) for f in past_sar2_files]
            future_sar2_files = [unidecode(f.replace('-sar.tif', '')) for f in future_sar2_files]
    # Find the common files in both folders
    if training_method == 'standard':
        common_files = list(set(sar_files) & set(mask_files))
    elif training_method == 'temporal_consistency':
        if num_dates == 1:
            common_files = list(set(sar_files) & set(mask_files) & set(past_sar_files) & set(future_sar_files))
        elif num_dates == 2:
            common_files = list(set(sar_files) & set(mask_files) & set(past_sar_files) & set(future_sar_files) & set(past_sar2_files) & set(future_sar2_files))
    common_indexes = [int(f.split('-')[1]) for f in common_files]
    # Create a dataframe from common_files and common_indexes and sort it by id
    tiles_data_frame = pandas.DataFrame({'index': common_indexes, 'id': common_files})
    tiles_data_frame.set_index('index', inplace=True)
    tiles_data_frame = tiles_data_frame.sort_values(by=['index'])

    # tiles_data_frame = tiles_data_frame.sample(frac=1).reset_index(drop=True)
    tiles_data_frame['split'] = 'valid'
    num_rows = len(tiles_data_frame)
    train_rows = int(num_rows * 0.8)
    tiles_data_frame.loc[tiles_data_frame.head(train_rows).index, 'split'] = 'train'

    print('There are a total of {} tiles'.format(num_rows))
    return tiles_data_frame

# def create_tiles_file_pipeline(config, pre_2020):
#     data_dir = config['TEMP_DATA_DIR']
#     print('data dir:', data_dir)
#     if pre_2020:
#         prefix = 'PRE'
#     else:
#         prefix = 'POST'
#     images_file = data_dir + config[f'{prefix}_20_SAR_FILE']
#     masks_file = data_dir + config[f'{prefix}_20_MASK_FILE']
#
#     training_method = config['TRAINING_METHOD']
#     if training_method == 'temporal_consistency':
#         num_dates = int(config['TEMPORAL_CONSISTENCY_NUM_DATES'])
#         past_images_file = data_dir + config[f'{prefix}_20_PAST_SAR_FILE']
#         future_images_file = data_dir + config[f'{prefix}_20_FUTURE_SAR_FILE']
#         if num_dates > 1:
#             past_images2_file = data_dir + config[f'{prefix}_20_PAST_SAR2_FILE']
#             future_images2_file = data_dir + config[f'{prefix}_20_FUTURE_SAR2_FILE']
#
#     sar_tile_ids = h5py.File(images_file, 'r')['indices']
#     # sar_tile_indices = [True]*sar_tile_ids.shape[0]
#     mask_tile_ids = h5py.File(masks_file, 'r')['indices']
#     # mask_tile_indices = [True]*mask_tile_ids.shape[0]
#
#     if training_method == 'temporal_consistency':
#         past_sar_tile_ids = h5py.File(past_images_file, 'r')['indices']
#         # past_sar_tile_indices = [True]*past_sar_tile_ids.shape[0]
#         future_sar_tile_ids = h5py.File(future_images_file, 'r')['indices']
#         # future_sar_tile_indices = [True]*future_sar_tile_ids.shape[0]
#         if num_dates > 1:
#             past_sar2_tile_ids = h5py.File(past_images2_file, 'r')['indices']
#             # past_sar2_tile_indices = [True]*past_sar2_tile_ids.shape[0]
#             future_sar2_tile_ids = h5py.File(future_images2_file, 'r')['indices']
#             # future_sar2_tile_indices = [True]*future_sar2_tile_ids.shape[0]
#
#     # Find the common files in both folders
#     if training_method == 'standard':
#         common_tile_ids = list(set(sar_tile_ids) & set(mask_tile_ids))
#     elif training_method == 'temporal_consistency':
#         if num_dates == 1:
#             common_tile_ids = list(set(sar_tile_ids) & set(mask_tile_ids) & set(past_sar_tile_ids) & set(future_sar_tile_ids))
#         elif num_dates == 2:
#             common_tile_ids = list(set(sar_tile_ids) & set(mask_tile_ids) & set(past_sar_tile_ids) & set(future_sar_tile_ids) & set(past_sar2_tile_ids) & set(future_sar2_tile_ids))
#
#
#     # Create a dataframe from common_files and common_indexes and sort it by id
#     tiles_data_frame = pandas.DataFrame({'id': common_tile_ids})
#     # tiles_data_frame.set_index('index', inplace=True)
#     # tiles_data_frame = tiles_data_frame.sort_values(by=['index'])
#     tiles_data_frame = tiles_data_frame.sample(frac=1).reset_index(drop=True)
#     tiles_data_frame['split'] = 'train'
#     num_rows = len(tiles_data_frame)
#     valid_rows = int(num_rows * 0.2)
#     tiles_data_frame.loc[tiles_data_frame.head(valid_rows).index, 'split'] = 'valid'
#
#     print('There are a total of {} tiles'.format(num_rows))
#     return tiles_data_frame


def config_update_casts(config):
    # Convert int values to int
    for key in ['EPOCHS', 'PATCH_SIZE', 'BATCH_SIZE', 'NUM_WORKERS', 'EARLY_STOP_NUM_EPOCHS',
                'TEMPORAL_CONSISTENCY_START_EPOCH', 'REDUCE_LR_PLATEAU_PATIENCE', 'TRANSFORMER_PATCH_SIZE', 'EMBED_DIM',
                'WINDOW_SIZE', 'TEMPORAL_CONSISTENCY_NUM_DATES']:
        config[key] = int(config[key])
    # Convert float values to float
    for key in ['LEARNING_RATE', 'MLP_RATIO', 'DROP_RATE', 'DROP_PATH_RATE', 'TEMPORAL_CONSISTENCY_WEIGHT',
                'STANDARD_TRAINING_WEIGHT', 'TEMPORAL_CONSISTENCY_POWER', 'REDUCE_LR_PLATEAU_FACTOR']:
        config[key] = float(config[key])
    for key in ['SAVE_MODEL_ON_ALL_EPOCHS', 'SAVE_MODEL_ON_LAST_EPOCH', 'QKV_BIAS', 'APE',
                'PATCH_NORM', 'USE_CHECKPOINT']:
        if config[key] == "TRUE":
            config[key] = True
        else:
            config[key] = False
    if config['RANDOM_SEED'] != 'NONE':
        config['RANDOM_SEED'] = int(config['RANDOM_SEED'])
    if config['QK_SKALE'] != 'NONE':
        config['QK_SKALE'] = float(config['QK_SKALE'])
    else:
        config['QK_SKALE'] = None

    for key in ['DEPTHS', 'NUM_HEADS']:
        config[key] = tuple([int(x) for x in config[key][1:-1].split(',')])
    return config

def main():

    file_name = '/tmp/sweden.geojson'
    download_country_boundaries('SWE', 'ADM2', file_name)
    get_region_boundaries('Sala kommun', file_name)


# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
