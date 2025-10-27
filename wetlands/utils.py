import os
import random
import numpy as np
import pandas
import torch
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


def create_tiles_file(config):
    data_dir = config['DATA_DIR']

    images_dir = data_dir + config[f'TRAIN_SAR_DIR']
    masks_dir = data_dir + config[f'TRAIN_MASK_DIR']

    sar_files = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.tif')]
    sar_files.sort()
    mask_files.sort()
    # Remove the -sar.tif suffix from the file name
    sar_files = [unidecode(f.replace('-sar.tif', '')) for f in sar_files]
    mask_files = [unidecode(f.replace(f'-ndwi_mask.tif', '')) for f in mask_files]
    # Find the common files in both folders
    common_files = list(set(sar_files) & set(mask_files))
    common_indexes = [int(f.split('-')[1]) for f in common_files]
    # Create a dataframe from common_files and common_indexes and sort it by id
    tiles_data_frame = pandas.DataFrame({'index': common_indexes, 'id': common_files})
    tiles_data_frame.set_index('index', inplace=True)
    tiles_data_frame = tiles_data_frame.sort_values(by=['index'])

    tiles_data_frame = tiles_data_frame.sample(frac=1.).reset_index(drop=True)
    tiles_data_frame['split'] = 'valid'
    num_rows = len(tiles_data_frame)
    train_rows = int(num_rows * 0.8)
    tiles_data_frame.loc[tiles_data_frame.head(train_rows).index, 'split'] = 'train'

    print('There are a total of {} tiles'.format(num_rows))
    return tiles_data_frame

def create_tiles_file_paper_test(config):
    patch_size = config['PATCH_SIZE']
    data_dir = config['DATA_DIR']
    print('data dir:', data_dir)


    sar_files = [f for f in os.listdir(f'{data_dir}sar_tiles/deepaqua_test_dataset_no_nov_pre_2020_full_{patch_size}x{patch_size}') if f.endswith('.tif')]
    mask_files = [f for f in os.listdir(f'{data_dir}ndwi_masks_tiles/deepaqua_test_dataset_no_nov_pre_2020_full_{patch_size}x{patch_size}') if f.endswith('.tif')]

    sar_files.sort()
    mask_files.sort()
    # Remove the -sar.tif suffix from the file name
    sar_files = [unidecode(f.replace('-sar.tif', '')) for f in sar_files]
    mask_files = [unidecode(f.replace(f'-ndwi_mask.tif', '')) for f in mask_files]
    common_files = sorted(list(set(sar_files) & set(mask_files)))
    common_indexes = [int(f.split('-')[1]) for f in common_files]

    tiles_data_frame = pandas.DataFrame({'index': common_indexes, 'id': common_files})
    tiles_data_frame.set_index('index', inplace=True)
    tiles_data_frame = tiles_data_frame.sort_values(by=['id'])

    # tiles_data_frame = tiles_data_frame.sample(frac=1.).reset_index(drop=True)
    tiles_data_frame['split'] = 'test'

    num_rows = len(tiles_data_frame)

    print('There are a total of {} tiles'.format(num_rows))
    return tiles_data_frame


def create_tiles_file_paper_test_local(config):
    patch_size = config['PATCH_SIZE']
    data_dir = config['TEMP_DATA_DIR']
    print('data dir:', data_dir)
    prefix = 'PRE'

    images_dir = data_dir + config[f'{prefix}_20_SAR_DIR']
    masks_dir = data_dir + config[f'{prefix}_20_MASK_DIR']

    sar_files = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.tif')]
    sar_files.sort()
    mask_files.sort()
    # Remove the -sar.tif suffix from the file name
    sar_files = [unidecode(f.replace('-sar.tif', '')) for f in sar_files]
    mask_files = [unidecode(f.replace(f'-ndwi_mask.tif', '')) for f in mask_files]
    # Find the common files in both folders
    common_files = list(set(sar_files) & set(mask_files))
    common_indexes = [int(f.split('-')[1]) for f in common_files]
    # Create a dataframe from common_files and common_indexes and sort it by id
    tiles_data_frame = pandas.DataFrame({'index': common_indexes, 'id': common_files})
    tiles_data_frame.set_index('index', inplace=True)
    tiles_data_frame = tiles_data_frame.sort_values(by=['index'])

    tiles_data_frame = tiles_data_frame.sample(frac=1).reset_index(drop=True)
    tiles_data_frame['split'] = 'train'

    # sar_files2 = [f for f in os.listdir(f'D:\\work\\pycharm_data\\sar_tiles\\deepaqua_test_dataset_no_nov_pre_2020_full_{patch_size}x{patch_size}') if f.endswith('.tif')]
    # mask_files2 = [f for f in os.listdir(f'D:\\work\\pycharm_data\\ndwi_masks_tiles\\deepaqua_test_dataset_no_nov_pre_2020_full_dynamic_{patch_size}x{patch_size}') if f.endswith('.tif')]
    sar_files2 = [f for f in os.listdir(f'D:\\work\\pycharm_data\\sar_tiles\\deepaqua_test_dataset_no_nov_pre_2020_full_{patch_size}x{patch_size}') if f.endswith('.tif')]
    mask_files2 = [f for f in os.listdir(f'D:\\work\\pycharm_data\\ndwi_masks_tiles\\deepaqua_test_dataset_no_nov_pre_2020_full_{patch_size}x{patch_size}') if f.endswith('.tif')]
    # sar_files2 = [f for f in os.listdir('D:\\work\\pycharm_data\\sar_tiles\\deepaqua_test_dataset_no_nov_pre_2020_64x64') if f.endswith('.tif')]
    # mask_files2 = [f for f in os.listdir('D:\\work\\pycharm_data\\ndwi_masks_tiles\\deepaqua_test_dataset_no_nov_pre_2020_64x64') if f.endswith('.tif')]

    sar_files2.sort()
    mask_files2.sort()
    # mask_type = 'dynamic'
    # Remove the -sar.tif suffix from the file name
    sar_files2 = [unidecode(f.replace('-sar.tif', '')) for f in sar_files2]
    mask_files2 = [unidecode(f.replace(f'-ndwi_mask.tif', '')) for f in mask_files2]
    common_files2 = sorted(list(set(sar_files2) & set(mask_files2)))
    common_indexes2 = [int(f.split('-')[1]) for f in common_files2]

    tiles_data_frame2 = pandas.DataFrame({'index': common_indexes2, 'id': common_files2})
    tiles_data_frame2.set_index('index', inplace=True)
    tiles_data_frame2 = tiles_data_frame2.sort_values(by=['id'])

    # tiles_data_frame2 = tiles_data_frame2.sample(frac=1.).reset_index(drop=True)
    tiles_data_frame2['split'] = 'valid'

    tiles_data_frame = pandas.concat([tiles_data_frame, tiles_data_frame2])
    num_rows = len(tiles_data_frame)

    print('There are a total of {} tiles'.format(num_rows))
    return tiles_data_frame


def config_update_casts(config):
    # Convert int values to int
    for key in ['EPOCHS', 'PATCH_SIZE', 'BATCH_SIZE', 'NUM_WORKERS', 'EARLY_STOP_NUM_EPOCHS',
                'TEMPORAL_CONSISTENCY_START_EPOCH', 'REDUCE_LR_PLATEAU_PATIENCE', 'TRANSFORMER_PATCH_SIZE', 'EMBED_DIM',
                'WINDOW_SIZE', 'TEMPORAL_CONSISTENCY_NUM_DATES', 'UNET_INIT_DIM', 'UNET_BLOCKS', 'DEPTH', 'KERNEL_SIZE',
                'NUM_FIRST_LVL_CHANNELS', 'K']:
        if key in config:
            config[key] = int(config[key])
    # Convert float values to float
    for key in ['LEARNING_RATE', 'MLP_RATIO', 'DROP_RATE', 'DROP_PATH_RATE', 'TEMPORAL_CONSISTENCY_WEIGHT',
                'STANDARD_TRAINING_WEIGHT', 'TEMPORAL_CONSISTENCY_POWER', 'REDUCE_LR_PLATEAU_FACTOR']:
        if key in config:
            config[key] = float(config[key])
    for key in ['SAVE_MODEL_ON_ALL_EPOCHS', 'SAVE_MODEL_ON_LAST_EPOCH', 'QKV_BIAS', 'APE',
                'PATCH_NORM', 'USE_CHECKPOINT']:
        if key in config:
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
        if key in config:
            config[key] = tuple([int(x) for x in config[key][1:-1].split(',')])
    return config