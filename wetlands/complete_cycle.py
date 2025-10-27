import shutil
import time
from dotenv import load_dotenv, dotenv_values
from wetlands import train_model, evaluate_performance, generate_ndwi, generate_sar
from utils import config_update_casts
import argparse
import traceback
import pandas as pd
import re
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=ascii)
parser.add_argument("--array_num", type=int)


# def initialise():



def update_name(config, test_name):
    def strip_name(name):
        name = re.sub('_run_\d+$', '', name)
        name = re.sub('_20\d\d$', '', name)
        return name
    model_dir = config['MODELS_DIR']
    outputs_dir = config['DATA_DIR']
    models_info = pd.read_csv(outputs_dir + model_dir + 'model_info.csv')
    past_test_names = models_info['run_name'].map(strip_name).values
    if test_name in past_test_names:
        version = 2
        while test_name + f'_ver_{version}' in past_test_names:
            version += 1
        test_name = test_name + f'_ver_{version}'
    return test_name


def main():
    if __name__ == '__main__':
            load_dotenv()
            config = dotenv_values()
            config = config_update_casts(config)
            test_name = config['TEST_NAME']
            num_trials = 5
            for i in range(num_trials):
                train_model.full_cycle(config, test_name + '_run_' + str(i))
                evaluate_performance.main(config, test_name + '_run_' + str(i), best_epoch=True, final_epoch=False)


if __name__ == '__main__':
    main()
