import glob
import json
import shutil
import sys, traceback
import time
from dotenv import load_dotenv, dotenv_values
from wetlands import train_model_pipeline, evaluate_performance_pipeline, aggregate_results
from utils import config_update_casts
import argparse
import os
import traceback
from slack_exception_logger import SlackExceptionLogger
import requests
import pandas as pd
import re
import requests


def send_slack_message(message):
    slack_hook = "###"
    payload = {
        "text": message}
    r = requests.post(url=slack_hook, json=payload)

def update_name(config, test_name):
    def strip_name(name):
        name = re.sub('_run_\d+$', '', name)
        name = re.sub('_20\d\d$', '', name)
        return name
    model_dir = config['MODELS_DIR']
    outputs_dir = config['OUTPUTS_DIR']
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
        slack_hook = "##"
        slack_logger = SlackExceptionLogger(slack_hook, "#alvis-messages")
        try:
            load_dotenv()
            config = dotenv_values()
            config = config_update_casts(config)
            start_time = time.time()
            test_name = config['TEST_NAME']
            num_trials = 3
            test_dataset = 'deepaqua_test_dataset_no_nov'
            # test_name = update_name(config, test_name)
            for i in range(num_trials):
                train_model_pipeline.full_cycle(config, test_name + '_2018_run_' + str(i), True)
                train_model_pipeline.full_cycle(config, test_name + '_2020_run_' + str(i), False)
                evaluate_performance_pipeline.main(config, test_name + '_run_' + str(i), dataset_name=test_dataset,
                                                   best_epoch=True, final_epoch=False, all_epochs=False)
            aggregate_results.main(config, test_name, test_dataset)
            payload = {
                "text": f'----------------------------------------------------------------------------------------------------------\nFinished. Total time: {time.time()-start_time}'}
            r = requests.post(url=slack_hook, json=payload)
        except Exception as e:
            payload = {
                "text": f'----------------------------------------------------------------------------------------------------------'}
            r = requests.post(url=slack_hook, json=payload)
            slack_logger.push_to_slack(e)
            print(traceback.format_exc())


if __name__ == '__main__':
    main()
