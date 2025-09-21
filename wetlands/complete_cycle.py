import json
import sys
import time
from datetime import datetime
from dotenv import load_dotenv, dotenv_values
import os
from wetlands import train_model_pipeline, evaluate_performance_pipeline, aggregate_results, train_model_pipeline_orebro_eval, train_model_pipeline_save_val_results
import traceback
from slack_exception_logger import SlackExceptionLogger
import requests


def main():
    if __name__ == '__main__':
        slack_hook = "https://hooks.slack.com/services/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"
        slack_logger = SlackExceptionLogger(slack_hook, "#alvis-messages")
        try:
            load_dotenv()
            config = dotenv_values()
            # print(json.dumps(config, indent=4))

            # generate_ndwi.full_cycle()
            # generate_sar.full_cycle()
            start_time = time.time()
            date = datetime.today().strftime('%Y-%m-%d')
            test_name = 'old_2020_test_real'
            num_trials = 1
            test_dataset = 'deepaqua_test_dataset_no_nov'
            # test_dataset = 'tavvavuoma_wide'
            for i in range(num_trials):
                train_model_pipeline.full_cycle(test_name + '_2018_run_' + str(i), True)
                train_model_pipeline.full_cycle(test_name + '_2020_run_' + str(i), False)
                # train_model_pipeline_orebro_eval.full_cycle(test_name + '_run_' + str(i), False)
                # train_model_pipeline_save_val_results.full_cycle(test_name + '_2020_run_' + str(i), False)
                evaluate_performance_pipeline.main(test_name + '_run_' + str(i), dataset_name=test_dataset,
                                                   best_epoch=True, final_epoch=False, all_epochs=False)
            # aggregate_results.main(test_name, test_dataset)
            # map_wetlands.full_cycle()
            # estimate_water.main()
            # performance_evaluator.full_cycle()
            # evaluate_performance.main()
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
    start = time.time()
    main()
    end = time.time()
    total_time = end - start
    print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
