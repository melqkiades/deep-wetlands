import json
import sys
import time

from dotenv import load_dotenv, dotenv_values

from wetlands import train_model_pipeline, evaluate_performance_pipeline#, aggregate_results, train_model_pipeline_orebro_eval


def main():
    if __name__ == '__main__':
        load_dotenv()
        config = dotenv_values()
        # print(json.dumps(config, indent=4))

        # generate_ndwi.full_cycle()
        # generate_sar.full_cycle()
        # test_name = 't_02_lr5^-5_restored_tiles_test_cor_earlystop_new'
        # test_name = 'standard_baseline_lr5^-5_redlrplateau_corrected_final3'
        test_name = 'standard_speed_test'
        num_trials = 5
        test_dataset = 'deepaqua_test_dataset_no_nov'
        for i in range(num_trials):
            train_model_pipeline.full_cycle(test_name + '_run_' + str(i), True)
            # train_model_pipeline_orebro_eval.full_cycle(test_name + '_run_' + str(i), False)
            # train_model_pipeline.full_cycle(test_name + '_run_' + str(i), False)
            # evaluate_performance_pipeline.main(test_name + '_run_' + str(i), dataset_name=test_dataset,
            #                                    best_epoch=True, final_epoch=False, all_epochs=False)
        # aggregate_results.main(test_name, test_dataset)
        # map_wetlands.full_cycle()
        # estimate_water.main()
        # performance_evaluator.full_cycle()
        # evaluate_performance.main()

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    total_time = end - start
    print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
