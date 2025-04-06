import json
import sys
import time
from dotenv import load_dotenv, dotenv_values

from wetlands import train_model_pipeline_local, evaluate_performance_pipeline, map_wetlands, estimate_water, performance_evaluator # generate_ndwi, generate_sar,


def main():
    if __name__ == '__main__':
        load_dotenv()
        config = dotenv_values()
        print(json.dumps(config, indent=4))
        test_name = 'tr_test_train_is_val'
        num_trials = 5
        test_dataset = 'deepaqua_test_dataset_no_nov'
        for i in range(num_trials):
            train_model_pipeline_local.full_cycle(test_name + '_run_' + str(i), True)
            # train_model_pipeline_orebro_eval.full_cycle(test_name + '_run_' + str(i), False)
            # train_model_pipeline.full_cycle(test_name + '_run_' + str(i), False)
            # evaluate_performance_pipeline.main(test_name + '_run_' + str(i), dataset_name=test_dataset,
            #                                    best_epoch=True, final_epoch=False, all_epochs=False)

        # # generate_ndwi.full_cycle()
        # # generate_sar.full_cycle()
        # train_model.full_cycle()
        # #map_wetlands.full_cycle()
        # #estimate_water.main()
        # #performance_evaluator.full_cycle()

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    total_time = end - start
    print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
