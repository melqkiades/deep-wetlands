# import os
# import glob
# from natsort import natsorted
# import csv
# import numpy as np


# def main(test_name, test_dataset):
#     directories = glob.glob(f'C:\\Users\\ioia4268\\data\\performance_evaluator\\{test_dataset}\\{test_name}_run*\\')
#     directories = natsorted(directories)
#     filenames = [os.path.basename(filename) for filename in glob.glob(directories[0]+'*_performance*.csv')]
#
#     for filename in filenames:
#         statistics = {}
#
#         for directory in directories:
#             with open(directory + filename, newline='') as csvfile:
#                 spamreader = csv.reader(csvfile)
#                 for name, value in spamreader:
#                     if name == 'Area':
#                         area_name = value
#                         if area_name not in statistics:
#                             statistics[area_name] = {}
#                     else:
#                         if name not in statistics[area_name]:
#                             statistics[area_name][name] = []
#                         statistics[area_name][name].append(float(value))
#         with open(directories[0] + 'mean_' + filename, 'w', newline='') as csvfile:
#             spamwriter = csv.writer(csvfile, delimiter=',')
#             for area_name in statistics:
#                 spamwriter.writerow(['Area', area_name])
#                 for statistic in statistics[area_name]:
#                     spamwriter.writerow([statistic, np.mean(statistics[area_name][statistic]), ''] + statistics[area_name][statistic])
#     compiled_results = pd.read_csv(f'{outputs_dir}{performance_evaluator_dir}{dataset_name}/compiled_results.csv')
#     new_results = {'Test name': [test_name+'_0-'+str(num_preds-1)]}
#     for area in areas:
#         new_results[area + ' iou'] = [mean_results[area]['iou']]
#     new_results = pd.DataFrame.from_dict(new_results)
#     updated_results = pd.concat([compiled_results, new_results], join='outer')
#     if config['ENSEMBLE'] == 'YES':
#         performance_temp = {}
#         with open(f'{outputs_dir}{performance_evaluator_dir}{dataset_name}/{test_name}_ensemble/ensemble_performance_split.csv', 'r', encoding='utf-8') as f:
#             for line in csv.reader(f):
#                 if line[0] == 'Area':
#                     area = line[1]
#                     performance_temp[area] = {}
#                 else:
#                     if '.' in line[1]:
#                         performance_temp[area][line[0]] = float(line[1])
#                     else:
#                         performance_temp[area][line[0]] = int(line[1])
#         ensemble_results = {'Test name': [test_name + '_ENSEMBLE']}
#         for area in areas:
#             ensemble_results[area + ' iou'] = [performance_temp[area]['iou']]
#         ensemble_results = pd.DataFrame.from_dict(ensemble_results)
#         updated_results = pd.concat([updated_results, ensemble_results], join='outer')
#     updated_results.to_csv(f'{outputs_dir}{performance_evaluator_dir}{dataset_name}/compiled_results.csv', index=False, na_rep='N/A')


# test_names = ['standard_baseline_lr5^-5_redlrplateau_corrected_final3', 't_01_2_pow2_5ep_a2',
#               't_02_2_2ep_baseline', 't_01_2_pow2_5ep_chained']
# for test_name in test_names:
# main('standard_baseline_lr2^-5', 'deepaqua_test_dataset_no_nov')

import csv
import glob
import numpy as np
import pandas as pd
from natsort import natsorted


def main(config, test_name, dataset_name='deepaqua_test_dataset_no_nov_pre_2020'):
    outputs_dir = config['OUTPUTS_DIR']
    performance_evaluator_dir = config['EVALUATION_DIR']
    performance_data = []
    important_metrics = ['iou']
    metrics = []
    print(f'{outputs_dir}{performance_evaluator_dir}{dataset_name}/{test_name}_run_*/')
    for directory in glob.glob(f'{outputs_dir}{performance_evaluator_dir}{dataset_name}/{test_name}_run_*/'):
        print(directory)
        performance_temp = {}
        with open(directory + 'best_epoch_performance_split.csv', 'r', encoding='utf-8') as f:
            # print(directory + 'best_epoch_performance_split.csv')
            for line in csv.reader(f):
                # print(line)
                if line[0] == 'Area':
                    area = line[1]
                    performance_temp[area] = {}
                else:
                    metric = line[0]
                    if metric not in metrics:
                        metrics.append(metric)
                    if line[1] == 'nan':
                        performance_temp[area][line[0]] = 0.
                    elif '.' in line[1]:
                        performance_temp[area][line[0]] = float(line[1])
                    else:
                        performance_temp[area][line[0]] = int(line[1])
        performance_data.append(performance_temp)
    num_preds = len(performance_data)
    mean_results = {}
    areas = list(performance_data[0].keys())
    areas.remove('global')
    areas = natsorted(areas) + ['global']
    for area in areas:
        mean_results[area] = {}
        for metric in metrics:
            mean_results[area][metric] = np.mean([performance_data[i][area][metric] for i in range(len(performance_data))])
    with open(f'{outputs_dir}{performance_evaluator_dir}{dataset_name}/{test_name}_run_0/compiled_results.csv', 'w', encoding='utf-8', newline='') as f:
        spamwriter = csv.writer(f)
        for area in areas:
            spamwriter.writerow(['Area', area])
            for metric in metrics:
                spamwriter.writerow([metric, mean_results[area][metric]])
            spamwriter.writerow(['', ''])
    compiled_results = pd.read_csv(f'{outputs_dir}{performance_evaluator_dir}{dataset_name}/compiled_results.csv')
    new_results = {'Test name': [test_name+'_0-'+str(num_preds-1)]}
    for area in areas:
        new_results[area + ' iou'] = [mean_results[area]['iou']]
    new_results = pd.DataFrame.from_dict(new_results)
    updated_results = pd.concat([compiled_results, new_results], join='outer')
    # if config['ENSEMBLE'] == 'YES':
    #     performance_temp = {}
    #     with open(f'{outputs_dir}{performance_evaluator_dir}{dataset_name}/{test_name}_ensemble/ensemble_performance_split.csv', 'r', encoding='utf-8') as f:
    #         for line in csv.reader(f):
    #             if line[0] == 'Area':
    #                 area = line[1]
    #                 performance_temp[area] = {}
    #             else:
    #                 if '.' in line[1]:
    #                     performance_temp[area][line[0]] = float(line[1])
    #                 else:
    #                     performance_temp[area][line[0]] = int(line[1])
    #     ensemble_results = {'Test name': [test_name + '_ENSEMBLE']}
    #     for area in areas:
    #         ensemble_results[area + ' iou'] = [performance_temp[area]['iou']]
    #     ensemble_results = pd.DataFrame.from_dict(ensemble_results)
    #     updated_results = pd.concat([updated_results, ensemble_results], join='outer')
    updated_results.to_csv(f'{outputs_dir}{performance_evaluator_dir}{dataset_name}/compiled_results.csv', index=False, na_rep='N/A')