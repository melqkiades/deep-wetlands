import glob
import os
import shutil
import cv2
import numpy as np
import pandas
import seaborn
import tqdm
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from evaluation import semantic_segmentation_evaluator
from model import model_factory
from wetlands import utils, viz_utils, map_wetlands, wandb_utils, noise_filters, jaccard_similarity
import rasterio as rio
from jaccard_similarity import calculate_intersection_over_union
from pathlib import Path
from skimage import io
import time
from networks.vision_transformer import SwinUnet as ViT_seg
import torch
import shutil


def convert_area_name_to_color(area_name):
    area_name_to_color = {'hjalstaviken':'red', 'hornborgasjon':'blue', 'svartadalen':'green'}
    if area_name in area_name_to_color:
        return area_name_to_color[area_name]
    else:
        return 'red'


def iterate(config, test_name, dataset_name, prediction_data_dict, annotated_data_dict, description, split_by_date=False):
    data_dir = config['DATA_DIR']
    performance_evaluator_dir = config['EVALUATION_DIR']
    # 1. Iterate all the annotated images and extract the date
    ious = {}
    accuracies = {}
    predictions = {}
    annotations = {}
    ious_list = []
    if not split_by_date:
        correct_values = {'hjalstaviken':{'Pixel accuracy': 0.96, 'IOU':0.68, 'Precision':0.81, 'Recall':0.81, 'F1':0.81},
                          'hornborgasjon':{'Pixel accuracy': 0.98, 'IOU':0.94, 'Precision':0.98, 'Recall':0.96, 'F1':0.97},
                          'svartadalen':{'Pixel accuracy': 0.97, 'IOU':0.88, 'Precision':0.98, 'Recall':0.9, 'F1':0.93}}

    performance_dir = data_dir + performance_evaluator_dir + f'{dataset_name}/{test_name}/'
    if not os.path.isdir(performance_dir):
        Path(performance_dir).mkdir(parents=True, exist_ok=True)
    model_performance_dir = f'{performance_dir}{description}_performance/'
    if not os.path.isdir(model_performance_dir):
        Path(model_performance_dir).mkdir(parents=True, exist_ok=True)
    results_dir = data_dir + config['RESULTS_DIR'] + f'{dataset_name}/{test_name}/'
    model_results_dir = f'{results_dir}{description}_exported_images/'
    annotations_dir = data_dir + config['ANNOTATED_DATA_DIR'] + f'{dataset_name}/'
    annotated_files = [filename for filename in os.listdir(annotations_dir) if 'annotated_vh' in filename and filename.endswith('.tif')]
    print('Annotated files:')
    print(annotated_files)
    for annotated_file in annotated_files:
        area_name = annotated_file.split('_')[0].lower()
        year = annotated_file.split('_')[-1].split('-')[0]
        if split_by_date:
            if year in ['2018', '2019']:
                date_period = '2018'
            elif year in ['2020', '2021', '2022']:
                date_period = '2020'
            area_name = area_name + '_' + date_period
        if area_name not in ious.keys():
            ious[area_name] = []
            accuracies[area_name] = []
            predictions[area_name] = []
            annotations[area_name] = []
        # Open the annotated file
        annotations[area_name].append(annotated_data_dict[annotated_file])

        # Locate the prediction file
        prediction_file = model_results_dir+ annotated_file.lower().replace('annotated_vh', 'mosaic').replace('.tif',
                                                                                                             f'_sar_VH.tif')
        # Open the prediction file
        prediction_data = prediction_data_dict[os.path.basename(prediction_file)].astype(int)
        predictions[area_name].append(prediction_data)

        iou = jaccard_similarity.calculate_intersection_over_union(prediction_data, annotated_data_dict[annotated_file])
        ious[area_name].append(iou)
        ious_list.append(iou)
        accuracy = (annotated_data_dict[annotated_file] == prediction_data).sum() / (annotated_data_dict[annotated_file].shape[0] * annotated_data_dict[annotated_file].shape[1])
        accuracies[area_name].append(accuracy)
    image_results_df = pandas.DataFrame({'filename': annotated_files, 'iou': ious_list})
    image_results_df.to_csv(f'{model_performance_dir}filename_ious.csv')
    all_area_names = list(ious.keys())
    ious['global'] = []
    predictions['global'] = []
    annotations['global'] = []
    accuracies['global'] = []
    if not split_by_date:
        correct_values['global'] = []
    for area_name in all_area_names:
        ious['global'] += ious[area_name]
        predictions['global'] += predictions[area_name]
        annotations['global'] += annotations[area_name]
        accuracies['global'] += accuracies[area_name]
        if not split_by_date:
            correct_values['global'] += correct_values[area_name]
    for area_name in ious.keys():
        print('\n\nAREA: ', area_name)
        result = semantic_segmentation_evaluator.eval_semantic_segmentation(predictions[area_name], annotations[area_name])

        print(result)

        print(f'Mean {test_name} {description} IOU', np.asarray(ious[area_name]).mean())
        print(f'Mean {test_name} {description} Accuracy', np.asarray(accuracies[area_name]).mean())

        confusion_matrix = semantic_segmentation_evaluator.calc_semantic_segmentation_confusion(predictions[area_name], annotations[area_name])
        print(confusion_matrix)
        TP = confusion_matrix[1, 1]
        TN = confusion_matrix[0, 0]
        FP = confusion_matrix[0, 1]
        FN = confusion_matrix[1, 0]

        print(f'{TP}\t{TN}\t{FP}\t{FN}')
        print('True Positives:', TP)
        print('True Negatives:', TN)
        print('False Positives:', FP)
        print('False Negatives:', FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)
        iou = TP / (TP + FP + FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        if not split_by_date:
            print('Pixel accuracy:', accuracy, accuracy-correct_values[area_name]['Pixel accuracy'])
            print('IOU:', iou, iou-correct_values[area_name]['IOU'])
            print('Precision:', precision, precision-correct_values[area_name]['Precision'])
            print('Recall:', recall, recall-correct_values[area_name]['Recall'])
            print('F1 Score:', f1_score, f1_score-correct_values[area_name]['F1'])
        else:
            print('Pixel accuracy:', accuracy, accuracy)
            print('IOU:', iou, iou)
            print('Precision:', precision, precision)
            print('Recall:', recall, recall)
            print('F1 Score:', f1_score, f1_score)
        # print('Area under curve:', aucs_dataframe.iloc[0][area_name])

        metrics = {
            'accuracy': accuracy,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': TP,
            'true_negatives': TN,
            'false_positives': FP,
            'false_negatives': FN,
            # 'area_under_curve': aucs_dataframe.iloc[0][area_name]
        }

        # Export metrics to CSV
        if not split_by_date:
            metrics_file = f'{performance_dir}{description}_performance.csv'
        else:
            metrics_file = f'{performance_dir}{description}_performance_split.csv'
        with open(metrics_file, 'a') as f:
            f.write("%s,%s\n" % ('Area', area_name))
            for key in metrics.keys():
                f.write("%s,%s\n" % (key, metrics[key]))

        # # if area_name is not 'global':
        #     # ConfusionMatrixDisplay.from_predictions(annotations, predictions, display_labels=['Water', 'Land']).plot()
        #     # Flatten the arrays
        #     area_annotations = np.asarray(annotations[area_name]).flatten()
        #     area_predictions = np.asarray(predictions[area_name]).flatten()
        #     ConfusionMatrixDisplay.from_predictions(area_annotations, area_predictions, display_labels=['Soil', 'Water'], cmap=plt.cm.Blues)
        #     ConfusionMatrixDisplay.from_predictions(area_annotations, area_predictions, normalize='true', display_labels=['Soil', 'Water'], cmap=plt.cm.Blues, values_format='.2%')
        #     ConfusionMatrixDisplay.from_predictions(area_annotations, area_predictions, normalize='pred', display_labels=['Soil', 'Water'], cmap=plt.cm.Blues, values_format='.2%')
        #     ConfusionMatrixDisplay.from_predictions(area_annotations, area_predictions, normalize='all', display_labels=['Soil', 'Water'], cmap=plt.cm.Blues, values_format='.2%')
        #     # plt.show()
        #
        #     cmat = [[TP, FN], [FP, TN]]
        #
        #     plt.figure(figsize=(6, 6))
        #     ax = seaborn.heatmap(cmat / np.sum(cmat), cmap="Reds", annot=True, fmt='.2%', square=1, linewidth=2.)
        #     ax.set_title(test_name)
        #     ax.xaxis.set_ticklabels(['Water', 'Soil'])
        #     ax.yaxis.set_ticklabels(['Water', 'Soil'])
        #     plt.xlabel("Predictions")
        #     plt.ylabel("Real values")


def visualize_predicted_image(config, image, model, device, file_name, test_name, dataset_name, description):
    if test_name == 'otsu':
        pred_mask = otsu_threshold(image)
    elif test_name == 'otsu_gaussian':
        kernel_size = os.getenv('OTSU_GAUSSIAN_KERNEL_SIZE')
        pred_mask = otsu_gaussian_threshold(image, int(kernel_size))
    elif test_name == 'thresholding_2018':
        pred_mask = threshold_method(image, 0.6486486486486487)
    elif test_name == 'thresholding_2020':
        pred_mask = threshold_method(image, 0.36236236236236236)
    else:
        pred_mask = map_wetlands.predict_water_mask(config, image, model, device, pad=True)

    unique, counts = np.unique(pred_mask, return_counts=True)
    results = dict(zip(unique, counts))
    if not 0. in results:
        results[0.] = 0
    if not 1. in results:
        results[1.] = 0
    image_date = file_name.split('_')[2]
    satellite = file_name.split('_')[-1]
    results['Date'] = image_date
    results['Satellite'] = satellite
    results['File_name'] = file_name

    images_dir = config['DATA_DIR'] + config['RESULTS_DIR'] + f'{dataset_name}/{test_name}/{description}_exported_images/'

    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)
    img = Image.fromarray(np.uint8((pred_mask) * 255), 'L')
    img.save(images_dir + file_name + '_pred_bw.png')

    return results, pred_mask


def get_prediction_image(config, tiff_image, tiff_file, model, device, test_name, dataset_name, description):

    file_name = os.path.basename(tiff_file).split('.')[0]
    results, prediction_image = visualize_predicted_image(config, tiff_image, model, device, file_name, test_name, dataset_name, description)
    return results, prediction_image


def otsu_threshold(image):
    image = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype('uint8')

    # Apply Otsu's thresholding on image
    threshold, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholded_image = 1 - ((thresholded_image - thresholded_image.min()) / (thresholded_image.max() - thresholded_image.min()))

    return thresholded_image


def otsu_gaussian_threshold(image, kernel_size=5):
    image = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype('uint8')

    # Apply Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    threshold, thresholded_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholded_image = 1 - ((thresholded_image - thresholded_image.min()) / (thresholded_image.max() - thresholded_image.min()))

    return thresholded_image


def threshold_method(image, threshold):

    # 2018: Optimal threshold: 0.6486486486486487 	Dice score: 0.8768594116989151 	Noise filter: gaussian
    # 2020: Optimal threshold 0.36236236236236236 	Dice score: 0.9218205699274742 	Noise filter: gaussian
    
    # threshold = 0.6486486486486487  # 2018
    # threshold = 0.36236236236236236  # 2020
    denoised_image = noise_filters.get_noise_filters()['gaussian'](image)
    water_prediction = (denoised_image < threshold).astype(float)
    
    return water_prediction


def plot_results(config, test_name, dataset_name, description):
    data_dir = config['DATA_DIR']
    charts_dir = data_dir + config['CHARTS_DIR'] + f'{dataset_name}/{test_name}/'
    results_dir = data_dir + config['RESULTS_DIR'] + f'{dataset_name}/{test_name}/'
    results_file = f'{results_dir}{description}_water_estimates.csv'
    data_frame = pandas.read_csv(results_file, usecols=['1.0', 'Date'], index_col=["Date"],  parse_dates=["Date"])
    data_frame.plot(title=test_name)
    plt.savefig(f'{charts_dir}{description}_water_estimates.png')


def update_water_estimates(config, test_name, dataset_name, description):
    data_dir = config['DATA_DIR']
    charts_dir = data_dir + config['CHARTS_DIR'] + f'{dataset_name}/{test_name}/'
    results_dir = data_dir + config['RESULTS_DIR'] + f'{dataset_name}/{test_name}/'
    results_file = f'{results_dir}{description}_water_estimates.csv'

    data_frame = pandas.read_csv(results_file, usecols=['1.0', 'Date', 'File_name'],  parse_dates=["Date"])
    print(data_frame.size)
    print(data_frame.columns.values)
    data_frame['area_name'] = data_frame.apply(lambda x:x['File_name'].split('_')[0], axis=1)
    data_frame['color'] = data_frame.apply(lambda x: convert_area_name_to_color(x['area_name']), axis=1)
    data_frame = data_frame.sort_values(by=['area_name', 'Date'])
    ious = {}
    days = {}
    ious_skip_winter = {}
    dates_skip_winter = {}
    for index, row in data_frame.iterrows():
        area_name = row['area_name']
        file_name = row['File_name'] + '_pred_bw.png'
        date = row['Date']
        if area_name not in ious:
            ious[area_name] = [0.]
            prev_prediction = io.imread(f'{results_dir}{description}_exported_images/'+ file_name)
            days[area_name] = [0.]
            prev_date = date
            ious_skip_winter[area_name] = [0.]
            dates_skip_winter[area_name] = [date]
        else:
            days_passed = (date - prev_date).days
            prediction = io.imread(f'{results_dir}{description}_exported_images/'+ file_name)
            if days_passed < 60:
                temp_iou = calculate_intersection_over_union(prev_prediction, prediction)
                ious[area_name].append(temp_iou)
                ious_skip_winter[area_name].append(temp_iou)
            else:
                ious[area_name].append(0)
                ious_skip_winter[area_name].append(0)
                ious_skip_winter[area_name].append(0)
                dates_skip_winter[area_name].append(dates_skip_winter[area_name][-1] + pandas.Timedelta("1 day"))
            prev_prediction = prediction[:]
            days[area_name].append(days[area_name][-1] + days_passed)
            dates_skip_winter[area_name].append(date)
            prev_date = date
    ious_list = []
    days_list = []
    for area_name in ious.keys():
        ious_list += ious[area_name]
        days_list += days[area_name]
    data_frame['ious'] = ious_list
    data_frame['days'] = days_list
    data_frame.drop(['File_name'], axis=1, inplace=True)
    data_frame = data_frame[data_frame['Date'].dt.month.isin([4, 5, 6, 7, 8, 9, 10, 11])]
    print(data_frame.size)
    print(data_frame.columns.values)

    data_frame.plot(x='Date', y='1.0', kind='scatter', title=f'{test_name} {description} [{dataset_name}]', c='color', s=0.7**2)
    plt.savefig(f'{charts_dir}{description}_scatter_new_water_estimates_filtered.png')
    plt.clf()
    areas_under_curve = {}
    for area_name in ious.keys():
        days_skip_winter = [0.]
        for i in range(1, len(dates_skip_winter[area_name])):
            days_skip_winter.append((dates_skip_winter[area_name][i] - dates_skip_winter[area_name][0]).days)
        area_under_curve = np.trapz(ious_skip_winter[area_name], x=days_skip_winter) / days_skip_winter[-1]
        areas_under_curve[area_name] = [area_under_curve]
        plt.plot(dates_skip_winter[area_name], ious_skip_winter[area_name], label='ious')
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.title(f'{description}_{area_name} auc: {str(round(area_under_curve,2))}')
        plt.savefig(f'{charts_dir}{description}_ious_{area_name}_new_water_estimates_filtered.png')
        plt.clf()
    auc_dataframe = pandas.DataFrame.from_dict(areas_under_curve)
    auc_dataframe.to_csv(f'{results_dir}{description}_areas_under_curve.csv')
    data_frame.to_csv(f'{results_dir}{description}_new_water_estimates_filtered.csv')


def full_cycle(config, test_name, dataset_name, images_dict, model_paths, description):
    patch_size = config['PATCH_SIZE']
    device = utils.get_device()
    results_dir = config['DATA_DIR'] + config['RESULTS_DIR'] + f'{dataset_name}/{test_name}/'
    training_method = config['TRAINING_METHOD']
    if training_method == 'multitemporal_data':
        num_input_channels = 3
    else:
        num_input_channels = 1
    if test_name not in ['otsu', 'otsu_gaussian', 'thresholding_2018', 'thresholding_2020']:
        model_2018 = ViT_seg(img_size=patch_size, num_classes=2, patch_size=config['TRANSFORMER_PATCH_SIZE'], input_channels=num_input_channels,
                    embed_dim=config['EMBED_DIM'], depths=config['DEPTHS'], num_heads=config['NUM_HEADS'],
                    window_size=config['WINDOW_SIZE'], mlp_ratio=config['MLP_RATIO'], qkv_bias=config['QKV_BIAS'],
                    qk_scale=config['QK_SKALE'], drop_rate=config['DROP_RATE'], drop_path_rate=config['DROP_PATH_RATE'],
                    ape=config['APE'], patch_norm=config['PATCH_NORM'], use_checkpoint=config['USE_CHECKPOINT']).cuda()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(model_paths[0], map_location=device)
        pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
        model_2018.swin_unet.load_state_dict(pretrained_dict, strict=False)
        model_2020 = ViT_seg(img_size=patch_size, num_classes=2, patch_size=config['TRANSFORMER_PATCH_SIZE'], input_channels=num_input_channels,
                    embed_dim=config['EMBED_DIM'], depths=config['DEPTHS'], num_heads=config['NUM_HEADS'],
                    window_size=config['WINDOW_SIZE'], mlp_ratio=config['MLP_RATIO'], qkv_bias=config['QKV_BIAS'],
                    qk_scale=config['QK_SKALE'], drop_rate=config['DROP_RATE'], drop_path_rate=config['DROP_PATH_RATE'],
                    ape=config['APE'], patch_norm=config['PATCH_NORM'], use_checkpoint=config['USE_CHECKPOINT']).cuda()
        pretrained_dict = torch.load(model_paths[1], map_location=device)
        pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
        model_2020.swin_unet.load_state_dict(pretrained_dict, strict=False)
    else:
        model_2018 = None
        model_2020 = None

    results_list = []
    prediction_data = {}

    for tiff_file in tqdm.tqdm(images_dict):
        if not tiff_file.endswith('.tif'):
            continue
        year = int(tiff_file.split('_')[-3].split('-')[0])
        if year < 2020:
            results, prediction_image = get_prediction_image(config, images_dict[tiff_file], tiff_file, model_2018, device, test_name, dataset_name, description)
        else:
            results, prediction_image = get_prediction_image(config, images_dict[tiff_file], tiff_file, model_2020, device, test_name, dataset_name, description)

        results_list.append(results)
        prediction_data[tiff_file] = prediction_image

    data_frame = pandas.DataFrame(results_list)
    data_frame['Date'] = data_frame['Date'].apply(pandas.to_datetime).dt.date
    print(data_frame.head())
    data_frame.to_csv(f'{results_dir}{description}_water_estimates.csv')

    return prediction_data


def main(config, test_name, dataset_name='deepaqua_test_dataset_no_nov', best_epoch=True, all_epochs=False, final_epoch=False):
    data_dir = config['DATA_DIR']
    outputs_dir = config['OUTPUTS_DIR']
    results_dir = config['RESULTS_DIR']
    performance_evaluator_dir = config['EVALUATION_DIR']
    models_dir = outputs_dir + config['MODELS_DIR']
    manual_annotations_dir = config['ANNOTATED_DATA_DIR']
    training_method = config['TRAINING_METHOD']
    if not os.path.isdir(data_dir + results_dir+ f'{dataset_name}/{test_name}/'):
        Path(data_dir + results_dir+ f'{dataset_name}/{test_name}/').mkdir(parents=True, exist_ok=True)
    charts_dir = config['CHARTS_DIR']
    if not os.path.isdir(data_dir + charts_dir + f'{dataset_name}/{test_name}/'):
        Path(data_dir + charts_dir + f'{dataset_name}/{test_name}/').mkdir(parents=True, exist_ok=True)
    patch_size = config['PATCH_SIZE']
    tiff_dir = data_dir + config['SAR_DIR'] + dataset_name

    if not os.path.exists(tiff_dir):
        raise FileNotFoundError(f'The folder containing the TIFF files does not exist: {tiff_dir}')

    filenames = sorted([os.path.basename(x) for x in glob.glob(tiff_dir+'/*.tif')])
    if training_method == 'multitemporal_data':
        filenames_past = sorted([os.path.basename(x) for x in glob.glob(tiff_dir+'_past'+'/*.tif')])
        filenames_future = sorted([os.path.basename(x) for x in glob.glob(tiff_dir+'_future'+'/*.tif')])
        combined_filenames = []
        for filename in filenames:
            filename_group = [filename]
            date = filename.split('_')[-3]
            found = False
            for filename_past in filenames_past:
                if date in filename_past:
                    filename_group.insert(0, filename_past)
                    found = True
                    break
            if not found:
                continue
            else:
                found = False
                for filename_future in filenames_future:
                    if date in filename_future:
                        filename_group.append(filename_future)
                        found = True
                        break
                if found:
                    combined_filenames.append(filename_group)
    images_dict = {}
    incomplete_images = 0

    with rio.open(config['DATA_DIR'] + config['SAR_DIR'] + config['PRE_20_MINMAX_IMAGE']) as src:
        dataset_array = src.read()
        minValue_2018 = np.nanpercentile(dataset_array, 1)
        maxValue_2018 = np.nanpercentile(dataset_array, 99)
    with rio.open(config['DATA_DIR'] + config['SAR_DIR'] + config['POST_20_MINMAX_IMAGE']) as src:
        dataset_array = src.read()
        minValue_2020 = np.nanpercentile(dataset_array, 1)
        maxValue_2020 = np.nanpercentile(dataset_array, 99)
    training_method = config['TRAINING_METHOD']
    if training_method != 'multitemporal_data':
        for tiff_file in tqdm.tqdm(sorted(filenames)):
            if '2014' in tiff_file or'2015' in tiff_file or '2016' in tiff_file or '2017' in tiff_file or '2018' in tiff_file or '2019' in tiff_file:
                minValue = minValue_2018
                maxValue = maxValue_2018
            else:
                minValue = minValue_2020
                maxValue = maxValue_2020
            image = viz_utils.load_image(tiff_dir + '/' + tiff_file, ignore_nan=True, min_value=minValue, max_value=maxValue)
            if image is None:
                incomplete_images += 1
            else:
                images_dict[tiff_file] = image
    else:
        drop_indices = []
        for i in range(len(combined_filenames)):
            tiff_files = combined_filenames[i]
            tiff_file = tiff_files[1]
            if '2014' in tiff_file or '2015' in tiff_file or '2016' in tiff_file or '2017' in tiff_file or '2018' in tiff_file or '2019' in tiff_file:
                minValue = minValue_2018
                maxValue = maxValue_2018
            else:
                minValue = minValue_2020
                maxValue = maxValue_2020
            image = viz_utils.load_image(tiff_dir + '/' + tiff_file, ignore_nan=False, min_value=minValue, max_value=maxValue)
            past_image = viz_utils.load_image(tiff_dir + '_past' + '/' + tiff_files[0], ignore_nan=False, min_value=minValue, max_value=maxValue)
            future_image = viz_utils.load_image(tiff_dir + '_future' + '/' + tiff_files[2], ignore_nan=False, min_value=minValue,
                                              max_value=maxValue)
            if not image.shape == past_image.shape == future_image.shape:
                print(f'Misaligned shapes {image.shape} {past_image.shape} {future_image.shape}')
                drop_indices.append(i)
                continue
            images_dict[tiff_file] = np.stack([past_image, image, future_image], 0)
        for drop_index in reversed(drop_indices):
            del combined_filenames[drop_index]

    print(f'There were a total of {incomplete_images} incomplete images')
    annotated_data_dict = {}
    if dataset_name in ['deepaqua_test_dataset_no_nov', 'deepaqua_test_dataset']:
        annotations_dir = data_dir + manual_annotations_dir + dataset_name + '/'
        annotated_files = [filename for filename in os.listdir(annotations_dir) if 'annotated_vh' in filename and filename.endswith('.tif')]
        for annotated_file in annotated_files:
            # Open the annotated file
            annotated_image = io.imread(annotations_dir + annotated_file)
            annotated_image[annotated_image==0.5] = 0

            array_min, array_max = np.nanmin(annotated_image), np.nanmax(annotated_image)
            annotated_data_dict[annotated_file] = ((annotated_image - array_min) / (array_max - array_min)).astype(int)
    model_data = pandas.read_csv(models_dir + 'model_info.csv')
    if best_epoch:
        description = 'best_epoch'
        evaluation_pipeline(config, test_name, dataset_name, images_dict, annotated_data_dict, description, model_data)
    if final_epoch:
        description = 'final_epoch'
        evaluation_pipeline(config, test_name, dataset_name, images_dict, annotated_data_dict, description, model_data)
    # if all_epochs:
    #     final_epoch_num = np.minimum(model_data.loc[(model_data['test_name'] == test_name) &
    #            (model_data['training_date'] == '2018-07-04')]['final_epoch'].values[-1],
    #                                  model_data.loc[(model_data['test_name'] == test_name) &
    #            (model_data['training_date'] == '2020-06-23')]['final_epoch'].values[-1])
    #     for epoch_num in range(1, final_epoch_num + 1):
    #         description = f'epoch_{epoch_num}'
    #         evaluation_pipeline(test_name, dataset_name, images_dict, annotated_data_dict, description, model_data, config, patch_size)
    # print(f'{outputs_dir}{performance_evaluator_dir}{test_name}.zip', f'{data_dir}{performance_evaluator_dir}')
    # shutil.make_archive(f'{outputs_dir}{performance_evaluator_dir}{test_name}', 'zip',
    #                      f'{data_dir}{performance_evaluator_dir}')
    # shutil.unpack_archive(f'{outputs_dir}{performance_evaluator_dir}{test_name}.zip',f'{outputs_dir}{performance_evaluator_dir}')
    # os.remove(f'{outputs_dir}{performance_evaluator_dir}{test_name}.zip')
    # print(f'{outputs_dir}{results_dir}{test_name}.zip', f'{data_dir}{results_dir}')
    # shutil.make_archive(f'{outputs_dir}{results_dir}{test_name}', 'zip',
    #                     data_dir + f'{results_dir}')
    # shutil.unpack_archive(
    #     f'{outputs_dir}{results_dir}{test_name}.zip',
    #     f'{outputs_dir}{results_dir}')
    # os.remove(f'{outputs_dir}{results_dir}{test_name}.zip')
    # print(f'{outputs_dir}{charts_dir}{test_name}.zip', f'{data_dir}{charts_dir}')
    # shutil.make_archive(f'{outputs_dir}{charts_dir}{test_name}', 'zip',
    #                     data_dir + f'{charts_dir}')
    # shutil.unpack_archive(
    #     f'{outputs_dir}{charts_dir}{test_name}.zip',
    #     f'{outputs_dir}{charts_dir}')
    # os.remove(f'{outputs_dir}{charts_dir}{test_name}.zip')
    print('Eval finished')


def evaluation_pipeline(config, test_name, dataset_name, images_dict, annotated_data_dict, description, model_data):
    models_dir = config['OUTPUTS_DIR'] + config['MODELS_DIR']
    training_date_2018 = config['PRE_20_TRAIN_DATE']
    training_date_2020 = config['POST_20_TRAIN_DATE']
    if description == 'best_epoch':
        model_paths = (models_dir +\
               model_data.loc[(model_data['test_name'] == test_name.replace('_run_', '_2018_run_')) &
               (model_data['training_date'] == training_date_2018)]['run_name'].values[-1] + '/best_model.pth',
               models_dir +\
               model_data.loc[(model_data['test_name'] == test_name.replace('_run_', '_2020_run_')) &
               (model_data['training_date'] == training_date_2020)]['run_name'].values[-1] + '/best_model.pth')
    elif description == 'final_epoch':
        model_paths = (models_dir +\
               model_data.loc[(model_data['test_name'] == test_name.replace('_run_', '_2018_run_')) &
               (model_data['training_date'] == training_date_2018)]['run_name'].values[-1] + '/final_epoch.pth',
               models_dir +\
               model_data.loc[(model_data['test_name'] == test_name.replace('_run_', '_2020_run_')) &
               (model_data['training_date'] == training_date_2020)]['run_name'].values[-1] + '/final_epoch.pth')
    elif description[:6] == 'epoch_':
        epoch_num = int(description[6:])
        model_paths = (models_dir +\
               model_data.loc[(model_data['test_name'] == test_name.replace('_run_', '_2018_run_')) &
               (model_data['training_date'] == training_date_2018)]['run_name'].values[-1] + f'/epoch_{epoch_num}_model.pth',
               models_dir +\
               model_data.loc[(model_data['test_name'] == test_name.replace('_run_', '_2020_run_')) &
               (model_data['training_date'] == training_date_2020)]['run_name'].values[-1] + f'/epoch_{epoch_num}_model.pth')
    prediction_data = full_cycle(config, test_name, dataset_name, images_dict, model_paths, description)
    plot_results(config, test_name, dataset_name, description)
    update_water_estimates(config, test_name, dataset_name, description)
    if dataset_name in ['deepaqua_test_dataset_no_nov', 'deepaqua_test_dataset']:
        iterate(config, test_name, dataset_name, prediction_data, annotated_data_dict, description, split_by_date=True)
