import os
import shutil
import cv2
import numpy as np
import pandas
import seaborn
import tqdm
from dotenv import load_dotenv
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
import csv


def convert_area_name_to_color(area_name):
    area_name_to_color = {'hjalstaviken':'red', 'hornborgasjon':'blue', 'svartadalen':'green'}
    if area_name in area_name_to_color:
        return area_name_to_color[area_name]
    else:
        return 'red'


def convert_annotated_data_to_png(dataset_name):
    annotations_dir = os.getenv('ANNOTATED_DATA_DIR') + '/' + dataset_name
    band = 'vis-gray'
    viz_utils.transform_ndwi_tiff_to_grayscale_png(annotations_dir, band)


# def rename_prediction_data(test_name, epoch_num, dataset_name):
#     results_dir = os.getenv('RESULTS_DIR') + f'/{dataset_name}/{test_name}'
#     performance_dir = os.getenv('EVALUATION_DIR') + f'/{dataset_name}/{test_name}'
#     if not os.path.isdir(performance_dir):
#         Path(performance_dir).mkdir(parents=True, exist_ok=True)
#     # Create subfolder to calculate the performance of the current model
#     model_performance_dir = f'{performance_dir}/epoch_{epoch_num}_performance/'
#     if not os.path.isdir(model_performance_dir):
#         Path(model_performance_dir).mkdir(parents=True, exist_ok=True)
#
#     # performance_dir = '/tmp/descending_otsu_flacksjon_exported_images/'
#     predictions_dir = f'{results_dir}/epoch_{epoch_num}_exported_images/'
#     [shutil.copyfile(predictions_dir + f, model_performance_dir + f[:-11] + f'pred_bw.png') for f in os.listdir(predictions_dir) if not f.startswith('[0-9]+') and f.endswith('_pred_bw.png')]


def copy_annotated_images(test_name, epoch_num, dataset_name):
    annotations_dir = os.getenv('ANNOTATED_DATA_DIR') + '/' + dataset_name
    performance_dir = os.getenv('EVALUATION_DIR') + f'/{dataset_name}/{test_name}'
    model_performance_dir = f'{performance_dir}/epoch_{epoch_num}_performance/'
    annotated_files = [filename for filename in os.listdir(annotations_dir) if filename.endswith('.png')]
    print('Annotated files:')
    print(annotated_files)
    [shutil.copyfile(annotations_dir +'/'+ f, model_performance_dir + f.lower()) for f in annotated_files]


def iterate(test_name, dataset_name, prediction_data_dict, annotated_data_dict, description, split_by_date=False):

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

    performance_dir = os.getenv('EVALUATION_DIR') + f'/{dataset_name}/{test_name}'
    if not os.path.isdir(performance_dir):
        Path(performance_dir).mkdir(parents=True, exist_ok=True)
    model_performance_dir = f'{performance_dir}/{description}_performance/'
    if not os.path.isdir(model_performance_dir):
        Path(model_performance_dir).mkdir(parents=True, exist_ok=True)
    results_dir = os.getenv('RESULTS_DIR') + f'/{dataset_name}/{test_name}'
    model_results_dir = f'{results_dir}/{description}_exported_images'
    aucs_dataframe = pandas.read_csv(f'{results_dir}/{description}_areas_under_curve.csv')
    annotations_dir = os.getenv('ANNOTATED_DATA_DIR') + '/' + dataset_name + '/'
    annotated_files = [filename for filename in os.listdir(annotations_dir) if 'annotated_vh' in filename and filename.endswith('.png')]
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
        # annotated_image = Image.open(annotations_dir + annotated_file).convert('L')
        # patch_size = int(os.getenv('PATCH_SIZE'))
        # new_width = (annotated_image.width // patch_size) * patch_size
        # new_height = (annotated_image.height // patch_size) * patch_size
        # annotated_image = annotated_image.crop((0, 0, new_width, new_height))
        # annotated_data = np.array(annotated_image)
        # array_min, array_max = np.nanmin(annotated_data), np.nanmax(annotated_data)
        # annotated_data = ((annotated_data - array_min) / (array_max - array_min)).astype(int)
        annotations[area_name].append(annotated_data_dict[annotated_file])

        # Locate the prediction file
        # prediction_file = model_results_dir + '/' + annotated_file.replace('annotated_vh', 'mosaic').replace('.png', f'_sar_VH_pred_bw.png')
        prediction_file = model_results_dir + '/' + annotated_file.lower().replace('annotated_vh', 'mosaic').replace('.png',
                                                                                                             f'_sar_VH.tif')

        # Check if the prediction file exists
        # if not os.path.isfile(prediction_file):
        #     print(f'Prediction file {prediction_file} does not exist')
        #     continue

        # Open the prediction file
        # prediction_image = Image.open(prediction_file).convert('L')
        prediction_data = prediction_data_dict[os.path.basename(prediction_file)].astype(int)

        # # prediction_image = prediction_image.crop((0, 0, new_width, new_height))
        # prediction_data = np.array(prediction_image)
        # # pred_min, pred_max = numpy.nanmin(prediction_data), numpy.nanmax(prediction_data)
        # array_min, array_max = np.nanmin(prediction_data), np.nanmax(prediction_data)
        # prediction_data = ((prediction_data - array_min) / (array_max - array_min)).astype(int)
        predictions[area_name].append(prediction_data)

        iou = jaccard_similarity.calculate_intersection_over_union(prediction_data, annotated_data_dict[annotated_file])
        ious[area_name].append(iou)
        ious_list.append(iou)
        accuracy = (annotated_data_dict[annotated_file] == prediction_data).sum() / (annotated_data_dict[annotated_file].shape[0] * annotated_data_dict[annotated_file].shape[1])
        accuracies[area_name].append(accuracy)
    image_results_df = pandas.DataFrame({'filename': annotated_files, 'iou': ious_list})
    image_results_df.to_csv(f'{model_performance_dir}/filename_ious.csv')
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
            metrics_file = f'{performance_dir}/{description}_performance.csv'
        else:
            metrics_file = f'{performance_dir}/{description}_performance_split.csv'
        with open(metrics_file, 'a') as f:
            f.write("%s,%s\n" % ('Area', area_name))
            for key in metrics.keys():
                f.write("%s,%s\n" % (key, metrics[key]))

        # ConfusionMatrixDisplay.from_predictions(annotations, predictions, display_labels=['Water', 'Land']).plot()
        # Flatten the arrays
        area_annotations = np.asarray(annotations[area_name]).flatten()
        area_predictions = np.asarray(predictions[area_name]).flatten()
        ConfusionMatrixDisplay.from_predictions(area_annotations, area_predictions, display_labels=['Soil', 'Water'], cmap=plt.cm.Blues)
        ConfusionMatrixDisplay.from_predictions(area_annotations, area_predictions, normalize='true', display_labels=['Soil', 'Water'], cmap=plt.cm.Blues, values_format='.2%')
        ConfusionMatrixDisplay.from_predictions(area_annotations, area_predictions, normalize='pred', display_labels=['Soil', 'Water'], cmap=plt.cm.Blues, values_format='.2%')
        ConfusionMatrixDisplay.from_predictions(area_annotations, area_predictions, normalize='all', display_labels=['Soil', 'Water'], cmap=plt.cm.Blues, values_format='.2%')
        # plt.show()

        cmat = [[TP, FN], [FP, TN]]

        plt.figure(figsize=(6, 6))
        ax = seaborn.heatmap(cmat / np.sum(cmat), cmap="Reds", annot=True, fmt='.2%', square=1, linewidth=2.)
        ax.set_title(test_name)
        ax.xaxis.set_ticklabels(['Water', 'Soil'])
        ax.yaxis.set_ticklabels(['Water', 'Soil'])
        plt.xlabel("Predictions")
        plt.ylabel("Real values")


def visualize_predicted_image(image, model, device, file_name, test_name, dataset_name, description):
    results_dir = os.getenv('RESULTS_DIR') + f'/{dataset_name}/{test_name}'

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
        pred_mask = map_wetlands.predict_water_mask(image, model, device)

    unique, counts = np.unique(pred_mask, return_counts=True)
    results = dict(zip(unique, counts))
    image_date = file_name.split('_')[2]
    satellite = file_name.split('_')[-1]
    results['Date'] = image_date
    results['Satellite'] = satellite
    results['File_name'] = file_name

    images_dir = f'{results_dir}/{description}_exported_images/'

    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)

    # Plotting SAR
    # plt.imshow(image[:width, :height], cmap='gray')
    # plt.imsave(images_dir + file_name + '_sar.png', image)
    # plt.imsave(images_dir + file_name + '_sar_bw.png', image, cmap='gray')

    # Plotting prediction
    # plt.imshow(pred_mask)
    # plt.imsave(images_dir + file_name + '_pred.png', pred_mask)
    img = Image.fromarray(np.uint8((pred_mask) * 255), 'L')
    img.save(images_dir + file_name + '_pred_bw.png')

    return results, pred_mask


def get_prediction_image(tiff_image, tiff_file, model, device, test_name, dataset_name, description):

    file_name = os.path.basename(tiff_file).split('.')[0]
    results, prediction_image = visualize_predicted_image(tiff_image, model, device, file_name, test_name, dataset_name, description)
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


def plot_results(test_name, dataset_name, description):

    charts_dir = os.getenv('CHARTS_DIR') + f'/{dataset_name}/{test_name}'
    results_dir = os.getenv('RESULTS_DIR') + f'/{dataset_name}/{test_name}'
    results_file = f'{results_dir}/{description}_water_estimates.csv'
    data_frame = pandas.read_csv(results_file, usecols=['1.0', 'Date'], index_col=["Date"],  parse_dates=["Date"])
    data_frame.plot(title=test_name)
    plt.savefig(f'{charts_dir}/{description}_water_estimates.png')


def update_water_estimates(test_name, dataset_name, description):
    charts_dir = os.getenv('CHARTS_DIR') + f'/{dataset_name}/{test_name}'
    results_dir = os.getenv('RESULTS_DIR') + f'/{dataset_name}/{test_name}'

    results_file = f'{results_dir}/{description}_water_estimates.csv'
    data_frame = pandas.read_csv(results_file, usecols=['1.0', 'Date', 'File_name'],  parse_dates=["Date"])
    print(data_frame.size)
    print(data_frame.columns.values)
    data_frame['area_name'] = data_frame.apply(lambda x:x['File_name'].split('_')[0], axis=1)
    data_frame['color'] = data_frame.apply(lambda x: convert_area_name_to_color(x['area_name']), axis=1)
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
            prev_prediction = io.imread(f'{results_dir}/{description}_exported_images/'+ file_name)
            days[area_name] = [0.]
            prev_date = date
            ious_skip_winter[area_name] = [0.]
            dates_skip_winter[area_name] = [date]
        else:
            days_passed = (date - prev_date).days
            prediction = io.imread(f'{results_dir}/{description}_exported_images/'+ file_name)
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
    # data_frame = data_frame[data_frame['Date'].dt.year.isin([2018, 2019, 2020, 2021, 2022])]
    print(data_frame.size)
    print(data_frame.columns.values)

    data_frame.plot(x='Date', y='1.0', kind='scatter', title=f'{test_name} {description} [{dataset_name}]', c='color', s=0.7**2)
    plt.savefig(f'{charts_dir}/{description}_scatter_new_water_estimates_filtered.png')
    plt.clf()
    areas_under_curve = {}
    for area_name in ious.keys():
        days_skip_winter = [0.]
        for i in range(1, len(dates_skip_winter[area_name])):
            days_skip_winter.append((dates_skip_winter[area_name][i] - dates_skip_winter[area_name][0]).days)
        # area_data_frame = data_frame.loc[data_frame['area_name']==area_name]
        # ious_array = area_data_frame['ious'].values
        # days_array = area_data_frame['days'].values
        # for i in range(days_array[:].size - 1, 0, -1):
        #     if days_array[i] - days_array[i-1] >= 60:
        #         days_array = np.insert(days_array, i, days_array[i-1] + 1)
        #         ious_array = np.insert(ious_array, i, 0)
        # area_data_frame.plot(x='Date', y='ious', title=f'{test_name} ep. {epoch_num}_{area_name} [{dataset_name}] auc: {str(round(area_under_curve,2))}')
        area_under_curve = np.trapz(ious_skip_winter[area_name], x=days_skip_winter) / days_skip_winter[-1]
        areas_under_curve[area_name] = [area_under_curve]
        plt.plot(dates_skip_winter[area_name], ious_skip_winter[area_name], label='ious')
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.title(f'{description}_{area_name} auc: {str(round(area_under_curve,2))}')
        plt.savefig(f'{charts_dir}/{description}_ious_{area_name}_new_water_estimates_filtered.png')
        plt.clf()
    auc_dataframe = pandas.DataFrame.from_dict(areas_under_curve)
    auc_dataframe.to_csv(f'{results_dir}/{description}_areas_under_curve.csv')
    data_frame.to_csv(f'{results_dir}/{description}_new_water_estimates_filtered.csv')


def full_cycle(test_name, dataset_name, images_dict, model_paths, description):
    load_dotenv()

    device = utils.get_device()
    results_dir = os.getenv('RESULTS_DIR') + f'/{dataset_name}/{test_name}'
    if test_name not in ['otsu', 'otsu_gaussian', 'thresholding_2018', 'thresholding_2020']:
        cnn_type = os.getenv('CNN_TYPE')
        model_2018 = model_factory.load_model(cnn_type, model_paths[0], device)
        model_2020 = model_factory.load_model(cnn_type, model_paths[1], device)
    else:
        model_2018 = None
        model_2020 = None

    results_list = []
    prediction_data = {}

    for tiff_file in tqdm.tqdm(images_dict) :
        if not tiff_file.endswith('.tif'):
            continue
        year = int(tiff_file.split('_')[-3].split('-')[0])
        if year < 2020:
            results, prediction_image = get_prediction_image(images_dict[tiff_file], tiff_file, model_2018, device, test_name, dataset_name, description)
        else:
            results, prediction_image = get_prediction_image(images_dict[tiff_file], tiff_file, model_2020, device, test_name, dataset_name, description)

        results_list.append(results)
        prediction_data[tiff_file] = prediction_image

    data_frame = pandas.DataFrame(results_list)
    data_frame['Date'] = data_frame['Date'].apply(pandas.to_datetime).dt.date
    print(data_frame.head())
    data_frame.to_csv(f'{results_dir}/{description}_water_estimates.csv')

    return prediction_data


def main(test_name, dataset_name='deepaqua_test_dataset_no_nov', best_epoch=True, all_epochs=False, final_epoch=False):
    load_dotenv()

    results_dir = os.getenv('RESULTS_DIR') + f'/{dataset_name}/{test_name}'
    if not os.path.isdir(results_dir):
        Path(results_dir).mkdir(parents=True, exist_ok=True)
    charts_dir = os.getenv('CHARTS_DIR') + f'/{dataset_name}/{test_name}'
    if not os.path.isdir(charts_dir):
        Path(charts_dir).mkdir(parents=True, exist_ok=True)
    if dataset_name in ['deepaqua_test_dataset_no_nov', 'deepaqua_test_dataset']:
        convert_annotated_data_to_png(dataset_name)
    patch_size = int(os.getenv('PATCH_SIZE'))

    tiff_dir = 'C:/Users/ioia4268/data/sar/' + dataset_name

    if not os.path.exists(tiff_dir):
        raise FileNotFoundError(f'The folder containing the TIFF files does not exist: {tiff_dir}')

    filenames = next(os.walk(tiff_dir), (None, None, []))[2]  # [] if no file

    images_dict = {}
    incomplete_images = 0

    # with rio.open('C:\\Users\\ioia4268\\data\\sar\\Örebro län\\Orebro lan_mosaic_2018-07-04_sar_VH.tif') as src:
    #     dataset_array = src.read()
    #     minValue_2018 = np.nanpercentile(dataset_array, 1)
    #     maxValue_2018 = np.nanpercentile(dataset_array, 99)
    # with rio.open('C:\\Users\\ioia4268\\data\\sar\\Örebro län\\Orebro lan_mosaic_2020-06-23_sar_VH.tif') as src:
    #     dataset_array = src.read()
    #     minValue_2020 = np.nanpercentile(dataset_array, 1)
    #     maxValue_2020 = np.nanpercentile(dataset_array, 99)

    for tiff_file in tqdm.tqdm(sorted(filenames)):
        if not tiff_file.endswith('.tif'):
            continue
        # if '2014' in tiff_file or'2015' in tiff_file or '2016' in tiff_file or '2017' in tiff_file or '2018' in tiff_file or '2019' in tiff_file:
            # minValue = minValue_2018
            # maxValue = maxValue_2018
        # else:
            # minValue = minValue_2020
            # maxValue = maxValue_2020
        image = viz_utils.load_image(tiff_dir + '/' + tiff_file, skip_nan=False, nan_to_zero=False)#, min_value=minValue, max_value=maxValue)
        if image is None:
            incomplete_images += 1
        else:
            images_dict[tiff_file] = image

    # image = viz_utils.load_image("C:/Users/ioia4268/Downloads/tav_2014-10-05_sar_VH.tif", skip_nan=False, nan_to_zero=False)
    print(f'There were a total of {incomplete_images} incomplete images')
    annotated_data_dict = {}
    if dataset_name in ['deepaqua_test_dataset_no_nov', 'deepaqua_test_dataset']:
        annotations_dir = os.getenv('ANNOTATED_DATA_DIR') + '/' + dataset_name + '/'
        annotated_files = [filename for filename in os.listdir(annotations_dir) if 'annotated_vh' in filename and filename.endswith('.png')]
        for annotated_file in annotated_files:
            # Open the annotated file
            # annotated_image = Image.open(annotations_dir + annotated_file).convert('L')
            annotated_image = viz_utils.load_image(annotations_dir + annotated_file, skip_nan=False, nan_to_zero=False)

            # new_width = (annotated_image.width // patch_size) * patch_size
            # new_height = (annotated_image.height // patch_size) * patch_size
            # annotated_image = annotated_image.crop((0, 0, new_width, new_height))
            annotated_data = np.array(annotated_image)
            array_min, array_max = np.nanmin(annotated_data), np.nanmax(annotated_data)
            annotated_data_dict[annotated_file] = ((annotated_data - array_min) / (array_max - array_min)).astype(int)
    model_dir = os.getenv('MODELS_DIR')
    model_data = pandas.read_csv(model_dir + '/model_info.csv')
    if best_epoch:
        description = 'best_epoch'
        evaluation_pipeline(test_name, dataset_name, images_dict, annotated_data_dict, description, model_data)
    if final_epoch:
        description = 'final_epoch'
        evaluation_pipeline(test_name, dataset_name, images_dict, annotated_data_dict, description, model_data)
    if all_epochs:
        final_epoch_num = np.minimum(model_data.loc[(model_data['test_name'] == test_name) &
               (model_data['training_date'] == '2018-07-04')]['final_epoch'].values[-1],
                                     model_data.loc[(model_data['test_name'] == test_name) &
               (model_data['training_date'] == '2020-06-23')]['final_epoch'].values[-1])
        for epoch_num in range(1, final_epoch_num + 1):
            description = f'epoch_{epoch_num}'
            evaluation_pipeline(test_name, dataset_name, images_dict, annotated_data_dict, description, model_data)


def evaluation_pipeline(test_name, dataset_name, images_dict, annotated_data_dict, description, model_data):
    if description == 'best_epoch':
        if not 'deepaqua_big' in test_name:
            model_paths = (os.getenv('MODELS_DIR') + '/'\
                   + model_data.loc[(model_data['test_name'] == test_name.replace('_run_', '_2018_run_'))]['run_name'].values[-1] + '/best_model.pth',
                   os.getenv('MODELS_DIR') + '/' \
                   + model_data.loc[(model_data['test_name'] == test_name.replace('_run_', '_2020_run_'))]['run_name'].values[-1] + '/best_model.pth')
        else:
            model_paths = ("C:/Users/ioia4268/data/models/big-2018.pth", "C:/Users/ioia4268/data/models/big-2020.pth")
    elif description == 'final_epoch':
        model_paths = (os.getenv('MODELS_DIR') + '/'\
               + model_data.loc[(model_data['test_name'] == test_name.replace('_run_', '_2018_run_')) &
               (model_data['training_date'] == '2018-07-04')]['run_name'].values[-1] + '/final_epoch.pth',
               os.getenv('MODELS_DIR') + '/' \
               + model_data.loc[(model_data['test_name'] == test_name.replace('_run_', '_2020_run_')) &
               (model_data['training_date'] == '2020-06-23')]['run_name'].values[-1] + '/final_epoch.pth')
    elif description[:6] == 'epoch_':
        epoch_num = int(description[6:])
        model_paths = (os.getenv('MODELS_DIR') + '/'\
               + model_data.loc[(model_data['test_name'] == test_name.replace('_run_', '_2018_run_')) &
               (model_data['training_date'] == '2018-07-04')]['run_name'].values[-1] + f'/epoch_{epoch_num}_model.pth',
               os.getenv('MODELS_DIR') + '/' \
               + model_data.loc[(model_data['test_name'] == test_name.replace('_run_', '_2020_run_')) &
               (model_data['training_date'] == '2020-06-23')]['run_name'].values[-1] + f'/epoch_{epoch_num}_model.pth')
    prediction_data = full_cycle(test_name, dataset_name, images_dict, model_paths, description)
    plot_results(test_name, dataset_name, description)
    update_water_estimates(test_name, dataset_name, description)
    if dataset_name in ['deepaqua_test_dataset_no_nov', 'deepaqua_test_dataset']:
        iterate(test_name, dataset_name, prediction_data, annotated_data_dict, description, split_by_date=True)


# start = time.time()
# for i in range(5):
# main(f'standard_baseline_lr5^-5_redlrplateau_corrected_final3_run_0', dataset_name='deepaqua_test_dataset_no_nov', best_epoch=True, final_epoch=False)
# end = time.time()
# total_time = end - start
# print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
