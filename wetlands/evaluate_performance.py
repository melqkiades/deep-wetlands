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


def convert_area_name_to_color(area_name):
    area_name_to_color = {'hjalstaviken':'red', 'hornborgasjon':'blue', 'svartadalen':'green'}
    return area_name_to_color[area_name]


def convert_annotated_data_to_png():
    annotations_dir = os.getenv('ANNOTATED_DATA_DIR') + '/hhs_seven_months'
    band = 'vis-gray'
    viz_utils.transform_ndwi_tiff_to_grayscale_png(annotations_dir, band)


def rename_prediction_images(test_name):
    results_dir = os.getenv('RESULTS_DIR')
    performance_dir = os.getenv('EVALUATION_DIR')
    if not os.path.isdir(performance_dir):
        os.mkdir(performance_dir)
    # Create subfolder to calculate the performance of the current model
    model_performance_dir = f'{performance_dir}/{test_name}_hhs_seven_months_performance/'
    if not os.path.isdir(model_performance_dir):
        os.mkdir(model_performance_dir)

    # performance_dir = '/tmp/descending_otsu_flacksjon_exported_images/'
    predictions_dir = f'{results_dir}/{test_name}_hhs_seven_months_exported_images/'
    [shutil.copyfile(predictions_dir + f, model_performance_dir + f[:-11] + f'_hhs_seven_months_pred_bw.png') for f in os.listdir(predictions_dir) if not f.startswith('[0-9]+') and f.endswith('_pred_bw.png')]


def copy_annotated_images(test_name):
    annotations_dir = os.getenv('ANNOTATED_DATA_DIR') + '/hhs_seven_months'
    performance_dir = os.getenv('EVALUATION_DIR')
    model_performance_dir = f'{performance_dir}/{test_name}_hhs_seven_months_performance/'
    annotated_files = [filename for filename in os.listdir(annotations_dir) if filename.endswith('.png')]
    print('Annotated files:')
    print(annotated_files)
    [shutil.copyfile(annotations_dir +'/'+ f, model_performance_dir + f.lower()) for f in annotated_files]


def iterate(test_name):

    # 1. Iterate all the annotated images and extract the date
    ious = {}
    accuracies = {}
    predictions = {}
    annotations = {}
    correct_values = {'hjalstaviken':{'Pixel accuracy': 0.96, 'IOU':0.68, 'Precision':0.81, 'Recall':0.81, 'F1':0.81},
                      'hornborgasjon':{'Pixel accuracy': 0.98, 'IOU':0.94, 'Precision':0.98, 'Recall':0.96, 'F1':0.97},
                      'svartadalen':{'Pixel accuracy': 0.97, 'IOU':0.88, 'Precision':0.98, 'Recall':0.9, 'F1':0.93}}

    performance_dir = os.getenv('EVALUATION_DIR')
    model_performance_dir = f'{performance_dir}/{test_name}_hhs_seven_months_performance/'
    annotated_files = [filename for filename in os.listdir(model_performance_dir) if 'annotated_vh' in filename]
    print('Annotated files:')
    print(annotated_files)
    for annotated_file in annotated_files:
        area_name = annotated_file.split('_')[0]
        if area_name not in ious.keys():
            ious[area_name] = []
            accuracies[area_name] = []
            predictions[area_name] = []
            annotations[area_name] = []
        # Open the annotated file
        annotated_image = Image.open(model_performance_dir + annotated_file).convert('L')
        patch_size = int(os.getenv('PATCH_SIZE'))
        new_width = (annotated_image.width // patch_size) * patch_size
        new_height = (annotated_image.height // patch_size) * patch_size
        annotated_image = annotated_image.crop((0, 0, new_width, new_height))
        annotated_data = np.array(annotated_image)
        array_min, array_max = np.nanmin(annotated_data), np.nanmax(annotated_data)
        annotated_data = ((annotated_data - array_min) / (array_max - array_min)).astype(int)
        annotations[area_name].append(annotated_data)

        # Locate the prediction file
        prediction_file = model_performance_dir + annotated_file.replace('annotated_vh', 'mosaic').replace('.png', '_sar_VH__hhs_seven_months_pred_bw.png')

        # Check if the prediction file exists
        if not os.path.isfile(prediction_file):
            print(f'Prediction file {prediction_file} does not exist')
            continue

        # Open the prediction file
        prediction_image = Image.open(prediction_file).convert('L')

        prediction_image = prediction_image.crop((0, 0, new_width, new_height))
        prediction_data = np.array(prediction_image)
        # pred_min, pred_max = numpy.nanmin(prediction_data), numpy.nanmax(prediction_data)
        array_min, array_max = np.nanmin(prediction_data), np.nanmax(prediction_data)
        prediction_data = ((prediction_data - array_min) / (array_max - array_min)).astype(int)
        predictions[area_name].append(prediction_data)

        iou = jaccard_similarity.calculate_intersection_over_union(prediction_data, annotated_data)
        ious[area_name].append(iou)
        accuracy = (annotated_data == prediction_data).sum() / (annotated_data.shape[0] * annotated_data.shape[1])
        accuracies[area_name].append(accuracy)
    for area_name in ious.keys():
        print('\n\nAREA: ', area_name)
        result = semantic_segmentation_evaluator.eval_semantic_segmentation(predictions[area_name], annotations[area_name])

        print(result)

        print(f'Mean {test_name} IOU', np.asarray(ious[area_name]).mean())
        print(f'Mean {test_name} Accuracy', np.asarray(accuracies[area_name]).mean())

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
        print('Pixel accuracy:', accuracy, accuracy-correct_values[area_name]['Pixel accuracy'])
        print('IOU:', iou, iou-correct_values[area_name]['IOU'])
        print('Precision:', precision, precision-correct_values[area_name]['Precision'])
        print('Recall:', recall, recall-correct_values[area_name]['Recall'])
        print('F1 Score:', f1_score, f1_score-correct_values[area_name]['F1'])

        metrics = {
            'accuracy': accuracy,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': TP,
            'true_negatives': TN,
            'false_positives': FP,
            'false_negatives': FN
        }

        # Export metrics to CSV
        metrics_file = f'{performance_dir}/{test_name}_hhs_seven_months_performance.csv'
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
        # plt.show()


def visualize_predicted_image(image, model, device, file_name, test_name):
    patch_size = int(os.getenv('PATCH_SIZE'))
    results_dir = os.getenv('RESULTS_DIR')

    width = image.shape[0] - image.shape[0] % patch_size
    height = image.shape[1] - image.shape[1] % patch_size
    if test_name == 'otsu':
        pred_mask = otsu_threshold(image)
    elif test_name == 'otsu_gaussian':
        kernel_size = os.getenv('OTSU_GAUSSIAN_KERNEL_SIZE')
        # test_name += '_' + kernel_size
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

    images_dir = f'{results_dir}/{test_name}_hhs_seven_months_exported_images/'

    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)

    # Plotting SAR
    plt.imshow(image[:width, :height], cmap='gray')
    plt.imsave(images_dir + file_name + '_sar.png', image)
    plt.imsave(images_dir + file_name + '_sar_bw.png', image, cmap='gray')

    # Plotting prediction
    plt.imshow(pred_mask)
    plt.imsave(images_dir + file_name + '_pred.png', pred_mask)
    img = Image.fromarray(np.uint8((pred_mask) * 255), 'L')
    img.save(images_dir + file_name + '_pred_bw.png')

    return results


def get_prediction_image(tiff_file, band, model, device, test_name):
    image = viz_utils.load_image(tiff_file, band, ignore_nan=True)

    if image is None:
        return None

    file_name = os.path.basename(tiff_file).split('.')[0]
    results = visualize_predicted_image(image, model, device, file_name, test_name)
    return results


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


def plot_results(test_name):

    charts_dir = os.getenv('CHARTS_DIR') + '/' + test_name
    results_dir = os.getenv('RESULTS_DIR')
    # results_file = '/tmp/water_estimates_flacksjon_2018-07.csv'
    results_file = f'{results_dir}/{test_name}_hhs_seven_months_water_estimates.csv'
    data_frame = pandas.read_csv(results_file, usecols=['1.0', 'Date'], index_col=["Date"],  parse_dates=["Date"])

    data_frame.plot(title=test_name)

    plt.savefig(f'{charts_dir}/hhs_seven_months_water_estimates.png')
    # plt.show()


def update_water_estimates(test_name):
    charts_dir = os.getenv('CHARTS_DIR') + '/' + test_name
    results_dir = os.getenv('RESULTS_DIR')

    results_file = f'{results_dir}/{test_name}_hhs_seven_months_water_estimates.csv'
    data_frame = pandas.read_csv(results_file, usecols=['1.0', 'Date', 'File_name'],  parse_dates=["Date"])
    print(data_frame.size)
    print(data_frame.columns.values)
    data_frame['area_name'] = data_frame.apply(lambda x:x['File_name'].split('_')[0], axis=1)
    data_frame['color'] = data_frame.apply(lambda x: convert_area_name_to_color(x['area_name']), axis=1)
    ious = {}
    days = {}
    gt_ious = []
    for index, row in data_frame.iterrows():
        area_name = row['area_name']
        file_name = row['File_name'] + '_pred_bw.png'
        date = row['Date']
        if area_name not in ious:
            ious[area_name] = [0.]
            prev_prediction = rio.open(f'{results_dir}/{test_name}_hhs_seven_months_exported_images/'+ file_name).read()
            days[area_name] = [0.]
            prev_date = date
        else:
            prediction = rio.open(f'{results_dir}/{test_name}_hhs_seven_months_exported_images/'+ file_name).read()
            ious[area_name].append(calculate_intersection_over_union(prev_prediction, prediction))
            prev_prediction = prediction[:]
            days[area_name].append(days[area_name][-1] + (date-prev_date).days)
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

    data_frame.plot(x='Date', y='1.0', kind='scatter', title=f'{test_name} [hhs_seven_months]', c='color')
    plt.savefig(f'{charts_dir}/scatter_hhs_seven_months_new_water_estimates_filtered.png')
    plt.clf()
    for area_name in ious.keys():
        area_data_frame = data_frame.loc[data_frame['area_name']==area_name]
        area_under_curve = np.trapz(area_data_frame['ious'], x=area_data_frame['days'])/area_data_frame['days'].values[-1]
        area_data_frame.plot(x='Date', y='ious', title=f'{test_name}_{area_name} [hhs_seven_months] auc: {str(area_under_curve)}')
        plt.savefig(f'{charts_dir}/ious_{area_name}_hhs_seven_months_new_water_estimates_filtered.png')
        plt.clf()
    # plt.show()

    data_frame.to_csv(f'{results_dir}/{test_name}_hhs_seven_months_new_water_estimates_filtered.csv')


def full_cycle(test_name):
    load_dotenv()

    sar_polarization = os.getenv('SAR_POLARIZATION')
    device = utils.get_device()
    tiff_dir = 'C:/Users/ioia4268/data/sar/hhs_seven_months'
    results_dir = os.getenv('RESULTS_DIR')

    if not os.path.exists(tiff_dir):
        raise FileNotFoundError(f'The folder containing the TIFF files does not exist: {tiff_dir}')

    filenames = next(os.walk(tiff_dir), (None, None, []))[2]  # [] if no file
    print(filenames)
    if test_name not in ['otsu', 'otsu_gaussian', 'thresholding_2018', 'thresholding_2020']:
        model_file_2018 = os.getenv('MODEL_FILE_EVALUATE_2018')
        model_file_2020 = os.getenv('MODEL_FILE_EVALUATE_2020')
        cnn_type = os.getenv('CNN_TYPE')
        model_2018 = model_factory.load_model(cnn_type, model_file_2018, device)
        model_2020 = model_factory.load_model(cnn_type, model_file_2020, device)
    else:
        model_2018 = None
        model_2020 = None

    results_list = []
    incomplete_images = 0

    for tiff_file in tqdm.tqdm(sorted(filenames)):
        if not tiff_file.endswith('.tif'):
            continue
        year = int(tiff_file.split('_')[2].split('-')[0])
        if year in [2018, 2019]:
            results = get_prediction_image(tiff_dir + '/' + tiff_file, sar_polarization, model_2018, device, test_name)
        elif year in [2020, 2021, 2022]:
            results = get_prediction_image(tiff_dir + '/' + tiff_file, sar_polarization, model_2020, device, test_name)

        if results is None:
            incomplete_images += 1
        else:
            results_list.append(results)

    print(f'There were a total of {incomplete_images} incomplete images')

    data_frame = pandas.DataFrame(results_list)
    data_frame['Date'] = data_frame['Date'].apply(pandas.to_datetime).dt.date
    print(data_frame.head())
    data_frame.to_csv(f'{results_dir}/{test_name}_hhs_seven_months_water_estimates.csv')


def main():
    load_dotenv()

    results_dir = os.getenv('RESULTS_DIR')
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    test_name = os.getenv('TEST_NAME')
    charts_dir = os.getenv('CHARTS_DIR') + '/' + test_name
    if not os.path.isdir(charts_dir):
        os.mkdir(charts_dir)

    full_cycle(test_name)
    plot_results(test_name)
    update_water_estimates(test_name)
    convert_annotated_data_to_png()
    rename_prediction_images(test_name)
    copy_annotated_images(test_name)
    iterate(test_name)




# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
