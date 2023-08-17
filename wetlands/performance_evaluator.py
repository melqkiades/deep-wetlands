
import os
import shutil
import time

import numpy
import seaborn
from PIL import Image
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from evaluation import semantic_segmentation_evaluator
from wetlands import viz_utils, jaccard_similarity


def rename_annotated_files():
    # Rename files
    annotations_dir = os.getenv('ANNOTATED_DATA_DIR')
    for filename in os.listdir(annotations_dir):
        if filename.endswith('.tif'):
            new_filename = filename.lower().replace('mask_sar_vh', 'annotated_vh')
            new_filename = new_filename.replace('iak_hornborgasjon', 'hornborgasjon_annotated_vh')
            os.rename(os.path.join(annotations_dir, filename), os.path.join(annotations_dir, new_filename))


def convert_annotated_data_to_png():
    annotations_dir = os.getenv('ANNOTATED_DATA_DIR')
    band = 'vis-gray'
    viz_utils.transform_ndwi_tiff_to_grayscale_png(annotations_dir, band)


def rename_prediction_images(model_name):
    results_dir = os.getenv('RESULTS_DIR')
    study_area = os.getenv('STUDY_AREA')
    performance_dir = os.getenv('EVALUATION_DIR')
    if not os.path.isdir(performance_dir):
        os.mkdir(performance_dir)
    # Create subfolder to calculate the performance of the current model
    model_performance_dir = f'{performance_dir}/{model_name}_{study_area}_performance/'
    if not os.path.isdir(model_performance_dir):
        os.mkdir(model_performance_dir)

    # performance_dir = '/tmp/descending_otsu_flacksjon_exported_images/'
    predictions_dir = f'{results_dir}/{model_name}_{study_area}_exported_images/'
    [shutil.copyfile(predictions_dir + f, model_performance_dir + f[:8] + f'_{study_area}_pred_bw.png') for f in os.listdir(predictions_dir) if not f.startswith('[0-9]+') and f.endswith('_pred_bw.png')]


def copy_annotated_images(model_name):
    annotations_dir = os.getenv('ANNOTATED_DATA_DIR')
    performance_dir = os.getenv('EVALUATION_DIR')
    study_area = os.getenv('STUDY_AREA')
    model_performance_dir = f'{performance_dir}/{model_name}_{study_area}_performance/'
    annotated_files = [filename for filename in os.listdir(annotations_dir) if filename.lower().startswith(study_area +'_annotated_vh_') and filename.endswith('.png')]
    print('Annotated files:')
    print(annotated_files)
    [shutil.copyfile(annotations_dir + f, model_performance_dir + f.lower()) for f in annotated_files]


def iterate(model_name):

    # 1. Iterate all the annotated images and extract the date
    ious = []
    accuracies = []
    predictions = []
    annotations = []

    performance_dir = os.getenv('EVALUATION_DIR')
    study_area = os.getenv('STUDY_AREA')
    model_performance_dir = f'{performance_dir}/{model_name}_{study_area}_performance/'
    annotated_files = [filename for filename in os.listdir(model_performance_dir) if filename.startswith(study_area + '_annotated_vh_')]
    print('Annotated files:')
    print(annotated_files)
    annotated_file_prefix = study_area + '_annotated_vh_'
    prefix_length = len(annotated_file_prefix)
    for annotated_file in annotated_files:
        # Extract the date
        date_str = annotated_file[prefix_length:prefix_length+10].replace('-', '')

        # Open the annotated file
        annotated_image = Image.open(model_performance_dir + annotated_file).convert('L')
        patch_size = int(os.getenv('PATCH_SIZE'))
        new_width = (annotated_image.width // patch_size) * patch_size
        new_height = (annotated_image.height // patch_size) * patch_size
        annotated_image = annotated_image.crop((0, 0, new_width, new_height))
        annotated_data = numpy.array(annotated_image)
        array_min, array_max = numpy.nanmin(annotated_data), numpy.nanmax(annotated_data)
        annotated_data = ((annotated_data - array_min) / (array_max - array_min)).astype(int)
        annotations.append(annotated_data)

        # Locate the prediction file
        prediction_file = model_performance_dir + date_str + f'_{study_area}_pred_bw.png'
        # Open the prediction file
        prediction_image = Image.open(prediction_file).convert('L')

        prediction_image = prediction_image.crop((0, 0, new_width, new_height))
        prediction_data = numpy.array(prediction_image)
        # pred_min, pred_max = numpy.nanmin(prediction_data), numpy.nanmax(prediction_data)
        array_min, array_max = numpy.nanmin(prediction_data), numpy.nanmax(prediction_data)
        prediction_data = ((prediction_data - array_min) / (array_max - array_min)).astype(int)
        predictions.append(prediction_data)

        iou = jaccard_similarity.calculate_intersection_over_union(prediction_data, annotated_data)
        ious.append(iou)
        accuracy = (annotated_data == prediction_data).sum() / (annotated_data.shape[0] * annotated_data.shape[1])
        accuracies.append(accuracy)

    result = semantic_segmentation_evaluator.eval_semantic_segmentation(predictions, annotations)

    print(result)

    print(f'Mean {model_name} IOU', numpy.asarray(ious).mean())
    print(f'Mean {model_name} Accuracy', numpy.asarray(accuracies).mean())

    confusion_matrix = semantic_segmentation_evaluator.calc_semantic_segmentation_confusion(predictions, annotations)
    print(confusion_matrix)
    TP = confusion_matrix[1, 1]
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]

    print(TP, TN, FP, FN)

    # ConfusionMatrixDisplay.from_predictions(annotations, predictions, display_labels=['Water', 'Land']).plot()
    # Flatten the arrays
    annotations = numpy.asarray(annotations).flatten()
    predictions = numpy.asarray(predictions).flatten()
    ConfusionMatrixDisplay.from_predictions(annotations, predictions, display_labels=['Soil', 'Water'], cmap=plt.cm.Blues)
    ConfusionMatrixDisplay.from_predictions(annotations, predictions, normalize='true', display_labels=['Soil', 'Water'], cmap=plt.cm.Blues, values_format='.2%')
    ConfusionMatrixDisplay.from_predictions(annotations, predictions, normalize='pred', display_labels=['Soil', 'Water'], cmap=plt.cm.Blues, values_format='.2%')
    ConfusionMatrixDisplay.from_predictions(annotations, predictions, normalize='all', display_labels=['Soil', 'Water'], cmap=plt.cm.Blues, values_format='.2%')
    plt.show()

    cmat = [[TP, FN], [FP, TN]]

    plt.figure(figsize=(6, 6))
    ax = seaborn.heatmap(cmat / numpy.sum(cmat), cmap="Reds", annot=True, fmt='.2%', square=1, linewidth=2.)
    ax.set_title(model_name)
    ax.xaxis.set_ticklabels(['Water', 'Soil'])
    ax.yaxis.set_ticklabels(['Water', 'Soil'])
    plt.xlabel("Predictions")
    plt.ylabel("Real values")
    plt.show()


def full_cycle():
    load_dotenv()

    model_name = os.getenv('MODEL_NAME')

    convert_annotated_data_to_png()
    rename_prediction_images(model_name)
    copy_annotated_images(model_name)
    iterate(model_name)


def main():
    full_cycle()


# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
