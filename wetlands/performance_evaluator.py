import os
import shutil
import time

import numpy
import seaborn
import six
from PIL import Image
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from wetlands import jaccard_similarity, viz_utils


# 1. Convert all the TIFF annotated data into grayscale images
# 2.

def convert_annotated_data_to_png():
    # annotated_file = '/tmp/annotated/Svartadalen_annotated_vh_2018-05-05.tif'
    annotations_dir = '/Users/frape/Projects/DeepWetlands/Datasets/wetlands/Annotated data/'
    band = 'vis-gray'
    viz_utils.transform_ndwi_tiff_to_grayscale_png(annotations_dir, band)


def rename_prediction_images(model_name):

    performance_dir = '/tmp/performance_evaluator/'
    if not os.path.isdir(performance_dir):
        os.mkdir(performance_dir)

    # performance_dir = '/tmp/descending_otsu_flacksjon_exported_images/'
    predictions_dir = f'/tmp/descending_{model_name}_flacksjon_exported_images/'
    [shutil.copyfile(predictions_dir + f, performance_dir + f[:8] + f'_{model_name}_bw.png') for f in os.listdir(predictions_dir) if not f.startswith('[0-9]+') and f.endswith('_pred_bw.png')]


def iterate(model_name):

    # 1. Iterate all the annotated images and extract the date
    ious = []
    accuracies = []
    predictions = []
    annotations = []

    folder = '/tmp/performance_evaluator/'
    annotated_files = [filename for filename in os.listdir(folder) if filename.startswith('Svartadalen_annotated_vh_')]
    # print(annotated_files)
    for annotated_file in annotated_files:
        # Extract the date
        date_str = annotated_file[25:35].replace('-', '')

        # Open the annotated file
        annotated_image = Image.open(folder + annotated_file).convert('L')
        patch_size = int(os.getenv('PATCH_SIZE'))
        new_width = (annotated_image.width // patch_size) * patch_size
        new_height = (annotated_image.height // patch_size) * patch_size
        annotated_image = annotated_image.crop((0, 0, new_width, new_height))
        annotated_data = numpy.array(annotated_image)
        array_min, array_max = numpy.nanmin(annotated_data), numpy.nanmax(annotated_data)
        annotated_data = ((annotated_data - array_min) / (array_max - array_min)).astype(int)
        annotations.append(annotated_data)

        # Locate the prediction file
        # prediction_file = folder + date_str + '_pred_bw.png'
        # prediction_file = folder + date_str + '_otsu_bw.png'
        prediction_file = folder + date_str + f'_{model_name}_bw.png'
        # Open the prediction file
        prediction_image = Image.open(prediction_file).convert('L')
        prediction_image = prediction_image.crop((0, 0, new_width, new_height))
        prediction_data = numpy.array(prediction_image)
        # pred_min, pred_max = numpy.nanmin(prediction_data), numpy.nanmax(prediction_data)
        array_min, array_max = numpy.nanmin(prediction_data), numpy.nanmax(prediction_data)
        prediction_data = ((prediction_data - array_min) / (array_max - array_min)).astype(int)
        predictions.append(prediction_data)

        # print(prediction_data)
        # print('\nDeepAqua')
        iou = jaccard_similarity.calculate_intersection_over_union(prediction_data, annotated_data)
        ious.append(iou)
        accuracy = (annotated_data == prediction_data).sum() / (annotated_data.shape[0] * annotated_data.shape[1])
        accuracies.append(accuracy)
        # print('IOU', iou)

        result = eval_semantic_segmentation([prediction_data], [annotated_data])
        confusion_matrix = calc_semantic_segmentation_confusion([prediction_data], [annotated_data])
        # plt.imshow(prediction_data)
        # plt.show()
        # plt.clf()
        # plt.imshow(annotated_data)
        # plt.show()
        # plt.clf()
        # calculate_confusion_matrix(prediction_data, annotated_data)
        # print(result)
        # print(result2)

        # break

    result = eval_semantic_segmentation(predictions, annotations)
    confusion_matrix = calc_semantic_segmentation_confusion([prediction_data], [annotated_data])

    print(result)
    print(confusion_matrix)
    print(f'Mean {model_name} IOU', numpy.asarray(ious).mean())
    print(f'Mean {model_name} Accuracy', numpy.asarray(accuracies).mean())

    TP = confusion_matrix[1, 1]
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]

    print(TP, TN, FP, FN)

    cmat = [[TP, FN], [FP, TN]]

    plt.figure(figsize=(6, 6))
    ax = seaborn.heatmap(cmat / numpy.sum(cmat), cmap="Reds", annot=True, fmt='.2%', square=1, linewidth=2.)
    ax.set_title(model_name)
    ax.xaxis.set_ticklabels(['Water', 'Soil'])
    ax.yaxis.set_ticklabels(['Water', 'Soil'])
    plt.xlabel("Predictions")
    plt.ylabel("Real values")
    plt.show()


def calculate_iou():

    # file1 = '/tmp/tmp_otsu_selected/bw_20180505_annotated.png'
    # file1 = '/tmp/tmp_otsu_selected/20180505_pred_bw.png'
    date = '20180505'
    # date = '20180704'
    # date = '20211016'
    annotated_file = f'/tmp/tmp_otsu_selected/{date}_annotated_bw.png'
    im = Image.open(annotated_file).convert('L')
    patch_size = int(os.getenv('PATCH_SIZE'))
    new_width = (im.width // patch_size) * patch_size
    new_height = (im.height // patch_size) * patch_size
    im = im.crop((0, 0, new_width, new_height))
    # im = im.crop((0, 0, 896, 448))
    # im.load()
    annotated_data = numpy.array(im)

    # DeepAqua prediction
    prediction_file = f'/tmp/tmp_otsu_selected/{date}_pred_bw.png'
    img = Image.open(prediction_file).convert('L')
    img = img.crop((0, 0, new_width, new_height))
    prediction_data = numpy.array(img)
    print('\nDeepAqua')
    iou = jaccard_similarity.calculate_intersection_over_union(prediction_data, annotated_data)

    # Otsu prediction
    prediction_file = f'/tmp/tmp_otsu_selected/{date}_otsu_bw.png'
    img = Image.open(prediction_file).convert('L')
    img = img.crop((0, 0, new_width, new_height))
    prediction_data = numpy.array(img)
    print('\nOtsu')
    iou = jaccard_similarity.calculate_intersection_over_union(prediction_data, annotated_data)

    # Otsu Gaussian
    prediction_file = f'/tmp/tmp_otsu_selected/{date}_otsu_gaussian_bw.png'
    img = Image.open(prediction_file).convert('L')
    img = img.crop((0, 0, new_width, new_height))
    prediction_data = numpy.array(img)
    print('\nOtsu Gaussian')
    iou = jaccard_similarity.calculate_intersection_over_union(prediction_data, annotated_data)


def calculate_confusion_matrix(Y_pred, Y_val):
    FP = len(numpy.where(Y_pred - Y_val == 1)[0])
    FN = len(numpy.where(Y_pred - Y_val == -1)[0])
    TP = len(numpy.where(Y_pred + Y_val == 2)[0])
    TN = len(numpy.where(Y_pred + Y_val == 0)[0])
    cmat = [[TP, FN], [FP, TN]]

    plt.figure(figsize=(6, 6))
    seaborn.heatmap(cmat / numpy.sum(cmat), cmap="Reds", annot=True, fmt='.2%', square=1, linewidth=2.)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.show()



def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    """Collect a confusion matrix.
    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.
    Args:
        pred_labels (iterable of numpy.ndarray): A collection of predicted
            labels. The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label.
        gt_labels (iterable of numpy.ndarray): A collection of ground
            truth labels. The shape of a ground truth label array is
            :math:`(H, W)`, and its corresponding prediction label should
            have the same shape.
            A pixel with value :obj:`-1` will be ignored during evaluation.
    Returns:
        numpy.ndarray:
        A confusion matrix. Its shape is :math:`(n\_class, n\_class)`.
        The :math:`(i, j)` th element corresponds to the number of pixels
        that are labeled as class :math:`i` by the ground truth and
        class :math:`j` by the prediction.
    """
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = 0
    confusion = numpy.zeros((n_class, n_class), dtype=numpy.int64)
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()

        # Dynamically expand the confusion matrix if necessary.
        lb_max = numpy.max((pred_label, gt_label))
        if lb_max >= n_class:
            expanded_confusion = numpy.zeros(
                (lb_max + 1, lb_max + 1), dtype=numpy.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        # Count statistics from valid pixels.
        mask = gt_label >= 0
        confusion += numpy.bincount(
            n_class * gt_label[mask].astype(int) +
            pred_label[mask], minlength=n_class**2).reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')
    return confusion


def calc_semantic_segmentation_iou(confusion):
    """Calculate Intersection over Union with a given confusion matrix.
    The definition of Intersection over Union (IoU) is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.
    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    Args:
        confusion (numpy.ndarray): A confusion matrix. Its shape is
            :math:`(n\_class, n\_class)`.
            The :math:`(i, j)` th element corresponds to the number of pixels
            that are labeled as class :math:`i` by the ground truth and
            class :math:`j` by the prediction.
    Returns:
        numpy.ndarray:
        An array of IoUs for the :math:`n\_class` classes. Its shape is
        :math:`(n\_class,)`.
    """
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0) -
                       numpy.diag(confusion))
    iou = numpy.diag(confusion) / iou_denominator
    return iou


def eval_semantic_segmentation(pred_labels, gt_labels):
    """Evaluate metrics used in Semantic Segmentation.
    This function calculates Intersection over Union (IoU), Pixel Accuracy
    and Class Accuracy for the task of semantic segmentation.
    The definition of metrics calculated by this function is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.
    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{mIoU} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{Pixel Accuracy} =  \
        \\frac \
        {\\sum_{i=1}^k N_{ii}} \
        {\\sum_{i=1}^k \\sum_{j=1}^k N_{ij}}`
    * :math:`\\text{Class Accuracy} = \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`
    * :math:`\\text{Mean Class Accuracy} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`
    The more detailed description of the above metrics can be found in a
    review on semantic segmentation [#]_.
    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.
    .. [#] Alberto Garcia-Garcia, Sergio Orts-Escolano, Sergiu Oprea, \
    Victor Villena-Martinez, Jose Garcia-Rodriguez. \
    `A Review on Deep Learning Techniques Applied to Semantic Segmentation \
    <https://arxiv.org/abs/1704.06857>`_. arXiv 2017.
    Args:
        pred_labels (iterable of numpy.ndarray): A collection of predicted
            labels. The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label.
            For example, this is a list of labels
            :obj:`[label_0, label_1, ...]`, where
            :obj:`label_i.shape = (H_i, W_i)`.
        gt_labels (iterable of numpy.ndarray): A collection of ground
            truth labels. The shape of a ground truth label array is
            :math:`(H, W)`, and its corresponding prediction label should
            have the same shape.
            A pixel with value :obj:`-1` will be ignored during evaluation.
    Returns:
        dict:
        The keys, value-types and the description of the values are listed
        below.
        * **iou** (*numpy.ndarray*): An array of IoUs for the \
            :math:`n\_class` classes. Its shape is :math:`(n\_class,)`.
        * **miou** (*float*): The average of IoUs over classes.
        * **pixel_accuracy** (*float*): The computed pixel accuracy.
        * **class_accuracy** (*numpy.ndarray*): An array of class accuracies \
            for the :math:`n\_class` classes. \
            Its shape is :math:`(n\_class,)`.
        * **mean_class_accuracy** (*float*): The average of class accuracies.
    """
    # Evaluation code is based on
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/
    # score.py#L37
    confusion = calc_semantic_segmentation_confusion(
        pred_labels, gt_labels)
    iou = calc_semantic_segmentation_iou(confusion)
    pixel_accuracy = numpy.diag(confusion).sum() / confusion.sum()
    class_accuracy = numpy.diag(confusion) / numpy.sum(confusion, axis=1)

    return {'iou': iou, 'miou': numpy.nanmean(iou),
            'pixel_accuracy': pixel_accuracy,
            'class_accuracy': class_accuracy,
            'mean_class_accuracy': numpy.nanmean(class_accuracy)}


def toy_example():

    pred_matrix = numpy.asarray([[1, 1], [1, 1]])
    gt_matrix = numpy.asarray([[1, 0], [1, 1]])

    calculate_confusion_matrix(pred_matrix, gt_matrix)
    confusion_matrix = calc_semantic_segmentation_confusion([pred_matrix], [gt_matrix])
    TP = confusion_matrix[1, 1]
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]

    print(TP, TN, FP, FN)

    cmat = [[TP, FN], [FP, TN]]

    plt.figure(figsize=(6, 6))
    ax = seaborn.heatmap(cmat / numpy.sum(cmat), cmap="Reds", annot=True, fmt='.2%', square=1, linewidth=2.)
    ax.xaxis.set_ticklabels(['Positive', 'Negative'])
    ax.yaxis.set_ticklabels(['Positive', 'Negative'])
    plt.xlabel("Predictions")
    plt.ylabel("Real values")
    plt.show()


def full_cycle():
    load_dotenv()

    model_name = os.getenv('MODEL_NAME')
    # model_name = 'otsu'
    # model_name = 'otsu_gaussian_95'
    # model_name = 'Orebro lan_mosaic_2018-07-04_64x64_sar_VH_20-epochs_0.00005-lr_42-rand'
    # model_name = 'pred'

    # convert_annotated_data_to_png()
    rename_prediction_images(model_name)
    iterate(model_name)
    # toy_example()


def main():
    full_cycle()


# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
