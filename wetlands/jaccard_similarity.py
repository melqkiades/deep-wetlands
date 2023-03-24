import time

import numpy as np
from matplotlib import pyplot as plt


def show_image():

    image1 = np.array(
        [
            [1, 0],
            [0, 1]
        ]
    )
    plt.imshow(image1, cmap='gray')
    plt.show()
    plt.clf()

    image2 = np.array(
        [
            [1, 1],
            [0, 1]
        ]
    )

    plt.imshow(image2, cmap='gray')
    plt.show()
    plt.clf()


def calculate_intersection_over_union(prediction_image, true_image):

    smooth = 1e-6
    prediction_image = prediction_image > 0.5
    true_image = true_image > 0.5
    intersection = (prediction_image & true_image).sum() + smooth
    union = (prediction_image | true_image).sum() + smooth
    intersection_over_union = intersection / union

    return intersection_over_union


def iou(y_true, y_pred, n_class):
    # IOU = TP/(TP+FN+FP)
    IOU = []
    for c in range(n_class):
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))

        n = TP
        d = float(TP + FP + FN + 1e-12)

        iou = np.divide(n, d)
        IOU.append(iou)

    return np.mean(IOU)


def main():
    show_image()
    calculate_intersection_over_union(None, None)

# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
