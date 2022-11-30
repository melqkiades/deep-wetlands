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


def calculate_intersection_over_union(image1, image2):
    # image1 = np.array(
    #     [
    #         [1, 0],
    #         [0, 1]
    #     ]
    # )
    #
    # image2 = np.array(
    #     [
    #         [1, 0],
    #         [1, 0]
    #     ]
    # )

    # image1 = np.array(
    #     [
    #         [1, 0, 1],
    #         [0, 1, 0],
    #         [1, 0, 1],
    #     ]
    # )
    #
    # image2 = np.array(
    #     [
    #         [1, 0, 0],
    #         [0, 1, 0],
    #         [1, 1, 1],
    #     ]
    # )

    image1 = image1 > 0.5
    image2 = image2 > 0.5
    # print('\nImage 1:\n', image1)
    # print('\nBinary 1:\n', binary1)


    SMOOTH = 1e-6

    intersection = image1 & image2
    union = image1 | image2

    # print('Intersection:\n', intersection)
    # print('\nUnion:\n', union)

    intersection_over_union = (intersection.sum() + SMOOTH) / (union.sum() + SMOOTH)

    print('IOU', intersection_over_union)

    # intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    # union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0
    #
    # iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

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
