# The following is an example of how to utilize our Sen1Floods11 dataset for training a FCNN. In this example, we train
# and validate on hand-labeled chips of flood events. However, our dataset includes several other options that are
# detailed in the README. To replace the dataset, as outlined further below, simply replace the train, test, and
# validation split csv's, and download the corresponding dataset.
import sys
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import cv2
import glob
import rasterio as rio
import pandas as pd
from natsort import natsorted


filenames = natsorted(glob.glob('C:/Users/ioia4268/data/performance_evaluator/deepaqua_test_dataset_no_nov/standard2/*.csv'))
lines = [['hjalstaviken'], ['accuracy'], ['iou'], ['precision'], ['recall'], ['f1_score'], ['true_positives'], ['true_negatives'], ['false_positives'], ['false_negatives'], ['auc'], ['hornborgasjon'], ['accuracy'], ['iou'], ['precision'], ['recall'], ['f1_score'], ['true_positives'], ['true_negatives'], ['false_positives'], ['false_negatives'], ['auc'], ['svartadalen'], ['accuracy'], ['iou'], ['precision'], ['recall'], ['f1_score'], ['true_positives'], ['true_negatives'], ['false_positives'], ['false_negatives'], ['auc']]
for filename in filenames:
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        count = 0
        for row in spamreader:
            lines[count].append(row[1])
            count += 1
with open(os.path.dirname(filename) +'/compiled_results.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerows(lines)