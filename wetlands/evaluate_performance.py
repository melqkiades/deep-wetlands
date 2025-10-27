import glob
import json
import os
import sys
import time
import random
import matplotlib.pyplot as plt
import wandb
# from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
from networks.vision_transformer import SwinUnet as ViT_seg
from loss_functions import loss_function_factory
from model import model_factory
from wetlands import utils, map_wetlands, viz_utils
from wetlands.jaccard_similarity import calculate_intersection_over_union
from skimage import io
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
# from torchvision.transforms import v2 as T
# torch.set_float32_matmul_precision("high")

def plot_data(input, target, prediction):
    batch_size = input.shape[0]
    fig, ax = plt.subplots(batch_size,3)
    for i in range(batch_size):
        ax[i,0].imshow(input[i][0])
        ax[i,1].imshow(target[i][0])
        ax[i,2].imshow(prediction[i][0])
    plt.show()

# Class for Unet-based projection model
class Unet(nn.Module):

    def __init__(self, training_method, input_dim=572, num_first_level_channels=64, depth=5, conv_kernel_size=3, conv_stride=1,
                 max_pool_size=2, up_conv_kernel_size=2, up_conv_stride=2, conv_init=None):
        super().__init__()

        self.training_method = training_method
        self.input_dim = input_dim
        self.num_input_channels = 1
        self.num_first_level_channels = num_first_level_channels
        self.depth = depth
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.max_pool_size = max_pool_size
        self.up_conv_kernel_size = up_conv_kernel_size
        self.up_conv_stride = up_conv_stride
        self.conv_init = conv_init
        self.depth_num_channels = [self.num_first_level_channels]
        for i in range(self.depth - 1):
            self.depth_num_channels.append(self.depth_num_channels[-1] * 2)
        self.encoder_layers = nn.ModuleList([self.unet_block(self.num_input_channels, self.depth_num_channels[0]),
                                             nn.MaxPool2d(2, return_indices=True)])
        for i in range(len(self.depth_num_channels) - 2):
            self.encoder_layers.append(
                self.unet_block(self.depth_num_channels[i], self.depth_num_channels[i + 1]))
            self.encoder_layers.append(nn.MaxPool2d(2, return_indices=True))
        self.bottleneck = self.unet_block(self.depth_num_channels[-2], self.depth_num_channels[-1])
        self.decoder_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            self.unet_block(self.depth_num_channels[-1] + self.depth_num_channels[-2],
                            self.depth_num_channels[-2])])
        for i in range(len(self.depth_num_channels) - 2, 0, -1):
            self.decoder_layers.append(nn.Upsample(scale_factor=2))
            self.decoder_layers.append(
                self.unet_block(self.depth_num_channels[i] + self.depth_num_channels[i - 1],
                                self.depth_num_channels[i - 1]))
        if self.training_method == "supervised":
            self.final_layer = nn.Conv2d(self.depth_num_channels[0], 1, 1)

    def forward(self, x):
        skip_connections = []
        for i in range(0, len(self.encoder_layers), 2):
            x = self.encoder_layers[i](x)
            skip_connections.insert(0, x)
            x, max_indices_temp = self.encoder_layers[i + 1](x)

        x = self.bottleneck(x)

        for i in range(0, len(self.decoder_layers), 2):
            x = self.decoder_layers[i](x)
            x = torch.cat([skip_connections[i // 2], x], 1)
            x = self.decoder_layers[i + 1](x)

        if self.training_method == "supervised":
            return torch.sigmoid(self.final_layer(x))
        elif self.training_method == "unsupervised":
            return x

    def conv_block(self, num_in_channels, num_out_channels):
        conv_layer = nn.Conv2d(num_in_channels, num_out_channels, self.conv_kernel_size, self.conv_stride,
                               padding='same')
        if self.conv_init == 'He':
            torch.nn.init.kaiming_uniform_(conv_layer.weight)
        conv_block = nn.Sequential(conv_layer,
                                   nn.BatchNorm2d(num_out_channels),
                                   nn.ReLU())
        return conv_block

    def unet_block(self, num_in_channels, num_out_channels):
        unet_block = nn.Sequential(self.conv_block(num_in_channels, num_out_channels),
                                   self.conv_block(num_out_channels, num_out_channels))
        return unet_block


# Class for CNN prediction model
class Prediction_Module(nn.Module):

    def __init__(self, num_input_channels=4, num_output_channels=2, conv_init=None):
        super().__init__()

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.conv_init = conv_init
        self.conv_layer = nn.Conv2d(self.num_input_channels, self.num_output_channels, kernel_size=1, stride=1,
                                    padding=0)
        if self.conv_init == "He":
            torch.nn.init.kaiming_uniform_(self.conv_layer.weight)
        self.bn_layer = nn.BatchNorm2d(self.num_output_channels)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn_layer(x)
        return x


class CFDDataset_in_memory_test(Dataset):
    def __init__(self, config, dataset):
        self.dataset = dataset
        self.num_images = self.dataset.shape[0]
        self.patch_size = int(config['PATCH_SIZE'])
        data_dir = config['DATA_DIR']
        self.sar_dir = data_dir + config[f'TEST_SAR_DIR']
        self.sar_images = np.zeros((self.num_images, 1, self.patch_size, self.patch_size), np.float32)
        self.masks_dir = data_dir + config[f'TEST_MASK_DIR']
        self.mask_images = np.zeros((self.num_images, 1, self.patch_size, self.patch_size), np.float32)
        for i in range(self.num_images):
            index_ = self.dataset.iloc[i]['id']
            sar_path = self.sar_dir + str(index_) + '-sar.tif'
            mask_path = self.masks_dir + str(index_) + f'-ndwi_mask.tif'

            # Read image
            self.sar_images[i][0] = io.imread(sar_path)

            # Read image
            self.mask_images[i][0] = io.imread(mask_path)
        # Convert to Pytorch tensor
        self.sar_images = torch.from_numpy(self.sar_images)
        self.mask_images = torch.from_numpy(self.mask_images)

    def __getitem__(self, index):
        sar_image = self.sar_images[index]
        mask_image = self.mask_images[index]
        return sar_image, mask_image

    def __len__(self):
        return len(self.dataset)


# Compute and return various different iou metrics.
def computeIOU(output, target):
    nan_pixels = target==255
    target[nan_pixels] = torch.nan
    output[nan_pixels] = 0
    im_intersection = output * torch.nan_to_num(target, 0.)
    im_union = torch.nan_to_num(target, 0.) + output - im_intersection
    im_iou = (torch.sum(im_intersection, dim=[1, 2, 3]) + .0000001) / (torch.sum(im_union, dim=[1, 2, 3]) + .0000001)
    mean_im_iou = torch.mean(im_iou)
    num_im = im_iou.shape[0]
    output = output.flatten()
    target = target.flatten()
    intersection = output * torch.nan_to_num(target, 0.)
    union = torch.nan_to_num(target, 0.) + output - intersection
    iou = (torch.sum(intersection) + .0000001) / (torch.sum(union) + .0000001)
    return mean_im_iou.cpu().numpy(), num_im, iou.cpu().numpy(), im_iou.numpy()


# Compute per-pixel accuracy
def computeAccuracy(output, target):
    output = output.flatten()
    target = target.flatten()
    output = output[target!=255]
    target = target[target!=255]

    correct = torch.sum(output.eq(target))
    if len(target) == 0:
        print('ERROR')

    return correct.float() / len(target)


# Mach predicted classes to water/land by assigning each class to the area that is has the greatest iou with.
def matchSegmentationResultToOriginalLabel(resultMaps, referenceMaps):
    referenceMaps = referenceMaps.astype(int)
    original_shape = resultMaps.shape
    resultMaps = resultMaps.flatten()
    referenceMaps = referenceMaps.flatten()
    ##ADDING 1 to not keep any zero value
    ##Otherwise zero is an object here (Impervious surfaces)
    resultMaps = resultMaps + 1
    referenceMaps = referenceMaps + 1

    ##Finding unique values
    resultMapUniqueVals = np.unique(resultMaps)
    resultMaps[referenceMaps == 0] = 0

    referenceMapUniqueVals, referenceMapUniqueCounts = np.unique(referenceMaps, return_counts=True)
    if 0 in referenceMapUniqueVals:
        nan_pos = np.where(referenceMapUniqueVals == 0)[0][0]
        referenceMapUniqueVals = np.delete(referenceMapUniqueVals, nan_pos)
        referenceMapUniqueCounts = np.delete(referenceMapUniqueCounts, nan_pos)
    referenceSortingIndices = np.argsort(-referenceMapUniqueCounts)
    referenceMapUniqueVals = referenceMapUniqueVals[referenceSortingIndices]

    resultToReferenceRelationMatrix = np.zeros((len(resultMapUniqueVals), len(referenceMapUniqueVals)))
    for resultIndex, resultUniqueVal in enumerate(resultMapUniqueVals):
        resultUniqueValIndicator = np.copy(resultMaps)
        resultUniqueValIndicator[resultUniqueValIndicator != resultUniqueVal] = 0
        for referenceIndex, referenceUniqueVal in enumerate(referenceMapUniqueVals):
            referenceUniqueValIndicator = np.copy(referenceMaps)
            referenceUniqueValIndicator[referenceUniqueValIndicator != referenceUniqueVal] = 0
            resultReferenceIntersection = resultUniqueValIndicator * referenceUniqueValIndicator
            numIntersection = len(np.argwhere(resultReferenceIntersection))
            num_union = len(np.argwhere(resultUniqueValIndicator + referenceUniqueValIndicator))
            resultToReferenceRelationMatrix[resultIndex, referenceIndex] = (numIntersection + 0.0001) / (num_union + 0.0001)

    resultMapReassigned = np.zeros(resultMaps.shape)

    for resultIndex, resultUniqueVal in enumerate(resultMapUniqueVals):
        matchesCorrespondingToThisVal = resultToReferenceRelationMatrix[resultIndex, :]
        maximizingIndex = np.argsort(matchesCorrespondingToThisVal)[-1]
        resultMapOptimumMatch = referenceMapUniqueVals[maximizingIndex]
        resultMapReassigned[resultMaps == resultUniqueVal] = resultMapOptimumMatch

    ##Subtracting 1 to keep values as it were
    resultMapReassigned = resultMapReassigned - 1

    resultMapReassigned = np.reshape(resultMapReassigned, original_shape).astype(int)

    return resultMapReassigned


def get_dataloader(config, data):
    dataset = CFDDataset_in_memory_test(config, data)
    print('Num test data:', len(dataset))
    batch_size = int(config['BATCH_SIZE'])
    num_workers = int(config['NUM_WORKERS'])
    dataloader = DataLoader(
          dataset,
          batch_size=batch_size,
          drop_last=False,
          num_workers=num_workers,
          pin_memory=True
        )
    return dataloader



def evaluate(config, models, dataloader, device):
    outputs = []
    targets = []
    for i in range(len(models)):
        models[i].eval()
        temp_outputs = []
        temp_targets = []


        for input, target in dataloader:
            input = input.to(device)
            target = target.to(device)

            with torch.set_grad_enabled(False):
                output = models[i](input)
            temp_outputs.append(output.cpu().numpy().flatten())
            temp_targets.append(target.cpu().numpy().flatten())

        outputs.append(np.concatenate(temp_outputs))
        targets.append(np.concatenate(temp_targets))
    ensemble_outputs = np.mean(np.stack(outputs), axis=0)
    ensemble_targets = targets[0]
    iou = intersection_over_union(ensemble_outputs.copy(), ensemble_targets.copy(), mask_values=True)
    single_ious = []
    for i in range(len(models)):
        single_ious.append(intersection_over_union(outputs[i].copy(), targets[i].copy(), mask_values=True))
    # val_loss = np.mean(losses)
    # val_iou = np.mean(ious)
    print('Ensemble test IOU', iou)
    print('Single test IOUs', single_ious)
    print('Mean single test IOUs', np.mean(single_ious))
    # current_lr = scheduler.get_last_lr()
    # if scheduler is not None:
    #     scheduler.step(val_iou)
    metrics = {
        'valid_iou': iou,
        # 'lr': current_lr
    }

    return metrics



def evaluate_unsupervised(config, models, prediction_models, dataloader, device):
    outputs_ensemble = []
    targets_ensemble = []
    for i in range(len(models)):
        models[i].eval()
        prediction_models[i].eval()
        valid_predictions = []
        valid_targets = []

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = models[i](inputs)
            projections = prediction_models[i](outputs)
            _, predictions = torch.max(projections, 1)

            valid_predictions.append(predictions.cpu().numpy())
            valid_targets.append(targets.cpu().numpy())
            # valid_images.append(inputs.cpu().numpy())

        valid_predictions = np.concatenate(valid_predictions,axis=0)
        valid_targets = np.concatenate(valid_targets, axis=0)


        matched_model_seg = np.expand_dims(matchSegmentationResultToOriginalLabel(valid_predictions, valid_targets), 1)
        outputs_ensemble.append(matched_model_seg.flatten())
        targets_ensemble = [valid_targets.flatten()]


    ensemble_outputs = np.mean(np.stack(outputs_ensemble), axis=0)
    ensemble_targets = targets_ensemble[0]
    iou = intersection_over_union(ensemble_outputs.copy(), ensemble_targets.copy(), mask_values=True)
    tp, fp, tn, fn = confusion_matrix(ensemble_outputs.copy(), ensemble_targets.copy(), mask_values=True)
    pa = (tp+tn)/(tp+tn+fp+fn)
    pr = (tp)/(tp+fp)
    re = (tp)/(tp+fn)
    f1 = 2*pr*re/(pr+re)
    print('Ensemble:')
    print('Test IOU', iou)
    print('tp, fp, tn, fn', tp, fp, tn, fn)
    print('pa', pa)
    print('pr', pr)
    print('re', re)
    print('f1', f1)

    single_ious = []
    single_pas = []
    single_prs = []
    single_res = []
    single_f1s = []
    for i in range(len(models)):
        single_ious.append(intersection_over_union(outputs_ensemble[i].copy(), targets_ensemble[0].copy(), mask_values=True))
        temp_tp, temp_fp, temp_tn, temp_fn = confusion_matrix(outputs_ensemble[i].copy(), targets_ensemble[0].copy(), mask_values=True)
        temp_pr = (temp_tp)/(temp_tp+temp_fp)
        temp_re = (temp_tp)/(temp_tp+temp_fn)
        single_pas.append((temp_tp+temp_tn)/(temp_tp+temp_tn+temp_fp+temp_fn))
        single_prs.append(temp_pr)
        single_res.append(temp_re)
        single_f1s.append(2*temp_pr*temp_re/(temp_pr+temp_re))

    print('Mean:')
    print('Test IOU', np.mean(single_ious))
    # print('tp, fp, tn, fn', tp, fp, tn, fn)
    print('pa', np.mean(single_pas))
    print('pr', np.mean(single_prs))
    print('re', np.mean(single_res))
    print('f1', np.mean(single_f1s))

    # print('Val IOU', iou)
    metrics = {
    'valid_iou': iou,}

    # print('Ensemble test IOU', iou)
    # print('Single test IOUs', single_ious)
    # print('Mean single test IOUs', np.mean(single_ious))

    return metrics


def save_model(model, model_dir, model_file):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file)
    torch.save(model.state_dict(), model_path)
    print(f'Model successfully saved to {model_path}')


def evaluate_single_image(model, tiles_data, images_dir, ndwi_masks_dir, device):
    i = 120
    model.eval()
    index = tiles_data[tiles_data.split == 'valid'].iloc[i]['id']
    image_path = images_dir + str(index) + '-sar.tif'
    sar_image = rio.open(image_path).read()
    print(sar_image.shape)

    ndwi_image_path = ndwi_masks_dir + str(index) + '-ndwi_mask.tif'
    ndwi_image = rio.open(ndwi_image_path).read()
    print(ndwi_image.shape)
    print(ndwi_image_path)

    # sar_image = sar_image.transpose((2, 1, 0))[None, :]
    batch_sar_image = sar_image[None, :]
    print(batch_sar_image.shape)
    batch_sar_image = torch.from_numpy(batch_sar_image.astype(np.float32)).to(device)
    pred_image = model(batch_sar_image).cpu().detach().numpy()
    # pred_image = pred_image.squeeze().transpose((1, 0))
    pred_image = pred_image.squeeze()
    # pred_image = (pred_image * 255.0).astype("uint8")
    plt.imshow(pred_image)
    plt.show()
    plt.clf()

    iou = calculate_intersection_over_union(ndwi_image[0], pred_image[0])
    print('IOU', iou)

    return sar_image, pred_image, ndwi_image


def full_cycle(config, test_name):
    training_method = config['TRAINING_METHOD']

    num_first_level_channels = int(config['NUM_FIRST_LVL_CHANNELS'])
    depth = int(config['DEPTH'])
    kernel_size = int(config['KERNEL_SIZE'])
    K = int(config['K'])

    tiles_data = utils.create_tiles_file_test(config)


    # Check is GPU is enabled
    device = utils.get_device()

    dataloader = get_dataloader(config, tiles_data)
    # return
    patch_size = config['PATCH_SIZE']
    if training_method == 'supervised':
        models = []
        for directory in glob.glob(config['DATA_DIR'] + config['MODELS_DIR'] + test_name + '_run_*'):
            model = Unet(training_method, patch_size, num_first_level_channels, depth, conv_init="He", conv_kernel_size=3).cuda()
            pretrained_dict = torch.load(f'{directory}/best_model.pth', map_location=device)
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
            model.load_state_dict(pretrained_dict, strict=True)
            model = torch.nn.DataParallel(model)
            model.to(device)
            models.append(model)
    elif training_method == 'unsupervised':
        models = []
        prediction_models = []
        for directory in glob.glob(config['DATA_DIR'] + config['MODELS_DIR'] + test_name + '_run_*'):
            model = Unet(training_method, patch_size, num_first_level_channels, depth, conv_init="He", conv_kernel_size=kernel_size).cuda()
            prediction_model = Prediction_Module(num_first_level_channels, K, conv_init="He").cuda()
            pretrained_dict = torch.load(f'{directory}/final_epoch.pth', map_location=device)
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
            model.load_state_dict(pretrained_dict, strict=True)
            prediction_pretrained_dict = torch.load(f'{directory}/final_epoch_prediction_model.pth',
                                                    map_location=device)
            prediction_pretrained_dict = {k[7:]: v for k, v in prediction_pretrained_dict.items()}
            prediction_model.load_state_dict(prediction_pretrained_dict, strict=True)
            model = torch.nn.DataParallel(model)
            model.to(device)
            prediction_model = torch.nn.DataParallel(prediction_model)
            prediction_model.to(device)
            models.append(model)
            prediction_models.append(prediction_model)

    if training_method == 'supervised':
        val_metrics = evaluate(config,
            models,
            dataloader,
            device
        )
    elif training_method == 'unsupervised':
        val_metrics = evaluate_unsupervised(config, models, prediction_models,
            dataloader,
            device,
        )




def intersection_over_union(y_pred, y_true, mask_values=True):#, compare_outputs=False):

    smooth = 1e-6
    if mask_values:
        y_pred[y_true == -1.] = 0.
        y_true[y_true == -1.] = 0.
    y_pred = y_pred > 0.5
    y_true = y_true > 0.5
    # y_pred = torch.argmax(y_pred, dim=1)
    # if not compare_outputs:
    #     y_true = torch.squeeze(y_true.to(torch.int))
    # else:
    #     y_true = torch.argmax(y_true, dim=1)
    intersection = (y_pred & y_true).sum() + smooth
    union = (y_pred | y_true).sum() + smooth
    iou = intersection / union

    return iou


def confusion_matrix(y_pred, y_true, mask_values=True):
    smooth = 1e-6
    if mask_values:
        y_pred[y_true == -1.] = 0.
        y_true[y_true == -1.] = 0.
    y_pred = y_pred > 0.5
    y_true = y_true > 0.5
    tp = (y_pred & y_true).sum()
    fp = (y_pred & np.logical_not(y_true)).sum()
    tn = (np.logical_not(y_pred) & np.logical_not(y_true)).sum()
    fn = (np.logical_not(y_pred) & y_true).sum()

    return tp, fp, tn, fn