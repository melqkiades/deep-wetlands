import json
import os
import time
import random
import wandb
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from loss_functions import loss_function_factory
from model import model_factory
from wetlands import utils, map_wetlands, viz_utils
from skimage import io
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms


# Class for Unet-based projection model
class Unet(nn.Module):

    def __init__(self, training_method, input_dim=512, num_first_level_channels=64, depth=5, conv_kernel_size=3, conv_stride=1,
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


class CFDDataset_in_memory(Dataset):
    def __init__(self, config, dataset, augment):
        self.dataset = dataset
        self.augment = augment
        self.num_images = self.dataset.shape[0]
        self.patch_size = int(config['PATCH_SIZE'])
        data_dir = config['DATA_DIR']
        self.sar_dir = data_dir + config[f'TRAIN_SAR_DIR']
        self.sar_images = np.zeros((self.num_images, 1, self.patch_size, self.patch_size), np.float32)
        self.masks_dir = data_dir + config[f'TRAIN_MASK_DIR']
        self.mask_images = np.zeros((self.num_images, 1, self.patch_size, self.patch_size), np.float32)
        self.training_method = config['TRAINING_METHOD']
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
        if self.training_method == 'unsupervised':
            aug_sar_image = transforms.GaussianBlur(5, sigma=(1.0, 2.0))(sar_image)
        if self.augment:
            if random.random() > 0.5:
                sar_image = TF.hflip(sar_image)
                mask_image = TF.hflip(mask_image)
                if self.training_method == 'unsupervised':
                    aug_sar_image = TF.hflip(aug_sar_image)
            if random.random() > 0.5:
                sar_image = TF.vflip(sar_image)
                mask_image = TF.vflip(mask_image)
                if self.training_method == 'unsupervised':
                    aug_sar_image = TF.vflip(aug_sar_image)
        if self.training_method == 'supervised':
            return sar_image, mask_image
        elif self.training_method == 'unsupervised':
            return sar_image, aug_sar_image, mask_image

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

    referenceMapUniqueVals, referenceMapUniqueCounts = np.unique(referenceMaps, return_counts=True)
    if 256 in referenceMapUniqueVals:
        nan_pos = np.where(referenceMapUniqueVals == 256)[0][0]
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
            resultToReferenceRelationMatrix[resultIndex, referenceIndex] = numIntersection / num_union

    resultMapReassigned = np.zeros(resultMaps.shape)

    for resultIndex, resultUniqueVal in enumerate(resultMapUniqueVals):
        matchesCorrespondingToThisVal = resultToReferenceRelationMatrix[resultIndex, :]
        maximizingIndex = np.argsort(matchesCorrespondingToThisVal)[-1]
        resultMapOptimumMatch = referenceMapUniqueVals[maximizingIndex]
        resultMapReassigned[resultMaps == resultUniqueVal] = resultMapOptimumMatch

    ##Subtracting 1 to keep values as it were
    resultMapReassigned = resultMapReassigned - 1

    resultMapReassigned = np.reshape(resultMapReassigned, original_shape).astype(int)

    return torch.from_numpy(resultMapReassigned)


def get_dataloaders(config, data):
    datasets = {'train': CFDDataset_in_memory(config, data[data.split == 'train'], True),
                'valid': CFDDataset_in_memory(config, data[data.split == 'valid'], False)}
    print('Num train data:', len(datasets['train']))
    print('Num val data:', len(datasets['valid']))
    batch_size = int(config['BATCH_SIZE'])
    num_workers = int(config['NUM_WORKERS'])
    dataloaders = {
        'train': DataLoader(
          datasets['train'],
          batch_size=batch_size,
          shuffle=True,
          num_workers=num_workers,
          pin_memory=True
        ),
        'valid': DataLoader(
          datasets['valid'],
          batch_size=batch_size,
          drop_last=False,
          num_workers=num_workers,
          pin_memory=True
        )
    }
    return dataloaders


def train(config, model, dataloader, criterion, optimizer, device, scheduler):
    model.train(True)
    losses = []
    ious = []
    for input, target in dataloader:
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            output = model(input)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            iou = intersection_over_union(output, target)
            losses.append(loss.cpu().detach().numpy())
            ious.append(iou.cpu().detach().numpy())
    scheduler.step()
    # prof.export_chrome_trace("trace.json")
    train_loss = np.mean(losses)
    train_iou = np.mean(ious)
    print('Train loss', train_loss, 'Train IOU', train_iou)

    metrics = {
        'train_loss': train_loss,
        'train_iou': train_iou,
    }

    return metrics

def evaluate(config, model, dataloader, criterion, device):
    model.eval()
    losses = []
    ious = []

    for input, target in dataloader:
        input = input.to(device)
        target = target.to(device)

        with torch.set_grad_enabled(False):
            output = model(input)
            loss = criterion(output, target)
            iou = intersection_over_union(output, target)
            losses.append(loss.cpu().detach().numpy())
            ious.append(iou.cpu().detach().numpy())

    val_loss = np.mean(losses)
    val_iou = np.mean(ious)
    print('Val loss', val_loss, 'Val IOU', val_iou)
    # current_lr = scheduler.get_last_lr()
    metrics = {
        'valid_loss': val_loss,
        'valid_iou': val_iou,
        # 'lr': current_lr
    }

    return metrics


def train_unsupervised(config, model, model_aug, prediction_model, dataloader, optimizer, device, epoch, scheduler):
    K = int(config['K'])
    sim_loss_mult = float(config['SIM_LOSS_MULT'])
    disim_loss_mult = float(config['DISIM_LOSS_MULT'])
    secondary_losses_epoch = int(config['SECONDARY_LOSSES_EPOCH'])

    model.train(True)
    model_aug.train(True)
    prediction_model.train(True)
    train_loss = 0
    train_deep_clustering_loss = 0
    train_deep_clustering_loss_aug = 0
    train_similarity_loss = 0
    train_disimilarity_loss = 0

    train_predictions = []
    train_targets = []

    for inputs, inputs_aug, targets in dataloader:
        inputs = inputs.to(device)
        inputs_aug = inputs_aug.to(device)
        targets = targets.to(device)
        batch_num_im = inputs.shape[0]

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            randomShufflingIndices = torch.randperm(inputs_aug.shape[0])
            inputs_aug_shuffled = inputs_aug[randomShufflingIndices, :, :, :]
            outputs = model(inputs)
            projections = prediction_model(outputs)
            outputs_aug = model_aug(inputs_aug)
            projections_aug = prediction_model(outputs_aug)
            outputs_aug_shuffled = model_aug(inputs_aug_shuffled)
            projections_aug_shuffled = prediction_model(outputs_aug_shuffled)

            _, predictions = torch.max(projections, 1)
            values, counts = torch.unique(predictions, sorted=True, return_counts=True)
            _, predictions_aug = torch.max(projections_aug, 1)
            values_aug, counts_aug = torch.unique(predictions_aug, sorted=True, return_counts=True)
            _, predictions_aug_shuffled = torch.max(projections_aug_shuffled, 1)
            values_aug_shuffled, counts_aug_shuffled = torch.unique(predictions_aug_shuffled, sorted=True, return_counts=True)
            weights = torch.zeros((K,), device=device)
            class_counts = torch.ones((K,), device=device)
            for j in range(values.shape[0]):
                class_counts[values[j]] += counts[j]
            for j in range(values_aug.shape[0]):
                class_counts[values_aug[j]] += counts_aug[j]
            for j in range(values_aug_shuffled.shape[0]):
                class_counts[values_aug_shuffled[j]] += counts_aug_shuffled[j]
            for j in range(K):
                weights[j] = 0.00001/(class_counts[j] + 0.00001)
            weights = weights/torch.sum(weights)
            deep_clustering_loss_func = nn.CrossEntropyLoss(weight=weights)
            deep_clustering_loss = deep_clustering_loss_func(projections, predictions)
            deep_clustering_loss_aug = deep_clustering_loss_func(projections_aug, predictions_aug)
            similarity_loss = nn.L1Loss()(projections, projections_aug)
            disimilarity_loss = -nn.L1Loss()(projections, projections_aug_shuffled)

            if epoch - 1 < secondary_losses_epoch:
                total_loss = (deep_clustering_loss + deep_clustering_loss_aug) / 2
            else:
                total_loss = (deep_clustering_loss + deep_clustering_loss_aug + sim_loss_mult * similarity_loss +
                            disim_loss_mult * disimilarity_loss) / (2 + sim_loss_mult + disim_loss_mult)

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.detach().cpu().numpy() * batch_num_im
            train_deep_clustering_loss += deep_clustering_loss.detach().cpu().numpy() * batch_num_im
            train_deep_clustering_loss_aug += deep_clustering_loss_aug.detach().cpu().numpy() * batch_num_im
            train_similarity_loss += similarity_loss.detach().cpu().numpy() * batch_num_im
            train_disimilarity_loss += disimilarity_loss.detach().cpu().numpy() * batch_num_im

            train_predictions.append(predictions.cpu().numpy())
            train_targets.append(targets.cpu().numpy())
            # train_images.append(inputs.cpu().numpy())
    scheduler.step()
    train_predictions = np.concatenate(train_predictions,axis=0)
    train_targets = np.concatenate(train_targets, axis=0)
    matched_model_seg = matchSegmentationResultToOriginalLabel(train_predictions, train_targets).unsqueeze(1)

    mean_im_iou, num_im, iou,  sep_im_iou = computeIOU(matched_model_seg, torch.from_numpy(train_targets))
    accuracy = computeAccuracy(matched_model_seg, torch.from_numpy(train_targets))



    train_loss = train_loss / num_im
    train_deep_clustering_loss = train_deep_clustering_loss / num_im
    train_deep_clustering_loss_aug = train_deep_clustering_loss_aug / num_im
    train_similarity_loss = train_similarity_loss / num_im
    train_disimilarity_loss = train_disimilarity_loss / num_im


    train_metrics = {'train_iou':iou, 'train_im_iou':mean_im_iou, 'train_acc':accuracy,
                     'train_loss': train_loss, 'train_deep_clustering_loss': train_deep_clustering_loss,
                     'train_deep_clustering_loss_aug': train_deep_clustering_loss_aug,
                     'train_similarity_loss': train_similarity_loss, 'train_disimilarity_loss': train_disimilarity_loss}
    return train_metrics

def evaluate_unsupervised(config, model, model_aug, prediction_model, dataloader, device, epoch):
    K = int(config['K'])
    sim_loss_mult = float(config['SIM_LOSS_MULT'])
    disim_loss_mult = float(config['DISIM_LOSS_MULT'])
    secondary_losses_epoch = int(config['SECONDARY_LOSSES_EPOCH'])

    model.eval()
    model_aug.eval()
    prediction_model.eval()
    valid_loss = 0
    valid_deep_clustering_loss = 0
    valid_deep_clustering_loss_aug = 0
    valid_similarity_loss = 0
    valid_disimilarity_loss = 0


    valid_predictions = []
    valid_targets = []

    for inputs, inputs_aug, targets in dataloader:
        inputs = inputs.to(device)
        inputs_aug = inputs_aug.to(device)
        targets = targets.to(device)
        batch_num_im = inputs.shape[0]

        randomShufflingIndices = torch.randperm(inputs_aug.shape[0])
        inputs_aug_shuffled = inputs_aug[randomShufflingIndices, :, :, :]
        outputs = model(inputs)
        projections = prediction_model(outputs)
        outputs_aug = model_aug(inputs_aug)
        projections_aug = prediction_model(outputs_aug)
        outputs_aug_shuffled = model_aug(inputs_aug_shuffled)
        projections_aug_shuffled = prediction_model(outputs_aug_shuffled)

        _, predictions = torch.max(projections, 1)
        values, counts = torch.unique(predictions, sorted=True, return_counts=True)
        _, predictions_aug = torch.max(projections_aug, 1)
        values_aug, counts_aug = torch.unique(predictions_aug, sorted=True, return_counts=True)
        _, predictions_aug_shuffled = torch.max(projections_aug_shuffled, 1)
        values_aug_shuffled, counts_aug_shuffled = torch.unique(predictions_aug_shuffled, sorted=True, return_counts=True)
        weights = torch.zeros((K,), device=device)
        class_counts = torch.ones((K,), device=device)
        for j in range(values.shape[0]):
            class_counts[values[j]] += counts[j]
        for j in range(values_aug.shape[0]):
            class_counts[values_aug[j]] += counts_aug[j]
        for j in range(values_aug_shuffled.shape[0]):
            class_counts[values_aug_shuffled[j]] += counts_aug_shuffled[j]
        for j in range(K):
            weights[j] = 0.00001/(class_counts[j] + 0.00001)
        weights = weights/torch.sum(weights)
        deep_clustering_loss_func = nn.CrossEntropyLoss(weight=weights)
        deep_clustering_loss = deep_clustering_loss_func(projections, predictions)
        deep_clustering_loss_aug = deep_clustering_loss_func(projections_aug, predictions_aug)
        similarity_loss = nn.L1Loss()(projections, projections_aug)
        disimilarity_loss = -nn.L1Loss()(projections, projections_aug_shuffled)

        if epoch - 1 < secondary_losses_epoch:
            total_loss = (deep_clustering_loss + deep_clustering_loss_aug) / 2
        else:
            total_loss = (deep_clustering_loss + deep_clustering_loss_aug + sim_loss_mult * similarity_loss +
                        disim_loss_mult * disimilarity_loss) / (2 + sim_loss_mult + disim_loss_mult)

        valid_loss += total_loss.detach().cpu().numpy() * batch_num_im
        valid_deep_clustering_loss += deep_clustering_loss.detach().cpu().numpy() * batch_num_im
        valid_deep_clustering_loss_aug += deep_clustering_loss_aug.detach().cpu().numpy() * batch_num_im
        valid_similarity_loss += similarity_loss.detach().cpu().numpy() * batch_num_im
        valid_disimilarity_loss += disimilarity_loss.detach().cpu().numpy() * batch_num_im

        valid_predictions.append(predictions.cpu().numpy())
        valid_targets.append(targets.cpu().numpy())

    valid_predictions = np.concatenate(valid_predictions,axis=0)
    valid_targets = np.concatenate(valid_targets, axis=0)
    matched_model_seg = matchSegmentationResultToOriginalLabel(valid_predictions, valid_targets).unsqueeze(1)

    mean_im_iou, num_im, iou,  sep_im_iou = computeIOU(matched_model_seg, torch.from_numpy(valid_targets))
    accuracy = computeAccuracy(matched_model_seg, torch.from_numpy(valid_targets))



    valid_loss = valid_loss / num_im
    valid_deep_clustering_loss = valid_deep_clustering_loss / num_im
    valid_deep_clustering_loss_aug = valid_deep_clustering_loss_aug / num_im
    valid_similarity_loss = valid_similarity_loss / num_im
    valid_disimilarity_loss = valid_disimilarity_loss / num_im


    valid_metrics = {'valid_iou':iou, 'valid_im_iou':mean_im_iou, 'valid_acc':accuracy,
                     'valid_loss': valid_loss, 'valid_deep_clustering_loss': valid_deep_clustering_loss,
                     'valid_deep_clustering_loss_aug': valid_deep_clustering_loss_aug,
                     'valid_similarity_loss': valid_similarity_loss, 'valid_disimilarity_loss': valid_disimilarity_loss}
    return valid_metrics


def save_model(model, model_dir, model_file):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file)
    torch.save(model.state_dict(), model_path)
    print(f'Model successfully saved to {model_path}')


def full_cycle(config, test_name):
    # Configure the wandb run
    wandb.login(key='###')
    wandb_config = config.copy()
    training_method = config['TRAINING_METHOD']
    if training_method == 'supervised':
        project_name = "wetlands_supervised"
    elif training_method == "unsupervised":
        project_name = "wetlands"
    wandb.init(project=project_name, config=wandb_config, name=test_name)
    run_name = wandb.run.name
    wandb.run.define_metric("valid_iou", summary="max")
    wandb.run.define_metric("valid_loss", summary="min")
    wandb.run.define_metric("train_iou", summary="max")
    wandb.run.define_metric("train_loss", summary="min")

    n_epochs = config['EPOCHS']
    learning_rate = config['LEARNING_RATE']
    seed = config['RANDOM_SEED']
    if seed != 'NONE':
        seed = seed
    model_dir = config['MODELS_DIR']
    outputs_dir = config['DATA_DIR']
    if config['SAVE_MODEL_ON_ALL_EPOCHS']:
        save_model_on_all_epochs = True
    else:
        save_model_on_all_epochs = False
    if config['SAVE_MODEL_ON_LAST_EPOCH']:
        save_model_on_last_epoch = True
    else:
        save_model_on_last_epoch = False

    num_first_level_channels = config['NUM_FIRST_LVL_CHANNELS']
    depth = config['DEPTH']
    kernel_size = config['KERNEL_SIZE']
    K = config['K']

    if seed != 'NONE':
        utils.plant_random_seed(seed)

    tiles_data = utils.create_tiles_file(config)
    training_date = config['TRAIN_DATE']

    # Check is GPU is enabled
    device = utils.get_device()

    dataloaders = get_dataloaders(config, tiles_data)
    print('Dataloaders length:', len(dataloaders['train']), len(dataloaders['valid']))
    if training_method == 'supervised':
        model = Unet(training_method, config['PATCH_SIZE'], num_first_level_channels, depth, conv_init="He", conv_kernel_size=3).cuda()
        criterion = loss_function_factory.create_loss_function('dice')
        optimizer = torch.optim.AdamW(list(model.parameters()), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(dataloaders['train']), T_mult=2, eta_min=0,
                                                                         last_epoch=-1)
        model = torch.nn.DataParallel(model)
        model.to(device)
    elif training_method == 'unsupervised':
        model = Unet(training_method, config['PATCH_SIZE'], num_first_level_channels, depth, conv_init="He", conv_kernel_size=kernel_size).cuda()
        model_aug = Unet(training_method, config['PATCH_SIZE'], num_first_level_channels, depth, conv_init="He",
                       conv_kernel_size=kernel_size).cuda()
        prediction_model = Prediction_Module(num_first_level_channels, K, conv_init="He").cuda()
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(model_aug.parameters()) +
                                      list(prediction_model.parameters()), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(dataloaders['train']), T_mult=2,
                                                                 eta_min=0, last_epoch=-1)
        model = torch.nn.DataParallel(model)
        model.to(device)
        model_aug = torch.nn.DataParallel(model_aug)
        model_aug.to(device)
        prediction_model = torch.nn.DataParallel(prediction_model)
        prediction_model.to(device)

    print(model)

    print('Model parameters', sum(param.numel() for param in model.parameters()))

    max_score = 0
    val_ious = []
    best_epoch = 0
    for epoch in range(1, n_epochs + 1):
        print("\nEpoch {}/{} {}".format(epoch, n_epochs, time.strftime("%Y/%m/%d-%H:%M:%S")))
        print("-" * 10)
        if training_method == 'supervised':
            train_metrics = train(config,
                model,
                dataloaders["train"],
                criterion,
                optimizer,
                device,
                scheduler
            )
            val_metrics = evaluate(config,
                model,
                dataloaders['valid'],
                criterion,
                device
            )
        elif training_method == 'unsupervised':
            train_metrics = train_unsupervised(config, model, model_aug, prediction_model,
                dataloaders["train"],
                optimizer,
                device,
                epoch,
                scheduler
            )
            val_metrics = evaluate_unsupervised(config, model, model_aug, prediction_model,
                dataloaders['valid'],
                device,
                epoch
            )

        metrics = {
            **train_metrics, **val_metrics,
        }

        print('Train loss: {}, Val loss: {}'.format(metrics['train_loss'], metrics['valid_loss']))
        wandb.log(metrics)

        val_ious.append(metrics['valid_iou'])
        if metrics['valid_iou'] > max_score:
            max_score = metrics['valid_iou']
            best_epoch = epoch
            print(f'New best model found on epoch {epoch}. Validation IoU: {max_score}')
            save_model(model, os.path.join(outputs_dir, model_dir, run_name), 'best_model.pth')
            if training_method == 'unsupervised':
                save_model(prediction_model, os.path.join(outputs_dir, model_dir, run_name), 'best_prediction_model.pth')
        if save_model_on_all_epochs:
            save_model(model, os.path.join(outputs_dir, model_dir, run_name), f'epoch_{epoch}.pth')
            if training_method == 'unsupervised':
                save_model(prediction_model, os.path.join(outputs_dir, model_dir, run_name), f'prediction_epoch_{epoch}.pth')
    if save_model_on_last_epoch:
        save_model(model, os.path.join(outputs_dir, model_dir, run_name), f'final_epoch.pth')
        if training_method == 'unsupervised':
            save_model(prediction_model, os.path.join(outputs_dir, model_dir, run_name), 'final_epoch_prediction_model.pth')
    if not os.path.isfile(outputs_dir + model_dir + 'model_info.csv'):
        columns = pd.DataFrame.from_dict({'run_name': [], 'test_name': [],
                                           'training_date': [], 'training_method': [],
                                           'max_epochs': [], 'learning_rate': [],
                                           'best_epoch': [],
                                           'max_val_iou': [], 'final_epoch': []})
        columns.to_csv(outputs_dir + model_dir + 'model_info.csv', index=False, na_rep='N/A')
    models_info = pd.read_csv(outputs_dir + model_dir + 'model_info.csv')
    new_info = pd.DataFrame.from_dict({'run_name':[run_name], 'test_name':[test_name],
                             'training_date': [str(training_date)], 'training_method': [training_method],
                             'max_epochs':[n_epochs], 'learning_rate':[learning_rate],
                             'best_epoch':[best_epoch],
                             'max_val_iou':[max_score], 'final_epoch':[epoch]})
    updated_info = pd.concat([models_info, new_info], join='outer')
    updated_info.to_csv(outputs_dir + model_dir + 'model_info.csv', index=False, na_rep='N/A')
    wandb.finish()


def intersection_over_union(y_pred, y_true):

    smooth = 1e-6
    y_pred = y_pred[:, 0].view(-1) > 0.5
    y_true = y_true[:, 0].view(-1) > 0.5
    intersection = (y_pred & y_true).sum() + smooth
    union = (y_pred | y_true).sum() + smooth
    iou = intersection / union

    return iou




def main():
    if __name__ == '__main__':
        full_cycle()