import json
import os
import sys
import time
import matplotlib.pyplot as plt
import wandb
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
from networks.vision_transformer import SwinUnet as ViT_seg
from loss_functions import loss_function_factory
from model import model_factory
from wetlands import utils, map_wetlands, viz_utils, complete_cycle
from wetlands.jaccard_similarity import calculate_intersection_over_union
from skimage import io
# torch.set_float32_matmul_precision("high")


class CFDDataset_in_memory(Dataset):
    def __init__(self, config, dataset, pre_2020):
        self.dataset = dataset
        self.num_images = self.dataset.shape[0]
        self.patch_size = int(config['PATCH_SIZE'])
        self.training_method = config['TRAINING_METHOD']
        if pre_2020:
            prefix = 'PRE'
        else:
            prefix = 'POST'
        data_dir = config['DATA_DIR']
        self.sar_dir = data_dir + config[f'{prefix}_20_SAR_DIR']
        if self.training_method != 'multitemporal_data':
            self.sar_images = np.zeros((self.num_images, 1, self.patch_size, self.patch_size), np.float32)
            self.sar_current_image_index = 0
        else:
            self.sar_images = np.zeros((self.num_images, 3, self.patch_size, self.patch_size), np.float32)
            self.sar_current_image_index = 1
        self.masks_dir = data_dir + config[f'{prefix}_20_MASK_DIR']
        self.mask_images = np.zeros((self.num_images, 1, self.patch_size, self.patch_size), np.float32)
        self.mask_type = config['MASK_TYPE']
        self.num_dates = int(config['TEMPORAL_CONSISTENCY_NUM_DATES'])
        if not self.training_method == 'standard':
            self.past_sar_dir = data_dir + config[f'{prefix}_20_PAST_SAR_DIR']
            self.future_sar_dir = data_dir + config[f'{prefix}_20_FUTURE_SAR_DIR']
            if self.training_method == 'temporal_consistency':
                self.past_sar_images = np.zeros((self.num_images, 1, self.patch_size, self.patch_size), np.float32)
                self.future_sar_images = np.zeros((self.num_images, 1, self.patch_size, self.patch_size), np.float32)
                if self.num_dates > 1:
                    self.past_images2_dir = data_dir + config[f'{prefix}_20_PAST_SAR2_DIR']
                    self.past_sar_images2 = np.zeros((self.num_images, 1, self.patch_size, self.patch_size), np.float32)
                    self.future_images2_dir = data_dir + config[f'{prefix}_20_FUTURE_SAR2_DIR']
                    self.future_sar_images2 = np.zeros((self.num_images, 1, self.patch_size, self.patch_size), np.float32)
        for i in range(self.num_images):
            index_ = self.dataset.iloc[i]['id']
            sar_path = self.sar_dir + str(index_) + '-sar.tif'
            mask_path = self.masks_dir + str(index_) + f'-{self.mask_type}_mask.tif'
            if not self.training_method == 'standard':
                past_sar_path = self.past_sar_dir + str(index_) + '-sar.tif'
                future_sar_path = self.future_sar_dir + str(index_) + '-sar.tif'
                if self.training_method == 'temporal_consistency' and self.num_dates > 1:
                    past_sar2_path = self.past_images2_dir + str(index_) + '-sar.tif'
                    future_sar2_path = self.future_images2_dir + str(index_) + '-sar.tif'

            # Read image
            self.sar_images[i][self.sar_current_image_index] = io.imread(sar_path)

            # Read image
            self.mask_images[i][0] = io.imread(mask_path)
            if self.training_method == 'temporal_consistency':
                self.past_sar_images[i][0] = io.imread(past_sar_path)
                self.future_sar_images[i][0] = io.imread(future_sar_path)
                if self.num_dates > 1:
                    self.past_sar_images2[i][0] = io.imread(past_sar2_path)
                    self.future_sar_images2[i][0] = io.imread(future_sar2_path)
            elif self.training_method == 'multitemporal_data':
                self.sar_images[i][0] = io.imread(past_sar_path)
                self.sar_images[i][2] = io.imread(future_sar_path)

        # Convert to Pytorch tensor
        self.sar_images = torch.from_numpy(self.sar_images)
        self.mask_images = torch.from_numpy(self.mask_images)
        if self.training_method == 'temporal_consistency':
            self.past_sar_images = torch.from_numpy(self.past_sar_images)
            self.future_sar_images = torch.from_numpy(self.future_sar_images)
            if self.num_dates > 1:
                self.past_sar_images2 = torch.from_numpy(self.past_sar_images2)
                self.future_sar_images2 = torch.from_numpy(self.future_sar_images2)

    def __getitem__(self, index):
        if self.training_method == 'standard' or self.training_method == 'multitemporal_data':
            return self.sar_images[index], self.mask_images[index]
        elif self.num_dates == 1:
            return self.sar_images[index], self.mask_images[index], self.past_sar_images[index], self.future_sar_images[index]
        elif self.num_dates == 2:
            return self.sar_images[index], self.mask_images[index], self.past_sar_images[index], self.future_sar_images[index], \
                self.past_sar_images2[index], self.future_sar_images2[index]

    def __len__(self):
        return len(self.dataset)


def get_dataloaders(config, data, pre_2020):
    datasets = {'train': CFDDataset_in_memory(config, data[data.split == 'train'], pre_2020),
                'valid': CFDDataset_in_memory(config, data[data.split == 'valid'], pre_2020)}
    print('Num train data:', len(datasets['train']))
    print('Num val data:', len(datasets['valid']))
    batch_size = config['BATCH_SIZE']
    num_workers = config['NUM_WORKERS']
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


def train(model, dataloader, criterion, optimizer, device):
    model.train(True)
    losses = []
    ious = []
    for input, target in dataloader:
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            output = model(input)
            loss = criterion(output, torch.squeeze(target, 1), softmax=True, ignore_class_zero=True)

            loss.backward()
            optimizer.step()

            iou = intersection_over_union(output, target)
            losses.append(loss.cpu().detach().numpy())
            ious.append(iou.cpu().detach().numpy())

    train_loss = np.mean(losses)
    train_iou = np.mean(ious)
    print('Train loss', train_loss, 'Train IOU', train_iou)

    metrics = {
        'train_loss': train_loss,
        'train_iou': train_iou,
    }

    return metrics


def evaluate(model, dataloader, scheduler, criterion, device):
    model.eval()
    losses = []
    ious = []

    for input, target in dataloader:
        input = input.to(device)
        target = target.to(device)

        with torch.set_grad_enabled(False):
            output = model(input)
            loss = criterion(output, torch.squeeze(target, 1), softmax=True, ignore_class_zero=True)
            iou = intersection_over_union(output, target)
            losses.append(loss.cpu().detach().numpy())
            ious.append(iou.cpu().detach().numpy())

    val_loss = np.mean(losses)
    val_iou = np.mean(ious)
    print('Val loss', val_loss, 'Val IOU', val_iou)
    if scheduler is not None:
        scheduler.step(val_iou)
    metrics = {
        'val_loss': val_loss,
        'val_iou': val_iou,
    }

    return metrics


def train_temporal_consistency(config, model, dataloader, criterion, optimizer, device, epoch):
    model.train(True)
    losses = []
    ious = []
    base_losses = []
    past_consistency_losses = []
    future_consistency_losses = []
    past_consistency_losses2 = []
    future_consistency_losses2 = []
    past_ious = []
    future_ious = []
    past_ious2 = []
    future_ious2 = []
    temporal_consistency_metric = config['TEMPORAL_CONSISTENCY_METRIC']
    temporal_consistency_scaling = config['TEMPORAL_CONSISTENCY_SCALING']
    if temporal_consistency_metric == 'iou':
        temporal_consistency_criterion = loss_function_factory.create_loss_function('dice_swin')
    elif temporal_consistency_metric == 'absolute_difference':
        temporal_consistency_criterion = loss_function_factory.create_loss_function('abs_dif')
    temporal_consistency_weight = config['TEMPORAL_CONSISTENCY_WEIGHT']
    standard_training_weight = config['STANDARD_TRAINING_WEIGHT']
    temporal_consistency_starting_epoch = config['TEMPORAL_CONSISTENCY_START_EPOCH']
    temporal_consistency_num_dates = config['TEMPORAL_CONSISTENCY_NUM_DATES']
    temporal_consistency_power = config['TEMPORAL_CONSISTENCY_POWER']

    if temporal_consistency_num_dates == 1:
        for input, target, past_sar, future_sar in dataloader:
            input = input.to(device)
            target = target.to(device)
            past_sar = past_sar.to(device)
            future_sar = future_sar.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output = model(input)
                past_output = model(past_sar)
                future_output = model(future_sar)
                loss = criterion(output, torch.squeeze(target, 1), softmax=True, ignore_class_zero=True)
                if temporal_consistency_scaling == 'sq_dif':
                    past_square_difs = torch.mean(torch.square(input - past_sar), (1, 2, 3))
                    past_weights = (1 - past_square_difs)#/torch.mean(1 - past_square_difs)
                    future_square_difs = torch.mean(torch.square(input - future_sar), (1, 2, 3))
                    future_weights = (1 - future_square_difs)# / torch.mean(1 - future_square_difs)
                    past_consistency_loss = temporal_consistency_criterion.comparison(past_output, output, batch_weights=past_weights, softmax=True, ignore_class_zero=True)
                    future_consistency_loss = temporal_consistency_criterion.comparison(future_output, output, batch_weights=future_weights, softmax=True, ignore_class_zero=True)
                elif temporal_consistency_scaling == 'none':
                    past_consistency_loss = temporal_consistency_criterion.comparison(past_output, output, softmax=True, ignore_class_zero=True)
                    future_consistency_loss = temporal_consistency_criterion.comparison(future_output, output, softmax=True, ignore_class_zero=True)
                if epoch < temporal_consistency_starting_epoch:
                    current_temporal_consistency_weight = 0.
                    current_standard_training_weight = 1.
                else:
                    current_temporal_consistency_weight = temporal_consistency_weight
                    current_standard_training_weight = standard_training_weight
                total_loss = current_standard_training_weight * loss + current_temporal_consistency_weight * (torch.pow(past_consistency_loss,temporal_consistency_power) + torch.pow(future_consistency_loss,temporal_consistency_power))

                total_loss.backward()
                optimizer.step()

                iou = intersection_over_union(output, target)
                past_iou = intersection_over_union(past_output, output, True)
                future_iou = intersection_over_union(future_output, output, True)
                losses.append(total_loss.cpu().detach().numpy())
                base_losses.append(loss.cpu().detach().numpy())
                past_consistency_losses.append(past_consistency_loss.cpu().detach().numpy())
                future_consistency_losses.append(future_consistency_loss.cpu().detach().numpy())
                ious.append(iou.cpu().detach().numpy())
                past_ious.append(past_iou.cpu().detach().numpy())
                future_ious.append(future_iou.cpu().detach().numpy())
    elif temporal_consistency_num_dates == 2:
        for input, target, past_sar, future_sar, past_sar2, future_sar2 in dataloader:
            input = input.to(device)
            target = target.to(device)
            past_sar = past_sar.to(device)
            future_sar = future_sar.to(device)
            past_sar2 = past_sar2.to(device)
            future_sar2 = future_sar2.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output = model(input)
                past_output = model(past_sar)
                future_output = model(future_sar)
                past_output2 = model(past_sar2)
                future_output2 = model(future_sar2)
                loss = criterion(output, torch.squeeze(target, 1), softmax=True, ignore_class_zero=True)
                if temporal_consistency_scaling == 'sq_dif':
                    past_square_difs = torch.mean(torch.square(input - past_sar), (1, 2, 3))
                    past_weights = (1 - past_square_difs)#/torch.mean(1 - past_square_difs)
                    past2_square_difs = torch.mean(torch.square(past_sar - past_sar2), (1, 2, 3))
                    past2_weights = (1 - past2_square_difs)# / torch.mean(1 - past2_square_difs)
                    future_square_difs = torch.mean(torch.square(input - future_sar), (1, 2, 3))
                    future_weights = (1 - future_square_difs)# / torch.mean(1 - future_square_difs)
                    future2_square_difs = torch.mean(torch.square(future_sar - future_sar2), (1, 2, 3))
                    future2_weights = (1 - future2_square_difs)# / torch.mean(1 - future2_square_difs)
                    past_consistency_loss = temporal_consistency_criterion.comparison(past_output, output, batch_weights=past_weights, softmax=True, ignore_class_zero=True)
                    future_consistency_loss = temporal_consistency_criterion.comparison(future_output, output, batch_weights=future_weights, softmax=True, ignore_class_zero=True)
                    past_consistency_loss2 = temporal_consistency_criterion.comparison(past_output2, past_output, batch_weights=past2_weights, softmax=True, ignore_class_zero=True)
                    future_consistency_loss2 = temporal_consistency_criterion.comparison(future_output2, future_output, batch_weights=future2_weights, softmax=True, ignore_class_zero=True)
                elif temporal_consistency_scaling == 'none':
                    past_consistency_loss = temporal_consistency_criterion.comparison(past_output, output, softmax=True, ignore_class_zero=True)
                    future_consistency_loss = temporal_consistency_criterion.comparison(future_output, output, softmax=True, ignore_class_zero=True)
                    past_consistency_loss2 = temporal_consistency_criterion.comparison(past_output2, past_output, softmax=True, ignore_class_zero=True)
                    future_consistency_loss2 = temporal_consistency_criterion.comparison(future_output2, future_output, softmax=True, ignore_class_zero=True)
                if epoch < temporal_consistency_starting_epoch:
                    current_temporal_consistency_weight = 0.
                    current_standard_training_weight = 1.
                else:
                    current_temporal_consistency_weight = temporal_consistency_weight
                    current_standard_training_weight = standard_training_weight
                total_loss = current_standard_training_weight * loss + current_temporal_consistency_weight * (
                            torch.pow(past_consistency_loss,temporal_consistency_power) + torch.pow(future_consistency_loss,temporal_consistency_power)
                            + torch.pow(past_consistency_loss2,temporal_consistency_power) + torch.pow(future_consistency_loss2,temporal_consistency_power))

                total_loss.backward()
                optimizer.step()

                iou = intersection_over_union(output, target)
                past_iou = intersection_over_union(past_output, output, True)
                future_iou = intersection_over_union(future_output, output, True)
                past_iou2 = intersection_over_union(past_output2, past_output, True)
                future_iou2 = intersection_over_union(future_output2, future_output, True)
                losses.append(total_loss.cpu().detach().numpy())
                base_losses.append(loss.cpu().detach().numpy())
                past_consistency_losses.append(past_consistency_loss.cpu().detach().numpy())
                future_consistency_losses.append(future_consistency_loss.cpu().detach().numpy())
                past_consistency_losses2.append(past_consistency_loss2.cpu().detach().numpy())
                future_consistency_losses2.append(future_consistency_loss2.cpu().detach().numpy())
                ious.append(iou.cpu().detach().numpy())
                past_ious.append(past_iou.cpu().detach().numpy())
                future_ious.append(future_iou.cpu().detach().numpy())
                past_ious2.append(past_iou2.cpu().detach().numpy())
                future_ious2.append(future_iou2.cpu().detach().numpy())

    train_loss = np.mean(losses)
    train_base_loss = np.mean(base_losses)
    train_past_consistency_loss = np.mean(past_consistency_losses)
    train_future_consistency_loss = np.mean(future_consistency_losses)
    train_iou = np.mean(ious)
    train_past_iou = np.mean(past_ious)
    train_future_iou = np.mean(future_ious)
    if temporal_consistency_num_dates > 1:
        train_past_consistency_loss2 = np.mean(past_consistency_losses2)
        train_future_consistency_loss2 = np.mean(future_consistency_losses2)
        train_past_iou2 = np.mean(past_ious2)
        train_future_iou2 = np.mean(future_ious2)
    print('Train loss', train_loss, 'Train IOU', train_iou)

    metrics = {
        'train_loss': train_loss,
        'train_base_loss': train_base_loss,
        'train_past_consistency_loss': train_past_consistency_loss,
        'train_future_consistency_loss': train_future_consistency_loss,
        'train_iou': train_iou,
        'train_past_iou': train_past_iou,
        'train_future_iou': train_future_iou,
    }
    if temporal_consistency_num_dates > 1:
        metrics['train_past_consistency_loss2'] = train_past_consistency_loss2
        metrics['train_future_consistency_loss2'] = train_future_consistency_loss2
        metrics['train_past_iou2'] = train_past_iou2
        metrics['train_future_iou2'] = train_future_iou2

    return metrics

def evaluate_temporal_consistency(config, model, dataloader, criterion, scheduler, device, epoch):
    model.eval()
    losses = []
    ious = []
    base_losses = []
    past_consistency_losses = []
    future_consistency_losses = []
    past_consistency_losses2 = []
    future_consistency_losses2 = []
    past_ious = []
    future_ious = []
    past_ious2 = []
    future_ious2 = []
    temporal_consistency_metric = config['TEMPORAL_CONSISTENCY_METRIC']
    temporal_consistency_scaling = config['TEMPORAL_CONSISTENCY_SCALING']
    if temporal_consistency_metric == 'iou':
        temporal_consistency_criterion = loss_function_factory.create_loss_function('dice_swin')
    elif temporal_consistency_metric == 'absolute_difference':
        temporal_consistency_criterion = loss_function_factory.create_loss_function('abs_dif')
    temporal_consistency_weight = config['TEMPORAL_CONSISTENCY_WEIGHT']
    standard_training_weight = config['STANDARD_TRAINING_WEIGHT']
    temporal_consistency_starting_epoch = config['TEMPORAL_CONSISTENCY_START_EPOCH']
    temporal_consistency_num_dates = config['TEMPORAL_CONSISTENCY_NUM_DATES']
    temporal_consistency_power = config['TEMPORAL_CONSISTENCY_POWER']

    if temporal_consistency_num_dates == 1:
        for input, target, past_sar, future_sar in dataloader:
            input = input.to(device)
            target = target.to(device)
            past_sar = past_sar.to(device)
            future_sar = future_sar.to(device)

            with torch.set_grad_enabled(False):
                output = model(input)
                past_output = model(past_sar)
                future_output = model(future_sar)
                loss = criterion(output, torch.squeeze(target, 1), softmax=True, ignore_class_zero=True)
                if temporal_consistency_scaling == 'sq_dif':
                    past_square_difs = torch.mean(torch.square(input - past_sar), (1, 2, 3))
                    past_weights = (1 - past_square_difs)#/torch.mean(1 - past_square_difs)
                    future_square_difs = torch.mean(torch.square(input - future_sar), (1, 2, 3))
                    future_weights = (1 - future_square_difs)# / torch.mean(1 - future_square_difs)
                    past_consistency_loss = temporal_consistency_criterion.comparison(past_output, output, batch_weights=past_weights, softmax=True, ignore_class_zero=True)
                    future_consistency_loss = temporal_consistency_criterion.comparison(future_output, output, batch_weights=future_weights, softmax=True, ignore_class_zero=True)
                elif temporal_consistency_scaling == 'none':
                    past_consistency_loss = temporal_consistency_criterion.comparison(past_output, output, softmax=True, ignore_class_zero=True)
                    future_consistency_loss = temporal_consistency_criterion.comparison(future_output, output, softmax=True, ignore_class_zero=True)
                if epoch < temporal_consistency_starting_epoch:
                    current_temporal_consistency_weight = 0.
                    current_standard_training_weight = 1.
                else:
                    current_temporal_consistency_weight = temporal_consistency_weight
                    current_standard_training_weight = standard_training_weight
                total_loss = current_standard_training_weight * loss + current_temporal_consistency_weight * (torch.pow(past_consistency_loss,temporal_consistency_power) + torch.pow(future_consistency_loss,temporal_consistency_power))

                iou = intersection_over_union(output, target)
                past_iou = intersection_over_union(past_output, output, True)
                future_iou = intersection_over_union(future_output, output, True)
                losses.append(total_loss.cpu().detach().numpy())
                base_losses.append(loss.cpu().detach().numpy())
                past_consistency_losses.append(past_consistency_loss.cpu().detach().numpy())
                future_consistency_losses.append(future_consistency_loss.cpu().detach().numpy())
                ious.append(iou.cpu().detach().numpy())
                past_ious.append(past_iou.cpu().detach().numpy())
                future_ious.append(future_iou.cpu().detach().numpy())
    elif temporal_consistency_num_dates == 2:
        for input, target, past_sar, future_sar, past_sar2, future_sar2 in dataloader:
            input = input.to(device)
            target = target.to(device)
            past_sar = past_sar.to(device)
            future_sar = future_sar.to(device)
            past_sar2 = past_sar2.to(device)
            future_sar2 = future_sar2.to(device)

            with torch.set_grad_enabled(False):
                output = model(input)
                past_output = model(past_sar)
                future_output = model(future_sar)
                past_output2 = model(past_sar2)
                future_output2 = model(future_sar2)
                loss = criterion(output, torch.squeeze(target, 1), softmax=True, ignore_class_zero=True)
                if temporal_consistency_scaling == 'sq_dif':
                    past_square_difs = torch.mean(torch.square(input - past_sar), (1, 2, 3))
                    past_weights = (1 - past_square_difs)#/torch.mean(1 - past_square_difs)
                    past2_square_difs = torch.mean(torch.square(past_sar - past_sar2), (1, 2, 3))
                    past2_weights = (1 - past2_square_difs)# / torch.mean(1 - past2_square_difs)
                    future_square_difs = torch.mean(torch.square(input - future_sar), (1, 2, 3))
                    future_weights = (1 - future_square_difs)# / torch.mean(1 - future_square_difs)
                    future2_square_difs = torch.mean(torch.square(future_sar - future_sar2), (1, 2, 3))
                    future2_weights = (1 - future2_square_difs)# / torch.mean(1 - future2_square_difs)
                    past_consistency_loss = temporal_consistency_criterion.comparison(past_output, output, batch_weights=past_weights, softmax=True, ignore_class_zero=True)
                    future_consistency_loss = temporal_consistency_criterion.comparison(future_output, output, batch_weights=future_weights, softmax=True, ignore_class_zero=True)
                    past_consistency_loss2 = temporal_consistency_criterion.comparison(past_output2, past_output, batch_weights=past2_weights, softmax=True, ignore_class_zero=True)
                    future_consistency_loss2 = temporal_consistency_criterion.comparison(future_output2, future_output, batch_weights=future2_weights, softmax=True, ignore_class_zero=True)
                elif temporal_consistency_scaling == 'none':
                    past_consistency_loss = temporal_consistency_criterion.comparison(past_output, output, softmax=True, ignore_class_zero=True)
                    future_consistency_loss = temporal_consistency_criterion.comparison(future_output, output, softmax=True, ignore_class_zero=True)
                    past_consistency_loss2 = temporal_consistency_criterion.comparison(past_output2, past_output, softmax=True, ignore_class_zero=True)
                    future_consistency_loss2 = temporal_consistency_criterion.comparison(future_output2, future_output, softmax=True, ignore_class_zero=True)
                if epoch < temporal_consistency_starting_epoch:
                    current_temporal_consistency_weight = 0.
                    current_standard_training_weight = 1.
                else:
                    current_temporal_consistency_weight = temporal_consistency_weight
                    current_standard_training_weight = standard_training_weight
                total_loss = current_standard_training_weight * loss + current_temporal_consistency_weight * (
                            torch.pow(past_consistency_loss,temporal_consistency_power) + torch.pow(future_consistency_loss,temporal_consistency_power)
                            + torch.pow(past_consistency_loss2,temporal_consistency_power) + torch.pow(future_consistency_loss2,temporal_consistency_power))

                iou = intersection_over_union(output, target)
                past_iou = intersection_over_union(past_output, output, True)
                future_iou = intersection_over_union(future_output, output, True)
                past_iou2 = intersection_over_union(past_output2, past_output, True)
                future_iou2 = intersection_over_union(future_output2, future_output, True)
                losses.append(total_loss.cpu().detach().numpy())
                base_losses.append(loss.cpu().detach().numpy())
                past_consistency_losses.append(past_consistency_loss.cpu().detach().numpy())
                future_consistency_losses.append(future_consistency_loss.cpu().detach().numpy())
                past_consistency_losses2.append(past_consistency_loss2.cpu().detach().numpy())
                future_consistency_losses2.append(future_consistency_loss2.cpu().detach().numpy())
                ious.append(iou.cpu().detach().numpy())
                past_ious.append(past_iou.cpu().detach().numpy())
                future_ious.append(future_iou.cpu().detach().numpy())
                past_ious2.append(past_iou2.cpu().detach().numpy())
                future_ious2.append(future_iou2.cpu().detach().numpy())

    val_loss = np.mean(losses)
    val_base_loss = np.mean(base_losses)
    val_past_consistency_loss = np.mean(past_consistency_losses)
    val_future_consistency_loss = np.mean(future_consistency_losses)
    val_iou = np.mean(ious)
    val_past_iou = np.mean(past_ious)
    val_future_iou = np.mean(future_ious)
    if temporal_consistency_num_dates > 1:
        val_past_consistency_loss2 = np.mean(past_consistency_losses2)
        val_future_consistency_loss2 = np.mean(future_consistency_losses2)
        val_past_iou2 = np.mean(past_ious2)
        val_future_iou2 = np.mean(future_ious2)
    print('Val loss', val_loss, 'Val IOU', val_iou)
    # current_lr = scheduler.get_last_lr()
    if scheduler is not None:
        scheduler.step(val_iou)

    metrics = {
        'val_loss': val_loss,
        'val_base_loss': val_base_loss,
        'val_past_consistency_loss': val_past_consistency_loss,
        'val_future_consistency_loss': val_future_consistency_loss,
        'val_iou': val_iou,
        'val_past_iou': val_past_iou,
        'val_future_iou': val_future_iou,
        # 'lr': current_lr
    }
    if temporal_consistency_num_dates > 1:
        metrics['val_past_consistency_loss2'] = val_past_consistency_loss2
        metrics['val_future_consistency_loss2'] = val_future_consistency_loss2
        metrics['val_past_iou2'] = val_past_iou2
        metrics['val_future_iou2'] = val_future_iou2

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


def full_cycle(config, test_name, pre_2020=True):
    # Configure the wandb run
    wandb.login(key='###')
    wandb_config = config.copy()
    del wandb_config['AGGREGATE_FUNCTION']
    del wandb_config['ANNOTATED_DATA_DIR']
    del wandb_config['CHARTS_DIR']
    del wandb_config['CLOUDY_PIXEL_PERCENTAGE']
    del wandb_config['COUNTRY_CODE']
    del wandb_config['DATA_DIR']
    del wandb_config['EVALUATION_DIR']
    del wandb_config['GEOJSON_FILE']
    del wandb_config['MODELS_DIR']
    del wandb_config['NDWI_INPUT']
    del wandb_config['ORBIT_PASS']
    del wandb_config['OTSU_GAUSSIAN_KERNEL_SIZE']
    del wandb_config['REGION_ADMIN_LEVEL']
    del wandb_config['REGION_NAME']
    del wandb_config['RESULTS_DIR']
    del wandb_config['OUTPUTS_DIR']
    del wandb_config['NDWI_MASK_TILES_DIR']
    del wandb_config['SAR_TILES_DIR']
    del wandb_config['SAR_DIR']
    del wandb_config['SAR_POLARIZATION']
    del wandb_config['STUDY_AREA']
    del wandb_config['WATER_INDEX']
    wandb.init(project="deepaqua", config=wandb_config, name=test_name)
    print(json.dumps(config, indent=4))
    run_name = wandb.run.name
    wandb.run.define_metric("val_iou", summary="max")
    wandb.run.define_metric("val_loss", summary="min")
    wandb.run.define_metric("train_iou", summary="max")
    wandb.run.define_metric("train_loss", summary="min")

    training_method = config['TRAINING_METHOD']
    n_epochs = config['EPOCHS']
    learning_rate = config['LEARNING_RATE']
    seed = config['RANDOM_SEED']
    if seed != 'NONE':
        seed = int(seed)
    model_dir = config['MODELS_DIR']
    outputs_dir = config['OUTPUTS_DIR']
    if config['SAVE_MODEL_ON_ALL_EPOCHS'] == 'True':
        save_model_on_all_epochs = True
    else:
        save_model_on_all_epochs = False
    if config['SAVE_MODEL_ON_LAST_EPOCH'] == 'True':
        save_model_on_last_epoch = True
    else:
        save_model_on_last_epoch = False
    if os.getenv('REDUCE_LR_PLATEAU') == 'True':
        reduce_lr_plateau = True
    else:
        reduce_lr_plateau = False

    early_stop_num_epochs = config['EARLY_STOP_NUM_EPOCHS']
    temporal_consistency_start_epoch = config['TEMPORAL_CONSISTENCY_START_EPOCH']
    num_dates = config['TEMPORAL_CONSISTENCY_NUM_DATES']
    temporal_consistency_weight = config['TEMPORAL_CONSISTENCY_WEIGHT']
    standard_training_weight = config['STANDARD_TRAINING_WEIGHT']
    temporal_consistency_power = config['TEMPORAL_CONSISTENCY_POWER']
    reduce_lr_plateau_patience = config['REDUCE_LR_PLATEAU_PATIENCE']
    mask_type = config['MASK_TYPE']

    if seed != 'NONE':
        utils.plant_random_seed(seed)

    tiles_data = utils.create_tiles_file_pipeline(config, pre_2020)
    if pre_2020:
        training_date = config['PRE_20_TRAIN_DATE']
    else:
        training_date = config['POST_20_TRAIN_DATE']

    tiff_file = config['DATA_DIR'] + config['SINGLE_TEST_FILE']
    tiff_image = viz_utils.load_image(tiff_file,  ignore_nan=True, skimage_read=False)

    # Check is GPU is enabled
    device = utils.get_device()

    dataloaders = get_dataloaders(config, tiles_data, pre_2020)
    # print('Data load time:'+ str(time.time()-load_time_start))
    # return
    if training_method == 'multitemporal_data':
        num_input_channels = 3
    else:
        num_input_channels = 1
    model = ViT_seg(img_size=int(config['PATCH_SIZE']), num_classes=2, patch_size=int(config['TRANSFORMER_PATCH_SIZE']),
                    input_channels=num_input_channels, embed_dim=int(config['EMBED_DIM']), depths=config['DEPTHS'],
                    num_heads=config['NUM_HEADS'], window_size=config['WINDOW_SIZE'], mlp_ratio=config['MLP_RATIO'],
                    qkv_bias=config['QKV_BIAS'], qk_scale=config['QK_SKALE'], drop_rate=config['DROP_RATE'],
                    drop_path_rate=config['DROP_PATH_RATE'], ape=config['APE'], patch_norm=config['PATCH_NORM'],
                    use_checkpoint=config['USE_CHECKPOINT']).cuda()
    print(model)

    print('Model parameters', sum(param.numel() for param in model.parameters()))
    criterion = loss_function_factory.create_loss_function('dice_swin')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if reduce_lr_plateau and reduce_lr_plateau_patience >=0:
        reduce_lr_plateau_factor = config['REDUCE_LR_PLATEAU_FACTOR']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=reduce_lr_plateau_patience, factor=reduce_lr_plateau_factor)
    else:
        scheduler = None
        reduce_lr_plateau_factor = 'N/A'

    model = torch.nn.DataParallel(model)
    model.to(device)
    max_score = 0
    val_ious = []
    best_epoch = 0
    for epoch in range(1, n_epochs + 1):
        # if epoch == 10:
        #     complete_cycle.send_slack_message('Epoch 30')
        #     pause_var=0
        print("\nEpoch {}/{} {}".format(epoch, n_epochs, time.strftime("%Y/%m/%d-%H:%M:%S")))
        print("-" * 10)
        if training_method == 'standard' or training_method == 'multitemporal_data':
            train_metrics = train(
                model,
                dataloaders["train"],
                criterion,
                optimizer,
                device
            )
            val_metrics = evaluate(
                model,
                dataloaders['valid'],
                scheduler,
                criterion,
                device
            )
        elif training_method == 'temporal_consistency':
            train_metrics = train_temporal_consistency(config, model,
                dataloaders["train"],
                criterion,
                optimizer,
                device,
                epoch
            )
            val_metrics = evaluate_temporal_consistency(config, model,
                dataloaders['valid'],
                criterion,
                scheduler,
                device,
                epoch
            )
        class_labels = {
            0: "land",
            1: "water",
        }

        #TEMP
        # pred_mask = map_wetlands.predict_water_mask(config, tiff_image, model, device)
        #
        # full_mask_img = wandb.Image(tiff_image, masks={
        #     "predictions": {
        #         "mask_data": pred_mask,
        #         "class_labels": class_labels
        #     },
        # }, caption=["Full water detection", "fwd", "fwdm"])

        # Count values of full_pred array

        # full_pred = wandb.Image(pred_mask, caption="Full prediction")

        metrics = {
            **train_metrics, **val_metrics,# 'full_pred': full_pred, 'full_mask': full_mask_img
        }

        print('Train loss: {}, Val loss: {}'.format(metrics['train_loss'], metrics['val_loss']))
        wandb.log(metrics)

        val_ious.append(metrics['val_iou'])
        if metrics['val_iou'] > max_score:
            max_score = metrics['val_iou']
            best_epoch = epoch
            print(f'New best model found on epoch {epoch}. Validation IoU: {max_score}')
            save_model(model, os.path.join(outputs_dir, model_dir, run_name), 'best_model.pth')
        if save_model_on_all_epochs:
            save_model(model, os.path.join(outputs_dir, model_dir, run_name), f'epoch_{epoch}.pth')
        stop_training = False
        if early_stop_num_epochs > 0:
            if training_method != 'temporal_consistency' or temporal_consistency_weight == 0.:
                if len(val_ious) > early_stop_num_epochs and np.max(val_ious[-early_stop_num_epochs:]) < np.max(val_ious):
                    stop_training = True
            elif len(val_ious) >= early_stop_num_epochs + temporal_consistency_start_epoch and np.max(val_ious[-early_stop_num_epochs:]) < np.max(val_ious):
                stop_training = True
        if stop_training:
            break
    if save_model_on_last_epoch:
        save_model(model, os.path.join(outputs_dir, model_dir, run_name), f'final_epoch.pth')
    # print(outputs_dir + model_dir + 'model_info.csv')
    models_info = pd.read_csv(outputs_dir + model_dir + 'model_info.csv')
    model_index = models_info.iloc[-1]['model_index'] + 1
    new_info = pd.DataFrame.from_dict({'model_index': [model_index], 'run_name':[run_name], 'test_name':[test_name],
                             'training_date': [str(training_date)], 'training_method': [training_method],
                             'num_dates':[num_dates], 'temporal_consistency_weight': [temporal_consistency_weight],
                             'standard_training_weight': [standard_training_weight],
                             'temporal_consistency_power':[temporal_consistency_power],
                             'temporal_consistency_start_epoch': [temporal_consistency_start_epoch],
                             'max_epochs':[n_epochs], 'learning_rate':[learning_rate],
                             'early_stop_num_epochs':[early_stop_num_epochs], 'best_epoch':[best_epoch],
                             'max_val_iou':[max_score], 'final_epoch':[epoch], 'mask_type':[mask_type],
                             'reduce_lr_plateau_patience':[reduce_lr_plateau_patience], 'reduce_lr_plateau_factor':[reduce_lr_plateau_factor]})
    updated_info = pd.concat([models_info, new_info], join='outer')
    updated_info.to_csv(outputs_dir + model_dir + 'model_info.csv', index=False, na_rep='N/A')
    wandb.finish()


def intersection_over_union(y_pred, y_true, compare_outputs=False):

    smooth = 1e-6
    # y_pred = y_pred[:, 0].view(-1) > 0.5
    # y_true = y_true[:, 0].view(-1) > 0.5
    y_pred = torch.argmax(y_pred, dim=1)
    if not compare_outputs:
        y_true = torch.squeeze(y_true.to(torch.int))
    else:
        y_true = torch.argmax(y_true, dim=1)
    intersection = (y_pred & y_true).sum() + smooth
    union = (y_pred | y_true).sum() + smooth
    iou = intersection / union

    return iou


def main():
    if __name__ == '__main__':
        full_cycle()
