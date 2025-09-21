import json
import os
import time
import matplotlib.pyplot as plt
import wandb
from dotenv import load_dotenv, dotenv_values
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
from loss_functions import loss_function_factory
from model import model_factory
from wetlands import utils, map_wetlands, viz_utils
from wetlands.jaccard_similarity import calculate_intersection_over_union
from skimage import io
import csv

rng = np.random.default_rng()
torch.set_float32_matmul_precision("high")

class CFDDataset(Dataset):
    def __init__(self, dataset, images_dir, masks_dir, past_images_dir=None, future_images_dir=None,
                 past_images2_dir=None, future_images2_dir=None):
        self.dataset = dataset
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        if past_images_dir is not None:
            self.past_images_dir = past_images_dir
            self.future_images_dir = future_images_dir
            if past_images2_dir is not None:
                self.past_images2_dir = past_images2_dir
                self.future_images2_dir = future_images2_dir
                self.num_dates = 2
            else:
                self.num_dates = 1
        else:
            self.num_dates = 0

    def __getitem__(self, index):
        index_ = self.dataset.iloc[index]['id']

        # Get image and mask file paths for specified index
        image_path = self.images_dir + str(index_) + '-sar.tif'
        mask_path = self.masks_dir + str(index_) + '-ndwi_mask.tif'
        if self.num_dates > 0:
            past_image_path = self.past_images_dir + str(index_) + '-sar.tif'
            future_image_path = self.future_images_dir + str(index_) + '-sar.tif'
            if self.num_dates > 1:
                past_image2_path = self.past_images2_dir + str(index_) + '-sar.tif'
                future_image2_path = self.future_images2_dir + str(index_) + '-sar.tif'

        # Read image
        image = rio.open(image_path).read()

        # Read image
        mask = rio.open(mask_path).read()
        if self.num_dates > 0:
            past_image = rio.open(past_image_path).read()
            future_image = rio.open(future_image_path).read()
            if self.num_dates > 1:
                past_image2 = rio.open(past_image2_path).read()
                future_image2 = rio.open(future_image2_path).read()
        # Convert to Pytorch tensor
        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        if self.num_dates > 0:
            past_image_tensor = torch.from_numpy(past_image.astype(np.float32))
            future_image_tensor = torch.from_numpy(future_image.astype(np.float32))
            if self.num_dates > 1:
                past_image2_tensor = torch.from_numpy(past_image2.astype(np.float32))
                future_image2_tensor = torch.from_numpy(future_image2.astype(np.float32))

        if self.num_dates == 0:
            return image_tensor, mask_tensor
        elif self.num_dates == 1:
            return image_tensor, mask_tensor, past_image_tensor, future_image_tensor
        elif self.num_dates == 2:
            return image_tensor, mask_tensor, past_image_tensor, future_image_tensor, past_image2_tensor, future_image2_tensor

    def __len__(self):
        return len(self.dataset)


class CFDDataset_in_memory(Dataset):
    def __init__(self, dataset, images_dir, masks_dir, past_images_dir=None, future_images_dir=None,
                 past_images2_dir=None, future_images2_dir=None):
        self.patch_size = int(os.getenv('PATCH_SIZE'))
        self.dataset = dataset
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        if '/ndwi_masks_tiles/' in masks_dir:
            self.mask_type = 'ndwi'
        elif '/otsu_masks_tiles/' in masks_dir:
            self.mask_type = 'otsu'
        if past_images_dir is not None:
            self.past_images_dir = past_images_dir
            self.future_images_dir = future_images_dir
            if past_images2_dir is not None:
                self.past_images2_dir = past_images2_dir
                self.future_images2_dir = future_images2_dir
                self.num_dates = 2
            else:
                self.num_dates = 1
        else:
            self.num_dates = 0
        self.num_images = self.dataset.shape[0]
        self.sar_images = np.zeros((self.num_images, 1, self.patch_size, self.patch_size), np.float32)
        self.ground_truth_masks = np.zeros((self.num_images, 1, self.patch_size, self.patch_size), np.float32)
        if self.num_dates > 0:
            self.past_sar_images = np.zeros((self.num_images, 1, self.patch_size, self.patch_size), np.float32)
            self.future_sar_images = np.zeros((self.num_images, 1, self.patch_size, self.patch_size), np.float32)
            if self.num_dates > 1:
                self.past_sar_images2 = np.zeros((self.num_images, 1, self.patch_size, self.patch_size), np.float32)
                self.future_sar_images2 = np.zeros((self.num_images, 1, self.patch_size, self.patch_size), np.float32)
        for i in range(self.num_images):
            index_ = self.dataset.iloc[i]['id']
            image_path = self.images_dir + str(index_) + '-sar.tif'
            mask_path = self.masks_dir + str(index_) + f'-{self.mask_type}_mask.tif'
            if self.num_dates > 0:
                past_image_path = self.past_images_dir + str(index_) + '-sar.tif'
                future_image_path = self.future_images_dir + str(index_) + '-sar.tif'
                if self.num_dates > 1:
                    past_image2_path = self.past_images2_dir + str(index_) + '-sar.tif'
                    future_image2_path = self.future_images2_dir + str(index_) + '-sar.tif'

            # Read image
            self.sar_images[i][0] = io.imread(image_path)

            # Read image
            self.ground_truth_masks[i][0] = io.imread(mask_path)
            if self.num_dates > 0:
                self.past_sar_images[i][0] = io.imread(past_image_path)
                self.future_sar_images[i][0] = io.imread(future_image_path)
                if self.num_dates > 1:
                    self.past_sar_images2[i][0] = io.imread(past_image2_path)
                    self.future_sar_images2[i][0] = io.imread(future_image2_path)
            # fig, axs = plt.subplots(1, 4)
            # axs[0].imshow(self.sar_images[i][0], cmap='Greys')
            # axs[1].imshow(self.ground_truth_masks[i][0], cmap='Greys')
            # axs[2].imshow(self.past_sar_images[i][0], cmap='Greys')
            # axs[3].imshow(self.future_sar_images[i][0], cmap='Greys')
            # plt.show()
        # Convert to Pytorch tensor
        self.sar_images = torch.from_numpy(self.sar_images)
        self.ground_truth_masks = torch.from_numpy(self.ground_truth_masks)
        if self.num_dates > 0:
            self.past_sar_images = torch.from_numpy(self.past_sar_images)
            self.future_sar_images = torch.from_numpy(self.future_sar_images)
            if self.num_dates > 1:
                self.past_sar_images2 = torch.from_numpy(self.past_sar_images2)
                self.future_sar_images2 = torch.from_numpy(self.future_sar_images2)

    def __getitem__(self, index):
        if self.num_dates == 0:
            return self.sar_images[index], self.ground_truth_masks[index]
        elif self.num_dates == 1:
            return self.sar_images[index], self.ground_truth_masks[index], self.past_sar_images[index], self.future_sar_images[index]
        elif self.num_dates == 2:
            return self.sar_images[index], self.ground_truth_masks[index], self.past_sar_images[index], self.future_sar_images[index], \
                self.past_sar_images2[index], self.future_sar_images2[index]

    def __len__(self):
        return len(self.dataset)


def get_dataloaders(data, batch_size, num_workers, images_dir, masks_dir, pre_2020):
    training_method = os.getenv('TRAINING_METHOD')
    if training_method == 'standard':
        datasets = {
            'train': CFDDataset_in_memory(data[data.split == 'train'], images_dir, masks_dir),
            'test': CFDDataset_in_memory(data[data.split == 'test'], images_dir, masks_dir)
        }
    elif training_method == 'temporal_consistency':
        num_dates = int(os.getenv('TEMPORAL_CONSISTENCY_NUM_DATES'))

        if pre_2020:
            past_images_dir = os.getenv('PRE_20_PAST_SAR_DIR') + '/'
            future_images_dir = os.getenv('PRE_20_FUTURE_SAR_DIR') + '/'
        else:
            past_images_dir = os.getenv('POST_20_PAST_SAR_DIR') + '/'
            future_images_dir = os.getenv('POST_20_FUTURE_SAR_DIR') + '/'
        # past_images_dir = os.getenv('PAST_SAR_DIR') + '/'
        # future_images_dir = os.getenv('FUTURE_SAR_DIR') + '/'
        if num_dates == 1:
            datasets = {
                'train': CFDDataset_in_memory(data[data.split == 'train'], images_dir, masks_dir, past_images_dir, future_images_dir),
                'test': CFDDataset_in_memory(data[data.split == 'test'], images_dir, masks_dir, past_images_dir, future_images_dir)
            }
        elif num_dates == 2:
            if pre_2020:
                past_images2_dir = os.getenv('PRE_20_PAST_SAR2_DIR') + '/'
                future_images2_dir = os.getenv('PRE_20_FUTURE_SAR2_DIR') + '/'
            else:
                past_images2_dir = os.getenv('POST_20_PAST_SAR2_DIR') + '/'
                future_images2_dir = os.getenv('POST_20_FUTURE_SAR2_DIR') + '/'
            # past_images2_dir = os.getenv('PAST_SAR2_DIR') + '/'
            # future_images2_dir = os.getenv('FUTURE_SAR2_DIR') + '/'
            datasets = {
                'train': CFDDataset_in_memory(data[data.split == 'train'], images_dir, masks_dir, past_images_dir,
                                    future_images_dir, past_images2_dir, future_images2_dir),
                'test': CFDDataset_in_memory(data[data.split == 'test'], images_dir, masks_dir, past_images_dir,
                                   future_images_dir, past_images2_dir, future_images2_dir)
            }
    dataloaders = {
        'train': DataLoader(
          datasets['train'],
          batch_size=batch_size,
          shuffle=True,
          num_workers=num_workers,
          pin_memory=True
        ),
        'test': DataLoader(
          datasets['test'],
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
    # step = 0
    # with torch.profiler.profile(
    #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/test'),
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True
    # ) as prof:
    for input, target in tqdm(dataloader, total=len(dataloader)):
        # prof.step()
        # if step >= 1 + 1 + 3:
        #     break
        # step += 1
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            output = model(input)
            loss = criterion(output, target)

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

    for input, target in tqdm(dataloader, total=len(dataloader)):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

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
    if scheduler is not None:
        scheduler.step(val_iou)
    metrics = {
        'val_loss': val_loss,
        'val_iou': val_iou,
        # 'lr': current_lr
    }

    return metrics


def train_temporal_consistency(model, dataloader, criterion, optimizer, device, epoch):
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
    temporal_consistency_criterion = loss_function_factory.create_loss_function('dice')
    temporal_consistency_weight = float(os.getenv('TEMPORAL_CONSISTENCY_WEIGHT'))
    standard_training_weight = float(os.getenv('STANDARD_TRAINING_WEIGHT'))
    temporal_consistency_starting_epoch = int(os.getenv('TEMPORAL_CONSISTENCY_START_EPOCH'))
    temporal_consistency_num_dates = int(os.getenv('TEMPORAL_CONSISTENCY_NUM_DATES'))
    temporal_consistency_power = float(os.getenv('TEMPORAL_CONSISTENCY_POWER'))

    if temporal_consistency_num_dates == 1:
        for input, target, past_sar, future_sar in tqdm(dataloader, total=len(dataloader)):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            past_sar = past_sar.to(device, non_blocking=True)
            future_sar = future_sar.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output = model(input)
                past_output = model(past_sar)
                future_output = model(future_sar)
                loss = criterion(output, target)
                past_consistency_loss = temporal_consistency_criterion(past_output, output)
                future_consistency_loss = temporal_consistency_criterion(future_output, output)
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
                past_iou = intersection_over_union(past_output, output)
                future_iou = intersection_over_union(future_output, output)
                losses.append(total_loss.cpu().detach().numpy())
                base_losses.append(loss.cpu().detach().numpy())
                past_consistency_losses.append(past_consistency_loss.cpu().detach().numpy())
                future_consistency_losses.append(future_consistency_loss.cpu().detach().numpy())
                ious.append(iou.cpu().detach().numpy())
                past_ious.append(past_iou.cpu().detach().numpy())
                future_ious.append(future_iou.cpu().detach().numpy())
    elif temporal_consistency_num_dates == 2:
        for input, target, past_sar, future_sar, past_sar2, future_sar2 in tqdm(dataloader, total=len(dataloader)):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            past_sar = past_sar.to(device, non_blocking=True)
            future_sar = future_sar.to(device, non_blocking=True)
            past_sar2 = past_sar2.to(device, non_blocking=True)
            future_sar2 = future_sar2.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output = model(input)
                past_output = model(past_sar)
                future_output = model(future_sar)
                past_output2 = model(past_sar2)
                future_output2 = model(future_sar2)
                loss = criterion(output, target)
                past_consistency_loss = temporal_consistency_criterion(past_output, output)
                future_consistency_loss = temporal_consistency_criterion(future_output, output)
                past_consistency_loss2 = temporal_consistency_criterion(past_output2, past_output)
                future_consistency_loss2 = temporal_consistency_criterion(future_output2, future_output)
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
                past_iou = intersection_over_union(past_output, output)
                future_iou = intersection_over_union(future_output, output)
                past_iou2 = intersection_over_union(past_output2, output)
                future_iou2 = intersection_over_union(future_output2, output)
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

def evaluate_temporal_consistency(model, dataloader, criterion, scheduler, device, epoch):
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
    temporal_consistency_criterion = loss_function_factory.create_loss_function('dice')
    temporal_consistency_weight = float(os.getenv('TEMPORAL_CONSISTENCY_WEIGHT'))
    standard_training_weight = float(os.getenv('STANDARD_TRAINING_WEIGHT'))
    temporal_consistency_starting_epoch = int(os.getenv('TEMPORAL_CONSISTENCY_START_EPOCH'))
    temporal_consistency_num_dates = int(os.getenv('TEMPORAL_CONSISTENCY_NUM_DATES'))
    temporal_consistency_power = float(os.getenv('TEMPORAL_CONSISTENCY_POWER'))

    if temporal_consistency_num_dates == 1:
        for input, target, past_sar, future_sar in tqdm(dataloader, total=len(dataloader)):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            past_sar = past_sar.to(device, non_blocking=True)
            future_sar = future_sar.to(device, non_blocking=True)

            with torch.set_grad_enabled(False):
                output = model(input)
                past_output = model(past_sar)
                future_output = model(future_sar)
                loss = criterion(output, target)
                past_consistency_loss = temporal_consistency_criterion(past_output, output)
                future_consistency_loss = temporal_consistency_criterion(future_output, output)
                if epoch < temporal_consistency_starting_epoch:
                    current_temporal_consistency_weight = 0.
                    current_standard_training_weight = 1.
                else:
                    current_temporal_consistency_weight = temporal_consistency_weight
                    current_standard_training_weight = standard_training_weight
                total_loss = current_standard_training_weight * loss + current_temporal_consistency_weight * (torch.pow(past_consistency_loss,temporal_consistency_power) + torch.pow(future_consistency_loss,temporal_consistency_power))

                iou = intersection_over_union(output, target)
                past_iou = intersection_over_union(past_output, output)
                future_iou = intersection_over_union(future_output, output)
                losses.append(total_loss.cpu().detach().numpy())
                base_losses.append(loss.cpu().detach().numpy())
                past_consistency_losses.append(past_consistency_loss.cpu().detach().numpy())
                future_consistency_losses.append(future_consistency_loss.cpu().detach().numpy())
                ious.append(iou.cpu().detach().numpy())
                past_ious.append(past_iou.cpu().detach().numpy())
                future_ious.append(future_iou.cpu().detach().numpy())
    elif temporal_consistency_num_dates == 2:
        for input, target, past_sar, future_sar, past_sar2, future_sar2 in tqdm(dataloader, total=len(dataloader)):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            past_sar = past_sar.to(device, non_blocking=True)
            future_sar = future_sar.to(device, non_blocking=True)
            past_sar2 = past_sar2.to(device, non_blocking=True)
            future_sar2 = future_sar2.to(device, non_blocking=True)

            with torch.set_grad_enabled(False):
                output = model(input)
                past_output = model(past_sar)
                future_output = model(future_sar)
                past_output2 = model(past_sar2)
                future_output2 = model(future_sar2)
                loss = criterion(output, target)
                past_consistency_loss = temporal_consistency_criterion(past_output, output)
                future_consistency_loss = temporal_consistency_criterion(future_output, output)
                past_consistency_loss2 = temporal_consistency_criterion(past_output2, past_output)
                future_consistency_loss2 = temporal_consistency_criterion(future_output2, future_output)
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
                past_iou = intersection_over_union(past_output, output)
                future_iou = intersection_over_union(future_output, output)
                past_iou2 = intersection_over_union(past_output2, output)
                future_iou2 = intersection_over_union(future_output2, output)
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
    index = tiles_data[tiles_data.split == 'test'].iloc[i]['id']
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
    batch_sar_image = torch.from_numpy(batch_sar_image.astype(np.float32)).to(device, non_blocking=True)
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


def full_cycle(test_name, pre_2020=True):
    config = dotenv_values()
    # Convert int values to int
    for key in ['EPOCHS', 'PATCH_SIZE', 'BATCH_SIZE', 'NUM_WORKERS', 'EARLY_STOP_NUM_EPOCHS', 'TEMPORAL_CONSISTENCY_START_EPOCH',
                'REDUCE_LR_PLATEAU_PATIENCE']:
        config[key] = int(config[key])
    # Convert float values to float
    for key in ['LEARNING_RATE']:
        config[key] = float(config[key])
    for key in ['SAVE_MODEL_ON_ALL_EPOCHS', 'SAVE_MODEL_ON_LAST_EPOCH', 'REDUCE_LR_PLATEAU']:
        if config[key] == "TRUE":
            config[key] = True
        else:
            config[key] = False

    training_method = config['TRAINING_METHOD']
    if config['RANDOM_SEED']!='NONE':
        config['RANDOM_SEED'] = int(config['RANDOM_SEED'])

    # Configure the wandb run
    wandb.login(key='7f266f9a3115da69acae90e8eb024ea65859454e')
    wandb_config = config.copy()
    del wandb_config['AGGREGATE_FUNCTION']
    del wandb_config['ANNOTATED_DATA_DIR']
    del wandb_config['CHARTS_DIR']
    del wandb_config['CLOUDY_PIXEL_PERCENTAGE']
    del wandb_config['COUNTRY_CODE']
    del wandb_config['DATA_DIR']
    del wandb_config['EVALUATION_DIR']
    del wandb_config['GEOJSON_FILE']
    del wandb_config['HOME_DIR']
    del wandb_config['MODELS_DIR']
    del wandb_config['NDWI_INPUT']
    del wandb_config['ORBIT_PASS']
    del wandb_config['OTSU_GAUSSIAN_KERNEL_SIZE']
    del wandb_config['REGION_ADMIN_LEVEL']
    del wandb_config['REGION_NAME']
    del wandb_config['RESULTS_DIR']
    del wandb_config['SAR_POLARIZATION']
    del wandb_config['STUDY_AREA']
    del wandb_config['TRAIN_CWD_DIR']
    del wandb_config['WATER_INDEX']
    del wandb_config['BASE_FILE_NAME']
    wandb.init(project="deepaqua", config=wandb_config, name=test_name)
    # wandb.init(project="sweeps", entity="deep-wetlands", config=config)
    config.update(wandb.config)
    print(json.dumps(config, indent=4))
    run_name = wandb.run.name
    wandb.run.define_metric("val_iou", summary="max")
    wandb.run.define_metric("val_loss", summary="min")
    wandb.run.define_metric("train_iou", summary="max")
    wandb.run.define_metric("train_loss", summary="min")

    # Set environment variables
    for key, value in config.items():
        os.environ[key] = str(value)

    n_epochs = int(os.getenv('EPOCHS'))
    learning_rate = float(os.getenv('LEARNING_RATE'))
    seed = os.getenv('RANDOM_SEED')
    if seed != 'NONE':
        seed = int(seed)
    batch_size = int(os.getenv('BATCH_SIZE'))
    num_workers = int(os.getenv('NUM_WORKERS'))
    model_dir = os.getenv('MODELS_DIR')
    loss_function_name = os.getenv('LOSS_FUNCTION')
    cnn_type = os.getenv('CNN_TYPE')
    band = os.getenv('SAR_POLARIZATION')
    tiff_dir = os.getenv('BULK_EXPORT_DIR')
    if os.getenv('SAVE_MODEL_ON_ALL_EPOCHS') == 'True':
        save_model_on_all_epochs = True
    else:
        save_model_on_all_epochs = False
    if os.getenv('SAVE_MODEL_ON_LAST_EPOCH') == 'True':
        save_model_on_last_epoch = True
    else:
        save_model_on_last_epoch = False
    if os.getenv('REDUCE_LR_PLATEAU') == 'True':
        reduce_lr_plateau = True
    else:
        reduce_lr_plateau = False

    early_stop_num_epochs = int(os.getenv('EARLY_STOP_NUM_EPOCHS'))
    temporal_consistency_start_epoch = int(os.getenv('TEMPORAL_CONSISTENCY_START_EPOCH'))
    num_dates = int(os.getenv('TEMPORAL_CONSISTENCY_NUM_DATES'))
    temporal_consistency_weight = float(os.getenv('TEMPORAL_CONSISTENCY_WEIGHT'))
    standard_training_weight = float(os.getenv('STANDARD_TRAINING_WEIGHT'))
    temporal_consistency_power = float(os.getenv('TEMPORAL_CONSISTENCY_POWER'))
    reduce_lr_plateau_patience = int(os.getenv('REDUCE_LR_PLATEAU_PATIENCE'))
    mask_type = os.getenv('MASK_TYPE')

    # e = int(os.getenv('TEMPORAL_CONSISTENCY_NUM_DATES'))
    # num_dates = int(os.getenv('TEMPORAL_CONSISTENCY_NUM_DATES'))

    if seed != 'NONE':
        utils.plant_random_seed(seed)

    tiles_data = utils.create_tiles_file_pipeline(pre_2020)
    if pre_2020:
        images_dir = os.getenv('PRE_20_SAR_DIR') + '/'
        masks_dir = os.getenv('PRE_20_MASK_DIR') + '/'
        tiles_data_file = os.getenv('PRE_20_TILES_FILE')
        training_date = os.getenv('PRE_20_TRAIN_DATE')
    else:
        images_dir = os.getenv('POST_20_SAR_DIR') + '/'
        masks_dir = os.getenv('POST_20_MASK_DIR') + '/'
        tiles_data_file = os.getenv('POST_20_TILES_FILE')
        training_date = os.getenv('POST_20_TRAIN_DATE')
    # images_dir = os.getenv('SAR_DIR') + '/'
    # masks_dir = os.getenv('NDWI_MASK_DIR') + '/'
    # tiles_data_file = os.getenv('TILES_FILE')

    tiff_file = os.getenv('SINGLE_TEST_FILE')
    tiff_path = os.path.join(tiff_dir, tiff_file)
    tiff_image = viz_utils.load_image(tiff_path,  skip_nan=False, skimage_read=False)
    # tiff_image2 = viz_utils.load_image(tiff_path, ignore_nan=True)

    # Check is GPU is enabled
    device = utils.get_device()

    # tiles_data = pd.read_csv(tiles_data_file)#.groupby('split').sample(frac=0.05)

    dataloaders = get_dataloaders(tiles_data, batch_size, num_workers, images_dir, masks_dir, pre_2020)

    # model = Unet(in_channels=1, out_channels=1, init_dim=unet_init_dim, num_blocks=unet_blocks)
    model = model_factory.create_model(cnn_type)
    print(model)
    print('Model parameters', sum(param.numel() for param in model.parameters()))
    # criterion = DiceLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCELoss()
    criterion = loss_function_factory.create_loss_function(loss_function_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if reduce_lr_plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=reduce_lr_plateau_patience, factor=0.5)
    else:
        scheduler = None

    model = torch.nn.DataParallel(model)
    model.to(device)
    max_score = 0
    val_ious = []
    best_epoch = 0
    for epoch in range(1, n_epochs + 1):
        # prof.step()
        print("\nEpoch {}/{} {}".format(epoch, n_epochs, time.strftime("%Y/%m/%d-%H:%M:%S")))
        print("-" * 10)
        # if epoch >= 3:
        #     break
        if training_method == 'standard':
            train_metrics = train(
                model,
                dataloaders["train"],
                criterion,
                optimizer,
                device
            )
            val_metrics = evaluate(
                model,
                dataloaders['test'],
                scheduler,
                criterion,
                device
            )
        elif training_method == 'temporal_consistency':
            train_metrics = train_temporal_consistency(
                model,
                dataloaders["train"],
                criterion,
                optimizer,
                device,
                epoch
            )
            val_metrics = evaluate_temporal_consistency(
                model,
                dataloaders['test'],
                criterion,
                scheduler,
                device,
                epoch
            )

        # mask_data = np.array([[1, 2, 2, ..., 2, 2, 1], ...])
        class_labels = {
            0: "land",
            1: "water",
        }

        pred_mask = map_wetlands.predict_water_mask(tiff_image, model, device)

        full_mask_img = wandb.Image(tiff_image, masks={
            "predictions": {
                "mask_data": pred_mask,
                "class_labels": class_labels
            },
        }, caption=["Full water detection", "fwd", "fwdm"])

        # Count values of full_pred array

        full_pred = wandb.Image(pred_mask, caption="Full prediction")

        metrics = {
            **train_metrics, **val_metrics, 'full_pred': full_pred, 'full_mask': full_mask_img
        }

        print('Train loss: {}, Val loss: {}'.format(metrics['train_loss'], metrics['val_loss']))
        wandb.log(metrics)

        val_ious.append(metrics['val_iou'])
        if metrics['val_iou'] > max_score:
            max_score = metrics['val_iou']
            best_epoch = epoch
            print(f'New best model found on epoch {epoch}. Validation IoU: {max_score}')
            save_model(model, os.path.join(model_dir, run_name), 'best_model.pth')
        if save_model_on_all_epochs:
            save_model(model, os.path.join(model_dir, run_name), f'epoch_{epoch}.pth')
        stop_training = False
        if early_stop_num_epochs > 0:
            if training_method == 'standard' or temporal_consistency_weight == 0.:
                if len(val_ious) > early_stop_num_epochs and np.max(val_ious[-early_stop_num_epochs:]) < np.max(val_ious):
                    stop_training = True
            elif len(val_ious) >= early_stop_num_epochs + temporal_consistency_start_epoch and np.max(val_ious[-early_stop_num_epochs:]) < np.max(val_ious):
                stop_training = True
        if stop_training:
            break
    if save_model_on_last_epoch:
        save_model(model, os.path.join(model_dir, run_name), f'final_epoch.pth')
    with open(model_dir + '/model_info.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([run_name, test_name, str(training_date), training_method, num_dates, temporal_consistency_weight, standard_training_weight, temporal_consistency_power,
                             temporal_consistency_start_epoch, n_epochs, learning_rate, early_stop_num_epochs, best_epoch, max_score, epoch, mask_type])
    wandb.finish()
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
    pred_mask = map_wetlands.predict_water_mask(tiff_image, model, device)
    # prof.export_chrome_trace("trace.json")
    # prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")
    plt.imshow(pred_mask)
    # plt.show()
    plt.clf()


def intersection_over_union(y_pred, y_true):

    smooth = 1e-6
    y_pred = y_pred[:, 0].view(-1) > 0.5
    y_true = y_true[:, 0].view(-1) > 0.5
    intersection = (y_pred & y_true).sum() + smooth
    union = (y_pred | y_true).sum() + smooth
    iou = intersection / union

    return iou


def load_and_test():
    model_file = os.getenv('MODEL_FILE')
    images_dir = os.getenv('SAR_DIR') + '/'
    ndwi_masks_dir = os.getenv('NDWI_MASK_DIR') + '/'
    cnn_type = os.getenv('CNN_TYPE')
    tiles_data_file = os.getenv('TILES_FILE')
    tiles_data = pd.read_csv(tiles_data_file)

    # Check is GPU is enabled
    device = utils.get_device()

    model = model_factory.load_model(cnn_type, model_file, device)
    evaluate_single_image(model, tiles_data, images_dir, ndwi_masks_dir, device)


def main():
    if __name__ == '__main__':
        print('BBBBBBBBBBBBBBBBBBBBB')
        load_dotenv()
        config = dotenv_values()
        # print(json.dumps(config, indent=4))

        full_cycle()
        # load_and_test()


# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
