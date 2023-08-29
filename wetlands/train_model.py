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


class CFDDataset(Dataset):
    def __init__(self, dataset, images_dir, masks_dir):
        self.dataset = dataset
        self.images_dir = images_dir
        self.masks_dir = masks_dir

    def __getitem__(self, index):
        index_ = self.dataset.iloc[index]['id']

        # Get image and mask file paths for specified index
        image_path = self.images_dir + str(index_) + '-sar.tif'
        mask_path = self.masks_dir + str(index_) + '-ndwi_mask.tif'

        # Read image
        image = rio.open(image_path).read()

        # Read image
        mask = rio.open(mask_path).read()

        # Convert to Pytorch tensor
        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.dataset)


def get_dataloaders(data, batch_size, num_workers, images_dir, masks_dir):
    datasets = {
        'train' : CFDDataset(data[data.split == 'train'], images_dir, masks_dir),
        'test' : CFDDataset(data[data.split == 'test'], images_dir, masks_dir)
    }

    dataloaders = {
        'train': DataLoader(
          datasets['train'],
          batch_size=batch_size,
          shuffle=True,
          num_workers=num_workers
        ),
        'test': DataLoader(
          datasets['test'],
          batch_size=batch_size,
          drop_last=False,
          num_workers=num_workers
        )
    }
    return dataloaders


def train(model, dataloader, criterion, optimizer, device):
    model.train(True)
    losses = []
    ious = []

    for input, target in tqdm(dataloader, total=len(dataloader)):
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

    train_loss = np.mean(losses)
    train_iou = np.mean(ious)
    print('Train loss', train_loss, 'Train IOU', train_iou)

    metrics = {
        'train_loss': train_loss,
        'train_iou': train_iou,
    }

    return metrics


def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = []
    ious = []

    for input, target in tqdm(dataloader, total=len(dataloader)):
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

    metrics = {
        'val_loss': val_loss,
        'val_iou': val_iou,
    }

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


def full_cycle():
    config = dotenv_values()
    # Convert int values to int
    for key in ['EPOCHS', 'PATCH_SIZE', 'BATCH_SIZE', 'NUM_WORKERS', 'RANDOM_SEED']:
        config[key] = int(config[key])
    # Convert float values to float
    for key in ['LEARNING_RATE']:
        config[key] = float(config[key])

    # Configure the wandb run
    wandb.init(project="test-project", entity="deep-wetlands", config=config)
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
    seed = int(os.getenv('RANDOM_SEED'))
    batch_size = int(os.getenv('BATCH_SIZE'))
    num_workers = int(os.getenv('NUM_WORKERS'))
    model_dir = os.getenv('MODELS_DIR')
    loss_function_name = os.getenv('LOSS_FUNCTION')
    cnn_type = os.getenv('CNN_TYPE')
    band = os.getenv('SAR_POLARIZATION')
    tiff_dir = os.getenv('BULK_EXPORT_DIR')

    utils.plant_random_seed(seed)

    utils.create_tiles_file()
    images_dir = os.getenv('SAR_DIR') + '/'
    masks_dir = os.getenv('NDWI_MASK_DIR') + '/'
    tiles_data_file = os.getenv('TILES_FILE')

    tiff_file = 'S1B_IW_GRDH_1SDV_20211115T052235_20211115T052300_029594_038826_6CF2.tif'
    tiff_path = os.path.join(tiff_dir, tiff_file)
    tiff_image = viz_utils.load_image(tiff_path, band, ignore_nan=True)

    # Check is GPU is enabled
    device = utils.get_device()

    tiles_data = pd.read_csv(tiles_data_file)

    dataloaders = get_dataloaders(tiles_data, batch_size, num_workers, images_dir, masks_dir)

    # model = Unet(in_channels=1, out_channels=1, init_dim=unet_init_dim, num_blocks=unet_blocks)
    model = model_factory.create_model(cnn_type)
    print(model)
    print('Model parameters', sum(param.numel() for param in model.parameters()))
    # criterion = DiceLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCELoss()
    criterion = loss_function_factory.create_loss_function(loss_function_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    max_score = 0
    for epoch in range(1, n_epochs + 1):
        print("\nEpoch {}/{} {}".format(epoch, n_epochs, time.strftime("%Y/%m/%d-%H:%M:%S")))
        print("-" * 10)

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
            criterion,
            device
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

        if metrics['val_iou'] > max_score:
            max_score = metrics['val_iou']
            model_file = f'{run_name}_best_model.pth'
            save_model(model, model_dir, model_file)
            print(f'New best model found on epoch {epoch}. Validation IoU: {max_score}')

    pred_mask = map_wetlands.predict_water_mask(tiff_image, model, device)

    plt.imshow(pred_mask)
    plt.show()
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
    load_dotenv()
    config = dotenv_values()
    print(json.dumps(config, indent=4))

    full_cycle()
    # load_and_test()


# start = time.time()
# main()
# end = time.time()
# total_time = end - start
# print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
