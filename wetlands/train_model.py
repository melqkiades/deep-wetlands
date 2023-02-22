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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio as rio

from wetlands import utils
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


def visualize_batch(batch, batch_size):
    fig, ax = plt.subplots(2, 4, figsize=(batch_size*3, batch_size*1.5))

    for i in range(batch_size):
        image = batch[0][i].cpu().numpy()
        mask = batch[1][i].cpu().numpy()

        image = image.transpose((1, 2, 0))
        mask = mask.transpose((1, 2, 0)).squeeze()

        image = (image * 255.0).astype("uint8")
        mask = (mask * 255.0).astype("uint8")

        ax[0, i].imshow(image, cmap='gray')
        ax[1, i].imshow(mask)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=32):
        super().__init__()

        self.encoder1 = self.block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.block(features, features*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.block(features*2, features*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.block(features*4, features*8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self.block(features*8, features*16)

        self.upconv4 = nn.ConvTranspose2d(features*16, features*8, kernel_size=2, stride=2)
        self.decoder4 = self.block(features*16, features*8)
        self.upconv3 = nn.ConvTranspose2d(features*8, features*4, kernel_size=2, stride=2)
        self.decoder3 = self.block(features*8, features*4)
        self.upconv2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = self.block(features*4, features*2)
        self.upconv1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = self.block(features*2, features)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.decoder4(torch.cat((self.upconv4(bottleneck), enc4), dim=1))
        dec3 = self.decoder3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.decoder2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.decoder1(torch.cat((self.upconv1(dec2), enc1), dim=1))

        return torch.sigmoid(self.conv(dec1))

    def block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
      )





class DiceLoss(nn.Module):
    def __init__(self, lambda_=1.):
        super(DiceLoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, y_pred, y_true):
        y_pred = y_pred[:, 0].view(-1)
        y_true = y_true[:, 0].view(-1)
        intersection = (y_pred * y_true).sum()
        dice_loss = (2. * intersection  + self.lambda_) / (
            y_pred.sum() + y_true.sum() + self.lambda_
        )
        return 1. - dice_loss


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


def load_model(model_file, device):
    loaded_model = Unet(in_channels=1)
    loaded_model.to(device)
    loaded_model.load_state_dict(torch.load(model_file, map_location=device))
    loaded_model.eval()

    print('Model file {} successfully loaded.'.format(model_file))

    return loaded_model


def plot_single_sar_image(model, tiles_data, images_dir, ndwi_masks_dir, device):
    i = 120
    model.eval()
    index = tiles_data[tiles_data.split == 'test'].iloc[i]['id']
    image_path = images_dir + str(index) + '-sar.tif'
    image = rio.open(image_path).read()
    print(image.shape)


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
    n_epochs = int(os.getenv('EPOCHS'))
    learning_rate = float(os.getenv('LEARNING_RATE'))
    seed = int(os.getenv('RANDOM_SEED'))
    batch_size = int(os.getenv('BATCH_SIZE'))
    num_workers = int(os.getenv('NUM_WORKERS'))
    model_dir = os.getenv('MODELS_DIR')
    region = os.getenv('REGION_ASCII_NAME')
    date = os.getenv('START_DATE')
    polarization = os.getenv('SAR_POLARIZATION')
    orbit_pass = os.getenv('ORBIT_PASS')

    config = {
        "learning_rate": learning_rate,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "random_seed": seed,
        "region": region,
        "date": date,
        "polarization": polarization,
        "orbit_pass": orbit_pass,
    }

    wandb.init(project="test-project", entity="deep-wetlands", config=config)
    run_name = wandb.run.name
    utils.plant_random_seed(seed)

    images_dir = os.getenv('SAR_DIR') + '/'
    masks_dir = os.getenv('NDWI_MASK_DIR') + '/'
    tiles_data_file = os.getenv('TILES_FILE')

    # Check is GPU is enabled
    device = utils.get_device()

    tiles_data = pd.read_csv(tiles_data_file)

    dataloaders = get_dataloaders(tiles_data, batch_size, num_workers, images_dir, masks_dir)

    model = Unet(in_channels=1)
    print(model)
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
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

        sar_image, predicted_image, ndwi_image = evaluate_single_image(
            model, tiles_data, images_dir, masks_dir, device)
        int_sar_image = np.array(predicted_image > 0.5).astype(int)
        ndwi_image = ndwi_image[0]
        int_ndwi_image = np.array(ndwi_image > 0.5).astype(int)
        mask_img = wandb.Image(sar_image, masks={
            "predictions": {
                "mask_data": int_sar_image,
                "class_labels": class_labels
            },
            "ndwi": {
                "mask_data": int_ndwi_image,
                "class_labels": class_labels
            }
        }, caption=["Water detection", "fd", "fds"])

        predicted_image = wandb.Image(predicted_image, caption="Predicted image")
        # mask_img = wandb.Image(mask_img, caption="Mask image")

        metrics = {
            **train_metrics, **val_metrics, 'prediction': predicted_image,
            'mask': mask_img
        }

        print('Train loss: {}, Val loss: {}'.format(metrics['train_loss'], metrics['val_loss']))
        wandb.log(metrics)

        if epoch == 2 or epoch % 5 == 0:
            model_name = utils.generate_model_file_name(epoch)
            model_file = f'{run_name}_{model_name}.pth'
            save_model(model, model_dir, model_file)

    model_name = os.getenv("MODEL_NAME")
    model_file = f'{run_name}_{model_name}.pth'
    save_model(model, model_dir, model_file)

    evaluate_single_image(model, tiles_data, images_dir, masks_dir, device)


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
    tiles_data_file = os.getenv('TILES_FILE')
    tiles_data = pd.read_csv(tiles_data_file)

    # Check is GPU is enabled
    device = utils.get_device()

    model = load_model(model_file, device)
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
