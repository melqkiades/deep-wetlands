
import os
import random
import time
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
from rasterio.plot import show


# tiles_data.head(3)

# index = tiles_data.iloc[10]['id']
# image_path = images_dir + str(index) + '-sar.tif'
# mask_path = masks_dir + str(index) + '-ndwi_mask.tif'
#
# # Open image file using Rasterio
# sar_image = rio.open(image_path)
# mask = rio.open(mask_path)
#
#
# # fig, ax = plt.subplots(1, 2, figsize=(8,8))
# # ax[0].imshow(image)
# # ax[1].imshow(mask);
#
# # Plot image and corresponding boundary
# # fig, ax = plt.subplots(1, 2, figsize=(8,8))
# # show(sar_image, ax=ax);
# fig, ax = plt.subplots(1, 2, figsize=(8,8))
# show(sar_image, ax=ax[0], cmap='gray')
# show(mask, ax=ax[1])
# # print(sar_image.count)
# print(sar_image.read(1).shape)


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
        # image = plt.imread(image_path)
        image = rio.open(image_path).read()
        # print('former image shape', image.shape)
        # image = image.transpose((2, 1, 0))
        # print('new image shape', image.shape)

        # Read image
        # mask = plt.imread(mask_path)
        mask = rio.open(mask_path).read()
        # print('former mask shape', mask.shape)
        # mask = mask.transpose((1, 0))[None, :]
        # print('new mask shape', mask.shape)

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

    for input, target in tqdm(dataloader, total=len(dataloader)):
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            output = model(input)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

    return loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    for input, target in tqdm(dataloader, total=len(dataloader)):
        input = input.to(device)
        target = target.to(device)

        with torch.set_grad_enabled(False):
            output = model(input)
            loss = criterion(output, target)

    return loss


def full_cycle():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cwd = '/Users/frape/Projects/DeepWetlands/src/deep-wetlands/external/data/'
    data_dir = '/Users/frape/Projects/DeepWetlands/Datasets/wetlands/'
    images_dir = data_dir + 'sar/'
    masks_dir = data_dir + 'ndwi_mask/'
    tiles_data_file = data_dir + 'tiles.csv'

    # Check is GPU is enabled
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Device: {}".format(device))

    # Get specific GPU model
    if str(device) == "cuda:0":
        print("GPU: {}".format(torch.cuda.get_device_name(0)))

    tiles_data = pd.read_csv(tiles_data_file)

    batch_size = 4
    num_workers = 0

    dataloaders = get_dataloaders(tiles_data, batch_size, num_workers, images_dir, masks_dir)
    train_batch = next(iter(dataloaders['train']))
    # visualize_batch(train_batch, batch_size)

    model = Unet(in_channels=1)
    print(model)
    input, target = next(iter(dataloaders['train']))
    pred = model(input)

    n_epochs = 15
    learning_rate = 0.0001
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    model.to(device)
    for epoch in range(1, n_epochs + 1):
        print("\nEpoch {}/{}".format(epoch, n_epochs))
        print("-" * 10)

        train_loss = train(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            device
        )
        val_loss = evaluate(
            model,
            dataloaders['test'],
            criterion,
            device
        )
        print('Train loss: {}, Val loss: {}'.format(
            train_loss.cpu().detach().numpy(),
            val_loss.cpu().detach().numpy())
        )

    i = 120
    model.eval()
    index = tiles_data[tiles_data.split == 'test' ].iloc[i]['id']
    image_path = images_dir + str(index) + '-sar.tif'
    image = rio.open(image_path).read()
    print(image.shape)
    plt.imshow(image[0], cmap='gray')
    plt.show()
    plt.clf()


    # image = image.transpose((2, 1, 0))[None, :]
    image = image[None, :]
    print(image.shape)
    image = torch.from_numpy(image.astype(np.float32)).to(device)
    pred = model(image).cpu().detach().numpy()
    # pred = pred.squeeze().transpose((1, 0))
    pred = pred.squeeze()
    # pred = (pred * 255.0).astype("uint8")
    plt.imshow(pred)
    plt.show()
    plt.clf()


def main():
    full_cycle()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
