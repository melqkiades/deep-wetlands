import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class CFDDataset(Dataset):
    def __init__(self, dataset, images_dir, masks_dir):
        self.dataset = dataset
        self.images_dir = images_dir
        self.masks_dir = masks_dir

    def __getitem__(self, index):
        index_ = self.dataset.iloc[index]['indices']

        # Get image and mask file paths for specified index
        image_path = self.images_dir + str(index_) + '.jpeg'
        mask_path = self.masks_dir + str(index_) + '.png'

        # Read image
        image = plt.imread(image_path)
        image = image.transpose((2, 1, 0))

        # Read image
        mask = plt.imread(mask_path)
        mask = mask.transpose((1, 0))[None, :]

        # Convert to Pytorch tensor
        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.dataset)
