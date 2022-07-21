

"""In the following, we create a function for generating pytorch dataloaders. we use `torch`'s `DataLoader` class to create a dataloader.  The dataloader manages fetching samples from the datasets (it can even fetch them in parallel using `num_workers`) and assembles batches of the datasets. """
from torch.utils.data import DataLoader

from crop_fields.cf_dataset import CFDDataset


def get_dataloaders(data, batch_size, num_workers):
    datasets = {
        'train' : CFDDataset(data[data.split == 'train']),
        'test' : CFDDataset(data[data.split == 'test'])
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