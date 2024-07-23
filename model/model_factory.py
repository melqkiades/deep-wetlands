import os

import torch

from model.old_unet import OldUnet
from model.unet import Unet

import segmentation_models_pytorch as smp
from collections import OrderedDict

def create_model(model_name):
    unet_init_dim = int(os.getenv('UNET_INIT_DIM'))
    unet_blocks = int(os.getenv('UNET_BLOCKS'))

    models_dict = {
        'unet': Unet(in_channels=1, out_channels=1, init_dim=unet_init_dim, num_blocks=unet_blocks),
        'old_unet': OldUnet(in_channels=1, out_channels=1),
        'unet_smp':  smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1),
        # 'unet2':  smp.Unet(encoder_name="inceptionv4", encoder_weights=None, in_channels=1, classes=1),
        'unet++':  smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1),
        'deeplabv3':  smp.DeepLabV3(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1),
        'deeplabv3+':  smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1),
    }

    if model_name not in models_dict.keys():
        raise ValueError(f'Unrecognized model: {model_name}')

    return models_dict[model_name]


def load_model(model_name, model_file, device):
    loaded_model = create_model(model_name)
    loaded_model.to(device)
    if os.path.basename(model_file) in ['big-2018.pth', 'big-2020.pth',
                                        'prime-2018', 'prime-2020.pth']:
        loaded_model.load_state_dict(torch.load(model_file, map_location=device))
    else:
        old_dict = torch.load(model_file, map_location=device)
        new_dict = OrderedDict([])
        for key in old_dict:
            new_dict[key[7:]] = old_dict[key]
        loaded_model.load_state_dict(new_dict)
    loaded_model.eval()

    print('Model file {} successfully loaded.'.format(model_file))

    return loaded_model
