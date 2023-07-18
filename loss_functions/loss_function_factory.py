from torch.nn import CrossEntropyLoss, BCELoss

from loss_functions.dice_loss import DiceLoss


def create_loss_function(loss_function_name):
    loss_function_dict = {
        'dice': DiceLoss(),
        'cross_entropy': CrossEntropyLoss(),
        'binary_cross_entropy': BCELoss()
    }

    if loss_function_name not in loss_function_dict.keys():
        raise ValueError(f'Unrecognized loss function: {loss_function_name}')

    return loss_function_dict[loss_function_name]
