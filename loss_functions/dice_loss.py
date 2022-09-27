
from torch import nn


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
