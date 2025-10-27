import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, lambda_=1e-5):
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

class DiceLossSwin(nn.Module):
    def __init__(self, n_classes):
        super(DiceLossSwin, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, batch_weights=None):
        target = target.float()
        smooth = 1e-5
        if batch_weights is None:
            intersect = torch.sum(score * target)
            y_sum = torch.sum(target * target)
            z_sum = torch.sum(score * score)
            loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        else:
            intersect = torch.sum(score * target, dim=(1,2))
            y_sum = torch.sum(target * target, dim=(1,2))
            z_sum = torch.sum(score * score, dim=(1,2))
            loss = torch.mean(batch_weights*(2 * intersect + smooth) / (z_sum + y_sum + smooth))
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False, ignore_class_zero=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        if ignore_class_zero:
            start_i = 1
        else:
            start_i = 0
        for i in range(start_i, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        if ignore_class_zero:
            return loss / (self.n_classes-1)
        else:
            return loss / self.n_classes

    def comparison(self, input1, input2, weight=None, softmax=False, ignore_class_zero=False, batch_weights=None):
        if softmax:
            input1 = torch.softmax(input1, dim=1)
            input2 = torch.softmax(input2, dim=1)
        if weight is None:
            weight = [1] * self.n_classes
        assert input1.size() == input2.size(), 'predict {} & target {} shape do not match'.format(input1.size(), input2.size())
        class_wise_dice = []
        loss = 0.0
        if ignore_class_zero:
            start_i = 1
        else:
            start_i = 0
        for i in range(start_i, self.n_classes):
            dice = self._dice_loss(input1[:, i], input2[:, i], batch_weights)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        if ignore_class_zero:
            return loss / (self.n_classes-1)
        else:
            return loss / self.n_classes