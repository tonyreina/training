import monai.losses as losses
import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, ce_weights=(0.1, 0.3, 0.6)):
        super(Loss, self).__init__()
        self.dice = losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor(ce_weights))

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return (dice + cross_entropy) / 2
