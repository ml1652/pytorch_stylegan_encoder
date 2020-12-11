# torch.log  and math.log is e based
import math
import torch
from torch import nn


class WingLoss(nn.Module):
    def __init__(self, w=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.w]
        delta_y2 = delta_y[delta_y >= self.w]
        loss1 = self.w * torch.log(1 + delta_y1 / self.epsilon)
        C = self.w - self.w * math.log(1 + self.w / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))