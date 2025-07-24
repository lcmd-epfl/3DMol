import torch
import torch.nn as nn
from torch.nn import functional as F


class MAE(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, preds, targets):
        loss = F.l1_loss(preds, targets)
        return loss


class Accuracy(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, preds, targets):
        pred = torch.zeros_like(preds, dtype=int)
        pred[torch.where(preds<0)]=-1
        pred[torch.where(preds>=0)]=1
        loss = ((pred==targets)*1.0).mean()
        return loss
