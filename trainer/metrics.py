import torch
from torch import nn
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
        pred = torch.sigmoid(preds)
        pred[torch.where(pred<0.5)]=0
        pred[torch.where(pred>=0.5)]=1
        loss = ((pred==targets)*1.0).mean()
        return loss
