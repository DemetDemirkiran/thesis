import torch
import torch.nn as nn
import torch.nn.functional as func


# Source: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, model_output, target):
        BCE_loss = func.binary_cross_entropy_with_logits(model_output, target)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss
