import torch
import torch.nn as nn


class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, pred, target):
        out = self.softmax(pred)
        positives = -torch.log(out[torch.where(target == 1)])
        negatives = -torch.log(1 - out[torch.where(target == 0)])

        return positives.sum() + negatives.sum()



class WCEL(CEL):

    def forward(self, out, target):
        out = self.softmax(out)
        positives = -torch.log(out[torch.where(target == 1)])
        negatives = -torch.log(1 - out[torch.where(target == 0)])

        w_p = (len(positives) + len(negatives)) / len(positives)
        w_n = (len(positives) + len(negatives)) / len(negatives)
        loss = w_p * positives.sum() + w_n * negatives.sum()

        if torch.isnan(loss):
            a=0

        return loss
