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
        # Clamp values to avoid underflow (log(0))
        out = torch.clamp(out, min=1e-6)
        positives = -torch.log(out[torch.where(target == 1)])
        negatives = -torch.log(torch.clamp(1 - out[torch.where(target == 0)], min=1e-6))

        w_p = (len(positives) + len(negatives)) / len(positives)
        w_n = (len(positives) + len(negatives)) / len(negatives)
        loss = w_p * positives.sum() + w_n * negatives.sum()

        return loss
