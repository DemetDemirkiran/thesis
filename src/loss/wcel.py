# Implemented according to the mathemetical function published by Wang et al for the paper ChestX-ray8
# @article{chestxray8,
#   author    = {Xiaosong Wang and
#                Yifan Peng and
#                Le Lu and
#                Zhiyong Lu and
#                Mohammadhadi Bagheri and
#                Ronald M. Summers},
#   title     = {ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on
#                Weakly-Supervised Classification and Localization of Common Thorax
#                Diseases},
#   journal   = {CoRR},
#   volume    = {abs/1705.02315},
#   year      = {2017},
#   url       = {http://arxiv.org/abs/1705.02315},
#   archivePrefix = {arXiv},
#   eprint    = {1705.02315},
#   timestamp = {Thu, 03 Oct 2019 13:13:22 +0200},
#   biburl    = {https://dblp.org/rec/journals/corr/WangPLLBS17.bib},
#   bibsource = {dblp computer science bibliography, https://dblp.org}
# }


import torch
import torch.nn as nn


class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, out, target):
        out = self.softmax(out)
        # Clamp values to avoid underflow (log(0))
        out = torch.clamp(out, min=1e-6)
        positives = -torch.log(out[torch.where(target == 1)])
        negatives = -torch.log(torch.clamp(1 - out[torch.where(target == 0)], min=1e-6))

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
