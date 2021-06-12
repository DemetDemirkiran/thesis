import torch
import numpy as np
import torch.nn as nn
from scipy.spatial.distance import cdist


class TripletAverage(nn.Module):
    def __init__(self, margin=1.2, threshold=0.5):
        super(TripletAverage, self).__init__()
        self.margin = margin
        self.threshold = threshold
        if self.margin is None:  # use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def loss(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[1]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda:
                y = y.cuda()
            ap_dist = torch.norm(anchor - pos, 2, dim=0).view(-1)
            an_dist = torch.norm(anchor - neg, 2, dim=0).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)
        return loss

    def forward(self, embedding, labels):
        embedding = torch.nn.functional.normalize(embedding)
        label_distances = cdist(labels.data.cpu(), labels.data.cpu(), 'jaccard')
        label_positives = label_distances <= self.threshold
        label_positives = label_positives - np.eye(label_positives.shape[0])
        anchor = []
        pos = []
        neg = []
        for i in range(label_positives.shape[0]):
            if label_positives[i, :].sum() > 0:
                positives = embedding[np.nonzero(label_positives[i, :])]
                negatives = embedding[np.nonzero(1 - label_positives[i, :])]
                anchor.append(embedding[i])
                pos.append(positives.mean(dim=0))
                neg.append(negatives.mean(dim=0))

        return self.loss(torch.stack(anchor), torch.stack(pos), torch.stack(neg))
