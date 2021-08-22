# """
# @misc{Tschernezki2020,
#   author = {Tschernezki, Vadim and Sanakoyeu, Artsiom and Ommer, Bj{\"o}rn,},
#   title = {PyTorch Implementation of ProxyNCA},
#   year = {2020},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/dichotomies/proxy-nca}},
# }
# """

import torch
from torch.nn import Parameter
import torch.nn.functional as F


def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0.1):
    # Optional: BNInception uses label smoothing, apply it for retraining also
    # "Rethinking the Inception Architecture for Computer Vision", p. 6
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    # double classes
    #return torch.cat((T,T), dim=-1)
    # same number of classes as input
    return T


class ProxyNCA(torch.nn.Module):
    def __init__(self,
                 nb_classes = 15,
                 sz_embedding = 2048,
                 smoothing_const=0.0,
                 scaling_x=1,
                 scaling_p=3
                 ):
        torch.nn.Module.__init__(self)
        # initialize proxies s.t. norm of each proxy ~1 through div by 8
        # i.e. proxies.norm(2, dim=1)) should be close to [1,1,...,1]
        # TODO: use norm instead of div 8, because of embedding size
        self.proxies = Parameter(torch.randn(nb_classes, sz_embedding) / 8)
        self.smoothing_const = smoothing_const
        self.scaling_x = scaling_x
        self.scaling_p = scaling_p

    def forward(self, X, T):
        P = F.normalize(self.proxies, p=2, dim=-1) * self.scaling_p
        X = F.normalize(X, p=2, dim=-1) * self.scaling_x
        D = torch.cdist(X, P.cuda()) ** 2
        T = binarize_and_smooth_labels(T, len(P), self.smoothing_const)
        # note that compared to proxy nca, positive included in denominator
        loss = torch.sum(-T * F.log_softmax(-D, -1), -1)
        return loss.mean()


if __name__ == '__main__':

    nb_classes = 100
    sz_batch = 32
    sz_embedding = 64
    X = torch.randn(sz_batch, sz_embedding).cuda()
    P = torch.randn(nb_classes, sz_embedding).cuda()
    T = torch.randint(low=0, high=nb_classes, size=[sz_batch]).cuda()
    criterion = ProxyNCA(nb_classes, sz_embedding).cuda()

    print(criterion(X, T.view(sz_batch)))