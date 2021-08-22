import torch
from torch.nn import BCEWithLogitsLoss, Module, AdaptiveMaxPool2d
from src.loss.wcel import WCEL, CEL
from src.loss.contrastive import ContrastiveAverage
from src.loss.triplet_avg import TripletAverage
from src.loss.proxynca import ProxyNCA
from src.loss.focal import FocalLoss

CLASS_LOSS_TYPE = ['wcel', 'cel', 'bce', 'focal']
METRIC_LOSS_TYPE = ['triplet', 'contrastive', 'proxy']

class Call_Loss:
    def __init__(self,
                 loss_type):
        super(Call_Loss, self).__init__()

        if loss_type not in CLASS_LOSS_TYPE:
            raise ValueError(
                'Wrong loss {} '.format(loss_type))
        else:
            self.loss_type_map = {
                'wcel': WCEL,
                'cel': CEL,
                'bce': BCEWithLogitsLoss
            }
            self.loss_type = loss_type

    def __call__(self):
        return self.loss_type_map[self.loss_type]()


class Loss_Wrapper(Module):
    def __init__(self, class_type, metric_type):
        super(Loss_Wrapper, self).__init__()

        self.loss_type_map = {
            'wcel': WCEL,
            'cel': CEL,
            'bce': BCEWithLogitsLoss,
            'contrastive': ContrastiveAverage,
            'triplet': TripletAverage,
            'proxy': ProxyNCA,
            'focal': FocalLoss,
            None: lambda *args: None
        }

        if class_type not in CLASS_LOSS_TYPE and class_type is not None:
            raise ValueError(
                'Wrong loss {} '.format(class_type))

        self.class_loss = self.loss_type_map[class_type]()

        if metric_type not in METRIC_LOSS_TYPE and metric_type is not None:
            raise ValueError(
                'Wrong loss {} '.format(metric_type))

        self.metric_loss = self.loss_type_map[metric_type]()
        self.pool = AdaptiveMaxPool2d(1)

    def forward(self, emb, preds, targets):

        if self.class_loss is None:
            cl = 0.0
        else:
            cl = self.class_loss(preds, targets)

        if self.metric_loss is None:
            ml = 0.0
        else:
            ml = self.metric_loss(self.pool(emb).squeeze(), targets)

        return cl + ml

