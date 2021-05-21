from src.loss.wcel import WCEL, CEL
from torch.nn import BCEWithLogitsLoss
LOSS_TYPE = ['wcel', 'cel', 'bce']


class Call_Loss:
    def __init__(self,
                 loss_type):
        super(Call_Loss, self).__init__()

        if loss_type not in LOSS_TYPE:
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
