import logging
import os
import torch
import yaml
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from src.logger import Logger
from src.model.call_model import Call_Model
from src.dataloader.chexpert import Chexpert, Chexpert_smaller
from src.dataloader.chest14 import XrayLoader14
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score, auc, average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_score
## Multi-label case for auc computation dependencies
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import RidgeClassifierCV

## binary case & roc_auc_score
from sklearn.linear_model import LogisticRegression

class EvalTest:
    def __init__(self, config, check_pnt=None, model=None):
        super(EvalTest, self).__init__()

        self.config = config
        self.mode = self.config['mode']
        self.model = self.config['model']
        self.test = self.config['test']
        # self.model = model

        if check_pnt is not None:
            self.checkpoint = check_pnt
            self.mode = "test"
            self.model = model
        else:
            self.checkpoint = self.test['checkpoint_path']
            self.mode = self.config['mode']

    def dataloader(self):
        data = self.test['dataset_path']
        if self.test['name'] == 'chest14':
            data = XrayLoader14(self.config)
        elif self.test['name'] == 'chex_small':
            data = Chexpert_smaller(self.config)
        else:
            data = Chexpert(self.config)

        ##  WRITE SAMPLIG SIZE FOR TRAINING
        batch_sz = self.test['batch']
        # learning_rate = self.training['learning_rate']
        dataloader = DataLoader(dataset=data, batch_size=batch_sz,
                                num_workers=0, pin_memory=True, drop_last=True)

        return dataloader

    def label_index(self):
        data = self.test['dataset_path']
        label_index = []

        if self.test['name'] == 'chest14':
            data = XrayLoader14(self.config)
            label_index.append(data[1])
        elif self.test['name'] == 'chex_small':
            data = Chexpert_smaller(self.config)
            label_index.append(data[1])
        else:
            data = Chexpert(self.config)
            label_index.append(data[1])

        return label_index

    def model_type(self):

        model_type = Call_Model(
            model_type=self.model['model_type'],
            num_classes=self.model['number_classes']
        )
        checkpnt = torch.load(self.checkpoint,
                              map_location=lambda storage, loc: storage.cpu())
        model_type = model_type()
        model_type.load_state_dict(checkpnt, strict=True)

        return model_type.cuda()

    def loss(self):
        # write a loss thats worth something
        loss = BCEWithLogitsLoss()
        return loss

    def area_under_curve(self, targets, out):

        with torch.no_grad():
            softmax = torch.nn.Softmax(-1)
            soft_out = softmax(out).cpu().numpy()
            targets = targets.cpu().numpy()

            fpr_d = dict()
            tpr_d = dict()
            auc_d = dict()
            for i in range(15):
                fpr_d[i], tpr_d[i], _ = roc_curve(targets[:, i], soft_out[:, i])
                auc_d[i] = auc(fpr_d[i], tpr_d[i])

        return auc_d, fpr_d, tpr_d

    def class_predictions(self, dataloader, model):
        pred_list = []
        label_list = []
        i = 0
        result_dict = dict()

        with torch.no_grad():
            total_accuracy = 0.0
            for images, targets in tqdm(dataloader):
                images = images.cuda()
                targets = targets.cuda()
                out = model(images)

                pred = out.data
                # pred = torch.nn.Sigmoid()(pred) > 0.5
                # pred = pred.cpu().numpy()
                # targets = targets.cpu().numpy()
                pred_list.append(pred)
                label_list.append(targets)
                i += 1
                # if i > 10:
                #     break
                # get prediction of images for target, do  it for every class then compute precision
        pred_list = torch.vstack(pred_list)
        label_list = torch.vstack(label_list)
        auc, fpr, tpr = self.area_under_curve(label_list, pred_list)
        return auc, fpr, tpr


    def __call__(self, *args, **kwargs):
        dataloader = self.dataloader()
        loss = self.loss()
        model = self.model_type()
        model.eval()
        area_under_curve = self.area_under_curve
        checkpoint = self.checkpoint

        result, fpr, tpr = self.class_predictions(dataloader, model)

        for k in np.sort(list(result.keys())):
            print('{:.4f} '.format(result[k]))

        return result, fpr, tpr


if __name__ == '__main__':
    from src.test.ev_test import EvalTest

    with open(os.path.abspath("/home/demet/PycharmProjects/thesis/configs/config.yaml"), 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    t = EvalTest(config)
    t()
    ...
