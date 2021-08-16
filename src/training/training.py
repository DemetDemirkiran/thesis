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
from src.loss.call_loss import Call_Loss, Loss_Wrapper
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
from src.loss.wcel import CEL, WCEL
import random
from sparsemax import Sparsemax


class Training:
    def __init__(self, config):
        super(Training, self).__init__()

        self.config = config
        self.mode = self.config['mode']
        self.training = self.config['train']
        # self.gpu = self.config['gpu']
        self.model = self.config['model']
        self.logger = logging.getLogger()

    def dataloader(self):

        data = self.training['data_path']
        if self.training['name'] == 'chest14':
            data = XrayLoader14(self.config)
        elif self.training['name'] == 'chex_small':
            data = Chexpert_smaller(self.config)
        else:
            data = Chexpert(self.config)

        ##  WRITE SAMPLIG SIZE FOR TRAINING
        batch_sz = self.training['batch']
        learning_rate = self.training['learning_rate']
        dataloader = DataLoader(dataset=data, batch_size=batch_sz,
                                num_workers=0, pin_memory=True, drop_last=True)

        return dataloader

    def model_type(self):

        model_type = Call_Model(
            model_type=self.model['model_type'],
            num_classes=self.model['number_classes']
        )

        return model_type().cuda()

    def call_loss(self):
        loss_type = Loss_Wrapper(self.config['train']['loss'], self.config['train']['metric'])
        # loss_type = Call_Loss(
        #     loss_type=self.config['train']['loss']
        # )
        return loss_type

    def logger(self):
        base_path = os.path.join(self.training['root_path'])
        if not os.path.exists(base_path):
            os.makedirs(os.path.join(base_path, "summary"))
            os.makedirs(os.path.join(base_path, "checkpoints"))
        with open(os.path.join(base_path, "config.yaml"), "w") as f:
            f.write(self.training.dump())

    def tb_writer(self):
        summary_path = os.path.join(
            self.config['log_dir'], self.config['experiment_name'], "summary")

        if not os.path.exists(summary_path):
            os.makedirs(summary_path)

        try:
            writer = SummaryWriter(log_dir=summary_path)
        except TypeError:
            writer = SummaryWriter(logdir=summary_path)
        return writer

    @staticmethod
    def add_tb_logs(writer, data, output_dict, loss, global_step, optimizer):
        writer.add_scalar('Loss/Total', loss['total'], global_step)
        writer.add_scalar('Loss/ID', loss['id'], global_step)
        writer.add_scalar('Loss/Metric', loss['metric'], global_step)
        writer.add_scalar('Loss/Recon', loss['recon'], global_step)

        # accuracy = eval.eval_accuracy(output_dict, data)
        # writer.add_scalar('Accuracy/Vision', accuracy['vision_preds'], global_step)

    def area_under_curve(self, targets, out):

        with torch.no_grad():
            softmax = torch.nn.Softmax(-1)
            soft_out = softmax(out).cpu().numpy()
            targets = targets.cpu().numpy()

            auc_calc = []
            for t, so in zip(targets, soft_out):
                # auc_calc.append(roc_auc_score(t, so))
                fpr, tpr, thresholds = roc_curve(t, so)
                auc_calc.append(auc(fpr, tpr))

        return auc_calc

    def get_mask_indices(self, num_labels, mask, known_labels=0, epoch=1):
            # sample random number of known labels during training
        if known_labels > 0:
            random.seed()
            num_known = random.randint(0, int(num_labels * 0.75))
        else:
            num_known = 0

        mask_indices = random.sample(range(num_labels), (num_labels - num_known))
        mask.scatter_(1, torch.Tensor(mask_indices).long().repeat(mask.shape[0], 1).cuda(), -1)
        return mask

    def __call__(self, *args, **kwargs):

        dataloader = self.dataloader()
        loss = self.call_loss()
        model = self.model_type()
        area_under_curve = self.area_under_curve
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=self.training['learning_rate']['base_rate']
                                     # momentum=self.training['learning_rate']['momentum']
                                     )
        if self.config['train']['metric'] == 'proxy':
            optimizer = torch.optim.Adam([
                                            {
                                                **{'params': list(model.parameters())},
                                            },
                                            {  # proxy nca parameters
                                                **{'params': list(loss.metric_loss.parameters())},
                                            }
                                         ],
                                         lr=self.training['learning_rate']['base_rate'],
                                         # momentum=self.training['learning_rate']['momentum']
                                         )

        lr_scheduler = StepLR(optimizer=optimizer,
                              step_size=self.training['learning_rate']['steps'],
                              gamma=self.training['learning_rate']['decay'])

        for epoch in range(self.training['start_epoch'],
                           self.training['end_epoch']):
            total_loss = 0.0
            total_accuracy = 0.0
            auc_calc = []

            for images, targets in tqdm(dataloader):
                optimizer.zero_grad()  # resets the gradients it needs to update
                images = images.cuda()
                targets = targets.cuda()
                if self.model['model_type'] == 'ctran':
                    out, emb, _ = model(images,
                                        self.get_mask_indices(self.config['model']['number_classes'],
                                        targets.clone())
                                        )

                else:
                    out, emb = model(images)
                # auc_calc.append(area_under_curve(targets, out))
                iter_loss = loss(emb, out, targets)
                pred = out.data
                pred = torch.nn.Sigmoid()(pred) > 0.5
                pred = pred.cpu().numpy()
                targets = targets.cpu().numpy()
                iter_accuracy = accuracy_score(targets, pred)
                total_accuracy += iter_accuracy / len(dataloader)
                iter_loss.backward()
                optimizer.step()
                total_loss += iter_loss.data / len(dataloader)
            # print(total_loss.data)
            # print('area under the curve {} for epoch {}'.format(np.mean(auc_calc), epoch))
            print(
                ('epoch {} total loss {} total accuracy {}'.format(epoch, total_loss, total_accuracy)))
            lr_scheduler.step()
            # plot(loss_per_epoch, epoch, bsz, lr)
            # Test script alternatively save each epoch ckpt and test each ckpt
            if epoch % self.config['train']['chkpnt_step'] == 0 and epoch > 0:

                out_path = os.path.join(self.config['log_dir'], self.config['train']['name'])
                out_path = os.path.join(out_path, self.config['experiment_name'] + self.config['model']['model_type'] +
                                        '_lr' + str(self.training['learning_rate']['base_rate']) +
                                        '_bs' + str(self.training['batch']))
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                torch.save(model.state_dict(), os.path.join(out_path, str(epoch) + '.pth'))


if __name__ == '__main__':
    from src.training.training import Training

    with open(os.path.abspath("/home/demet/PycharmProjects/thesis/configs/config.yaml"), 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    t = Training(config)
    t()
    ...
