import os
import cv2
import torch
import yaml
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from src.dataloader.utils import parse_csv

KEYS = ['No Finding',
        'Enlarged Cardiomediastinum',
        'Cardiomegaly',
        'Lung Opacity',
        'Lung Lesion',
        'Edema',
        'Consolidation',
        'Pneumonia',
        'Atelectasis',
        'Pneumothorax',
        'Pleural Effusion',
        'Pleural Other',
        'Fracture',
        'Support Devices',
        'Uncertainty']


class Chexpert:
    def __init__(self, config):
        self.config = config
        self.mode = self.config['mode']
        self.root = self.config['root_path']
        self.train_csv_path = os.path.join(self.root, 'train.csv')
        self.test_csv_path = os.path.join(self.root, 'valid.csv')
        self.transform = transforms.Compose([transforms.ToTensor()])

        if self.mode == 'train':
            self.header, self.csv = parse_csv(self.train_csv_path)
        elif 'test':
            self.header, self.csv = parse_csv(self.test_csv_path)
        else:
            raise ValueError('Unknown mode {}.'.format(self.mode))

        self.data = self.config['data']

    def __len__(self):
        return len(self.csv)

    @staticmethod
    def parse_label(data):

        # label whether each observation was mentioned as
        # confidently present (1),
        # confidently absent (0),
        # uncertainly present (-1),
        # or not mentioned (blank) for the dataset

        target = torch.zeros(len(KEYS))
        for idx, k in enumerate(KEYS):
            if k != 'Uncertainty':
                if data[k].split('\n')[0] == '1.0':
                    target[idx] = 1
                else:
                    target[idx] = 0

        if torch.sum(target) == 0:
            target[-1] = 1

        return target

    def __getitem__(self, index):
        data = {k: v for k, v in zip(self.header, self.csv[index])}
        img_path = os.path.split(self.root)[0]
        img_path = os.path.join(img_path, data['Path'])
        img = cv2.imread(img_path)
        img = cv2.resize(img, tuple(self.data['size']))
        img = self.transform(img)
        target = self.parse_label(data)
        return img, target


class Chexpert_smaller(Chexpert):
    def __init__(self, config):
        super(Chexpert_smaller, self).__init__(config)

        self.csv = self.csv[:56060]


if __name__ == '__main__':

    with open(os.path.abspath("/home/demet/PycharmProjects/thesis/configs/config_ubuntu.yaml"), 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    c = Chexpert(config)
    dl = DataLoader(c)

    for img_batch in tqdm(dl):
        ...
