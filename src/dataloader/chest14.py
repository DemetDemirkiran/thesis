import numpy
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torch.utils import data
import os
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import yaml

class_dict = {'Atelectasis': 0,
              'Cardiomegaly': 1,
              'Effusion': 2,
              'Infiltration': 3,
              'Mass': 4,
              'Nodule': 5,
              'Pneumonia': 6,
              'Pneumothorax': 7,
              'Consolidation': 8,
              'Edema': 9,
              'Emphysema': 10,
              'Fibrosis': 11,
              'Pleural_Thickening': 12,
              'Hernia': 13,
              'No Finding': 14
              }

def load_text(path):
    with open(path, 'r') as f:
        text = f.readlines()
        text = [t.split('\n')[0] for t in text]
    return text


def preprocess_txt(train_path, test_path, seed=2020):
    np.random.seed(seed)
    train_names = load_text(train_path)
    test_names = load_text(test_path)
    total_len = len(train_names) + len(test_names)
    train_set = np.random.choice(train_names, int(total_len * 0.7), replace=False)
    val_set = list(set(train_names).symmetric_difference(train_set))
    return train_set, val_set, test_names


def csv2dict(path):
    with open(path, 'r') as f:
        text = f.readlines()
    keys = text[0].split(',')
    text = text[1:]
    metadata = dict()
    for t in text:
        data = t.split(',')
        idx = data[0]
        metadata[idx] = dict()
        for en, k in enumerate(keys):
            if k != 'Finding Labels':
                metadata[idx][k] = data[en]
            else:
                labels = data[en]
                labels = labels.split('|')
                labels = [class_dict[l] for l in labels]
                metadata[idx]['labels'] = labels
    return metadata


class XrayLoader14(data.Dataset):

    def __init__(self, config, mode='train'):
        super(XrayLoader14, self).__init__()
        self.config = config
        self.mode = self.config['mode']
        self.root = self.config['root_path']
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.train_path = os.path.join(self.root, 'train_val_list.txt')
        self.test_path = os.path.join(self.root, 'test_list.txt')
        self.meta_path = os.path.join(self.root, 'Data_Entry_2017_v2020.csv')
        preprocess_txt(self.train_path, self.test_path)
        self.trainlist, self.evallist, self.testlist = preprocess_txt(self.train_path, self.test_path)
        self.metadata = csv2dict(self.meta_path)

    def __len__(self):
        if self.mode == 'train':
            return len(self.trainlist) #// 50
        elif self.mode == 'eval':
            return len(self.evallist) #// 50
        elif self.mode == 'test':
            return len(self.testlist) #// 50
        else:
            raise ValueError('Wrong mode')

    @staticmethod
    def parse_label(data):
        target = torch.zeros(len(class_dict))
        for idx, k in enumerate(class_dict):
            if data[k].split('\n')[0] == '1.0':
                target[idx] = 1
            else:
                target[idx] = 0
        return target


    def __getitem__(self, item):

        if self.mode == 'train':
            img_list = self.trainlist
        elif self.mode == 'eval':
            img_list = self.evallist
        elif self.mode == 'test':
            img_list = self.testlist
        else:
            raise ValueError('Wrong mode')

        # img_name = img_list[item]
        # img = Image.open(os.path.join(self.path, 'Chest_Dataset_Resize', img_name))
        # img = Image.open(os.path.join(self.path, img_name))
        # img_path = os.path.split(self.path)[0]

        img_name = img_list[item]
        # data = {k: v for k, v in zip(self.header, self.csv[item])}
        img_path = os.path.split(self.root)[0]
        img_path = os.path.join(self.root, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, tuple(self.config['data']['size']))
        img = self.transform(img)
        meta = self.metadata[img_name]
        label = torch.zeros(15, dtype=torch.float)
        label[meta['labels']] = 1
        c, h, w = img.shape
        if c != 3:
            raise ValueError('whoopsie image not 3 channels')

        return img, label


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    with open(os.path.abspath("/home/demet/PycharmProjects/thesis/configs/config.yaml"), 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    c = XrayLoader14(config)
    dl = DataLoader(c)

    for img, meta in tqdm(dl):
        #    plt.close('all')
        #    plt.imshow(img[0], cmap='gray')
        #    plt.show()
        ...
