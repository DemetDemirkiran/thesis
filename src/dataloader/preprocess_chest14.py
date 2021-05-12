import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
import os
from PIL import Image
from tqdm import tqdm

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


class preprocess(data.Dataset):

    def __init__(self, path, mode='preprocess'):
        super(preprocess, self).__init__()
        self.mode = mode
        self.path = path
        self.train_path = os.path.join(self.path, 'train_val_list.txt')
        self.test_path = os.path.join(self.path, 'test_list.txt')
        preprocess_txt(self.train_path, self.test_path)
        self.trainlist, self.evallist, self.testlist = preprocess_txt(self.train_path, self.test_path)
        self.preprocess = np.hstack([self.trainlist, self.evallist, self.testlist])

    def __len__(self):
        if self.mode == 'preprocess':
            return len(self.preprocess)
        else:
            raise ValueError('Wrong mode')

    def __getitem__(self, item):

        if self.mode == 'preprocess':
            img_list = self.preprocess
        else:
            raise ValueError('Wrong mode')

        img_name = img_list[item]
        img = Image.open(os.path.join(self.path, 'images', img_name))
        img = img.resize((256, 256))

        preprocess_imgs = '/home/demet/Desktop/Chest_Dataset/Chest_Dataset_Resize'
        if not os.path.exists(preprocess_imgs):
            os.mkdir(preprocess_imgs)

        img.save(os.path.join(preprocess_imgs, img_name))

        return img


if __name__ == '__main__':

    d = preprocess('/home/demet/Desktop/Chest_Dataset')
    for img in tqdm(d):
        # plt.close('all')
        # plt.imshow(img[0], cmap='gray')
        # plt.show()
        ...
