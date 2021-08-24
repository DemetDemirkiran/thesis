import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
import os
from PIL import Image
from tqdm import tqdm
import yaml
from src.dataloader.chexpert import Chexpert
from src.dataloader.chest14 import XrayLoader14
from torch.utils.data.dataloader import DataLoader
import torch

if __name__ == '__main__':

    means = []
    variances = []

    with open(os.path.abspath("/home/demet/PycharmProjects/thesis/configs/config_ubuntu.yaml"), 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    c = Chexpert(config)
    dl = DataLoader(c, batch_size=128)

    h, w = 224, 244
    pixels = len(c) * h * w
    sum = 0

    for img_batch, targets in tqdm(dl):
        sum += img_batch[:, 0, :].sum()
    mean = sum / pixels

    sequence_error = 0
    for img_batch, targets in tqdm(dl):
        sequence_error += ((img_batch[:, 0, :] - mean).pow(2)).sum()
    variance = torch.sqrt(sequence_error / pixels)

    print('mean: ', torch.mean(mean), ' variance: ', torch.mean(variance))
