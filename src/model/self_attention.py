import math
import torch
from torch import nn
from src.model.resnet import ResNet50


class Convolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Convolution, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1)

    def forward(self, x):

        return self.conv(x)


class Self_Attention(nn.Module):

    def __init__(self, in_dim, reduction_factor=8):
        super(Self_Attention, self).__init__()

        self.input = in_dim
        self.reduction = reduction_factor
        self.query = Convolution(self.input, self.input // self.reduction)
        self.key = Convolution(self.input, self.input // self.reduction)
        self.value = Convolution(self.input, self.input // self.reduction)
        self.out = Convolution(self.input // self.reduction, self.input)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        num_batch, channels, height, width = x.size()

        query_calc = self.query(x).view(num_batch, -1, width * height).permute(0, 2, 1)
        key_calc = self.key(x).view(num_batch, -1, width * height)
        value_calc = self.value(x).view(num_batch, -1, width * height)

        energy = torch.bmm(key_calc, query_calc)
        attention = self.softmax(energy)
        out = torch.bmm(attention, value_calc)
        out = self.out(out.view(num_batch,
                                channels // self.reduction, height, width))

        return self.gamma * out + x


class ResNet50_SA(ResNet50):
    def __init__(self, num_classes):
        super(ResNet50_SA, self).__init__(num_classes)

        self.self_attention = Self_Attention(2048)

    def forward(self, img):

        img = self.encoder(img)
        img = self.self_attention(img)
        img = self.fc(img.squeeze())

        return img
