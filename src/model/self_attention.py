import math
import torch
from torch import nn
from src.model.resnet import ResNet50


class Conv1x1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv1x1, self).__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1)

    def forward(self, x):
        return self.conv1x1(x)


class Self_Attention(nn.Module):

    def __init__(self, in_dim, reduction_factor=8):
        super(Self_Attention, self).__init__()
        self.in_dim = in_dim
        self.reduction_factor = reduction_factor
        self.query_conv = Conv1x1(self.in_dim, self.in_dim // self.reduction_factor)
        self.key_conv = Conv1x1(self.in_dim, self.in_dim // self.reduction_factor)
        self.value_conv = Conv1x1(self.in_dim, self.in_dim // self.reduction_factor)
        self.out_conv = Conv1x1(self.in_dim // self.reduction_factor, self.in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, channels, height, width = x.size()

        query_proj = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        key_proj = self.key_conv(x).view(m_batchsize, -1, width * height)
        value_proj = self.value_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(key_proj, query_proj)
        attention = self.softmax(energy)
        out = torch.bmm(attention, value_proj)
        out = self.out_conv(out.view(m_batchsize,
                                     channels // self.reduction_factor, height, width))
        return self.gamma * out + x


class ResNet50_SA(ResNet50):
    def __init__(self, num_classes):
        super(ResNet50_SA, self).__init__(num_classes)

        self.sa = Self_Attention(2048)

    def forward(self, img):
        img = self.encoder(img)
        img = self.sa(img)
        img = self.fc(img.squeeze())

        return img
