import math
from torch import nn
import torch.utils.model_zoo as model_zoo
from torchvision.models import ResNet
import torchvision.transforms as transforms
import torchvision.models as models


# resnet50 = models.resnet50(pretrained=True)


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.encoder = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, img):

        emb = self.encoder(img)
        img = self.fc(self.pool(emb).squeeze())

        return img, emb
