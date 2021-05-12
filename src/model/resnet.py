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

        self.encoder = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        self.fc = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, img):

        img = self.encoder(img)
        img = self.fc(img.squeeze())

        return img
