import timm
from torch import nn
from src.model.resnet import ResNet50


class Model_ViT(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Model_ViT, self).__init__()

        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, img):
        img = self.model(img)

        return img


class ViT_Hybrid(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ViT_Hybrid, self).__init__()

        self.model = timm.create_model('vit_base_resnet50_224_in21k', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, img):
        img = self.model(img)

        return img
