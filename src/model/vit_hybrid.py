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


class ViT_Hybrid(ResNet50):
    def __init__(self, num_classes, pretrained=True):
        super(ViT_Hybrid, self).__init__(num_classes)

        self.hybrid = Model_ViT(2048)

    def forward(self, img):

        img = self.encoder(img)
        img = self.fc(img.squeeze())
        img = self.model(img)

        return img
