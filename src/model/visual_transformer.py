import timm
from torch import nn


class ViT(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ViT, self).__init__()

        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, img):

        img = self.model(img)

        return img









