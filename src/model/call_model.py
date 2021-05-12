from src.model.resnet import ResNet50
from src.model.self_attention import ResNet50_SA
from src.model.resnet_cbam import ResNet50_CBAM
from src.model.visual_transformer import ViT
from src.model.vit_hybrid import ViT_Hybrid

MODEL_TYPE = ['resnet50', 'resnet_sa', 'cbam', 'vit', 'vit_hybrid']


class Call_Model:
    def __init__(self,
                 model_type,
                 num_classes):
        super(Call_Model, self).__init__()

        if model_type not in MODEL_TYPE:
            raise ValueError(
                'Wrong model {} '.format(model_type))
        else:
            self.model_type_map = {
                'resnet50': ResNet50,
                'resnet_sa': ResNet50_SA,
                'cbam': ResNet50_CBAM,
                'vit': ViT,
                'vit_hybrid': ViT_Hybrid
            }
            self.model_type = model_type
            self.num_classes = num_classes

    def __call__(self):
        return self.model_type_map[self.model_type](self.num_classes)
