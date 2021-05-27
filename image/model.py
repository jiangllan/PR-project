import torch.nn as nn
import torchvision.models as models


class ShopeeImageEmbeddingNet(nn.Module):
    def __init__(self, model):
        super(ShopeeImageEmbeddingNet, self).__init__()

        if model == "resnet18":
            model = models.resnet18(True)
        elif model == "resnet50":
            model = models.resnet50(True)
        elif model == "resnet101":
            model = models.resnet101(True)
        elif model == "resnet152":
            model = models.resnet152(True)
        else:
            raise NotImplementedError
        model.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        self.model = model

    def forward(self, img):
        out = self.model(img)
        return out
