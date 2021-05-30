import torch
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
        self.model = model

    def forward(self, img):
        out = self.model(img)
        return out


class TripletShopeeImageEmbeddingNet(nn.Module):
    def __init__(self, model):
        super(TripletShopeeImageEmbeddingNet, self).__init__()
        self.model = model

        if model == "resnet50":
            base_model = models.resnet50(True)
        elif model == "densenet201":
            base_model = models.densenet201(True)
        elif model == "wide_resnet50_2":
            base_model = models.wide_resnet50_2(True)
        else:
            raise NotImplementedError
        if "res" in model:
            base_model.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        base_model = nn.Sequential(*list(base_model.children())[:-1])
        self.base_model = base_model

    def forward(self, anchor, positive=None, negative=None):
        if positive is not None and negative is not None:
            anchor_feature = self.base_model(anchor)
            positive_feature = self.base_model(positive)
            negative_feature = self.base_model(negative)
            if "densenet" in self.model:
                anchor_feature = torch.nn.functional.adaptive_avg_pool2d(anchor_feature, output_size=(1, 1))
                positive_feature = torch.nn.functional.adaptive_avg_pool2d(positive_feature, output_size=(1, 1))
                negative_feature = torch.nn.functional.adaptive_avg_pool2d(negative_feature, output_size=(1, 1))
            return anchor_feature, positive_feature, negative_feature
        else:
            anchor_feature = self.base_model(anchor)
            if "densenet" in self.model:
                anchor_feature = torch.nn.functional.adaptive_avg_pool2d(anchor_feature, output_size=(1, 1))
            return anchor_feature