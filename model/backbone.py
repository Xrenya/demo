from abc import ABC, abstractmethod
import torch.nn as nn
from torchvision.models import alexnet


class Backbone(ABC, nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.backbone = None

    def forward(self, x):
        x = self.backbone(x)
        return x


class AlexNet(Backbone):
    def __init__(self,
                 pretrained: bool = False,
                 progress: bool = False,
                 layer_index: int = 10):
        super().__init__()
        model = alexnet(pretrained, progress)
        self.backbone = model.features[:layer_index + 1]