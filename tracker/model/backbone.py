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


class SiameseAlexNet(Backbone):
    def __init__(self, ):
        super(SiameseAlexNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),
        )
