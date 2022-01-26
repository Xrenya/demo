from head import RPNHead
from backbone import SiameseAlexNet
import torch.nn as nn
import torch.nn.functional as F


class SiamRPN(nn.Module):
    def __init__(self):
        super(SiamRPN, self).__init__()
        self.backbone = SiameseAlexNet()
        self.rpn_head = RPNHead()
        self.z = None
        self.x = None
        self.cls = None
        self.loc = None

    def template(self, z_f):
        self.z = self.backbone(z_f)

    def track(self, x):
        x = self.backbone(x)
        self.cls, self.loc = self.rpn_head(self.z, x)
        return {
            "cls": self.cls, "loc": self.loc
        }

    def forward(self, template, search):
        z = self.backbone(template)
        x = self.backbone(search)
        self.cls, self.loc = self.rpn_head(z, x)
        return {
            "cls": self.cls, "loc": self.loc
        }
