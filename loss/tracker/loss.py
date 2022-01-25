from torch.nn.modules.loss import _Loss
import torch.nn as nn
import torch


class SiamRPNLoss(_Loss):
    def __init__(self, loc_c: int = 10, cls_c: int = 5):
        super(SiamRPNLoss, self).__init__()
        self.conf_loss = nn.CrossEntropyLoss()
        self.loc_c = loc_c
        self.cls_c = cls_c

    def weight_l1_loss(self, pred_loc, label_loc, loss_weight):
        """Compute weight_l1_loss"""
        batch, channels, height, width = pred_loc.shape
        pred_loc = pred_loc.reshape(batch, 4, -1, height, width)
        diff = torch.abs(pred_loc - label_loc)
        diff = torch.sum(diff, axis=1).reshape(batch, -1, height, width)
        loss = diff * loss_weight
        return F.sum(loss) / batch

    def get_cls_loss(self, cls_pred, label, select):
        if len(select) == 0:
            return 0
        cls_pred = torch.index_select(cls_pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.conf_loss(cls_pred, label)
        #return F.nll_loss(pred, label)

    def cross_entropy_loss(self, cls_pred, label, pos_index, neg_index):
        batch, channels, height, width = cls_pred.shape
        cls_pred = cls_pred.reshape(batch, 2, self.loc_c // 2, height, width)
        cls_pred = torch.transpose(cls_pred, (0, 2, 3, 4, 1))
        cls_pred = cls_pred.reshape(-1, 2)
        label = label.reshape(-1)
        loss_pos = self.get_cls_loss(cls_pred, label, pos_index)
        loss_neg = self.get_cls_loss(cls_pred, label, neg_index)
        return loss_pos * 0.5 + loss_neg * 0.5

    def forward(self, cls_pred, loc_pred, label_cls, pos_index, neg_index,
                label_loc, label_loc_weight):
        """Compute loss"""
        loc_loss = self.weight_l1_loss(loc_pred, label_loc, label_loc_weight)
        cls_loss = self.cross_entropy_loss(
            cls_pred, label_cls, pos_index, neg_index
        )
        return {
            "cls_loss": loc_loss, "cls_loss": cls_loss
        }
