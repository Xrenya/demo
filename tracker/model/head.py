import torch.nn as nn
import torch.nn.functional as F


class RPN(nn.Module):
    """RPN Head"""
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        """
        Args:
            z_f: template features
            x_f: search features

        Returns:
            None
        """
        raise NotImplemented


class RPNHead(RPN):

    def __init__(self, anchor_num: int = 5):
        super(RPNHead, self).__init__()
        self.K = anchor_num

        self.cls_z = nn.Conv2d(256, 256 * 2 * self.K, kernel_size=3, stride=1, padding=0)
        self.reg_z = nn.Conv2d(256, 256 * 4 * self.K, kernel_size=3, stride=1, padding=0)

        self.cls_x = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.reg_x = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)

    def forward(self, z_ft, x_ft):
        N = z_ft.shape[0]

        cls_z = self.cls_z(z_ft)  # [N, 2K*256, 4, 4]
        reg_z = self.reg_z(z_ft)  # [N, 4K*256, 4, 4]

        cls_x = self.cls_x(x_ft)  # [N, 256, 20, 20]
        reg_x = self.reg_x(x_ft)  # [N, 256, 20, 20]

        # cross-correlation
        cls_z = cls_z.view(-1, 256, 4, 4)  # [N*2K, 256, 4, 4]
        cls_x = cls_x.view(1, -1, 20, 20)  # [1, N*256, 20, 20]
        pred_cls = F.conv2d(cls_x, cls_z, groups=N)  # [1, N*2K, 17, 17]
        pred_cls = pred_cls.view(N, -1, pred_cls.shape[2], pred_cls.shape[3])  # [N, 2K, 17, 17]

        reg_z = reg_z.view(-1, 256, 4, 4)  # [N*4K, 256, 4, 4]
        reg_x = reg_x.view(1, -1, 20, 20)  # [1, N*256, 20, 20]
        pred_reg = F.conv2d(reg_x, reg_z, groups=N)  # [1, N*4K, 17, 17]
        pred_reg = pred_reg.view(N, -1, pred_reg.shape[2], pred_reg.shape[3])  # [N, 4K, 17, 17]

        return pred_cls, pred_reg
