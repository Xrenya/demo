import torch.nn as nn


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


def conv2d_dw_group(x, kernel):
    batch, channel = kernel.shape[:2]
    x = x.view(1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class DepthwiseXCorr(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden: int,
                 out_channels: int,
                 kernel_size: int = 3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward_corr(self, kernel, input):
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        feature = conv2d_dw_group(input, kernel)
        return feature

    def forward(self, kernel, search):
        feature = self.forward_corr(kernel, search)
        out = self.head(feature)
        return out


class DepthwiseRPN(RPN):
    """DepthwiseRPN

    Args:
        batch (int): batch size for train, batch = 1 if test
        anchor_num  (int): number of anchors
        out_channels (int): hidden features channels

    Returns:
        cls (torch.tensor): class
        loc (torch.tensor): location
    """
    def __init__(self,
                 in_channels: int = 256,
                 anchor_num: int = 5,
                 out_channels: int = 256) -> None:
        super(DepthwiseRPN, self).__init__()
        self.cls = DepthwiseXCorr(
            in_channels=in_channels,
            hidden=2 * anchor_num,
            out_channels=out_channels
        )
        self.loc = DepthwiseXCorr(
            in_channels=in_channels,
            hidden=2 * anchor_num,
            out_channels=out_channels
        )

    def forward(self, z_f, x_f):
        """

        Args:
            z_f (torch.tensor): features
                [batch_size, channels, z_f height, z_f width]
            x_f (torch.tensor): features
                [batch_size, channels, x_f height, x_f width]

        Returns:
            cls (torch.tensor): TO DO
            loc (torch.tensor): TO DO
        """
        cls = self.cls(z_f, x_f)
        loc = self.cls(z_f, x_f)
        return cls, loc
