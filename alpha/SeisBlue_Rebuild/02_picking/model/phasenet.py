# -*- coding: utf-8 -*-
from torch import nn, cat, optim
import torch
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint


class DoubleConv(nn.Module):
    """(convolution => ReLU => Dropout) * 2"""
    def __init__(self, out_channels, kernel_size, dropout=0.1):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.LazyConv2d(out_channels, kernel_size, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, out_channels, kernel_size, pool_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pool_size),
            DoubleConv(out_channels, kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, out_channels, kernel_size, stride_size):
        super().__init__()
        self.up = nn.LazyConvTranspose2d(out_channels, kernel_size,
                                         stride=stride_size)
        self.double_conv = DoubleConv(out_channels, kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = cat((x1, x2), dim=1)
        x1 = self.double_conv(x1)
        return x1


class OutConv(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.conv = nn.Sequential(
            nn.LazyConv2d(num_class, (1, 1), padding='same'),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, num_class=3):
        super().__init__()
        # out_channels = [8, 16, 32, 64, 128]
        out_channels = [8, 11, 16, 22, 32]
        pool_size = (1, 2)
        kernel_size = (1, 7)

        self.inc = (DoubleConv(out_channels[0], kernel_size))
        self.down1 = (Down(out_channels[1], kernel_size, pool_size))
        self.up1 = (Up(out_channels[1], kernel_size, stride_size=pool_size))
        self.outc = (OutConv(num_class))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.up1(x2, x1)
        output = self.outc(x3)
        return output

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

# ======================================================


class Conv1dSame(nn.Module):
    """
    Add PyTorch compatible support for Tensorflow/Keras padding option: padding='same'.
    Discussions regarding feature implementation:
    https://discuss.pytorch.org/t/converting-tensorflow-model-to-pytorch-issue-with-padding/84224
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-598264120
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.cut_last_element = (
            kernel_size % 2 == 0 and stride == 1 and dilation % 2 == 1
        )
        self.padding = math.ceil((1 - stride + dilation * (kernel_size - 1)) / 2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding+1,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x):
        if self.cut_last_element:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)


class PhaseNet(nn.Module):
    def __init__(
        self, in_channels=3, classes=3, sampling_rate=100, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_samples = 3001
        self.sampling_rate = sampling_rate
        self.classes = classes
        self.kernel_size = 7
        self.stride = 4
        self.activation = torch.relu

        self.inc = nn.Conv1d(self.in_channels, 8, 1)
        self.in_bn = nn.BatchNorm1d(8)

        self.conv1 = Conv1dSame(8, 11, self.kernel_size, self.stride)
        self.bnd1 = nn.BatchNorm1d(11)

        self.conv2 = Conv1dSame(11, 16, self.kernel_size, self.stride)
        self.bnd2 = nn.BatchNorm1d(16)

        self.conv3 = Conv1dSame(16, 22, self.kernel_size, self.stride)
        self.bnd3 = nn.BatchNorm1d(22)

        self.conv4 = Conv1dSame(22, 32, self.kernel_size, self.stride)
        self.bnd4 = nn.BatchNorm1d(32)

        self.up1 = nn.ConvTranspose1d(
            32, 22, self.kernel_size, self.stride, padding=self.conv4.padding
        )
        self.bnu1 = nn.BatchNorm1d(22)

        self.up2 = nn.ConvTranspose1d(
            44,
            16,
            self.kernel_size,
            self.stride,
            padding=self.conv3.padding,
            output_padding=1,
        )
        self.bnu2 = nn.BatchNorm1d(16)

        self.up3 = nn.ConvTranspose1d(
            32, 11, self.kernel_size, self.stride, padding=self.conv2.padding
        )
        self.bnu3 = nn.BatchNorm1d(11)

        self.up4 = nn.ConvTranspose1d(22, 8, self.kernel_size, self.stride, padding=3)
        self.bnu4 = nn.BatchNorm1d(8)

        self.out = nn.ConvTranspose1d(16, self.classes, 1)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, logits=False):
        x_in = self.activation(self.in_bn(self.inc(x)))

        x1 = self.activation(self.bnd1(self.conv1(x_in)))
        x2 = self.activation(self.bnd2(self.conv2(x1)))
        x3 = self.activation(self.bnd3(self.conv3(x2)))
        x4 = self.activation(self.bnd4(self.conv4(x3)))

        x = torch.cat([self.activation(self.bnu1(self.up1(x4))), x3], dim=1)
        x = torch.cat([self.activation(self.bnu2(self.up2(x))), x2], dim=1)
        x = torch.cat([self.activation(self.bnu3(self.up3(x))), x1], dim=1)
        x = torch.cat([self.activation(self.bnu4(self.up4(x))), x_in], dim=1)

        x = self.out(x)
        if logits:
            return x
        else:
            return self.softmax(x)


if __name__ == "__main__":
    pass