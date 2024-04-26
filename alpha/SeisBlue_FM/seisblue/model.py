# -*- coding: utf-8 -*-
from torch import nn, cat, optim
import torch
import math


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


class FMNet(nn.Module):
    def __init__(
        self, in_channels=1, classes=3, sampling_rate=100, npts=256, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_samples = npts
        self.sampling_rate = sampling_rate
        self.classes = classes
        self.kernel_size = 11
        self.stride = 4
        self.activation = torch.relu
        self.pool = nn.MaxPool1d(20)
        self.dropout = nn.Dropout(p=0.1)

        self.inc = nn.Conv1d(self.in_channels, 256, 1)
        self.in_bn = nn.BatchNorm1d(256)

        self.conv1 = Conv1dSame(256, 128, self.kernel_size, self.stride)
        self.bnd1 = nn.BatchNorm1d(128)

        self.conv2 = Conv1dSame(128, 64, self.kernel_size, self.stride)
        self.bnd2 = nn.BatchNorm1d(64)

        self.conv3 = Conv1dSame(64, 32, self.kernel_size, self.stride)
        self.bnd3 = nn.BatchNorm1d(32)

        self.flatten = nn.Flatten(1, -1)
        self.fc1 = nn.Linear(128, 40)
        self.fc2 = nn.Linear(40, 3)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x_in = self.activation(self.in_bn(self.inc(x)))

        x1 = self.activation(self.bnd1(self.conv1(x_in)))
        # x1 = self.dropout(x1)
        x2 = self.activation(self.bnd2(self.conv2(x1)))
        # x2 = self.dropout(x2)
        x3 = self.activation(self.bnd3(self.conv3(x2)))
        # x3 = self.dropout(x3)

        x4 = self.flatten(x3)
        x5 = self.fc1(x4)
        # x5 = self.dropout(x5)
        x6 = self.fc2(x5)
        x_out = self.softmax(x6)

        return x_out


if __name__ == "__main__":
    pass