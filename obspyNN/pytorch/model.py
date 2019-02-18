from torch import nn
from torch.nn import functional as F
import torch

from torchsummaryX import summary


def conv1x3(in_, out):
    return nn.Conv2d(in_, out, (1, 3), padding=(0, 1))


class ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = conv1x3(in_ch, out_ch)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class StandardUnit(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        dropout_rate = 0.2
        self.unit = nn.Sequential(
            ConvRelu(in_ch, out_ch),
            nn.Dropout(dropout_rate),
            ConvRelu(out_ch, out_ch),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.unit(x)


class Pooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(x)


class UpSampling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(1, 2), stride=(1, 2))

    def forward(self, x):
        return self.up(x)


class pad(torch.nn.Module):
    def __init__(self, padding):
        super(pad, self).__init__()
        self.padding = nn.ZeroPad2d(padding)

    def forward(self, x):
        x = self.padding(x)
        return x


class Nest_Net(nn.Module):
    def __init__(self, in_ch, out_ch, start_filters=8):
        super().__init__()
        padding_size = (3, 4, 0, 0)
        n = start_filters
        nfilters = [n, n * 2, n * 4, n * 8, n * 16]

        self.pad = pad(padding_size)
        self.pool = Pooling()

        self.conv1_1 = StandardUnit(in_ch, nfilters[0])

        self.conv2_1 = StandardUnit(nfilters[0], nfilters[1])

        self.up1_2 = UpSampling(nfilters[1], nfilters[0])
        self.conv1_2 = StandardUnit(nfilters[0] * 2, nfilters[0])

        self.conv3_1 = StandardUnit(nfilters[1], nfilters[2])

        self.up2_2 = UpSampling(nfilters[2], nfilters[1])
        self.conv2_2 = StandardUnit(nfilters[1] * 2, nfilters[1])
        self.up1_3 = UpSampling(nfilters[1], nfilters[0])
        self.conv1_3 = StandardUnit(nfilters[0] * 3, nfilters[0])

        self.conv4_1 = StandardUnit(nfilters[2], nfilters[3])

        self.up3_2 = UpSampling(nfilters[3], nfilters[2])
        self.conv3_2 = StandardUnit(nfilters[2] * 2, nfilters[2])
        self.up2_3 = UpSampling(nfilters[2], nfilters[1])
        self.conv2_3 = StandardUnit(nfilters[1] * 3, nfilters[1])
        self.up1_4 = UpSampling(nfilters[1], nfilters[0])
        self.conv1_4 = StandardUnit(nfilters[0] * 4, nfilters[0])

        self.conv5_1 = StandardUnit(nfilters[3], nfilters[4])

        self.up4_2 = UpSampling(nfilters[4], nfilters[3])
        self.conv4_2 = StandardUnit(nfilters[3] * 2, nfilters[3])
        self.up3_3 = UpSampling(nfilters[3], nfilters[2])
        self.conv3_3 = StandardUnit(nfilters[2] * 3, nfilters[2])
        self.up2_4 = UpSampling(nfilters[2], nfilters[1])
        self.conv2_4 = StandardUnit(nfilters[1] * 4, nfilters[1])
        self.up1_5 = UpSampling(nfilters[1], nfilters[0])
        self.conv1_5 = StandardUnit(nfilters[0] * 5, nfilters[0])

        self.final = nn.Conv2d(nfilters[0], out_ch, kernel_size=1)

    def forward(self, x):
        pad = self.pad(x)

        conv1_1 = self.conv1_1(pad)
        pool1 = self.pool(conv1_1)

        conv2_1 = self.conv2_1(pool1)
        pool2 = self.pool(conv2_1)

        up1_2 = self.up1_2(conv2_1)
        conv1_2 = self.conv1_2(torch.cat([up1_2, conv1_1], 1))

        conv3_1 = self.conv3_1(pool2)
        pool3 = self.pool(conv3_1)

        up2_2 = self.up2_2(conv3_1)
        conv2_2 = self.conv2_2(torch.cat([up2_2, conv2_1], 1))
        up1_3 = self.up1_3(conv2_2)
        conv1_3 = self.conv1_3(torch.cat([up1_3, conv1_1, conv1_2], 1))

        conv4_1 = self.conv4_1(pool3)
        pool4 = self.pool(conv4_1)

        up3_2 = self.up3_2(conv4_1)
        conv3_2 = self.conv3_2(torch.cat([up3_2, conv3_1], 1))
        up2_3 = self.up2_3(conv3_2)
        conv2_3 = self.conv2_3(torch.cat([up2_3, conv2_1, conv2_2], 1))
        up1_4 = self.up1_4(conv2_3)
        conv1_4 = self.conv1_4(torch.cat([up1_4, conv1_1, conv1_2, conv1_3], 1))

        conv5_1 = self.conv5_1(pool4)

        up4_2 = self.up4_2(conv5_1)
        conv4_2 = self.conv4_2(torch.cat([up4_2, conv4_1], 1))
        up3_3 = self.up3_3(conv4_2)
        conv3_3 = self.conv3_3(torch.cat([up3_3, conv3_1, conv3_2], 1))
        up2_4 = self.up2_4(conv3_3)
        conv2_4 = self.conv2_4(torch.cat([up2_4, conv2_1, conv2_2, conv2_3], 1))
        up1_5 = self.up1_5(conv2_4)
        conv1_5 = self.conv1_5(torch.cat([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], 1))

        final = self.final(conv1_5)
        outputs = torch.sigmoid(final)
        outputs = outputs[:, :, :, 3:-4]
        return outputs


if __name__ == '__main__':
    summary(Nest_Net(1, 1), torch.zeros((1, 1, 1, 3001)))
