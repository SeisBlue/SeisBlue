import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Unet model from: 
https://github.com/milesial/Pytorch-UNet/tree/master/unet
"""


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        padding_size = (3, 4, 0, 0)
        nb_filter = [8, 16, 32, 64, 128]

        self.pad = pad(padding_size)
        self.inc = inconv(n_channels, nb_filter[0])
        self.down1 = down(nb_filter[0], nb_filter[1])
        self.down2 = down(nb_filter[1], nb_filter[2])
        self.down3 = down(nb_filter[2], nb_filter[3])
        self.down4 = down(nb_filter[3], nb_filter[3])
        self.up1 = up(nb_filter[4], nb_filter[2])
        self.up2 = up(nb_filter[3], nb_filter[1])
        self.up3 = up(nb_filter[2], nb_filter[0])
        self.up4 = up(nb_filter[1], nb_filter[0])
        self.outc = outconv(nb_filter[0], n_classes)

    def forward(self, x):
        x = self.pad(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = x[:, :, :, 3:-4]
        return F.sigmoid(x)


class pad(torch.nn.Module):
    def __init__(self, padding):
        super(pad, self).__init__()
        self.padding = nn.ZeroPad2d(padding)

    def forward(self, x):
        x = self.padding(x)
        return x


class double_conv(nn.Module):
    # (conv => BN => ReLU) * 2

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=(1, 2), stride=(1, 2))
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)

        return x
