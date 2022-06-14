import torch.nn as nn
import torch.nn.functional as F


class Deconvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Deconvolution, self).__init__()
        self.deconvolution = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.deconvolution(x)
        return x


class Deconvolution2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, padding):
        super(Deconvolution2, self).__init__()
        self.deconvolution = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=1, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.deconvolution(x)
        return x


class ResDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResDeconv, self).__init__()
        self.deconvolution = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels//2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels)
        )
        self.deconvolution2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        skip = self.deconvolution2(x)

        x = self.deconvolution(x)
        x = F.relu(x + skip)
        return x


"""
class ResDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResDeconv, self).__init__()
        self.deconvolution = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=kernel_size, stride=2,
                               padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels//2),
            nn.ConvTranspose2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels)
        )
        self.deconvolution2 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                                 kernel_size=kernel_size, stride=2, padding=padding)

    def forward(self, x):
        skip = self.deconvolution2(x)

        x = self.deconvolution(x)
        x = F.relu(x + skip)
        return x
"""

