import torch.nn as nn


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