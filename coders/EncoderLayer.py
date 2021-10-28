import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))
        return self.activation(self.bn(self.conv(x)))
