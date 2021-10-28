import torch.nn as nn


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, pad, activation="relu"):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=pad)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.Sigmoid()  # last layer before output is sigmoid, since we are using BCE as reconstruction loss

    def forward(self, x):
        return self.activation(self.deconv(x))
        return self.activation(self.bn(self.deconv(x)))

