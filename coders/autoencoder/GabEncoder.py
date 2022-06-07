import torch.nn as nn
from coders.EncoderLayer import MixPool, GaborConv2d


class GabEncoder(nn.Module):
    def __init__(self):
        super(GabEncoder, self).__init__()
        self.features = nn.Sequential(
            GaborConv2d(in_channels=3, out_channels=c, kernel_size=3, padding=1, stride=1),  # 32x32 => 16x16
            nn.ReLU(inplace=True),
            MixPool(2, 2, 0, 1),
            nn.BatchNorm2d(c),
            nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c * 2),
            nn.Conv2d(in_channels=c * 2, out_channels=c * 4, kernel_size=3, padding=1, stride=1),  # 16x16 => 8x8
            nn.ReLU(inplace=3),
            MixPool(2, 2, 0, 0.6),
            nn.BatchNorm2d(c * 4),
            nn.Conv2d(in_channels=c * 4, out_channels=c * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MixPool(2, 2, 0, 0.2),
            nn.BatchNorm2d(c * 8),
            nn.Conv2d(in_channels=c * 8, out_channels=c * 16, kernel_size=3, padding=1, stride=1),  # 8x8 => 4x4
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c * 16)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc = nn.Linear(c*16, latent_dims)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        # x = self.fc(x)
        return x