import torch.nn as nn
from coders.EncoderLayer import ResConv


class ResEncoder(nn.Module):
    def __init__(self, colour_channels, c, encoder_out_size, latent_dims):
        super(ResEncoder, self).__init__()
        self.features = nn.Sequential(
            ResConv(in_channels=colour_channels, out_channels=c, kernel_size=3, padding=1, stride=1),  # 96 -> 48
            ResConv(in_channels=c, out_channels=c * 2, kernel_size=3, padding=1, stride=1),  # 48 -> 24
            ResConv(in_channels=c * 2, out_channels=c * 4, kernel_size=3, padding=1, stride=1),  # 24 -> 12
            ResConv(in_channels=c * 4, out_channels=c * 8, kernel_size=3, padding=1, stride=1),  # 12 -> 6
            ResConv(in_channels=c * 8, out_channels=c * 16, kernel_size=3, padding=1, stride=1)  # 6 -> 3
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc = nn.Linear(c*16, latent_dims)
        self.fc_mu = nn.Linear(in_features=c * 16, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c * 16, out_features=latent_dims)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        # x = self.fc(x)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar