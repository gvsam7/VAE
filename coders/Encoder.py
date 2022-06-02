import torch.nn as nn
from coders.EncoderLayer import Conv1, Conv2


class Encoder(nn.Module):
    def __init__(self, color_channels, c, encoder_out_size, latent_dims):
        super(Encoder, self).__init__()
        self.features = nn.Sequential(
            Conv2(in_channels=3, out_channels=c, kernel_size=3, padding=1, stride=1),
            Conv1(in_channels=c, out_channels=c * 2, kernel_size=3, padding=1, stride=1),
            Conv2(in_channels=c * 2, out_channels=c * 4, kernel_size=3, padding=1, stride=1),
            Conv2(in_channels=c * 4, out_channels=c * 8, kernel_size=3, padding=1, stride=1),
            Conv1(in_channels=c * 8, out_channels=c * 16, kernel_size=3, padding=1, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_mu = nn.Linear(in_features=c * 16, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c * 16, out_features=latent_dims)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar