import torch.nn as nn
from coders.EncoderLayer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, color_channels, c, encoder_out_size, latent_dims):
        super().__init__()
        self.conv1 = EncoderLayer(in_channels=color_channels, out_channels=c, kernel=4, stride=2, pad=1) # out: c x 14 x 14
        self.conv2 = EncoderLayer(in_channels=c, out_channels=c*2, kernel=4, stride=2, pad=1)  # out: c x 7 x 7
        self.fc_mu = nn.Linear(in_features=c*2*encoder_out_size, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*2*encoder_out_size, out_features=latent_dims)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten batch of multichannel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar