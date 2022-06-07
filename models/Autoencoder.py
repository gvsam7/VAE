import torch.nn as nn
from coders.autoencoder.Encoder import Encoder
from coders.autoencoder.GabEncoder import GabEncoder
from coders.autoencoder.Decoder import Decoder


class Autoencoder(nn.Module):
    def __init__(self, encoder_type, color_channels, c, encoder_out_size, latent_dims):
        super(Autoencoder, self).__init__()
        if encoder_type == 'encoder':
            self.encoder = Encoder(color_channels, c, encoder_out_size, latent_dims)
        else:
            self.encoder = GabEncoder(color_channels, c, encoder_out_size, latent_dims)
        self.decoder = Decoder(color_channels, c, encoder_out_size, latent_dims)

    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon