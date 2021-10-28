import torch.nn as nn
from coders.DecoderLayer import DecoderLayer
c1 = 64


class Decoder(nn.Module):
    def __init__(self, color_channels, c, encoder_out_size, latent_dims):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(in_features=latent_dims, out_features=c * 2 * encoder_out_size)
        self.deconv2 = DecoderLayer(in_channels=c * 2, out_channels=c, kernel=4, stride=2, pad=1)
        self.deconv1 = DecoderLayer(in_channels=c, out_channels=color_channels, kernel=4, stride=2, pad=1, activation="sigmoid")

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), c1 * 2, 7, 7)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = self.deconv2(x)
        x = self.deconv1(x)
        return x
