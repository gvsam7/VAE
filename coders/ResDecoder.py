import torch.nn as nn
from coders.DecoderLayer import ResDeconv


class ResDecoder(nn.Module):
    def __init__(self, colour_channels, c, encoder_out_size, latent_dims):
        super(ResDecoder, self).__init__()
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*16*3*3)  # encoder_out_size)
        self.deconvolution = nn.Sequential(
            ResDeconv(in_channels=c * 16, out_channels=c * 8, kernel_size=3, padding=1, stride=1),  # 4x4 => 8x8
            ResDeconv(in_channels=c * 8, out_channels=c * 4, kernel_size=3, padding=1, stride=1),
            ResDeconv(in_channels=c * 4, out_channels=c * 2, kernel_size=3, padding=1, stride=1),  # 8x8 => 16x16
            ResDeconv(in_channels=c * 2, out_channels=c, kernel_size=3, padding=1, stride=1),
            ResDeconv(in_channels=c, out_channels=c//2, kernel_size=3, padding=1, stride=1),
            nn.ConvTranspose2d(in_channels=c//2, out_channels=colour_channels, kernel_size=3, padding=1, stride=1),  # 16x16 => 32x32
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        # x = x.reshape(x.shape[0], -1, 12, 12)
        x = x.reshape(x.shape[0], -1, 3, 3)
        x = self.deconvolution(x)
        return x



