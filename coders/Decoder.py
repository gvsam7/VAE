import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, color_channels, c, encoder_out_size, latent_dims):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*16*4*4)
        self.dec = nn.Sequential(
           nn.ConvTranspose2d(in_channels=c*16, out_channels=c*8, kernel_size=3, output_padding=1, padding=1, stride=2),  # 4x4 => 8x8
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c*8),
            nn.ConvTranspose2d(in_channels=c*8, out_channels=c*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c*4),
            nn.ConvTranspose2d(in_channels=c*4, out_channels=c*2, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c*2),
            nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c),
            nn.ConvTranspose2d(in_channels=c, out_channels=3, kernel_size=3, output_padding=1, padding=1, stride=2),  # 16x16 => 32x32
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        # x = x.view(x.size(0), capacity*16, 4, 4) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = self.dec(x)
        return x



