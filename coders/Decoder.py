import torch.nn as nn
from coders.DecoderLayer import Deconvolution, Deconvolution2

"""
 To truly have a reverse operation of the convolution, I need to ensure that the layer scales the input shape by a 
 factor of 2 (e.g.  4×4→8×8 ). For this, I can specify the parameter output_padding which adds additional values to 
 the output shape. Note that I do not perform zero-padding with this, but rather increase the output shape for 
 calculation.
"""


class Decoder(nn.Module):
    def __init__(self, colour_channels, c, encoder_out_size, latent_dims):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=latent_dims, out_features=c*16*encoder_out_size),
            nn.ReLU(inplace=True)
        )
        self.deconvolution = nn.Sequential(
            Deconvolution2(in_channels=c * 16, out_channels=c * 8, kernel_size=3, output_padding=1, padding=1,
                           stride=2),  # 4x4 => 8x8
            Deconvolution(in_channels=c * 8, out_channels=c * 4, kernel_size=3, padding=1, stride=1),
            Deconvolution2(in_channels=c * 4, out_channels=c * 2, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 8x8 => 16x16
            Deconvolution(in_channels=c * 2, out_channels=c, kernel_size=3, padding=1, stride=1),
            nn.ConvTranspose2d(in_channels=c, out_channels=colour_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.shape[0], -1, 12, 12)
        # x = x.view(x.size(0), capacity*16, 4, 4) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = self.deconvolution(x)
        return x



