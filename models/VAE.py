import torch
import torch.nn.functional as F
import torch.nn as nn
from coders.Encoder import Encoder
from coders.Decoder import Decoder
from coders.GaborEncoder import GaborEncoder
from coders.Gabor2Encoder import Gabor2Encoder
from coders.Gabor3Encoder import Gabor3Encoder

variational_beta = 1


class VAE(nn.Module):
    def __init__(self, encoder_type, color_channels, c, encoder_out_size, latent_dims):
        super(VAE, self).__init__()
        if encoder_type == 'encoder':
            self.encoder = Encoder(color_channels, c, encoder_out_size, latent_dims)
        elif encoder_type == 'gabor2encoder':
            self.encoder = Gabor2Encoder(color_channels, c, encoder_out_size, latent_dims)
        elif encoder_type == 'gabor3encoder':
            self.encoder = Gabor3Encoder(color_channels, c, encoder_out_size, latent_dims)
        else:
            self.encoder = GaborEncoder(color_channels, c, encoder_out_size, latent_dims)
        self.decoder = Decoder(color_channels, c, encoder_out_size, latent_dims)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def vae_loss(recon_x, x, mu, logvar):
        # recon_x is the probability of a multivariate Bernoulli distribution p.
        # -log(p(x)) is then the pixel-wise binary cross-entropy.
        # Averaging or not averaging the binary cross-entropy over all pixels here
        # is a subtle detail with big effect on training, since it changes the weight
        # we need to pick for the other loss term by several orders of magnitude.
        # Not averaging is the direct implementation of the negative log likelihood,
        # but averaging makes the weight of the other loss term independent of the image resolution.
        recon_loss = F.binary_cross_entropy(recon_x.view(-1, 1024), x.view(-1, 1024), reduction='sum')

        # KL-divergence between the prior distribution over latent vectors
        # (the one we are going to sample from when generating new images)
        # and the distribution estimated by the generator for the given image.
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + variational_beta * kldivergence


