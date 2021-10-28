import torch
import torch.nn.functional as F
import torch.nn as nn
from coders.Encoder import Encoder
from coders.Decoder import Decoder

variational_beta = 1


class VAE(nn.Module):
    def __init__(self, color_channels, c, encoder_out_size, latent_dims):

        super(VAE, self).__init__()
        self.encoder = Encoder(color_channels, c, encoder_out_size, latent_dims)
        self.decoder = Decoder(color_channels, c, encoder_out_size, latent_dims)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the parameterisation trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def vae_loss(recon_x, x, mu, logvar, dataset):
        """
        recon_x is the probability of a multivariate Bernoulli distribution p.
        -log(p(x)) is then the pixel-wise binary cross-entropy.
        Averaging or not averaging the binary cross-entropy over all pixels here
        is a subtle detail with big effect on training, since it changes the weight
        we need to pick for the other loss term by several orders of magnitude.
        Not averaging is the direct implementation of the negative log likelihood,
        but averaging makes the weight of the other loss term independent of the image resolution.
        """
        if dataset == "mnist" or dataset == "fashion-mnist":
            recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
        elif dataset == "cifar":
            recon_loss = F.binary_cross_entropy(recon_x.view(-1, 1024), x.view(-1, 1024), reduction='sum')  # cifar
        elif dataset == "stl":
            recon_loss = F.binary_cross_entropy(recon_x.view(-1, 9216), x.view(-1, 9216), reduction='sum')  # stl

        """KL-divergence between the prior distribution over latent vectors
        (the one we are going to sample from when generating new images)
        and the distribution estimated by the generator for the given image.
        """
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + variational_beta * kldivergence



