import torch
import torch.nn as nn
from coders.EncoderLayer import MixPool #, Gabor2Conv2d
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Gabor2Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, device="cpu", stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(Gabor2Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _pair(0), groups, bias, padding_mode)
        self.freq = nn.Parameter(
            (3.14 / 2) * 1.41 ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor))
        self.theta = nn.Parameter((3.14 / 8) * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor))
        self.psi = nn.Parameter(3.14 * torch.rand(out_channels, in_channels))
        self.sigma = nn.Parameter(3.14 / self.freq)
        self.x0 = torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0]
        self.y0 = torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0]
        self.device = device

    def forward(self, input_image):
        y, x = torch.meshgrid([torch.linspace(-self.x0 + 1, self.x0, self.kernel_size[0]),
                               torch.linspace(-self.y0 + 1, self.y0, self.kernel_size[1])])
        x = x.to(self.device)
        y = y.to(self.device)
        weight = torch.empty(self.weight.shape, requires_grad=False).to(self.device)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                sigma = self.sigma[i, j].expand_as(y)
                freq = self.freq[i, j].expand_as(y)
                theta = self.theta[i, j].expand_as(y)
                psi = self.psi[i, j].expand_as(y)

                rotx = x * torch.cos(theta) + y * torch.sin(theta)
                roty = -x * torch.sin(theta) + y * torch.cos(theta)

                g = torch.zeros(y.shape)

                g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2) / (sigma + 1e-3) ** 2))
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * 3.14 * sigma ** 2)
                weight[i, j] = g
                self.weight.data[i, j] = g
        return F.conv2d(input_image, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Gabor2Encoder(nn.Module):
    def __init__(self, color_channels, c, encoder_out_size, latent_dims):
        super(Gabor2Encoder, self).__init__()
        self.features = nn.Sequential(
            Gabor2Conv2d(in_channels=3, out_channels=c, kernel_size=3, padding=1, stride=1),  # 32x32 => 16x16
            nn.ReLU(inplace=True),
            MixPool(2, 2, 0, 1),
            nn.BatchNorm2d(c),
            nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c * 2),
            nn.Conv2d(in_channels=c * 2, out_channels=c * 4, kernel_size=3, padding=1, stride=1),  # 16x16 => 8x8
            nn.ReLU(inplace=True),
            MixPool(2, 2, 0, 0.6),
            nn.BatchNorm2d(c * 4),
            nn.Conv2d(in_channels=c * 4, out_channels=c * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MixPool(2, 2, 0, 0.2),
            nn.BatchNorm2d(c * 8),
            nn.Conv2d(in_channels=c * 8, out_channels=c * 16, kernel_size=3, padding=1, stride=1),  # 8x8 => 4x4
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c * 16)
        )
        # self.fc = nn.Linear(c*16*4*4, latent_dims)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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