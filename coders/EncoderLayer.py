import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(Conv1, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x


class Conv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(Conv2, self).__init__()
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.convlayer2(x)
        return x


class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(out_channels//2),
            nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        skip = self.conv2(x)

        x = self.conv(x)
        x = F.relu(x + skip)
        return x


class MixPool(nn.Module):
    def __init__(self, kernel_size, stride, padding, alpha):
        super(MixPool, self).__init__()
        alpha = torch.FloatTensor([alpha])
        self.alpha = nn.Parameter(alpha)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.alpha * F.max_pool2d(x, self.kernel_size, self.stride, self.padding) + (1 - self.alpha) * \
            F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return x


class GaborConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1, dilation=1, groups=1, bias=False,
                 padding_mode="zeros"):
        super(GaborConv2d, self).__init__()
        self.is_calculated = False

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              dilation,
                              groups,
                              bias,
                              padding_mode
                              )

        self.kernel_size = self.conv.kernel_size

        # Prevents dividing by zero
        self.delta = 1e-3

        # freq, theta, sigma are set up according to S. Meshgini,
        # A. Aghagolzadeh and H. Seyedarabi, "Face recognition using
        # Gabor filter bank, kernel principal component analysis
        # and support vector machine"
        self.freq = Parameter(
            (math.pi / 2)
            * math.sqrt(2)
            ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor),
            requires_grad=True
        )

        self.theta = Parameter(
            (math.pi / 8)
            * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor),
            requires_grad=True
        )

        self.sigma = Parameter(math.pi / self.freq, requires_grad=True)
        self.psi = Parameter(math.pi * torch.rand(out_channels, in_channels), requires_grad=True)

        self.x0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0],
            requires_grad=False
        )

        self.y0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0],
            requires_grad=False
        )

        self.y, self.x = torch.meshgrid(
            [
                torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]),
                torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1])
            ]
        )
        self.x = Parameter(self.x)
        self.y = Parameter(self.y)

        self.weight = Parameter(
            torch.empty(self.conv.weight.shape, requires_grad=True),
            requires_grad=True
        )

        self.register_parameter("freq", self.freq)
        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("psi", self.psi)
        self.register_parameter("x_shape", self.x0)
        self.register_parameter("y_shape", self.y0)
        self.register_parameter("x_grid", self.x)
        self.register_parameter("y_grid", self.y)
        self.register_parameter("weight", self.weight)

    def forward(self, input_tensor):
        if self.training:
            self.calculate_weights()
            self.is_calculated = False
        if not self.training:
            if not self.is_calculated:
                self.calculate_weights()
                self.is_calculated = True
        return self.conv(input_tensor)

    def calculate_weights(self):
        for i in range(self.conv.out_channels):
            for j in range(self.conv.in_channels):
                sigma = self.sigma[i, j].expand_as(self.y)
                freq = self.freq[i, j].expand_as(self.y)
                theta = self.theta[i, j].expand_as(self.y)
                psi = self.psi[i, j].expand_as(self.y)

                rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
                roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

                g = torch.exp(
                    -0.5 * ((rotx ** 2 + roty ** 2) / (sigma + self.delta) ** 2)
                )
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * math.pi * sigma ** 2)
                self.conv.weight.data[i, j] = g


class Gabor2Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, device=device, stride=1,
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
