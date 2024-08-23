import torch
import torch.nn as nn


class LazyConvGru(nn.Module):
    """
    Concat gates instead of input.
    """

    def __init__(self, out_channels, kernel_size):
        super().__init__()

        self.out_channels = out_channels
        padding = kernel_size // 2

        self.ih = nn.LazyConv2d(3 * out_channels, kernel_size, padding=padding)
        self.hh = nn.LazyConv2d(3 * out_channels, kernel_size, padding=padding)

    def forward(self, x, hx):
        # get previous output
        if hx is None:
            b, _, h, w = x.shape
            hx = torch.zeros(b, self.out_channels, h, w, device=x.device, dtype=x.dtype)

        # compute concatenated gates
        ih, hh = self.ih(x), self.hh(hx)
        ih_r, ih_z, ih_n = ih.chunk(3, dim=1)
        hh_r, hh_z, hh_n = hh.chunk(3, dim=1)
        r = torch.sigmoid(ih_r + hh_r)
        z = torch.sigmoid(ih_z + hh_z)
        n = torch.tanh(ih_n + r * hh_n)

        # compute hidden
        hx = (1 - z) * hx + z * n
        return hx


class Residual(nn.Sequential):
    """
    Residual block with connection from input to after.
    """

    def forward(self, input):
        x = input
        *before, residual, after = self
        for layer in before:
            x = layer(x)
        res = residual(input)
        x = after(x + res)
        return x


def res_block(out_channels, kernel_size, stride=1):
    padding = kernel_size // 2
    if stride != 1:
        downsample = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size, stride=stride, padding=padding),
            nn.Identity(),
        )
    else:
        downsample = nn.Identity()
    block = Residual(
        nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
        ),
        nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size, padding=padding),
            nn.Identity(),
        ),
        downsample,
        nn.ReLU(),
    )
    return block


def conv_encoder(out_channels):
    """
    Components:
    - Head with large kernel
    - 2 pairs of residual blocks with stride
    """
    head = nn.Sequential(
        nn.LazyConv2d(out_channels // 2, 7, stride=2, padding=3),
        nn.ReLU(),
    )
    encoder = nn.Sequential(
        res_block(out_channels // 2, 3, stride=2),
        res_block(out_channels // 2, 3),
        res_block(out_channels, 3, stride=2),
        res_block(out_channels, 3),
    )
    return nn.Sequential(head, encoder)


def global_pool_decoder(out_features):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.LazyLinear(out_features),
    )
