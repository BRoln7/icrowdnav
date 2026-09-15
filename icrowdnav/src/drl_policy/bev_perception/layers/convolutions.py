from functools import partial

import torch.nn as nn


class ConvBlock(nn.Module):
    """2D convolution followed by optional normalisation and activation."""

    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        stride=1,
        norm="bn",
        activation="relu",
        bias=False,
        transpose=False,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        padding = int((kernel_size - 1) / 2)
        conv_cls = nn.Conv2d if not transpose else partial(nn.ConvTranspose2d, output_padding=1)
        self.conv = conv_cls(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == "none":
            self.norm = None
        else:
            raise ValueError("Invalid norm {}".format(norm))

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            raise ValueError("Invalid activation {}".format(activation))

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
