from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import numpy as np

from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.convutils import same_padding, stride_minus_kernel_padding
from monai.networks.layers.factories import Conv


class SE3Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        strides: int | Sequence[int] = 1,
        kernel_size: int | Sequence[int] = 3,          
        adn_ordering: str = "NDA",
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
    )-> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.adn_ordering = adn_ordering
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.dropout_dim = dropout_dim
        self.groups = groups
        self.bias = bias
        self.conv_only = conv_only
        self.is_transposed = is_transposed
        # Split
        self.conv1 = self._get_conv_layer(in_channels, out_channels, strides, kernel=3, dilation=1)
        self.conv2 = self._get_conv_layer(in_channels, out_channels, strides, kernel=3, dilation=2)
        # self.conv3 = self._get_conv_layer(in_channels, out_channels, strides, kernel=7, dilation=1)
        # Fuse
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Hyper parameters
        L = 16 # minimal value of d
        r = 0.5 # reduction ratio for d
        d = max(L, int(out_channels * r))
        self.fully_connected = nn.Sequential(
            nn.Linear(out_channels, d), 
            # nn.BatchNorm1d(num_features=d), 
            nn.InstanceNorm1d(num_features=d),
            nn.PReLU(),
        ) # Fully connected layer
        self.weight_proj1 = nn.Linear(d, out_channels)
        self.weight_proj2 = nn.Linear(d, out_channels)
        # self.weight_proj3 = nn.Linear(d, out_channels)
        self.softmax_layer = nn.Softmax(dim=1) # Softmax layer

    def _get_conv_layer(self, in_channels, out_channels, strides, kernel, dilation=1):
        mod = Convolution(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=kernel,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
            dilation=dilation,
            conv_only=self.conv_only,
            is_transposed=self.is_transposed,
        )
        return mod
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split
        u1 = self.conv1(x)
        u2 = self.conv2(x)
        # u3 = self.conv3(x)
        u = torch.add(u1, u2)
        # u = torch.add(u, u3)

        # Fuse
        s = self.avg_pool(u).squeeze(-1).squeeze(-1).squeeze(-1)
        z = self.fully_connected(s)
        a1 = self.weight_proj1(z)
        a2 = self.weight_proj2(z)
        # a3 = self.weight_proj3(z)
        # a = torch.stack([a1, a2, a3], dim=1)
        a = torch.stack([a1, a2], dim=1)
        a = self.softmax_layer(a)
        # Attention weight for each branch
        w1 = a[:, 0, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        w2 = a[:, 1, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # w3 = a[:, 2, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Select
        v1 = torch.mul(u1, w1)
        v2 = torch.mul(u2, w2)
        # v3 = torch.mul(u3, w3)
        v = torch.add(v1, v2)
        # v = torch.add(v, v3)
        return v

class SE3ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        subunits: int = 2,
        adn_ordering: str = "NDA",
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        bias: bool = True,
        last_conv_only: bool = False,
        padding: Sequence[int] | int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential()
        self.residual = nn.Identity()
        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)
        if not padding:
            padding = same_padding(kernel_size, 1)

        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = SE3Block(
                schannels,
                out_channels,
                strides=sstrides,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=dropout_dim,
                bias=bias,
                conv_only=conv_only,
            )

            self.conv.add_module(f"unit{su:d}", unit)

            # after first loop set channels and strides to what they should be for subsequent units
            schannels = out_channels
            sstrides = 1

        # apply convolution to input to change number of output channels and size to match that coming from self.conv
        if np.prod(strides) != 1 or in_channels != out_channels:
            rkernel_size = kernel_size
            rpadding = padding

            if np.prod(strides) == 1:  # if only adapting number of channels a 1x1 kernel is used with no padding
                rkernel_size = 1
                rpadding = 0

            conv_type = Conv[Conv.CONV, 3]
            self.residual = conv_type(in_channels, out_channels, rkernel_size, strides, rpadding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = self.residual(x)  # create the additive residual from x
        cx: torch.Tensor = self.conv(x)  # apply x to sequence of operations
        return cx + res  # add the residual to the output

class SEResBlockV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        # self.conv = nn.Sequential(
        #     SE3Block(in_channels=in_channels, out_channels=out_channels, strides=stride, kernel_size=3),
        #     nn.InstanceNorm3d(out_channels),
        #     nn.PReLU(),
        #     Convolution(spatial_dims=3, in_channels=out_channels, out_channels=out_channels, strides=1, kernel_size=3),
        #     )
        self.conv = nn.Sequential(
            SE3Block(in_channels=in_channels, out_channels=out_channels, strides=1, kernel_size=3),
            # nn.InstanceNorm3d(out_channels),
            nn.ReLU(),
            Convolution(spatial_dims=3, in_channels=out_channels, out_channels=out_channels, strides=stride, kernel_size=3),
            )
        if stride == 1: # Bottleneck
            padding = 0
            kernel_size = 1
        else:
            padding = 1
            kernel_size = 3
        self.res = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.res(x)
        cx = self.conv(x)
        return cx + res

