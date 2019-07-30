import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def split_layer(total_channels, num_groups):
    split = [int(np.ceil(total_channels / num_groups)) for _ in range(num_groups)]
    split[num_groups - 1] += total_channels - sum(split)
    return split


class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels, kernal_size, stride):
        super(DepthwiseConv2D, self).__init__()
        padding = (kernal_size - 1) // 2

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernal_size, padding=padding, stride=stride, groups=in_channels)

    def forward(self, x):
        out = self.depthwise_conv(x)
        return out


class GroupConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, n_chunks=1):
        super(GroupConv2D, self).__init__()
        self.n_chunks = n_chunks
        split_in_channels = split_layer(in_channels, n_chunks)
        split_out_channels = split_layer(out_channels, n_chunks)

        if n_chunks == 1:
            self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        else:
            self.group_layers = nn.ModuleList()
            for idx in range(n_chunks):
                self.group_layers.append(nn.Conv2d(split_in_channels[idx], split_out_channels[idx], kernel_size=kernel_size))

    def forward(self, x):
        if self.n_chunks == 1:
            return self.group_conv(x)
        else:
            out = torch.tensor([]).to(x)
            for idx, each_chunk in enumerate(torch.chunk(x, self.n_chunks, dim=1)):
                out = torch.cat((out, self.group_layers[idx](each_chunk)), dim=1)
            return out


class MDConv(nn.Module):
    def __init__(self, out_channels, n_chunks, stride=1):
        super(MDConv, self).__init__()
        self.n_chunks = n_chunks
        split_out_channels = split_layer(out_channels, n_chunks)

        self.layers = nn.ModuleList()
        for idx in range(self.n_chunks):
            kernel_size = 2 * idx + 3
            self.layers.append(DepthwiseConv2D(split_out_channels[idx], kernal_size=kernel_size, stride=stride))

    def forward(self, x):
        out = torch.tensor([]).to(x)
        for idx, each_chunk in enumerate(torch.chunk(x, self.n_chunks, dim=1)):
            out = torch.cat((out, self.layers[idx](each_chunk)), dim=1)
        return out


# temp = torch.randn((16, 3, 32, 32))
# group = GroupConv2D(3, 16, n_chunks=2)
# print(group(temp).size())
