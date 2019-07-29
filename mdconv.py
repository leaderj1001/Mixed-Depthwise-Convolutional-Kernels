import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size):
        super(DepthwiseConv, self).__init__()
        padding = (kernal_size - 1) // 2

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernal_size, padding=padding, groups=in_channels)

    def forward(self, x):
        out = self.conv(x)
        return out


class MDConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_chunks):
        super(MDConv, self).__init__()
        self.n_chunks = n_chunks
        assert in_channels % n_chunks == 0 and out_channels % n_chunks == 0

        self.layers = nn.ModuleList([DepthwiseConv(in_channels // n_chunks, out_channels, kernal_size=idx * 2 + 3) for idx in range(n_chunks)])

    def forward(self, x):
        out = torch.tensor([])
        for idx, each_chunk in enumerate(torch.chunk(x, self.n_chunks, dim=1)):
            out = torch.cat((out, self.layers[idx](each_chunk)), dim=1)

        return out


temp = torch.randn((2, 32, 32, 32))
mdconv = MDConv(32, 64, 4)
print(mdconv(temp).size())
