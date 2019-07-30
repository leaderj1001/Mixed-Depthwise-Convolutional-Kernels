import torch
import torch.nn as nn
import torch.nn.functional as F

from mdconv import MDConv, GroupConv2D


def round_filters(filters):
    multiplier = 1.3
    divisor = 8
    min_depth = None
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return new_filters


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, out_channels, swish):
        super(SqueezeExcitation, self).__init__()
        self.activation = Swish() if swish else nn.ReLU()
        self.se_reduce = nn.Sequential(
            GroupConv2D(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU()
            self.activation
        )

        self.se_expand = nn.Sequential(
            GroupConv2D(out_channels, in_channels),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        se_tensor = torch.mean(x, dim=[2, 3], keepdim=True)
        out = self.se_expand(self.se_reduce(se_tensor))
        out = torch.sigmoid(out) * x

        return out


class MixBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_chunks, stride, expand_ratio, se_ratio, swish, expand_ksize, project_ksize):
        super(MixBlock, self).__init__()
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
        self.activation = Swish() if swish else nn.ReLU()

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                GroupConv2D(in_channels, in_channels * expand_ratio, n_chunks=expand_ksize),
                nn.BatchNorm2d(in_channels * expand_ratio),
                self.activation
            )

            self.mdconv = nn.Sequential(
                MDConv(in_channels * expand_ratio, n_chunks=n_chunks, stride=stride),
                nn.BatchNorm2d(in_channels * expand_ratio),
                self.activation
            )

            if self._has_se:
                num_reduced_filters = max(1, int(in_channels * se_ratio))
                self.squeeze_excitation = SqueezeExcitation(in_channels * expand_ratio, num_reduced_filters, swish)

                self.project_conv = nn.Sequential(
                    GroupConv2D(in_channels * expand_ratio, out_channels, n_chunks=project_ksize),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.project_conv = nn.Sequential(
                    GroupConv2D(in_channels * expand_ratio, out_channels, n_chunks=project_ksize),
                    nn.BatchNorm2d(out_channels),
                )
        else:
            self.mdconv = nn.Sequential(
                MDConv(in_channels, n_chunks=n_chunks, stride=stride),
                nn.BatchNorm2d(in_channels),
                self.activation
            )

            if self._has_se:
                num_reduced_filters = max(1, int(in_channels * se_ratio))
                self.squeeze_excitation = SqueezeExcitation(in_channels, num_reduced_filters, swish)

                self.project_conv = nn.Sequential(
                    GroupConv2D(in_channels, out_channels, n_chunks=project_ksize),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.project_conv = nn.Sequential(
                    GroupConv2D(in_channels * expand_ratio, out_channels, n_chunks=project_ksize),
                    nn.BatchNorm2d(out_channels),
                )

    def forward(self, x):
        if self.expand_ratio != 1:
            out = self.expand_conv(x)
            out = self.mdconv(out)

            if self._has_se:
                out = self.squeeze_excitation(out)
                out = self.project_conv(out)
            else:
                out = self.project_conv(out)
        else:
            out = self.mdconv(x)
            if self._has_se:
                out = self.squeeze_excitation(out)
                out = self.project_conv(out)
            else:
                out = self.project_conv(out)

        if self.stride == 1 and self.in_channels == self.out_channels:
            out = out + x
        return out


class MixNet(nn.Module):
    def __init__(self, num_classes=1000, model_name='s'):
        super(MixNet, self).__init__()
        self.head_hidden = 1536

        if model_name == 's':
            # in_channels, out_channels, n_chunks, stride, expqand_ratio, se_ratio
            small = [
                [16, 16, 1, 1, 1, None, False, 1, 1],
                [16, 24, 1, 2, 6, None, False, 2, 2],
                [24, 24, 1, 1, 3, None, False, 2, 2],
                [24, 40, 3, 2, 6, 0.5, True, 1, 1],
                [40, 40, 2, 1, 6, 0.5, True, 2, 2],

                [40, 40, 2, 1, 6, 0.5, True, 2, 2],
                [40, 40, 2, 1, 6, 0.5, True, 2, 2],
                [40, 80, 3, 2, 6, 0.25, True, 1, 2],
                [80, 80, 2, 1, 6, 0.25, True, 1, 2],
                [80, 80, 2, 1, 6, 0.25, True, 1, 2],

                [80, 120, 3, 1, 6, 0.5, True, 2, 2],
                [120, 120, 4, 1, 3, 0.5, True, 2, 2],
                [120, 120, 4, 1, 3, 0.5, True, 2, 2],
                [120, 200, 5, 2, 6, 0.5, True, 1, 1],
                [200, 200, 4, 1, 6, 0.5, True, 1, 2],

                [200, 200, 4, 1, 6, 0.5, True, 1, 2]
            ]

            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )

            layers = []
            for in_channels, out_channels, n_chunks, stride, expqand_ratio, se_ratio, swish, expand_ksize, project_ksize in small:
                layers.append(MixBlock(in_channels, out_channels, n_chunks, stride, expqand_ratio, se_ratio, swish, expand_ksize, project_ksize))

            last_out_channels = 200

        elif model_name == 'm':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(),
            )

            medium = [
                [24, 24, 1, 1, 1, None, False, 1, 1],
                [24, 32, 3, 2, 6, None, False, 2, 2],
                [32, 32, 1, 1, 3, None, False, 2, 2],
                [32, 40, 4, 2, 6, 0.5, True, 1, 1],
                [40, 40, 2, 1, 6, 0.5, True, 2, 2],

                [40, 40, 2, 1, 6, 0.5, True, 2, 2],
                [40, 40, 2, 1, 6, 0.5, True, 2, 2],
                [40, 80, 3, 2, 6, 0.25, True, 1, 1],
                [80, 80, 4, 1, 6, 0.25, True, 2, 2],
                [80, 80, 4, 1, 6, 0.25, True, 2, 2],

                [80, 80, 4, 1, 6, 0.25, True, 2, 2],
                [80, 120, 1, 1, 6, 0.5, True, 1, 1],
                [120, 120, 4, 1, 3, 0.5, True, 2, 2],
                [120, 120, 4, 1, 3, 0.5, True, 2, 2],
                [120, 120, 4, 1, 3, 0.5, True, 2, 2],

                [120, 200, 4, 2, 6, 0.5, True, 1, 1],
                [200, 200, 4, 1, 6, 0.5, True, 1, 2],
                [200, 200, 4, 1, 6, 0.5, True, 1, 2],
                [200, 200, 4, 1, 6, 0.5, True, 1, 2]
            ]

            layers = []
            for in_channels, out_channels, n_chunks, stride, expqand_ratio, se_ratio, swish, expand_ksize, project_ksize in medium:
                layers.append(
                    MixBlock(in_channels, out_channels, n_chunks, stride, expqand_ratio, se_ratio, swish, expand_ksize,
                             project_ksize))
            last_out_channels = 200

        elif model_name == 'l':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=round_filters(24), kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(round_filters(24)),
                nn.ReLU(),
            )

            medium = [
                [24, 24, 1, 1, 1, None, False, 1, 1],
                [24, 32, 3, 2, 6, None, False, 2, 2],
                [32, 32, 1, 1, 3, None, False, 2, 2],
                [32, 40, 4, 2, 6, 0.5, True, 1, 1],
                [40, 40, 2, 1, 6, 0.5, True, 2, 2],

                [40, 40, 2, 1, 6, 0.5, True, 2, 2],
                [40, 40, 2, 1, 6, 0.5, True, 2, 2],
                [40, 80, 3, 2, 6, 0.25, True, 1, 1],
                [80, 80, 4, 1, 6, 0.25, True, 2, 2],
                [80, 80, 4, 1, 6, 0.25, True, 2, 2],

                [80, 80, 4, 1, 6, 0.25, True, 2, 2],
                [80, 120, 1, 1, 6, 0.5, True, 1, 1],
                [120, 120, 4, 1, 3, 0.5, True, 2, 2],
                [120, 120, 4, 1, 3, 0.5, True, 2, 2],
                [120, 120, 4, 1, 3, 0.5, True, 2, 2],

                [120, 200, 4, 2, 6, 0.5, True, 1, 1],
                [200, 200, 4, 1, 6, 0.5, True, 1, 2],
                [200, 200, 4, 1, 6, 0.5, True, 1, 2],
                [200, 200, 4, 1, 6, 0.5, True, 1, 2]
            ]

            layers = []
            for in_channels, out_channels, n_chunks, stride, expqand_ratio, se_ratio, swish, expand_ksize, project_ksize in medium:
                layers.append(
                    MixBlock(round_filters(in_channels), round_filters(out_channels), n_chunks, stride, expqand_ratio, se_ratio, swish, expand_ksize,
                             project_ksize))

            last_out_channels = round_filters(200)

        self.layers = nn.Sequential(*layers)

        self.head_conv = nn.Sequential(
            nn.Conv2d(last_out_channels, self.head_hidden, kernel_size=1),
            nn.BatchNorm2d(self.head_hidden),
            nn.ReLU()
        )
        self.fc = nn.Linear(self.head_hidden, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.layers(out)
        out = self.head_conv(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


# temp = torch.randn((16, 3, 224, 224))
# mix = MixNet(model_name='s')
# print(mix(temp).size())
# print(get_model_parameters(mix))
