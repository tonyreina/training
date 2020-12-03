import torch
import torch.nn as nn


activations = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(0.01),
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax(dim=1),
    "none": nn.Identity(),
}

normalizations = {
    "instance": lambda n, _: nn.InstanceNorm3d(n, affine=True),
    "batch": lambda n, _: nn.BatchNorm3d(n),
    "group": lambda n, g: nn.GroupNorm(n, g),
    "none": lambda _, __: nn.Identity(),
}

convolutions = {"transpose": nn.ConvTranspose3d, "regular": nn.Conv3d}


class Unet3D(nn.Module):
    def __init__(self, in_channels, n_class, normalization, size):
        super(Unet3D, self).__init__()

        filters = [32, 64, 128, 256, size]
        self.filters = filters

        self.inp = filters[:-1]
        self.out = filters[1:]
        input_dim = filters[0]

        self.input_block = InputBlock(in_channels, input_dim, normalization)

        self.downsample = nn.ModuleList(
            [DownsampleBlock(i, o, normalization) for i, o in zip(self.inp, self.out)]
        )
        self.bottleneck = DownsampleBlock(filters[-1], filters[-1], normalization)
        upsample = [UpsampleBlock(filters[-1], filters[-1], normalization)]
        upsample.extend([UpsampleBlock(i, o, normalization)
                         for i, o in zip(reversed(self.out), reversed(self.inp))])
        self.upsample = nn.ModuleList(upsample)
        self.output = OutputLayer(input_dim, n_class, normalization="none")

    def forward(self, x):
        x = self.input_block(x)
        outputs = [x]

        for downsample in self.downsample:
            x = downsample(x)
            outputs.append(x)

        x = self.bottleneck(x)

        for upsample, skip in zip(self.upsample, reversed(outputs)):
            x = upsample(x, skip)

        x = self.output(x)

        return x


def _normalization(normalization, num_features, num_groups=16):
    if normalization in normalizations:
        return normalizations[normalization](num_features, num_groups)
    raise ValueError(f"Unknown normalization {normalization}")


def _activation(activation):
    if activation in activations:
        return activations[activation]
    raise ValueError(f"Unknown activation {activation}")


def conv_block_factory(in_channels, out_channels,
                       kernel_size=3, stride=1, padding=1,
                       conv_type="regular",
                       normalization="instance", activation="leaky_relu"):
    conv = convolutions[conv_type]
    conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, bias=normalization == "none")
    normalization = _normalization(normalization, out_channels)
    activation = _activation(activation)

    return nn.Sequential(conv, normalization, activation)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization):
        super(DownsampleBlock, self).__init__()
        self.conv1 = conv_block_factory(in_channels, out_channels, normalization=normalization, stride=2)
        self.conv2 = conv_block_factory(out_channels, out_channels, normalization=normalization)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization):
        super(UpsampleBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_conv = conv_block_factory(in_channels, out_channels,
                                                kernel_size=2, stride=2, padding=0,
                                                conv_type="transpose", normalization="none", activation="none")

        self.conv1 = conv_block_factory(2 * out_channels, out_channels, normalization=normalization)
        self.conv2 = conv_block_factory(out_channels, out_channels, normalization=normalization)

    def forward(self, x, skip):
        x = self.upsample_conv(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization):
        super(InputBlock, self).__init__()
        self.conv1 = conv_block_factory(in_channels, out_channels, normalization=normalization)
        self.conv2 = conv_block_factory(out_channels, out_channels, normalization=normalization)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, in_channels, n_class, normalization):
        super(OutputLayer, self).__init__()
        self.conv = conv_block_factory(in_channels, n_class, normalization=normalization,
                                       kernel_size=1, padding=0, activation="none")

    def forward(self, x):
        return self.conv(x)

