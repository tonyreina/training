import numpy as np
import torch
import torch.nn as nn

normalizations = {
    "instancenorm3d": nn.InstanceNorm3d,
    "instancenorm2d": nn.InstanceNorm2d,
    "batchnorm3d": nn.BatchNorm3d,
    "batchnorm2d": nn.BatchNorm2d,
}

convolutions = {
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "ConvTranspose2d": nn.ConvTranspose2d,
    "ConvTranspose3d": nn.ConvTranspose3d,
}


def get_norm(name, out_channels):
    if "groupnorm" in name:
        return nn.GroupNorm(32, out_channels, affine=True)
    return normalizations[name](out_channels, affine=True)


def get_conv(in_channels, out_channels, kernel_size, stride, dim, bias=False):
    conv = convolutions[f"Conv{dim}d"]
    padding = get_padding(kernel_size, stride)
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias)


def get_transp_conv(in_channels, out_channels, kernel_size, stride, dim):
    conv = convolutions[f"ConvTranspose{dim}d"]
    padding = get_padding(kernel_size, stride)
    output_padding = get_output_padding(kernel_size, stride, padding)
    return conv(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False)


def get_padding(kernel_size, stride):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]


def get_output_padding(kernel_size, stride, padding):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm, negative_slope, dim):
        super(ConvBlock, self).__init__()
        self.conv1 = get_conv(in_channels, out_channels, kernel_size, stride, dim)
        self.conv2 = get_conv(out_channels, out_channels, kernel_size, 1, dim)
        self.norm1 = get_norm(norm, out_channels)
        self.norm2 = get_norm(norm, out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, inp):
        out = self.lrelu(self.norm1(self.conv1(inp)))
        out = self.lrelu(self.norm2(self.conv2(out)))
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm, negative_slope, dim):
        super(UpsampleBlock, self).__init__()
        self.transp_conv = get_transp_conv(in_channels, out_channels, stride, stride, dim)
        self.conv_block = ConvBlock(2 * out_channels, out_channels, kernel_size, 1, norm, negative_slope, dim)

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(OutputBlock, self).__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size=1, stride=1, dim=dim, bias=True)

    def forward(self, inp):
        return self.conv(inp)
