import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _pair(inputs):
    if isinstance(inputs, int):
        outputs = (inputs, inputs)
    elif isinstance(inputs, list) or isinstance(inputs, tuple):
        if len(inputs) != 2:
            raise ValueError("Length of parameters should be TWO!")
        else:
            outputs = tuple(int(item) for item in inputs)
    else:
        raise TypeError("Not proper type!")

    return outputs


class SignConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample, use_bias):
        super(SignConv2d, self).__init__()
        # Now only check these combinations of parameters.
        if (kernel_size, stride) not in [(9, 4), (5, 2), (3, 1)]:
            raise ValueError("This pair of parameters (kernel_size, stride) has not been checked!")

        # Save the input parameters.
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        # kernel 和 stride 其实都分(h,w), _pair
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.upsample = bool(upsample)
        self.use_bias = bool(use_bias)

        # Define the parameters.
        if self.upsample:
            self.weight = nn.Parameter(torch.Tensor(
                self.in_channels, self.out_channels, *self.kernel_size
            ))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                self.out_channels, self.in_channels, *self.kernel_size
            ))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))  # F.conv中bias定义为 out_channels
        else:
            self.register_parameter("bias", None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            init.xavier_normal_(self.weight)
            if self.bias is not None:
                self.bias.zero_()

    # 编码是：补零(h,w)->(0,1)+卷积，解码对称的过程是：反卷积+裁剪
    def forward(self, inputs):
        if self.upsample:
            outputs = F.conv_transpose2d(inputs, self.weight, self.bias, self.stride, 0)  # padding_size=0
            outputs = outputs[
                      slice(None), slice(None),
                      self.kernel_size[0] // 2:self.kernel_size[0] // 2 - (self.kernel_size[0] - self.stride[0]),
                      self.kernel_size[1] // 2:self.kernel_size[1] // 2 - (self.kernel_size[1] - self.stride[1]),
                      ]
        else:
            outputs = F.conv2d(inputs, self.weight, self.bias, self.stride, tuple(k // 2 for k in self.kernel_size))

        return outputs


class Downsample2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_bias):
        super(Downsample2d, self).__init__()
        # Now only check these combinations of parameters
        if (kernel_size, stride) not in [(9, 4), (5, 2)]:
            raise ValueError("This pair of parameters (kernel_size, stride) has not been checked!")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.padding = tuple(k // 2 for k in self.kernel_size)
        self.stride = _pair(stride)
        self.use_bias = bool(use_bias)

        # Define the parameter
        self.weight = nn.Parameter(torch.Tensor(
            self.out_channels, self.in_channels, *self.kernel_size
        ))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            self.weight.normal_(0, math.sqrt(1 / fan_in))
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, inputs):
        return F.conv2d(inputs, self.weight, self.bias, self.stride, self.padding)


class Upsample2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_bias):
        super(Upsample2d, self).__init__()
        # Now only check these combinations of parameters
        if (kernel_size, stride) not in [(9, 4), (5, 2)]:
            raise ValueError("This pair of parameters (kernel_size, stride) has not been checked!")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(0)
        self.stride = _pair(stride)
        self.use_bias = bool(use_bias)

        # Define the parameter
        self.weight = nn.Parameter(torch.Tensor(
            self.in_channels, self.out_channels, *self.kernel_size
        ))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # Define the slice
        self._slice = [
            slice(None), slice(None),
            slice(self.kernel_size[0] // 2, self.kernel_size[0] // 2 - (self.kernel_size[0] - self.stride[0])),
            slice(self.kernel_size[1] // 2, self.kernel_size[1] // 2 - (self.kernel_size[1] - self.stride[1])),
        ]

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            self.weight.normal_(0, math.sqrt(1 / fan_in))
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, inputs):
        outputs = F.conv_transpose2d(inputs, self.weight, self.bias, self.stride, self.padding)
        outputs = outputs[self._slice]

        return outputs
