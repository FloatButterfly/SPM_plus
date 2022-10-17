import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from .quantizer import Quantizer
from ..ops import lower_bound, signum


class CDF_origin(nn.Module):
    def __init__(self, channels, filters=(3, 3, 3), init_scale=10.):  # 网络：(1,3,3,3,1)
        super(CDF, self).__init__()
        self._ch = int(channels)
        self._ft = (1,) + tuple(int(nf) for nf in filters) + (1,)
        init_scale = 10
        scale = init_scale ** (1 / (len(self._ft) - 1))

        self._matrices = nn.ParameterList()  # h
        self._biases = nn.ParameterList()  # b
        self._factors = nn.ParameterList()  # a

        for i in range(len(self._ft) - 1):
            # Define the matrices
            init = math.log(math.expm1(1 / scale / self._ft[i + 1]))
            matrix = nn.Parameter(torch.empty(self._ch, self._ft[i + 1], self._ft[i]).fill_(init))
            self._matrices.append(matrix)

            # Define the biases
            bias = nn.Parameter(torch.empty(self._ch, self._ft[i + 1], 1).uniform_(-0.5, 0.5))
            self._biases.append(bias)

            # Define the factor
            if i < len(self._ft) - 2:
                factor = nn.Parameter(torch.empty(self._ch, self._ft[i + 1], 1).fill_(0))
                self._factors.append(factor)
        # print("length of matrix", self._matrices)

    def forward(self, inputs, stop_gradient):
        # Inputs shape => [Channels, 1, Values]
        logits = inputs
        for i in range(len(self._ft) - 1):
            matrix = self._matrices[i]
            if stop_gradient:
                matrix = matrix.detach()
            matrix = nn.functional.softplus(matrix)
            logits = torch.matmul(matrix, logits)

            bias = self._biases[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self._ft) - 2:
                factor = self._factors[i]
                if stop_gradient:
                    factor = factor.detach()
                factor = torch.tanh(factor)
                logits += factor * torch.tanh(logits)

        return logits


class CDF(nn.Module):
    def __init__(self, channels, filters=(3, 3, 3), init_scale=10.):  # 网络：(1,3,3,3,1)
        super(CDF, self).__init__()
        self._ch = int(channels)
        self._ft = (1,) + tuple(int(nf) for nf in filters) + (1,)
        # init_scale = 10
        # scale = init_scale ** (1 / (len(self._ft) - 1))
        self.init_scale = float(init_scale)
        self.filters = tuple(int(f) for f in filters)
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))

        # self._matrices = nn.ParameterList()  # h
        # self._biases = nn.ParameterList()  # b
        # self._factors = nn.ParameterList()  # a

        # Define univariate density model
        for k in range(len(self.filters) + 1):
            # Weights
            H_init = np.log(np.expm1(1 / scale / filters[k + 1]))
            H_k = nn.Parameter(torch.ones((channels, filters[k + 1], filters[k])))  # apply softmax for non-negativity
            torch.nn.init.constant_(H_k, H_init)
            self.register_parameter('H_{}'.format(k), H_k)

            # Scale factors
            a_k = nn.Parameter(torch.zeros((channels, filters[k + 1], 1)))
            self.register_parameter('a_{}'.format(k), a_k)

            # Biases
            b_k = nn.Parameter(torch.zeros((channels, filters[k + 1], 1)))
            torch.nn.init.uniform_(b_k, -0.5, 0.5)
            self.register_parameter('b_{}'.format(k), b_k)

        # for i in range(len(self._ft) - 1):
        #     # Define the matrices
        #     init = math.log(math.expm1(1 / scale / self._ft[i + 1]))
        #     matrix = nn.Parameter(torch.empty(self._ch, self._ft[i + 1], self._ft[i]).fill_(init))
        #     self._matrices.append(matrix)
        #
        #     # Define the biases
        #     bias = nn.Parameter(torch.empty(self._ch, self._ft[i + 1], 1).uniform_(-0.5, 0.5))
        #     self._biases.append(bias)
        #
        #     # Define the factor
        #     if i < len(self._ft) - 2:
        #         factor = nn.Parameter(torch.empty(self._ch, self._ft[i + 1], 1).fill_(0))
        #         self._factors.append(factor)
        # print("length of matrix", self._matrices)

    def forward(self, inputs, stop_gradient):
        # Inputs shape => [Channels, 1, Values]
        logits = inputs

        for k in range(len(self.filters) + 1):
            H_k = getattr(self, 'H_{}'.format(str(k)))  # Weight
            a_k = getattr(self, 'a_{}'.format(str(k)))  # Scale
            b_k = getattr(self, 'b_{}'.format(str(k)))  # Bias

            if stop_gradient:
                H_k, a_k, b_k = H_k.detach(), a_k.detach(), b_k.detach()
            logits = torch.bmm(F.softplus(H_k), logits)  # [C,filters[k+1],*]
            logits = logits + b_k
            logits = logits + torch.tanh(a_k) * torch.tanh(logits)

        # for i in range(len(self._ft) - 1):
        #     matrix = self._matrices[i]
        #     if stop_gradient:
        #         matrix = matrix.detach()
        #     matrix = nn.functional.softplus(matrix)
        #     logits = torch.matmul(matrix, logits)
        #
        #     bias = self._biases[i]
        #     if stop_gradient:
        #         bias = bias.detach()
        #     logits += bias
        #
        #     if i < len(self._ft) - 2:
        #         factor = self._factors[i]
        #         if stop_gradient:
        #             factor = factor.detach()
        #         factor = torch.tanh(factor)
        #         logits += factor * torch.tanh(logits)

        return logits


class FullFactorizedModel(nn.Module):
    def __init__(self, channels, filters, likelihood_bound, sign_reversal=True):
        super(FullFactorizedModel, self).__init__()
        self._ch = int(channels)
        self._likelihood_bound = float(likelihood_bound)
        self._sign_reversal = sign_reversal

        self._cdf = CDF(self._ch, filters)
        self._quantizer = Quantizer()

        # Define the "optimize_integer_offset".
        self.quantiles = nn.Parameter(torch.zeros(self._ch, 1, 1))
        self.register_buffer("target", torch.zeros(self._ch, 1, 1))

    def forward(self, inputs):
        # Reshape the inputs.
        inputs = torch.transpose(inputs, 0, 1)
        shape = inputs.shape
        values = inputs.contiguous().view(self._ch, 1, -1)

        # Add noise or quantize.
        values = self._quantizer(values, self.training, self.quantiles)

        # Evaluate densities.
        lower = self._cdf(values - 0.5, stop_gradient=False)
        upper = self._cdf(values + 0.5, stop_gradient=False)
        if self._sign_reversal:
            sign = signum(lower + upper).detach()
            likelihood = torch.abs(
                torch.sigmoid(-sign * upper) - torch.sigmoid(-sign * lower)  # sign*num 相当于取绝对值
            )
        else:
            likelihood = torch.sigmoid(upper) - torch.sigmoid(lower)
        if self._likelihood_bound > 0:
            likelihood = lower_bound(likelihood, self._likelihood_bound)

        # Convert back to input tensor shape.
        values = values.view(*shape)
        values = torch.transpose(values, 0, 1)
        likelihood = likelihood.view(*shape)
        likelihood = torch.transpose(likelihood, 0, 1)

        return values, likelihood

    def integer_offset(self):
        logits = self._cdf(self.quantiles, stop_gradient=True)
        loss = torch.sum(torch.abs(logits - self.target))

        return loss

    def visualize(self, index, minval=-10, maxval=10, interval=0.1):
        # Get the default dtype and device.
        var = next(self.parameters())
        dtype, device = var.dtype, var.device

        # Compute the density.
        x = torch.arange(minval, maxval, interval, dtype=dtype, device=device)
        x_ = torch.zeros(self._ch, 1, len(x), dtype=dtype, device=device, requires_grad=True)
        with torch.no_grad():
            x_[index, 0, :] = x

        w_ = torch.sigmoid(self._cdf(x_, stop_gradient=True))
        w_.backward(torch.ones_like(w_))
        y = x_.grad[index, 0, :]

        # Convert the tensor to numpy array.
        x = x.cpu().numpy()
        y = y.cpu().numpy()

        plt.figure()
        plt.plot(x, y, 'r-')
        plt.show()

        return x, y
