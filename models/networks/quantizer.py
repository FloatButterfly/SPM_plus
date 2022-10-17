import torch
import torch.nn as nn

from torch.distributions.uniform import Uniform


class Quantizer(nn.Module):
    def forward(self, x, is_training, offset=0, noise_scale=None):
        if is_training:
            # TODO: parametrize noise
            if noise_scale is None:
                y = x + torch.empty_like(x).uniform_(-0.5, 0.5)
            else:
                y = x / noise_scale + torch.empty_like(x).uniform_(-0.5, 0.5)
                print("training! Use noise scale")
                # noise = noise_scale * torch.empty_like(x).uniform_(-0.5, 0.5)
                # uniform must be reparameterized for grad backward !
                # noise = Uniform(-0.5 * torch.abs(noise_scale), 0.5 * torch.abs(noise_scale)).sample()

        else:
            if noise_scale is None:
                y = torch.round(x - offset) + offset
            else:
                print("Testing! Use noise scale!")
                y = torch.round(x / noise_scale)
        return y
