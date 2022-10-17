import torch
import torch.nn as nn
from torch.nn import init
from ..ops import BinaryQuantize


class MyBinary(nn.module):
    def __init__(self, label_nc):
        super(MyBinary, self).__init__()
        size = (label_nc + 1, 1)
        self.Phi = nn.Parameter(init.xavier_normal_(torch.tensor(size)))
        # self.Bias = nn.Parameter(init.xavier_normal_(torch.Tensor(size)))
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def forward(self, y):
        # binarize_in = self.Phi + self.Bias)
        mask_matrix = BinaryQuantize(self.Phi, self.k, self.t)
        mask = mask_matrix.unsqueeze(0).unsqueeze(0)

        return mask
