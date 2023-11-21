import torch
from torch import nn


# RBM模型
class RBM(nn.Module):
    def __init__(self, n_vis, n_hid):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
        self.v_bias = nn.Parameter(torch.zeros(n_vis))

    def forward(self, v):
        h = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        v = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        return v, h

    def free_energy(self, v):
        v_term = torch.matmul(v, self.v_bias)
        w_x_h = torch.matmul(v, self.W.t()) + self.h_bias
        h_term = torch.sum(torch.log1p(torch.exp(w_x_h)), dim=1)
        return -h_term - v_term
