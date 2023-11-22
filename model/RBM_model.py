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

    def sample_h(self, v):
        p_h = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        sample_h = torch.bernoulli(p_h)
        return p_h, sample_h

    def sample_v(self, h):
        p_v = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        sample_v = torch.bernoulli(p_v)
        return p_v, sample_v

    def contrastive_divergence(self, data, learning_rate):
        v0 = data
        h0_prob, h0_sample = self.sample_h(v0)
        v1_prob, _ = self.sample_v(h0_sample)
        h1_prob, _ = self.sample_h(v1_prob)

        positive_grad = torch.matmul(h0_prob.T, v0)
        negative_grad = torch.matmul(h1_prob.T, v1_prob)

        self.W.data += learning_rate * (positive_grad - negative_grad) / data.size(0)
        self.v_bias.data += learning_rate * torch.mean(v0 - v1_prob, dim=0)
        self.h_bias.data += learning_rate * torch.mean(h0_prob - h1_prob, dim=0)

        # 计算loss
        positive_phase = torch.mean(torch.matmul(h0_prob.T, v0))
        negative_phase = torch.mean(torch.matmul(h1_prob.T, v1_prob))
        loss = negative_phase - positive_phase
        return loss
