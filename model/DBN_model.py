import torch
from torch import nn

from model.RBM_model import RBM


# DBN模型
class DBN(nn.Module):
    def __init__(self):
        super(DBN, self).__init__()
        self.rbm1 = RBM(784, 500)
        self.rbm2 = RBM(500, 200)
        self.rbm3 = RBM(200, 50)
        self.output_layer = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        v, h = self.rbm1(x)
        v, h = self.rbm2(h)
        v, h = self.rbm3(h)
        out = self.output_layer(h)
        return out
