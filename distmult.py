import torch
import torch.nn as nn


class DM(torch.nn.Module):
    def __init__(self, dm_dim):
        super(DM, self).__init__()
        self.r = nn.Parameter(torch.randn(dm_dim))

    def forward(self, emb, batch_ind):
        x = emb[batch_ind]
        x_s = x[:, 0, :]
        x_o = x[:, 1, :]
        x = torch.sum(x_s * self.r * x_o, dim=1)
        
        return torch.sigmoid(x)
        