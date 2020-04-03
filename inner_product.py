import torch
import torch.nn as nn


class IP(torch.nn.Module):
    def __init__(self):
        super(IP, self).__init__()

    def forward(self, emb, batch_ind):
        x = emb[batch_ind]
        x_s = x[:, 0, :]
        x_o = x[:, 1, :]
        x = torch.sum(x_s * x_o, 1)
        return torch.sigmoid(x)
