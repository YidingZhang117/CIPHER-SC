import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class NN(torch.nn.Module):
    def __init__(self, nn_dim, device):
        super(NN, self).__init__()
        fc_list = []
        for i in range(len(nn_dim)-2):
            fc_list.append((f"fc{i+1}", nn.Linear(nn_dim[i], nn_dim[i+1])))
            fc_list.append((f"relu{i+1}", nn.ReLU()))
        fc_list.append((f"fc{len(nn_dim)}", nn.Linear(nn_dim[-2], nn_dim[-1])))
        print(fc_list)
        self.fc = nn.Sequential(OrderedDict(fc_list))

    def forward(self, emb, batch_ind):
        x = emb[batch_ind]
        x = x.view(batch_ind.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
