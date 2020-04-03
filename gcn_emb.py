import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv


class GCN_EMB(torch.nn.Module):
    def __init__(self, num_node_features, gcn_dim, conv_type, dropout_ratio=None):
        super(GCN_EMB, self).__init__()
        self.conv_type = conv_type
        self.dropout_ratio = dropout_ratio
        if conv_type == "gcnconv":
            self.conv1 = GCNConv(num_node_features, gcn_dim[0])
            self.conv2 = GCNConv(gcn_dim[0], gcn_dim[1])
        elif conv_type == "chebconv":
            self.conv1 = ChebConv(num_node_features, gcn_dim[0], 2)
            self.conv2 = ChebConv(gcn_dim[0], gcn_dim[1], 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if "chebconv" in self.conv_type and self.dropout_ratio is not None:
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        if "chebconv" in self.conv_type and self.dropout_ratio is not None:
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index)
        return x
