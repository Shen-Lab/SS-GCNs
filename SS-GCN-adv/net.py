import torch
import torch.nn as nn


class net_gcn(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, adj, val_test=False):

        for ln in range(self.layer_num):
            x = torch.spmm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x


class net_gcn_2task(nn.Module):

    def __init__(self, embedding_dim, ss_class_num):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.ss_classifier = nn.Linear(embedding_dim[-2], ss_class_num, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, x_per, adj, adj_per, val_test=False):

        for ln in range(self.layer_num):
            x = torch.spmm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)

        if not val_test:
            for ln in range(self.layer_num):
                x_per = torch.spmm(adj_per, x_per)
                if ln == self.layer_num - 1:
                    break
                x_per = self.net_layer[ln](x_per)
                x_per = self.relu(x_per)
                x_per = self.dropout(x_per)
            x_per = self.ss_classifier(x_per)

        return x, x_per


