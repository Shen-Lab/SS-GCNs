import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

from gnns.gin_layer import GINLayer, ApplyNodeFunc, MLP

class GINNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params[0]
        hidden_dim = net_params[1]
        n_classes = net_params[2]
        dropout = 0.5
        self.n_layers = 2
        n_mlp_layers = 1               # GIN
        learn_eps = True              # GIN
        neighbor_aggr_type = 'mean' # GIN
        graph_norm = False      
        batch_norm = False
        residual = False
        self.n_classes = n_classes
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        for layer in range(self.n_layers):
            if layer == 0:
                mlp = MLP(n_mlp_layers, in_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, n_classes)
                
            
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, graph_norm, batch_norm, residual, 0, learn_eps))

        # Linear function for output of each layer
        # which maps the output of different layers into a prediction score

        self.linears_prediction = nn.Linear(hidden_dim, n_classes, bias=False)
        
        
    def forward(self, g, h, snorm_n, snorm_e):
        
        # list of hidden representation at each layer (including input)
        hidden_rep = []

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, snorm_n)
            hidden_rep.append(h)

        # score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2
        score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2

        return score_over_layer
      
 
class GINNet_ss(nn.Module):
    
    def __init__(self, net_params, num_par):
        super().__init__()
        in_dim = net_params[0]
        hidden_dim = net_params[1]
        n_classes = net_params[2]
        dropout = 0.5
        self.n_layers = 2
        n_mlp_layers = 1               # GIN
        learn_eps = True              # GIN
        neighbor_aggr_type = 'mean' # GIN
        graph_norm = False      
        batch_norm = False
        residual = False
        self.n_classes = n_classes
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        for layer in range(self.n_layers):
            if layer == 0:
                mlp = MLP(n_mlp_layers, in_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, n_classes)
                
            
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, graph_norm, batch_norm, residual, 0, learn_eps))

        # Linear function for output of each layer
        # which maps the output of different layers into a prediction score

        self.linears_prediction = nn.Linear(hidden_dim, n_classes, bias=False)
        self.classifier_ss = nn.Linear(hidden_dim, num_par, bias=False)
        
    def forward(self, g, h, snorm_n, snorm_e):
        
        # list of hidden representation at each layer (including input)
        hidden_rep = []

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, snorm_n)
            hidden_rep.append(h)

        score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2
        h_ss = self.classifier_ss(hidden_rep[0])

        return score_over_layer, h_ss
       
