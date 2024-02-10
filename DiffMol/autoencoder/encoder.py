from torch import nn
import torch
from torch_geometric.nn import BatchNorm, LayerNorm
from DiffMol.autoencoder.IPA.ipa_pooling import IPAPooling


class Encoder(nn.Module):
    def __init__(self, config, hidden_dim=32, out_node_dim=32, in_edge_dim=32, max_length=256, layers=2,
                 egnn_layers=4, pooling=True, residual=True, attn=False, stride=2, kernel=3, padding=1):
        super(Encoder, self).__init__()
        
        self.layers = layers
        self.poolings = nn.ModuleList([IPAPooling(config, hidden_dim=hidden_dim, stride=stride, kernel=kernel, padding=padding, attn=attn) for i in range(self.layers)])
    
        
    def forward(self, struc):
        # ipa
        for i in range(self.layers):
            struc = self.poolings[i](struc)
        return struc