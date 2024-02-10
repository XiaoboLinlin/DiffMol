
import torch
from torch import nn
from DiffMol.autoencoder.IPA.ipa_unpooling import IPAUnPooling

class Decoder(torch.nn.Module):
    def __init__(self, config, hidden_dim=32, ratio=2, layers=2, attn=False, out_node_dim=32, in_edge_dim=32, egnn_layers=4, residual=True):
        super(Decoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.layers = layers
        self.unpooling = nn.ModuleList()

        for i in range(self.layers):
            self.unpooling.append(
                IPAUnPooling(config, hidden_dim=self.hidden_dim, stride=2, kernel=3, padding=1, output_padding=1, attn=attn)
            )


    def forward(self, struc):

        for i in range(self.layers):
            # unpooling
            struc= self.unpooling[i](struc)

        return struc
