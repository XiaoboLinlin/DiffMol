from torch import nn
from DiffMol.autoencoder.encoder import Encoder
from DiffMol.autoencoder.decoder import Decoder

class AutoEncoder(nn.Module):
    def __init__(self,config, layers=2, mp_steps=4, num_types=27, type_dim=32, hidden_dim=32, out_node_dim=32, in_edge_dim=32,
                 output_pad_dim=1, output_res_dim=26, pooling=True, up_mlp=False, residual=True, noise=False, transpose=False, attn=False,
                 stride=2, kernel=3, padding=1):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(config, hidden_dim=hidden_dim, out_node_dim=hidden_dim,
                               in_edge_dim=hidden_dim, egnn_layers=mp_steps, layers=layers, pooling=pooling, residual=residual, attn=attn,
                               stride=stride, kernel=kernel, padding=padding)
        self.decoder = Decoder(config,hidden_dim=hidden_dim, ratio=2, layers=layers, attn=attn)
        
    def forward(self, struc):
        en_struc = self.encoder(struc) ## number of res decreases 1/4
        de_struc = self.decoder(en_struc)
        return de_struc
        