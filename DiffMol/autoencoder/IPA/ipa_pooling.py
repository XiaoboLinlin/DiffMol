
import torch
import torch.nn.functional as F
from DiffMol.IPA_general_feature.modules.mlp import MLP
from DiffMol.autoencoder.IPA.ipa import IPA
from torch_geometric.nn import BatchNorm, LayerNorm


class IPAPooling(torch.nn.Module):
    def __init__(self, config, hidden_dim=32, stride=2, kernel=3, padding=1, attn=False):
        super(IPAPooling, self).__init__()

        self.hidden_dim = hidden_dim
        self.stride = stride
        self.kernel = kernel
        self.padding = padding

        self.ipa = IPA(config)
       
    def forward(self, struc):

        ## get initial h and coords for pooling node

        # get number of node in one input graph and number of pooling node
        num_node = struc.trans.shape[1]
        num_pool_node = int(torch.div(num_node + 2 * self.padding - self.kernel, self.stride, rounding_mode='floor')) + 1

        # build mapping matrix to map original node to initial pooling node
        M = torch.zeros((num_pool_node, num_node + 2 * self.padding)).to(struc.trans.device)
        for i in range(num_pool_node):
            M[i, i * self.stride:(i * self.stride + self.kernel)] = 1 / self.kernel
            
        # create index to get coords for one graph and padding node (padding mode: same)
        index = [0] * self.padding + list(range(0, num_node)) + [num_node - 1] * self.padding
        struc.trans = struc.trans[:, index, :]
        trans_pool = M @ struc.trans # broadcast matrix multiplication
        struc.trans = torch.cat((struc.trans, trans_pool), dim=1) # B x (n + n_pool) x 3
        
        struc.rots = struc.rots[:, index, :, :]
        reshaped_rots = struc.rots.view(struc.rots.shape[0], struc.rots.shape[1], -1)
        rots_pool = M @ reshaped_rots # broadcast matrix multiplication
        rots_pool = rots_pool.view(rots_pool.shape[0], rots_pool.shape[1], struc.rots.shape[2], struc.rots.shape[3])
        struc.rots = torch.cat((struc.rots, rots_pool), dim=1) # B x (n + n_pool) x 3
        
        
        struc_ipa = self.ipa(struc) ## struc_new is a new instance
        struc.trans = struc_ipa.trans[:, (num_node + 2 * self.padding):, :]
        struc.rots = struc_ipa.rots[:, (num_node + 2 * self.padding):, :]
        
        return struc
