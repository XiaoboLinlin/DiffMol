
import torch
from DiffMol.autoencoder.IPA.ipa import IPA

class IPAUnPooling(torch.nn.Module):
    def __init__(self, config, hidden_dim=32, stride=2, kernel=3, padding=1, output_padding=1, attn=False):
        super(IPAUnPooling, self).__init__()

        self.stride = stride
        self.kernel = kernel
        self.padding = padding
        self.output_padding = output_padding

        self.ipa = IPA(config)


    def forward(self, struc):

        # initialize coords for pooling node
        num_node = struc.trans.shape[1]

        # size after padding
        aug_size = (num_node * self.stride - 1) + 2 * (self.kernel - self.padding - 1) + self.output_padding
        out_size = (num_node - 1) * self.stride - 2 * self.padding + (self.kernel - 1) + self.output_padding + 1
        M = torch.zeros((out_size, aug_size)).to(struc.trans.device)
        for i in range(out_size):
            M[i, i:(i + self.kernel)] = 1 / self.kernel

        ##### add same position and h on boundry, add average position and h in between #####
        avg = torch.stack([struc.trans[:, 0:-1, :], struc.trans[:, 1:, :]], dim=2).mean(dim=2) # B x (n-1) x 3
        tmp = torch.stack([struc.trans[:, 0:-1, :], avg], dim=2) # B x (n-1) x 2 x 3
        tmp = torch.flatten(tmp, start_dim=1, end_dim=2) # B x 2*(n-1) x 3
        struc.trans = torch.cat([struc.trans[:, 0:1, :],
                            tmp,
                            struc.trans[:, -1:, :].repeat(1,3,1)], dim=1)

        reshaped_rots = struc.rots.view(struc.rots.shape[0], struc.rots.shape[1], -1)
        avg = torch.stack([reshaped_rots[:, 0:-1, :], reshaped_rots[:, 1:, :]], dim=2).mean(dim=2) # B x (n-1) x F
        tmp = torch.stack([reshaped_rots[:, 0:-1, :], avg], dim=2) # B x (n-1) x 2 x F
        tmp = torch.flatten(tmp, start_dim=1, end_dim=2) # B x 2*(n-1) x F
        reshaped_rots = torch.cat([reshaped_rots[:, 0:1, :],
                       tmp,
                       reshaped_rots[:, -1:, :].repeat(1,3,1)], dim=1)

        assert reshaped_rots.shape[1] == M.shape[1]


        trans_pool = M @ struc.trans
        rots_pool = M @ reshaped_rots

        struc.trans = torch.cat((struc.trans, trans_pool), dim=1)
        
        rots_pool = torch.cat((reshaped_rots, rots_pool), dim=1)
        struc.rots = rots_pool.view(rots_pool.shape[0], rots_pool.shape[1], struc.rots.shape[2], struc.rots.shape[3])
        
        struc_ipa = self.ipa(struc)
        # keep pooling node
     
        struc.trans = struc_ipa.trans[:, aug_size:, :]
        struc.rots = struc_ipa.rots[:, aug_size:, :]

        return struc
