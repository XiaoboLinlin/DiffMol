from DiffMol.model.lightningAVE import LightningAVE
import torch
from DiffMol.utils.openfold_utils import T
from DiffMol.utils.geo_utils import compute_frenet_frames

class ModelAVE(LightningAVE):
    
    def frenet_frames(self, batch):
        coords, mask = batch
        coords = coords.float()
        mask = mask.float()

        ca_coords = coords[:, 1::3]
        trans = ca_coords - torch.mean(ca_coords, dim=1, keepdim=True)
        rots = compute_frenet_frames(trans, mask)
        return T(rots, trans), mask
    
    def loss_fn(self, struc, mask):
        eps = 1e-10
        origin_struc = T(struc.rots,struc.trans)
        de_struc = self.model(struc)
        rmsds = (eps + torch.sum((de_struc.trans - origin_struc.trans) ** 2, dim=-1)) ** 0.5
        return torch.sum(rmsds * mask) / torch.sum(mask)