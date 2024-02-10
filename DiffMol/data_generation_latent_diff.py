import torch
from DiffMol.autoencoder.data_module.data_module import DataModule
from DiffMol.model.modelAVE import ModelAVE

from tqdm import tqdm
from DiffMol.utils.geo_utils import compute_frenet_frames
from DiffMol.utils.openfold_utils import T

def generate_data_latent_diff(config):
    """
    Generate training data for diffusion model at latent space
    
    """

    # config = Config('config.yml')
    # # data module
    dm = DataModule(config)
    # model
    model = ModelAVE.load_from_checkpoint("/home/xiaobo/project/diffusion/DiffMol/runs/AVE/version_3/checkpoints/config.io.name=0_epoch=199.ckpt", config = config)

    dm.setup()
    train_loader = dm.diff_train_loader()
    print("hello there")

    def frenet_frames(batch):
        coords, mask = batch
        coords = coords.float()
        mask = mask.float()

        ca_coords = coords[:, 1::3]
        trans = ca_coords - torch.mean(ca_coords, dim=1, keepdim=True)
        rots = compute_frenet_frames(trans, mask)
        return T(rots, trans), mask

    n = 0
    device = torch.device("cuda")
    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        struc, mask = frenet_frames(batch)
        test_struc = T(struc.rots.to(device), struc.trans.to(device))
        encode_struc = model.model.encoder(test_struc)
        
        for i in range(encode_struc.trans.shape[0]):
            
            torch.save((encode_struc.trans[i].to('cpu'), encode_struc.rots[i].to('cpu')),f'data/diff_train/sample_{n}.pt')
            n += 1
        