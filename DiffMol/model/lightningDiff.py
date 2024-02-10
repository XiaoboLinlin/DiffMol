from abc import ABC, abstractmethod
from pytorch_lightning.core import LightningModule
from torch.optim import Adam
from DiffMol.autoencoder.autoencoder import AutoEncoder
from DiffMol.utils.openfold_utils import T
from DiffMol.diffusion.diffIPA import DiffIPA
import torch
from tqdm import tqdm

class LightningDiff(LightningModule):
    def __init__(self, config):
        super(LightningDiff, self).__init__()
        self.config = config
        self.model =  DiffIPA(config)
        self.setup = False
        print("Diff lightning start")

    @abstractmethod
    def loss_fn(struc_t, noise_t, time):
        raise NotImplemented
    
    def p_sample_loop(self, noise_scale, verbose=True):
        if not self.setup:
            self.setup_schedule()
            self.setup = True
        ts = self.sample_frames(self.config.sampling.batch_size, self.config.sampling.num_res)
        ts_seq = [ts]
        for i in tqdm(reversed(range(self.config.diffusion.n_timestep)), desc='sampling loop time step', total=self.config.diffusion.n_timestep, disable=not verbose):
            s = torch.Tensor([i] * self.config.sampling.batch_size).long().to(self.device)
            if i == 100:
                print("here")
            ts = self.p(ts, s, noise_scale)
            ts_seq.append(ts)
        return ts_seq

    def training_step(self, batch):
        # print(f"Current device: {self.device}")
        if not self.setup:
            self.setup_schedule()
            self.setup = True
        # trans, rots = batch
        # struc0 = T(rots,trans)
        struc0 = self.transform(batch)
        time = self.sample_timesteps(struc0.shape[0])
        struc_t, noise_t = self.q(struc0, time)
        loss = self.loss_fn(struc_t, noise_t, time) 
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return Adam(
			self.model.parameters(),
			lr=self.config.optimization.lr
		)
        
