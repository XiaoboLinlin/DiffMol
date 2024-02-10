from abc import ABC, abstractmethod
from pytorch_lightning.core import LightningModule
from torch.optim import Adam
from DiffMol.autoencoder.autoencoder import AutoEncoder

class LightningAVE(LightningModule):
    def __init__(self, config):
        super(LightningAVE, self).__init__()
        self.config = config
        self.model =  AutoEncoder(config)
        print("ModelAVE start")
    
    
    @abstractmethod
    def loss_fn(struc0, mask):
        raise NotImplemented

    @abstractmethod
    def frenet_frames(self, batch):
        raise NotImplemented

    def training_step(self, batch):
        struc0, mask = self.frenet_frames(batch)
        loss = self.loss_fn(struc0, mask) 
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return Adam(
			self.model.parameters(),
			lr=self.config.optimization.lr
		)
        
