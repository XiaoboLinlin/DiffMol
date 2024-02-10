import os
import glob
from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule

from DiffMol.diffusion.data_module.diff_dataset import DiffDataset
from DiffMol.autoencoder.data_module.load_path import load_filepaths


class DiffDataModule(LightningDataModule):

    def __init__(self, config):
        super(DiffDataModule, self).__init__()
        self.data_dir = "{}/{}".format(config.io.data_dir, config.io.diff_dataset_names)
        self.batch_size = config.training.batch_size
        self.num_workers = config.io.num_load_workers  # You can set a default here if not in config

    def setup(self, stage=None):
        # if stage == 'fit' or stage is None:
        self.dataset = DiffDataset(self.data_dir)
            
    def train_dataloader(self):
        print("Number of samples in dataset:", len(self.dataset))
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
