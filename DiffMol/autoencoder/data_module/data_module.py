import os
import glob
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as dataloader

from pytorch_lightning import LightningDataModule

from DiffMol.autoencoder.data_module.dataset import Dataset
from DiffMol.autoencoder.data_module.load_path import load_filepaths


class DataModule(LightningDataModule):

	def __init__(self, config):
		super(DataModule, self).__init__()

		self.name = config.io.name
		self.log_dir = config.io.log_dir
		self.data_dir = config.io.data_dir
		self.max_n_res = config.io.max_n_res
		self.min_n_res = config.io.min_n_res
		self.dataset_names = config.io.dataset_names
		self.dataset_size = config.io.dataset_size
		self.dataset_classes = config.io.dataset_classes
		self.batch_size = config.training.batch_size

	def setup(self, stage=None):

		# load filepaths
		dataset_filepath = os.path.join(self.log_dir, self.name, 'dataset.txt')
		if os.path.exists(dataset_filepath):
			with open(dataset_filepath) as file:
				filepaths = [line.strip() for line in file]
		else:
			filepaths = load_filepaths(self.data_dir, self.dataset_names, self.max_n_res, self.min_n_res, self.dataset_classes, self.dataset_size)
			with open(dataset_filepath, 'w') as file:
				for filepath in filepaths:
					file.write(filepath + '\n')

		# create dataset
		self.dataset = Dataset(filepaths, self.max_n_res, self.min_n_res)
		print(f'Number of samples: {len(filepaths)}')

	def train_dataloader(self):
		return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)


	def diff_train_loader(self):
		"""Load data for generating training data for latent diffusion model
		Returns:
			_type_: _description_
		"""
		return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)

   