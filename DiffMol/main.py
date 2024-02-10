import argparse
from DiffMol.utils.config import Config
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer, seed_everything
from DiffMol.autoencoder.data_module.data_module import DataModule
from DiffMol.model.modelAVE import ModelAVE
from DiffMol.data_generation_latent_diff import generate_data_latent_diff
from DiffMol.train_ave import train_ave
from DiffMol.train_latent_diff import train_latent_diff


def main(args):
  config = Config(args.config)
    # logger
  tb_logger = TensorBoardLogger(
    save_dir=config.io.log_dir,
    name=config.io.name
  )
  wandb_logger = WandbLogger(project=config.io.name)
    
  # checkpoint callback, save the model at specific intervals during training
  checkpoint_callback = ModelCheckpoint(
  every_n_epochs=config.training.checkpoint_every_n_epoch,
  filename='{config.io.name}_{epoch}',
  save_top_k=-1
  )
  # seed
  seed_everything(config.training.seed, workers=True)

  if args.train_autoencoder:
    train_ave(config, tb_logger, wandb_logger, checkpoint_callback)

  if args.generate_data_latent_diff:
    generate_data_latent_diff(config)
    
  if args.train_latent_diffusion_model:
    train_latent_diff(config, tb_logger, wandb_logger, checkpoint_callback)

  
  
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path for configuration file', required=True)
    parser.add_argument('--train_autoencoder', action='store_true', help='Flag to train the autoencoder')
    parser.add_argument('--generate_data_latent_diff', action='store_true', help='Flag to generate trainning data for latent diffusion model')
    parser.add_argument('--train_latent_diffusion_model', action='store_true', help='Flag to train the diffusion model in latent space')
    args = parser.parse_args()
    
    #run
    main(args)