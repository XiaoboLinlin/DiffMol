from pytorch_lightning.trainer import Trainer
from DiffMol.autoencoder.data_module.data_module import DataModule
from DiffMol.model.modelAVE import ModelAVE


def train_ave(config, tb_logger, wandb_logger, checkpoint_callback):
  
  # data module
  dm = DataModule(config)

  # model
  model = ModelAVE(config)

  # trainer
  trainer = Trainer(
  devices=1,
  accelerator='gpu',
  logger=[tb_logger, wandb_logger],
  strategy='ddp',
  deterministic=True,
  enable_progress_bar=False,
  log_every_n_steps=config.training.log_every_n_step,
  max_epochs=config.training.n_epoch,
  callbacks=[checkpoint_callback]
  )
  # run
  trainer.fit(model, dm)