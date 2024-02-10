
from pytorch_lightning.trainer import Trainer
from DiffMol.diffusion.data_module.diff_data_module import DiffDataModule
from DiffMol.model.modelDiff import ModelDiff



def train_latent_diff(config, tb_logger, wandb_logger, checkpoint_callback):
    # data module
    dm = DiffDataModule(config)

    # model
    model =  ModelDiff(config)

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