from pytorch_lightning import Trainer
from DiffMol.diffusion.data_module.diff_data_module import DiffDataModule  # Make sure to import your DiffDataModule class correctly
from DiffMol.utils.config import Config

config = Config('config.yml')

# Assuming your config object is set up like this:

# Create the DataModule
data_module = DiffDataModule(config)

# Call setup - this is normally called internally, but we can call it to test
data_module.setup()

# Get the train dataloader
train_dataloader = data_module.train_dataloader()

# Iterate over the dataloader to get the first batch
for batch in train_dataloader:
    trans, rots = batch
    print(f'Batch of trans: {trans}')
    print(f'Batch of rots: {rots}')
    break  # Only need to check the first batch for this test

# You can also test with a full Trainer run if you have a model and trainer set up
# model = YourModel()  # Replace with your actual model
# trainer = Trainer()
# trainer.fit(model, datamodule=data_module)