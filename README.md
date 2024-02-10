# DiffMol

This repository, named `DiffMol`, focuses on generating protein structures using a latent diffusion model. The core of the process involves training an autoencoder to represent protein structures in a compact, latent form, which is then utilized to train a diffusion model for generating new protein structures.

## How to Train Autoencoder

To begin training the autoencoder, execute the following command:

```bash
python DiffMol/main.py -c config_AVE.yml --train_autoencoder
```

This step involves using the configuration file `config_AVE.yml` to guide the training process of the autoencoder.

## Generate Latent Data for Training Latent Diffusion Model

Once the autoencoder is trained, the next step is to generate latent data necessary for training the latent diffusion model:

1. Modify the `data_generation_latent_diff.py` file to change the checkpoint path to the one you wish to load.
2. Execute the command below to generate the data:

```bash
python DiffMol/main.py -c config_AVE.yml --generate_data_latent_diff
```

## Train Latent Diffusion Model

To train the latent diffusion model with the generated latent data, use the following command:

```bash
python DiffMol/main.py -c config_diff.yml --train_latent_diffusion_model
```

## Sampling

For sampling from the trained latent diffusion model, execute:

```bash
python DiffMol/main_sample.py -c config_sample.yml
```

## Generate PDB File for Visualization

To visualize the generated protein structures, you can convert them into PDB files using the following command:

```bash
python evaluation/produce_pdb.py --input_dir /home/xiaobo/project/diffusion/DiffMol/data/sampling --output_dir /home/xiaobo/project/diffusion/DiffMol/data/sampling
```

### Note

When generating PDB files, ensure to modify the mask definition to match your hardware configuration, specifically, change to:

```python
mask = torch.ones(struc.trans.shape[0], struc.trans.shape[1]).to('cuda')
```

This adjustment is crucial for compatibility with CUDA-enabled devices.