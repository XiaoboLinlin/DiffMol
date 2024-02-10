how to train autoencoder:
python DiffMol/main.py -c config_AVE.yml --train_autoencoder

1. Train Autoencoder
python DiffMol/main.py -c config_AVE.yml --train_autoencoder

2. Generate latent data for training latent diffusion model
    (1) change checkpoint path that you want to load in data_generation_latent_diff.py
    (2) python DiffMol/main.py -c config_AVE.yml ----generate_data_latent_diff

3. Train latent diffusion model
python DiffMol/main.py -c config_diff.yml --train_latent_diffusion_model

4. Do sampling
python DiffMol/main_sample.py -c config_sample.yml

5. Generate PDB file for visualization

python evaluation/produce_pdb.py --input_dir /home/xiaobo/project/diffusion/DiffMol/data/sampling --output_dir /home/xiaobo/project/diffusion/DiffMol/data/sampling

note:
need to change: mask = torch.ones(struc.trans.shape[0], struc.trans.shape[1]).to('cuda')
