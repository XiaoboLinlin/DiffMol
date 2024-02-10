import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm, trange
from DiffMol.model.modelAVE import ModelAVE
from DiffMol.model.modelDiff import ModelDiff
from DiffMol.utils.config import Config
from DiffMol.utils.openfold_utils import T
from DiffMol.utils.geo_utils import compute_frenet_frames
import copy

def main(args):
    config = Config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    model_diff = ModelDiff.load_from_checkpoint(config.sampling.diff_checkpoint_dir, config = config)
    model_ave = ModelAVE.load_from_checkpoint(config.sampling.ave_checkpoint_dir, config = config)
    
    min_length = 80
    max_length = config.io.max_n_res
    
    
    # sample
    for length in trange(min_length, max_length + 1):
        for batch_idx in range(config.sampling.num_batches):
            mask = torch.cat([
				torch.ones((config.sampling.batch_size, length)),
				torch.zeros((config.sampling.batch_size, max_length - length))
			], dim=1).to(device)
            latent_struc = model_diff.p_sample_loop(config.sampling.noise_scale, verbose=False)
            struc0 = latent_struc[-1]
            struc_save = copy.deepcopy(struc0)
            
            # del latent_struc
            # torch.cuda.empty_cache()
            # struc0_trans = struc0.trans.to('cpu')
            # struc0_rots = struc0.rots.to('cpu')
            # struc0 = T(struc0_rots.to(model_ave.device), struc0_trans.to(model_ave.device))
            # decoded_struc = model_ave.model.decoder(struc0.to(model_ave.device))
            decoded_struc = model_ave.model.decoder(struc0)
            trans = decoded_struc.trans * mask.unsqueeze(-1)
            rots = compute_frenet_frames(trans, mask)
            for idx in range(config.sampling.batch_size):
                trans = decoded_struc.trans[idx].detach().to('cpu')
                trans_truncated = trans[:length]
                
                trans_diff = struc_save.trans[idx].detach().to('cpu')
                trans_diff_truncated = trans_diff[:length]
                
                torch.save(trans_truncated, f'data/sampling_test/sample_{batch_idx}_{idx}_{length}.pt')
                torch.save(trans_diff_truncated, f'data/sampling_test/diff_{batch_idx}_{idx}_{length}.pt')
                
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path for configuration file', required=True)
    args = parser.parse_args()

	# run
    main(args)