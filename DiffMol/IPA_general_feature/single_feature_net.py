
import torch
from torch import nn
from DiffMol.utils.sinusoidal import sinusoidal_encoding


class SingleFeatureNet(nn.Module):

	def __init__(self,
		c_s,
		c_pos_emb,
		n_timestep = None,
		c_timestep_emb = None
		):
		super(SingleFeatureNet, self).__init__()

		self.c_s = c_s
		self.c_pos_emb = c_pos_emb
		if n_timestep and c_timestep_emb:
			self.n_timestep = n_timestep
			self.c_timestep_emb = c_timestep_emb
			self.linear = nn.Linear(self.c_pos_emb + self.c_timestep_emb, self.c_s)
		else:
			self.linear = nn.Linear(self.c_pos_emb, self.c_s)
			

	def forward(self, struc, timesteps = None):
		# s: [b]

		b, max_n_res, device = struc.shape[0], struc.shape[1], struc.trans.device
		# [b, n_res, c_pos_emb]
		pos_emb = sinusoidal_encoding(torch.arange(max_n_res).to(device), max_n_res, self.c_pos_emb)
		pos_emb = pos_emb.unsqueeze(0).repeat([b, 1, 1])

		if timesteps is not None:
			# [b, n_res, c_timestep_emb]
			timestep_emb = sinusoidal_encoding(timesteps.view(b, 1), self.n_timestep, self.c_timestep_emb)
			timestep_emb = timestep_emb.repeat(1, max_n_res, 1)
			# timestep_emb = timestep_emb * mask.unsqueeze(-1)
			return self.linear(torch.cat([
				pos_emb,
				timestep_emb
			], dim=-1))
		else:
			return self.linear(torch.cat([
				pos_emb
			], dim=-1))
    
    