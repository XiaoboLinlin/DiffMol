from torch import nn
import torch
from DiffMol.IPA_general_feature.single_feature_net import SingleFeatureNet
from DiffMol.IPA_general_feature.pair_feature_net import PairFeatureNet
from DiffMol.IPA_general_feature.structure_net import StructureNet
from genie.model.pair_transform_net import PairTransformNet

class DiffIPA(nn.Module):
    def __init__(self,config):
        super(DiffIPA, self).__init__()
        self.single_feature_net = SingleFeatureNet(
            config.model.c_s,
            config.model.c_pos_emb,
            config.diffusion.n_timestep,
            config.model.c_timestep_emb
		)
        self.pair_feature_net = PairFeatureNet(
			config.model.c_s,
			config.model.c_p,
			config.model.c_pos_emb,
			config.model.template_type
		)
        # self.pair_transform_net = PairTransformNet(
		# 	config.model.c_p,
		# 	config.model.n_pair_transform_layer,
		# 	config.model.include_mul_update,
		# 	config.model.include_tri_att,
		# 	config.model.c_hidden_mul,
		# 	config.model.c_hidden_tri_att,
		# 	config.model.n_head_tri,
		# 	config.model.tri_dropout,
		# 	config.model.pair_transition_n
		# )
        self.structure_net = StructureNet(
			config.model.c_s,
			config.model.c_p,
			config.model.n_structure_layer,
			config.model.n_structure_block,
			config.model.c_hidden_ipa,
			config.model.n_head_ipa,
			config.model.n_qk_point,
			config.model.n_v_point,
			config.model.ipa_dropout,
			config.model.n_structure_transition_layer,
			config.model.structure_transition_dropout
		)
        
    def forward(self, struc, time, pair_tranform=False):
        s = self.single_feature_net(struc, time)
        p = self.pair_feature_net(s, struc)
        if pair_tranform:
            mask = torch.ones(struc.trans.shape[0], struc.trans.shape[1]).to('cuda')
            p_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
            p = self.pair_transform_net(p, p_mask)
        struc = self.structure_net(s, p, struc)
        return struc
        
        