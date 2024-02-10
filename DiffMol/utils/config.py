import yaml

class Config:
    """
    Please see get_default_settings() for detailed default setting
    """
    def __init__(self, config_file, defaults=None):
        self.config = self.load_config(config_file)
        default_settings = self.get_default_settings()
        combined_defaults = self.merge_dicts(default_settings, defaults or {})
        self.apply_defaults(self.config, combined_defaults)
        if isinstance(self.config, dict):
            self.__dict__.update(self.config)
        elif hasattr(self.config, '__dict__'):
            self.__dict__.update(self.config.__dict__)
        else:
            raise TypeError("Config object is not in the correct format.")

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return self.convert_to_dot_notation(config)

    def convert_to_dot_notation(self, config):
        if isinstance(config, dict):
            dot_notation_config = type('DotNotationConfig', (), {})()  # Create an empty object
            for key, value in config.items():
                setattr(dot_notation_config, key, self.convert_to_dot_notation(value))  # Set attributes
            return dot_notation_config
        elif isinstance(config, list):
            return [self.convert_to_dot_notation(item) for item in config]
        else:
            return config

    def apply_defaults(self, config, defaults):
        for key, default_value in defaults.items():
            if isinstance(default_value, dict):
                # If default_value is a dictionary, we need to recurse
                config_subdict = getattr(config, key, type('DotNotationConfig', (), {})())  # Get the sub-dict or create a new one if it doesn't exist
                self.apply_defaults(config_subdict, default_value)  # Recursively apply defaults
                setattr(config, key, config_subdict)  # Set the updated sub-dict back to the config
            else:
                # Only set the default if the key is not present in the config
                if not hasattr(config, key):
                    setattr(config, key, default_value)
        return config


    def merge_dicts(self, dict1, dict2):
        merged = dict1.copy()
        for key, value in dict2.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def get_default_settings(self):
        DEFAULTS = {
            'io': {
                'name': None,  # 'name'
                'max_n_res': 128,  # 'maximumNumResidues'
                'min_n_res': None,  # 'minimumNumResidues'
                'log_dir': 'runs',  # 'logDirectory'
                'data_dir': 'data',  # 'dataDirectory'
                'dataset_names': 'scope',  # 'data set for training autoencoder'
                'diff_dataset_names': 'diff_train',  # 'data set for training diffusion model'
                'dataset_size': None,  # 'datasetSize'
                'dataset_classes': None,  # 'datasetClasses'
                'num_load_workers': 4
            },
            
            'diffusion': {
                'n_timestep': 1000,  # 'numTimesteps'
                'schedule': 'cosine'  # 'schedule'
            },
            'model': {
                'c_s': 128,  # 'singleFeatureDimension'
                'c_p': 128,  # 'pairFeatureDimension'
                
                # single feature network
                'c_pos_emb': 128,  # 'positionalEmbeddingDimension'
                'c_timestep_emb': 128,  # 'timestepEmbeddingDimension'
                
                # pair feature network
                'relpos_k': 32,  # 'relativePositionK'
                'template_type': 'v1',  # 'templateType'
                
                # pair transform network
                'n_pair_transform_layer': 5,  # 'numPairTransformLayers'
                'include_mul_update': True,  # 'includeTriangularMultiplicativeUpdate'
                'include_tri_att': False,  # 'includeTriangularAttention'
                'c_hidden_mul': 128,  # 'triangularMultiplicativeHiddenDimension'
                'c_hidden_tri_att': 32,  # 'triangularAttentionHiddenDimension'
                'n_head_tri': 4,  # 'triangularAttentionNumHeads'
                'tri_dropout': 0.25,  # 'triangularDropout'
                'pair_transition_n': 4,  # 'pairTransitionN'
                
                # structure network
                'n_structure_layer': 5,  # 'numStructureLayers'
                'n_structure_block': 1,  # 'numStructureBlocks'
                'c_hidden_ipa': 16,  # 'ipaHiddenDimension'
                'n_head_ipa': 12,  # 'ipaNumHeads'
                'n_qk_point': 4,  # 'ipaNumQkPoints'
                'n_v_point': 8,  # 'ipaNumVPoints'
                'ipa_dropout': 0.1,  # 'ipaDropout'
                'n_structure_transition_layer': 1,  # 'numStructureTransitionLayers'
                'structure_transition_dropout': 0.1  # 'structureTransitionDropout'
            },
        
            'training': {
                'seed': 100,  # 'seed'
                'n_epoch': 10000,  # 'numEpoches'
                'batch_size': 4,  # 'batchSize'
                'log_every_n_step': 10,  # 'logEverySteps'
                'checkpoint_every_n_epoch': 100  # 'checkpointEveryEpoches'
            },
            'sampling': {
                'ave_checkpoint_dir': None, 
                'diff_checkpoint_dir': None, 
                'batch_size': 4,
                'num_batches': 1,
                'num_res': 32,
                'noise_scale': 0.6
            },
            'optimization': {
                'lr': 1e-4,  # 'learningRate'
            }
        }
        return DEFAULTS
        