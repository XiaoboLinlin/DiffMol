
import numpy as np
from torch.utils.data import Dataset
import os
import torch

class DiffDataset(Dataset):
    def __init__(self, data_dir):
        super(DiffDataset, self).__init__()
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.data_dir = data_dir
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        trans, rots = torch.load(file_path)
        trans = trans.detach()
        rots = rots.detach()
        return trans, rots