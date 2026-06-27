import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class LoRaDataset(Dataset):
    def __init__(self, file_path, subset_size=None, is_rogue_dataset=False):
        self.file_path = file_path
        self.is_rogue_dataset = is_rogue_dataset
        
       
        with h5py.File(file_path, 'r') as f:
            self.total_samples = f['data'].shape[0]
        
        if subset_size:
            np.random.seed(42)
            self.indices = np.random.choice(self.total_samples, subset_size, replace=False)
        else:
            self.indices = np.arange(self.total_samples)
            
        self.length = len(self.indices)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
       
        with h5py.File(self.file_path, 'r') as h5_file:
            data_row = h5_file['data'][actual_idx]
            original_label = h5_file['label'][0, actual_idx]
            
        device_label = int(original_label - 1)
        
       
        is_rogue_label = 1 if self.is_rogue_dataset else 0
        
        x = torch.tensor(data_row, dtype=torch.float32)
        y_device = torch.tensor(device_label, dtype=torch.long)
        y_rogue = torch.tensor(is_rogue_label, dtype=torch.long)
        
        return x, y_device, y_rogue