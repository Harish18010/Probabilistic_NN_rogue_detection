import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class LoRaDataset(Dataset):
    def __init__(self, file_path, subset_size=None):
        self.file_path = file_path
        self.h5_file = h5py.File(file_path, 'r')
        self.data = self.h5_file['data']
        self.labels = self.h5_file['label']
        
        self.total_samples = self.data.shape[0]
        
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
        
       
        data_row = self.data[actual_idx]
        
       
        original_label = self.labels[0, actual_idx]
        device_label = int(original_label - 1)
        
       
        is_rogue = 0 
        
        x = torch.tensor(data_row, dtype=torch.float32)
        y_device = torch.tensor(device_label, dtype=torch.long)
        y_rogue = torch.tensor(is_rogue, dtype=torch.long)
        
        return x, y_device, y_rogue
        
    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()