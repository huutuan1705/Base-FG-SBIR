import os
import pickle
import torch

from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSBIR_Dataset(Dataset):
    def __init__(self, hp, mode):
        self.hp = hp
        self.mode = mode
        
        coordinate_path = os.path.join(hp.root_dir, 'Dataset', hp.dataset_name, hp.dataset_name + '_Coordinate')
        self.root_dir = os.path.join(hp.root_dir, 'Dataset', hp.dataset_name)
        with open(coordinate_path, 'rb') as f:
            self.Coordinate = pickle.load(f)
            
        