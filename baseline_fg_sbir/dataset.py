import os
import pickle
import torch
import random
import torchvision.transforms.functional as F

from torch.utils.data import Dataset
from utils import get_transform
from random import randint
from PIL import Image

from rasterize import rasterize_sketch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSBIR_Dataset(Dataset):
    def __init__(self, hp, mode):
        self.hp = hp
        self.mode = mode
        
        coordinate_path = os.path.join(hp.root_dir, hp.dataset_name, hp.dataset_name + '_Coordinate')
        self.root_dir = os.path.join(hp.root_dir, hp.dataset_name)
        with open(coordinate_path, 'rb') as f:
            self.coordinate = pickle.load(f)
            
        self.train_sketch = [x for x in self.coordinate if 'train' in x]
        self.test_sketch = [x for x in self.coordinate if 'test' in x]
        
        self.train_transform = get_transform('train')
        self.test_transform = get_transform('test')
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_sketch)

        return len(self.test_sketch)
        
        
    def __getitem__(self, item):
        sample = {}
        
        if self.mode == 'train':
            sketch_path = self.train_sketch[item]
            
            positive_sample = '_'.join(self.train_sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            
            posible_list = list(range(len(self.train_sketch)))
            posible_list.remove(item)
            
            negative_item = posible_list[randint(0, len(posible_list)-1)]
            negative_sample = '_'.join(self.train_sketch[negative_item].split('/')[-1].split('_')[:-1])
            negative_path = os.path.join(self.root_dir, 'photo', negative_sample + '.png')
            
            vector_x = self.coordinate[sketch_path]
            sketch_img = rasterize_sketch(vector_x)
            # <class 'numpy.ndarray'>
            # print("Shape:", sketch_img.shape, "Dtype:", sketch_img.dtype)
               
            sketch_img = Image.fromarray(sketch_img).convert("RGB")
            
            positive_image = Image.open(positive_path).convert("RGB")
            negative_image = Image.open(negative_path).convert("RGB")
            
            # n_flip = random.random()
            # if n_flip > 0.5:
            #     sketch_img = F.hflip(sketch_img)
            #     positive_image = F.hflip(positive_image)
            #     negative_image = F.hflip(negative_image)
            
            sketch_img = self.train_transform(sketch_img)
            positive_image = self.train_transform(positive_image)
            negative_image = self.train_transform(negative_image)
            
            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path,
                      'positive_img': positive_image, 'positive_path': positive_sample,
                      'negative_img': negative_image, 'negative_path': negative_sample
                      } 
        
        elif self.mode == "test":
            sketch_path = self.test_sketch[item] 
            vector_x = self.coordinate[sketch_path]
            sketch_img = rasterize_sketch(vector_x)
            sketch_img = self.test_transform(Image.fromarray(sketch_img).convert("RGB"))
            
            positive_sample = '_'.join(self.test_sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            positive_image = self.test_transform(Image.open(positive_path).convert("RGB"))
            
            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path, 'Coordinate':vector_x,
                      'positive_img': positive_image, 'positive_path': positive_sample}
            
        return sample
            
            
                     
            