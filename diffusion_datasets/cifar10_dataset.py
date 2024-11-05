import sys
sys.path.append('..')
from diffusion_datasets.transforms import get_transform

from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image

import os


class CIFAR10Dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.root_dir = cfg['root_dir']
        self.transform = get_transform(cfg)
        
        self.c2i = {}
        self.images = []
        self.labels = []
        
        cls_idx = 0
        for cls in os.listdir(self.root_dir):
            self.c2i[cls] = cls_idx
            for img in sorted(os.listdir(os.path.join(self.root_dir, cls))):
                self.images.append(os.path.join(self.root_dir, cls, img))
                self.labels.append(self.c2i[cls])
                
            cls_idx += 1
            
        assert len(self.images) == len(self.labels)
            
        print(f"Loaded CIFAR10 dataset with {cls_idx} classes.")
        print(f"Classes: {self.c2i}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.transform(Image.open(self.images[idx])), self.labels[idx]
    