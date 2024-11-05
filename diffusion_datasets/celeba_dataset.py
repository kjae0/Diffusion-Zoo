import sys
sys.path.append('..')
from diffusion_datasets.transforms import get_transform

from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image

import os


class CelebADataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.root_dir = cfg['root_dir']
        self.transform = get_transform(cfg)
        
        self.images = []
        for img in sorted(os.listdir(self.root_dir)):
            self.images.append(os.path.join(self.root_dir, img))
            
        print(f"Total dataset: {len(self.images)}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.transform(Image.open(self.images[idx])), 0