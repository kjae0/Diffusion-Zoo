import sys
sys.path.append('..')

from datasets.transforms import get_transform
from datasets.cifar10_dataset import CIFAR10Dataset
from datasets.celeba_dataset import CelebADataset

def build_dataset(cfg):
    if cfg['name'] == 'cifar10':
        return CIFAR10Dataset(cfg)
    elif cfg['name'] == 'celeba':
        return CelebADataset(cfg)
    else:
        raise NotImplementedError(f"Unknown dataset type: {cfg['name']}")
    