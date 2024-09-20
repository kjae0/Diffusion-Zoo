from PIL import Image
from torchvision.utils import make_grid

import os
import cv2
import torch
import numpy as np
import imageio


def save_checkpoints(ckpt_path, ckpt, epoch, train_loss, val_loss=None):
    torch.save({
        'epoch': epoch,
        'checkpoint': ckpt,
        'train_loss': train_loss,
        'val_loss': val_loss
    }, ckpt_path)

def save_images_grid(save_path, images):
    """
    Args:
        save_path (_type_): save path
        images (list[torch.Tensor]): (N * M) x C x H x W, 0 to 1 normalized pytorch tensor.
    """
    if isinstance(images, list):
        images = torch.stack(images)  # Shape: (N*M, C, H, W)
    
    # Calculate grid size (assuming square grid)
    num_images = images.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    # Create the grid image using torchvision
    grid_img = make_grid(images, nrow=grid_size, padding=2)
    
    # Convert the tensor to a PIL Image
    ndarr = grid_img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(ndarr)
    
    # Save the image
    img.save(save_path)
    