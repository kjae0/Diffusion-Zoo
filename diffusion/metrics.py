import torch
import torch.nn.functional as F

def calculate_psnr(img1, img2, max_pixel_value=1.0):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.
    
    Args:
        img1: torch.Tensor, the first image (ground truth), shape (B, C, H, W) or (C, H, W)
        img2: torch.Tensor, the second image (reconstructed), shape (B, C, H, W) or (C, H, W)
        max_pixel_value: Maximum possible pixel value (1.0 if images are normalized between 0 and 1)
    
    Returns:
        psnr: Peak Signal-to-Noise Ratio (in dB)
    """
    # Ensure the inputs are float tensors
    img1 = img1.float()
    img2 = img2.float()
    
    # Compute Mean Squared Error (MSE)
    mse = F.mse_loss(img1, img2)
    
    if mse == 0:
        return float('inf')  # PSNR is infinite if MSE is 0
    
    # Compute PSNR
    psnr = 10 * torch.log10((max_pixel_value ** 2) / mse)
    
    return psnr

def calculate_mse(img1, img2):
    img1 = img1.float()
    img2 = img2.float()
    
    # Compute Mean Squared Error (MSE)
    mse = F.mse_loss(img1, img2)
    
    return mse
#   mport math
# import os

# import numpy as np
# import torch
# from einops import rearrange, repeat
# from pytorch_fid.fid_score import calculate_frechet_distance
# from pytorch_fid.inception import InceptionV3
# from torch.nn.functional import adaptive_avg_pool2d
# from tqdm.auto import tqdm


# def num_to_groups(num, divisor):
#     groups = num // divisor
#     remainder = num % divisor
#     arr = [divisor] * groups
#     if remainder > 0:
#         arr.append(remainder)
#     return arr


# class FIDEvaluation:
#     def __init__(
#         self,
#         batch_size,
#         dl,
#         sampler,
#         channels=3,
#         accelerator=None,
#         stats_dir="./results",
#         device="cuda",
#         num_fid_samples=50000,
#         inception_block_idx=2048,
#     ):
#         self.batch_size = batch_size
#         self.n_samples = num_fid_samples
#         self.device = device
#         self.channels = channels
#         self.dl = dl
#         self.sampler = sampler
#         self.stats_dir = stats_dir
#         self.print_fn = print if accelerator is None else accelerator.print
#         assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
#         block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
#         self.inception_v3 = InceptionV3([block_idx]).to(device)
#         self.dataset_stats_loaded = False

#     def calculate_inception_features(self, samples):
#         if self.channels == 1:
#             samples = repeat(samples, "b 1 ... -> b c ...", c=3)

#         self.inception_v3.eval()
#         features = self.inception_v3(samples)[0]

#         if features.size(2) != 1 or features.size(3) != 1:
#             features = adaptive_avg_pool2d(features, output_size=(1, 1))
#         features = rearrange(features, "... 1 1 -> ...")
#         return features

#     def load_or_precalc_dataset_stats(self):
#         path = os.path.join(self.stats_dir, "dataset_stats")
#         try:
#             ckpt = np.load(path + ".npz")
#             self.m2, self.s2 = ckpt["m2"], ckpt["s2"]
#             self.print_fn("Dataset stats loaded from disk.")
#             ckpt.close()
#         except OSError:
#             num_batches = int(math.ceil(self.n_samples / self.batch_size))
#             stacked_real_features = []
#             self.print_fn(
#                 f"Stacking Inception features for {self.n_samples} samples from the real dataset."
#             )
#             for _ in tqdm(range(num_batches)):
#                 try:
#                     real_samples = next(self.dl)
#                 except StopIteration:
#                     break
#                 real_samples = real_samples.to(self.device)
#                 real_features = self.calculate_inception_features(real_samples)
#                 stacked_real_features.append(real_features)
#             stacked_real_features = (
#                 torch.cat(stacked_real_features, dim=0).cpu().numpy()
#             )
#             m2 = np.mean(stacked_real_features, axis=0)
#             s2 = np.cov(stacked_real_features, rowvar=False)
#             np.savez_compressed(path, m2=m2, s2=s2)
#             self.print_fn(f"Dataset stats cached to {path}.npz for future use.")
#             self.m2, self.s2 = m2, s2
#         self.dataset_stats_loaded = True

#     @torch.inference_mode()
#     def fid_score(self):
#         if not self.dataset_stats_loaded:
#             self.load_or_precalc_dataset_stats()
#         self.sampler.eval()
#         batches = num_to_groups(self.n_samples, self.batch_size)
#         stacked_fake_features = []
#         self.print_fn(
#             f"Stacking Inception features for {self.n_samples} generated samples."
#         )
#         for batch in tqdm(batches):
#             fake_samples = self.sampler.sample(batch_size=batch)
#             fake_features = self.calculate_inception_features(fake_samples)
#             stacked_fake_features.append(fake_features)
#         stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
#         m1 = np.mean(stacked_fake_features, axis=0)
#         s1 = np.cov(stacked_fake_features, rowvar=False)

#         return calculate_frechet_distance(m1, s1, self.m2, self.s2)
