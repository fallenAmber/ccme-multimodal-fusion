"""
dataset_utils.py

PairedModuleDataset moved out of the notebook into a real, importable module.

WHY: On macOS, PyTorch DataLoader workers (num_workers > 0) use Python's
'spawn' start method, which re-imports everything a worker needs in a fresh
process - including your Dataset class. A class defined inline in a Jupyter
notebook lives in '__main__' and can't be re-imported by a spawned worker,
causing:
    AttributeError: Can't get attribute 'PairedModuleDataset' on <module '__main__'>
Moving the class into this file (and importing it in the notebook) fixes
this, because spawned workers CAN import a real module.

USAGE in your notebook, replacing the inline class definition:
    from dataset_utils import PairedModuleDataset

"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PairedModuleDataset(Dataset):
    """
    Returns (rgb_tensor, thermal_tensor, label).

    Thermal: single-channel (1,H,W), normalized with (mean=0.5, std=0.25).
    INFERNO colormap used only for visualization - NOT at model input.

    Synchronized augmentation:
      Geometric flips (hflip, vflip) are decided once per sample from a
      shared random draw and applied identically to both modalities.
      ColorJitter is RGB-only (photometric - no spatial component).
    """

    def __init__(self, df: pd.DataFrame, img_size: int = 224, split: str = "train"):
        self.df = df.reset_index(drop=True)
        self.training = split == "train"
        self.img_size = img_size

        # Resize pipelines - geometric augmentation done manually below
        self.rgb_resize = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
            ]
        )
        self.th_resize = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
            ]
        )
        # RGB-only photometric augmentation
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1
        )
        # Final normalization
        self.rgb_norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.th_norm = transforms.Compose(
            [
                transforms.ToTensor(),  # -> (1,H,W)
                transforms.Normalize((0.5,), (0.25,)),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        rgb = np.load(r["rgb_npy"])[:, :, ::-1].copy()  # BGR->RGB
        th = np.load(r["thermal_npy"])
        if th.ndim == 2:
            th = th[:, :, None]

        rgb_pil = self.rgb_resize(rgb)
        th_pil = self.th_resize(th)

        if self.training:
            # Synchronized geometric augmentation
            # One shared random decision applied to BOTH modalities
            if torch.rand(1).item() < 0.5:
                rgb_pil = transforms.functional.hflip(rgb_pil)
                th_pil = transforms.functional.hflip(th_pil)
            if torch.rand(1).item() < 0.3:
                rgb_pil = transforms.functional.vflip(rgb_pil)
                th_pil = transforms.functional.vflip(th_pil)
            # ColorJitter: RGB only (no spatial component)
            rgb_pil = self.color_jitter(rgb_pil)

        return (
            self.rgb_norm(rgb_pil),
            self.th_norm(th_pil),
            torch.tensor(int(r["label"]), dtype=torch.long),
        )
