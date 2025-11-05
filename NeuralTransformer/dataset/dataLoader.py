import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

from .SEEDIV_dataset2 import SEEDIVDataset2


# SEEDIV train set loader
# data_file : train_data
# label_file : train_label
# batch_size: batch
# val_frac
class SEEDIV_trainSetLoader(pl.LightningDataModule):
    def __init__(self, data_file, label_file, batch_size=32, val_frac=0.1):
        super().__init__()
        self.data_file = data_file
        self.label_file = label_file
        self.batch_size = batch_size
        self.val_frac   = val_frac
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_ds = SEEDIVDataset2(self.data_file, self.label_file)
            val_len = int(len(full_ds) * self.val_frac)
            train_len = len(full_ds) - val_len
            self.train_ds, self.val_ds = random_split(
                full_ds, [train_len, val_len],
                generator=torch.Generator().manual_seed(42)  # 可复现
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True,  num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,   batch_size=self.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)