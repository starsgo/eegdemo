import os
import argparse
import pickle

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import UnsupervisedPretrain

class LitModel_supervised_pretrain(pl.LightningModule):
    def __init__(self, args, save_path):
        super().__init__()
        self.args = args
        self.save_path = save_path
        self.T = 0.2
        self.model = UnsupervisedPretrain(emb_size=256, heads=8, depth=4,
                                          n_channels=64)

    def training_step(self, batch, batch_idx):

        # store the checkpoint every 5000 steps
        if self.global_step % 2000 == 0:
            self.trainer.save_checkpoint(
                filepath=f"{self.save_path}/epoch={self.current_epoch}_step={self.global_step}.ckpt"
            )

        prest_samples , labels= batch
        contrastive_loss = 0

        prest_masked_emb, prest_samples_emb = self.model(prest_samples, 0)

        # L2 normalize
        prest_samples_emb = F.normalize(prest_samples_emb, dim=1, p=2)
        prest_masked_emb = F.normalize(prest_masked_emb, dim=1, p=2)
        N = prest_samples.shape[0]

        # representation similarity matrix, NxN
        logits = torch.mm(prest_samples_emb, prest_masked_emb.t()) / self.T
        labels = torch.arange(N).to(logits.device)
        contrastive_loss += F.cross_entropy(logits, labels, reduction="mean")

        self.log("train_loss", contrastive_loss, on_step=True, on_epoch=True, prog_bar=True)
        return contrastive_loss

    def validation_step(self, batch, batch_idx):
        prest_samples, labels = batch
        contrastive_loss = 0

        prest_masked_emb, prest_samples_emb = self.model(prest_samples, 0)

        # L2 normalize
        prest_samples_emb = F.normalize(prest_samples_emb, dim=1, p=2)
        prest_masked_emb = F.normalize(prest_masked_emb, dim=1, p=2)
        N = prest_samples.shape[0]

        # representation similarity matrix, NxN
        logits = torch.mm(prest_samples_emb, prest_masked_emb.t()) / self.T
        labels = torch.arange(N).to(logits.device)
        contrastive_loss += F.cross_entropy(logits, labels, reduction="mean")

        self.log("val_loss", contrastive_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return contrastive_loss

    def configure_optimizers(self):
        # set optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

        # set learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.3
        )

        return [optimizer], [scheduler]
