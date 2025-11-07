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
from torchmetrics import Accuracy

from model import NeuralTransformer

class LitModel_neuralTransformer(pl.LightningModule):
    def __init__(self, args, save_path):
        super().__init__()
        self.args = args
        self.save_path = save_path
        self.in_chans = np.arange(1, 63)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=4)
        self.model = NeuralTransformer(EEG_size=args.eeg_size, patch_size=args.patch_size)

    def training_step(self, batch, batch_idx):
        # store the checkpoint every 5000 steps
        if self.global_step % 2000 == 0:
            self.trainer.save_checkpoint(
                filepath=f"{self.save_path}/epoch={self.current_epoch}_step={self.global_step}.ckpt"
            )
        x, y = batch
        out = self.model(x, self.in_chans)
        train_loss = self.criterion(out, y.long())
        train_acc = self.accuracy(out, y.long())

        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_acc", train_acc, prog_bar=True, on_epoch=True, on_step=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        print(x.shape)
        out = self.model(x, self.in_chans)
        val_loss = self.criterion(out, y.long())
        val_acc = self.accuracy(out, y.long())

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=True)
        return val_loss

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
