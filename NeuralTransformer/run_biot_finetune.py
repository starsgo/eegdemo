import argparse
import os

import torch
from pytorch_lightning.callbacks import BackboneFinetuning
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dataset import SEEDIVDataset2, SEEDIV_trainSetLoader2
from litModel.litBIOT import LitModel_supervised_pretrain
from fineTune import BiotTuner

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

def main(args):
    train_data_path = "/home/gxx/Documents/pythonProjects/datasets/dataset_SEED-IV/SEED-IV_test_data"
    train_label_path = "/home/gxx/Documents/pythonProjects/datasets/dataset_SEED-IV/SEED-IV_test_labels"


    dataModule = SEEDIV_trainSetLoader2(train_data_path, train_label_path, 32, 0.1)
    # define the trainer
    log_dir = "log-finetune"
    os.makedirs(log_dir, exist_ok=True)
    N_version = (
            len(os.listdir(os.path.join(log_dir))) + 1
    )
    # define the model
    save_path = f"{log_dir}/{N_version}-unsupervised/checkpoints"
    backbone = LitModel_supervised_pretrain.load_from_checkpoint(
        "log-pretrain/14-unsupervised/14-unsupervised/checkpoints/epoch=37_step=18000.ckpt",
        args=args, save_path="log-finetune").model
    model = BiotTuner(backbone, num_classes=4)

    logger = TensorBoardLogger(
        save_dir="./biotLog",
        version=f"{N_version}/checkpoints",
        name=log_dir,
    )
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        # strategy="ddp_notebook",
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=10,
    )
    trainer.fit(model, dataModule)
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        # strategy="ddp_notebook",
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=10,
        callbacks=[BackboneFinetuning(unfreeze_at_epoch=5)])
    trainer.fit(model, dataModule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_workers", type=int, default=32, help="number of workers")
    args = parser.parse_args(["--batch_size", "64", "--lr", "2e-3"])
    main(args)
