import argparse
import os

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dataset import SEEDIVDataset2, SEEDIV_trainSetLoader
from litModel.litBIOT import LitModel_supervised_pretrain

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def main(args):
    train_data_path = "/root/autodl-tmp/SEEDIV/SEED-IV_train_data"
    train_label_path = "/root/autodl-tmp/SEEDIV/SEED-IV_train_labels"
    # train_data_path = "../file_mmap.npz"
    # train_label_path = "../target_file_mmap.npz"

    # train_set = SEEDIVDataset2(train_data_path, train_label_path)
    # train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True, drop_last=True)

    dataModule = SEEDIV_trainSetLoader(train_data_path, train_label_path, 64, 0.1)
    # define the trainer
    log_dir = "log-pretrain"
    os.makedirs(log_dir, exist_ok=True)
    N_version = (
            len(os.listdir(os.path.join(log_dir))) + 1
    )
    # define the model
    save_path = f"{log_dir}/{N_version}-unsupervised/checkpoints"
    model = LitModel_supervised_pretrain(args, save_path)

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
        max_epochs=args.epochs,
    )
    # train the model
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