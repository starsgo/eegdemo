import argparse
import os


from dataset import SEEDIV_trainSetLoader
from litModel import LitModel_neuralTransformer

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


def main(args):
    train_data_path = "/home/gxx/Documents/pythonProjects/datasets/dataset_SEED-IV/testSEED-IV_train_data"
    train_label_path = "/home/gxx/Documents/pythonProjects/datasets/dataset_SEED-IV/testSEED-IV_train_labels"

    dataModule = SEEDIV_trainSetLoader(train_data_path, train_label_path, 64, 0.1)
    # define the trainer
    log_dir = "log-neuralTransformer"
    os.makedirs(log_dir, exist_ok=True)
    N_version = (
            len(os.listdir(os.path.join(log_dir))) + 1
    )
    # define the model
    save_path = f"{log_dir}/{N_version}-unsupervised/checkpoints"
    model = LitModel_neuralTransformer(args, save_path)

    logger = TensorBoardLogger(
        save_dir=log_dir + "/biotLog",
        version=f"{N_version}/checkpoints",
        name=log_dir,
    )
    trainer = pl.Trainer(
        #strategy=DDPStrategy(find_unused_parameters=False),
        strategy="ddp_notebook",
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
    parser.add_argument("--patch_size", type=int, default=200, help="patch size")
    parser.add_argument("--eeg_size", type=int, default=1600, help="eeg size")
    parser.add_argument("--num_workers", type=int, default=32, help="number of workers")
    args = parser.parse_args(["--batch_size", "64", "--lr", "2e-3"])
    main(args)