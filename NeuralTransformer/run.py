import copy

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import optim
from torch.utils.data import Subset, DataLoader

from model.biot import BIOTClassifier
from dataset.SEEDIV_dataset import SEEDIVDataset
from model.NeuralTransfomer import NeuralTransformer
from engine.train import train_one_epoch

class Config():
    train_data_path = "/root/autodl-tmp/SEEDIV/testSEED-IV_train_data"
    train_label_path = "/root/autodl-tmp/SEEDIV/testSEED-IV_train_labels"
    train_batch_size = 32 # 64
    test_batch_size = 10
    train_number_epochs = 80 # 100
    test_number_epochs = 40

    b, n, a, t = 32, 62, 8, 200
    in_chans = np.arange(1, n+1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

full_set = SEEDIVDataset(Config.train_data_path, Config.train_label_path)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_idx   = np.arange(len(full_set))
y_all   = full_set.target

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_idx, y_all)):
    print(f'\n===== Fold {fold + 1}/5 =====')
    train_set = Subset(full_set, train_idx)
    val_set = Subset(full_set, val_idx)

    train_loader = DataLoader(train_set, batch_size=Config.train_batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=Config.test_batch_size,
                            shuffle=False, drop_last=True)

    # net = NeuralTransformer(EEG_size=Config.a*Config.t, patch_size=Config.t)
    net = BIOTClassifier(n_fft=200, hop_length=200, depth=4, heads=8, n_classes=4)

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    best_val_acc = 0.0
    patience = 0
    for epoch in range(Config.train_number_epochs):
        print(f'\n--- Epoch {epoch + 1}/{Config.train_number_epochs} ---')
        val_acc = train_one_epoch(device=device, net=net, in_chans=Config.in_chans ,train_loader=train_loader, val_loader=val_loader,
                                  criterion=criterion, optimizer=optimizer, scheduler=scheduler)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(net.state_dict())
            torch.save(best_model_wts, f"checkpoints/best_model{fold}.pt")
            patience = 0
        else:
            patience += 1
            if patience >= 10:  # 连续 10  epoch 无提升
                print(f'Fold {fold + 1} early stop at epoch {epoch}')
                break

