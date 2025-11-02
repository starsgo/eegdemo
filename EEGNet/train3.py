from sklearn.model_selection import StratifiedKFold
import copy
from sched import scheduler

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
from EEGNet import EEGNet, Config
from dataset import EEGNetDataset
import seaborn as sns


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def reset_model():   # 每折重新初始化权重
    net = EEGNet(ch_nums=22, T=1248, class_dim=4).to(device)
    return net

full_set = EEGNetDataset('./bciciv_all5_exclude9.mat', './bciciv_all5_exclude9.mat')
kfold   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_idx   = np.arange(len(full_set))
y_all   = full_set.target

val_acc_list = []
test_acc_list = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_idx, y_all)):

    counter = []
    loss_history = []
    accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []
    patience = 0
    iteration_number = 0

    print(f'\n===== Fold {fold + 1}/5 =====')
    train_set = Subset(full_set, train_idx)
    val_set   = Subset(full_set, val_idx)

    train_loader = DataLoader(train_set, batch_size=Config.train_batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=Config.test_batch_size,
                              shuffle=False)

    net = reset_model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     patience=5, factor=0.5)
    best_val_acc = 0.0
    for epoch in range(Config.train_number_epochs):
        # ---------- 训练 ----------
        net.train()
        correct = total = 0
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y.long())
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == y).sum().item()
            train_loss += loss.item() * x.size(0)
            total += y.size(0)
        train_acc = correct / total
        train_loss /= total
        iteration_number += 1
        counter.append(iteration_number)
        accuracy_history.append(train_acc)
        loss_history.append(train_loss)

        # ---------- 验证 ----------
        net.eval()
        correct = total = 0
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = net(x)
                correct += (out.argmax(1) == y).sum().item()
                loss = criterion(out, y.long())
                val_loss += loss.item() * x.size(0)
                total += y.size(0)
        val_acc = correct / total
        val_loss /= total
        val_accuracy_history.append(val_acc)
        val_loss_history.append(val_loss)

        scheduler.step(val_acc)
        print(f'Fold {fold+1} Epoch {epoch:02d} | TrainAcc {train_acc:.4f} '
              f'| ValAcc {val_acc:.4f} | BestVal {best_val_acc:.4f}')

        # ---------- 早停 ----------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(net.state_dict())
            torch.save(best_model_wts, f"best_model_exclude9_fold{fold} .pt")
            patience = 0
        else:
            patience += 1
            if patience >= 10:  # 连续 10  epoch 无提升
                print(f'Fold {fold + 1} early stop at epoch {epoch}')
                break

    val_acc_list.append(best_val_acc)

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(counter, accuracy_history, label='Train Accuracy')
    plt.plot(counter, val_accuracy_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(counter, loss_history, label='Train Loss')
    plt.plot(counter, val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# -------------- 结果汇总 --------------
torch.save(net.state_dict(), 'eegnet_.pth')
print('\n===== 5-Fold CV Summary =====')
print('Val-Acc each fold:', val_acc_list)
print(f'Mean ± Std: {np.mean(val_acc_list):.4f} ± {np.std(val_acc_list):.4f}')