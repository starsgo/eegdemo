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

from sklearn.model_selection import train_test_split

full_set = EEGNetDataset('./bciciv_2a01T.mat', './bciciv_2a_1T.mat')

# idx_train_val, idx_test = train_test_split(range(len(full_set)), test_size=0.2, random_state=42,
#                                            stratify=full_set.target)
# train_val_set = torch.utils.data.Subset(full_set, idx_train_val)
# test_set = torch.utils.data.Subset(full_set, idx_test)
#
# train_val_labels = [full_set.target[i] for i in train_val_set.indices]
# idx_train, idx_val = train_test_split(range(len(train_val_set)), test_size=0.2, random_state=42,
#                                       stratify=train_val_labels)
# train_set = torch.utils.data.Subset(train_val_set, idx_train)
# val_set = torch.utils.data.Subset(train_val_set, idx_val)

# 1. 先拿全局下标
idx_train_val, idx_test = train_test_split(
    range(len(full_set)), test_size=0.2, random_state=42, stratify=full_set.target)

idx_train, idx_val = train_test_split(
    idx_train_val, test_size=0.2, random_state=42,
    stratify=[full_set.target[i] for i in idx_train_val])

# 2. 只建一次 Subset，索引直接对准 full_set
train_set = torch.utils.data.Subset(full_set, idx_train)
val_set   = torch.utils.data.Subset(full_set, idx_val)
test_set  = torch.utils.data.Subset(full_set, idx_test)

train_loader = DataLoader(train_set, batch_size=Config.train_batch_size,
                          shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=Config.test_batch_size,
                         shuffle=False)
val_loader = DataLoader(val_set, batch_size=Config.test_batch_size,
                        shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = EEGNet(ch_nums=22, T=1248, class_dim=4).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

counter = []
loss_history = []
accuracy_history = []
val_loss_history = []
val_accuracy_history = []

iteration_number = 0
patience_counter = 0
best_val_loss = np.inf
best_model_wts = copy.deepcopy(net.state_dict())

for epoch in range(0, Config.train_number_epochs):
    net.train()
    train_loss, train_correct = 0.0, 0
    total = 0

    for i, data in enumerate(train_loader, 0):
        item, target = data
        item, target = item.to(device), target.to(device)

        optimizer.zero_grad()
        output = net(item)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()

        predicted = torch.argmax(output, 1)
        train_correct += (predicted == target).sum().item()
        train_loss += loss.item() * item.size(0)
        total += target.size(0)

    train_accuracy = train_correct / total
    train_loss = train_loss / total
    print(f"Epoch {epoch}: Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}")

    iteration_number += 1
    counter.append(iteration_number)
    accuracy_history.append(train_accuracy)
    loss_history.append(train_loss)

    # 验证阶段
    net.eval()
    val_loss, val_correct = 0.0, 0
    val_total = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            item, target = data
            item, target = item.to(device), target.to(device)
            output = net(item)
            loss = criterion(output, target.long())

            val_pred = torch.argmax(output, 1)
            val_correct += (val_pred == target).sum().item()
            val_loss += loss.item() * item.size(0)
            val_total += target.size(0)

    val_accuracy = val_correct / val_total
    val_loss = val_loss / val_total
    val_accuracy_history.append(val_accuracy)
    val_loss_history.append(val_loss)

    print(f"Epoch {epoch}: Val Accuracy: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")

    # 学习率调度
    scheduler.step(val_loss)

    # 早停机制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(net.state_dict())
        patience_counter = 0
        # torch.save(best_model_wts, "best_model_exclude8.pt")
    else:
        patience_counter += 1
        if patience_counter >= 10:
            print("Early stopping triggered")
            break

# 加载最佳模型
# net.load_state_dict(best_model_wts)
# torch.save(net.state_dict(), 'eegnet_01at_22_1250.pth')

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

# 测试阶段
net.eval()
all_pred, all_true = [], []
correct = total = 0
for i, data in enumerate(test_loader, 0):
    with torch.no_grad():
        x, y = data
        x, y = x.to(device), y.to(device)
        out = net(x)
        pred = out.argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

        all_pred.extend(pred.cpu().numpy())
        all_true.extend(y.cpu().numpy())

acc = correct / total
print(f'Test Accuracy: {acc:.4f}')

# 分类报告和混淆矩阵
print(classification_report(all_true, all_pred, digits=4))

cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()