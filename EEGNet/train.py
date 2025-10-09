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

full_set = EEGNetDataset('./bciciv_2a01T.mat', './bciciv_2a01T.mat')

idx_train_val, idx_test = train_test_split(range(len(full_set)), test_size=0.2, random_state=42,stratify=full_set.target)          # 保持类别比例
train_val_set = torch.utils.data.Subset(full_set, idx_train_val)
test_set  = torch.utils.data.Subset(full_set, idx_test)

train_val_labels = [full_set.target[i] for i in train_val_set.indices]
idx_train, idx_val = train_test_split(range(len(train_val_set)), test_size=0.2, random_state=42,stratify=train_val_labels)          # 保持类别比例
train_set = torch.utils.data.Subset(train_val_set, idx_train)
val_set  = torch.utils.data.Subset(train_val_set, idx_val)

train_loader = DataLoader(train_set, batch_size=Config.train_batch_size,
                          shuffle=True, drop_last=True)
test_loader  = DataLoader(test_set,  batch_size=Config.test_batch_size,
                          shuffle=False)
val_loader  = DataLoader(val_set,  batch_size=Config.test_batch_size,
                          shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = EEGNet(ch_nums=22, T=1250, class_dim=4).to(device)
criterion = torch.nn.CrossEntropyLoss()
# criterion = nn.MultiMarginLoss()
# optimizer = optim.SGD(net.parameters(),lr=0.8)
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

counter = []
loss_history = []
accuracy_history = []

val_loss_history= []
val_accuracy_history =[]

iteration_number = 0
total = 0
correct = 0
classnum = 4

patience_counter = 0
best_val_loss = np.inf

for epoch in range(0, Config.train_number_epochs):
    net.train()
    train_loss, train_correct = 0.0, 0
    total = 0
    for i, data in enumerate(train_loader, 0):  # enumerate防止重复抽取到相同数据，数据取完就可以结束一个epoch
        item, target = data
        item, target = item.to(device), target.to(device)

        optimizer.zero_grad()  # grad归零
        output = net(item)  # 输出
        loss = criterion(output, target.long())  # 算loss,target原先为Tensor类型，指定target为long类型即可。
        loss.backward()  # 反向传播算当前grad
        optimizer.step()  # optimizer更新参数
        # 求ACC标准流程
        predicted = torch.argmax(output, 1)
        train_correct += (predicted == target).sum().item()
        train_loss += loss.item() * target.size(0)
        total += target.size(0)  # total += target.size

    train_accuracy = train_correct / total
    train_accuracy = np.array(train_accuracy)
    train_loss = train_loss / total
    train_loss = np.array(train_loss)
    print("Epoch number {}\n Current Accuracy {}\n Current loss {}\n".format(epoch, train_accuracy.item(), train_loss.item()))
    iteration_number += 1
    counter.append(iteration_number)
    accuracy_history.append(train_accuracy.item())
    loss_history.append(train_loss.item())

    net.eval()
    val_loss, val_correct = 0, 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            item, target = data
            item, target = item.to(device), target.to(device)
            output = net(item)
            loss = criterion(output, target.long())
            val_pred = torch.argmax(output, 1)
            val_correct += (val_pred == target).sum().item()
            val_loss += loss.item() *  target.size(0)
            total += target.size(0)
    val_accuracy = val_correct / total
    val_accuracy = np.array(val_accuracy)
    val_loss = val_loss / total
    val_loss = np.array(val_loss)
    # print("Epoch number {}\n Current Accuracy {}\n Current loss {}\n".format(epoch, val_accuracy.item(),
    #                                                                          val_loss.item()))
    val_accuracy_history.append(val_accuracy.item())
    val_loss_history.append(val_loss.item())
    # ---------- 调度器更新 ----------
    scheduler.step(val_loss)  # ① 学习率
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(net.state_dict())
        patience_counter = 0
        torch.save(best_model_wts, "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= 5:
            print("early stop")
            break



torch.save(net.state_dict(), 'eegnet_01at_22_1250.pth')

plt.plot(counter, accuracy_history, loss_history)
plt.plot(counter, val_accuracy_history, val_loss_history)
plt.show()


net.eval()  # 关键：推理模式
# 3. 测试循环
all_pred, all_true = [], []
correct = total = 0
for i, data in enumerate(test_loader, 0):  # enumerate防止重复抽取到相同数据，数据取完就可以结束一个epoch
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

# 4. 混淆矩阵 & 分类报告
print(classification_report(all_true, all_pred, digits=4))

cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()