import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from EEGNet import EEGNet, Config
from dataset import EEGNetDataset


# from train import net

# 1. 测试集：这里用后 60% 当测试（与训练无交集）
full_set = EEGNetDataset(file_path='./bciciv_2a_1E.mat', target_path='./bciciv_2a_1E.mat')

test_loader = DataLoader(full_set, batch_size=Config.test_batch_size,
                         shuffle=False, num_workers=0)

# 2. 加载模型权重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = EEGNet(ch_nums=22, T=1248, class_dim=4).to(device)
net.load_state_dict(torch.load('eegnet_01at_22_1250.pth', map_location=device))
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