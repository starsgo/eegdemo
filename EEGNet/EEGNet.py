
import torch
import torch.nn as nn


class Config():
    training_dir = "./data/training/"
    testing_dir = "./data/testing/"
    train_batch_size = 15 # 64
    test_batch_size = 15
    train_number_epochs = 150 # 100
    test_number_epochs = 20


class EEGNet(nn.Module):
    def __init__(self, ch_nums=16, T=250, class_dim=4):
        super().__init__()
        self.F1 = 16
        self.D  = 2
        self.kern = 125
        self.p  = 0.5
        self.F2 = self.F1 * self.D          # 32
        # 经过两层池化 (4, 8) 后时间维：T // (4*8) = T // 32
        self.fcin = self.F2 * (T // 64)

        # ---------- 1. 时间卷积 ----------
        self.conv1 = nn.Conv2d(1, self.F1, (1, self.kern), padding=(0, self.kern//2))
        self.bn1   = nn.BatchNorm2d(self.F1)

        # ---------- 2. 深度空间卷积 ----------
        self.conv2 = nn.Conv2d(self.F1, self.D*self.F1, (ch_nums, 1),
                               groups=self.F1, bias=False)
        self.bn2   = nn.BatchNorm2d(self.D*self.F1)
        self.pool2 = nn.AvgPool2d((1, 8))

        # ---------- 3. 可分离卷积 ----------
        self.conv3_1 = nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, 16),
                                 groups=self.D*self.F1, padding=(0, 8), bias=False)
        self.conv3_2 = nn.Conv2d(self.D*self.F1, self.F2, 1, bias=False)
        self.bn3     = nn.BatchNorm2d(self.F2)
        self.pool3   = nn.AvgPool2d((1, 8))

        # ---------- 4. 分类 ----------
        self.fc = nn.Linear(self.fcin, class_dim)

    def forward(self, x):
        # 1. 时间卷积
        x = self.bn1(self.conv1(x))
        # 2. 深度空间卷积
        x = self.bn2(self.conv2(x))
        x = nn.functional.elu(x)
        x = self.pool2(x)
        x = nn.functional.dropout(x, self.p, training=self.training)
        # 3. 可分离卷积
        x = self.conv3_2(self.conv3_1(x))
        x = self.bn3(x)
        x = nn.functional.elu(x)
        x = self.pool3(x)
        x = nn.functional.dropout(x, self.p, training=self.training)
        # 4. 全连接
        x = x.view(x.size(0), -1)      # flatten
        x = self.fc(x)
        return x


# ---------------- 测试 ----------------
if __name__ == '__main__':
    model = EEGNet(ch_nums=22, T=1250, class_dim=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = EEGNet(ch_nums=22, T=1250, class_dim=4).to(device)
    net.load_state_dict(torch.load('eegnet_01at_22_1250.pth', map_location=device))
    dummy = torch.randn(4, 1, 22, 1250)   # batch×1×channel×time
    out = model(dummy)
    out2 = net(dummy)
    print(out.shape)   # -> [4, 4]
    print(out)
    print(out2.shape)  # -> [4, 4]
    print(out2)