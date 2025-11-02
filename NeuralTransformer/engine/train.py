from sklearn.model_selection import StratifiedKFold
import copy
from sched import scheduler

import numpy as np
import torch
from matplotlib import pyplot as plt
from sympy import transpose
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset






def train_one_epoch(device, net, in_chans, train_loader, val_loader ,criterion, optimizer, scheduler):
    init_params = {}
    for name, param in net.named_parameters():
        # 选几个你关心的层（比如分类头、TemporalConv、Transformer块）
        if "head" in name or "conv" in name or "attn" in name:
            init_params[name] = param.data.clone()  # 克隆保存初始值

    # ---------- 训练 ----------
    net.train()
    train_acc = train_loss = 0.0
    batches = 0
    for i, (x, y) in enumerate(train_loader, 1):

        x, y = x.to(device), y.to(device)
        out = net(x, in_chans)
        loss = criterion(out, y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc += (out.argmax(1) == y).float().mean().item()
        train_loss += loss.item()
        batches += 1
        if i % 10 == 0 or i == len(train_loader):
            print(f"batch {i}/{len(train_loader)} "
                  f"train_loss={train_loss/batches} "
                  f"train_acc={train_acc/batches}")

    # ---------- 验证 ----------
    net.eval()
    val_acc = val_loss = 0.0
    batches = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader, 1):
            x, y = x.to(device), y.to(device)
            out = net(x, in_chans)
            loss = criterion(out, y.long())
            val_acc += (out.argmax(1) == y).float().mean().item()
            val_loss += loss.item()
            batches += 1
            if i % 10 == 0 or i == len(val_loader):
                print(f"batch {i}/{len(val_loader)}"
                      f"val_loss={val_loss / batches}"
                      f"val={val_acc / batches}")

    scheduler.step(val_acc)
    return val_acc