import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.fx.experimental.unification.multipledispatch.dispatcher import source
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from scipy.io import savemat, loadmat


# SEEDIVDataset2只能加载类似{'train_data': tensor:(b,t,c)}的mat文件
class SEEDIVDataset2(Dataset):
    def __init__(self, file_path, target_path, transform=None, target_transform=None):
        self.file_path = file_path
        self.target_path = target_path
        self.data = self.parse_data_file(file_path)
        self.target = self.parse_target_file(target_path)

        self.transform = transform
        self.target_transform = target_transform

    def parse_data_file(self, file_path):
        mat = loadmat(file_path)
        source = {k: v for k, v in mat.items() if not k.startswith('__')}
        # source {'train_data': tensor: (b, t, c)}
        if 'train_data' in source.keys():
            data = source['train_data']
        else:
            data = source['test_data']
        return np.array(data, dtype=np.float32)

    def parse_target_file(self, target_path):
        mat = loadmat(target_path)
        source = {k: v for k, v in mat.items() if not k.startswith('__')}
        # source {'train_labels': tensor: (1, b)}
        if 'train_labels' in source.keys():
            target = source['train_labels'].squeeze()
        else:
            target = source['test_labels'].squeeze()
        #(1, b) --> (b)
        return np.array(target, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index, :]
        item = torch.from_numpy(item)  # 先转 Tensor
        #(index t n)

        # for biot
        item = torch.transpose(item, -2, -1)
        mean = item.mean(dim=-1, keepdim=True)  # (index, C, 1)
        std = item.std(dim=-1, keepdim=True)  # (index, C, 1)
        item = (item - mean) / (std + 1e-8)

        target = self.target[index]
        #(index)
        if self.transform:
            item = self.transform(item)
        if self.target_transform:
            target = self.target_transform(target)
        return item, target
#
# file_path = "/home/gxx/Documents/pythonProjects/datasets/dataset_SEED-IV/SEED-IV_test_data"
# dataSet = SEEDIVDataset(file_path='./bciciv.mat', target_path='./bciciv.mat')
