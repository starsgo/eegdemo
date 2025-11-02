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


# ## 帮助函数
def show_plot(iteration, accuracy, loss):
    plt.plot(iteration, accuracy, loss)
    plt.show()

def test_show_plot(iteration, accuracy):
    plt.plot(iteration, accuracy)
    plt.show()

# # ## 用于配置的帮助类
# class Config():
#     training_dir = "./data/faces/training/"
#     testing_dir = "./data/faces/testing/"
#     train_batch_size = 48  # 64
#     test_batch_size = 48
#     train_number_epochs = 100  # 100
#     test_number_epochs = 20

class EEGNetDataset(Dataset):
    def __init__(self, file_path, target_path, transform=None, target_transform=None):
        self.file_path = file_path
        self.target_path = target_path
        self.data = self.parse_data_file(file_path)
        self.target = self.parse_target_file(target_path)

        self.transform = transform
        self.target_transform = target_transform

        # mat = loadmat('bciciv.mat')
        # source = {k: v for k, v in mat.items() if not k.startswith('__')}

    def parse_data_file(self, file_path):
        mat = loadmat(file_path)
        data = mat['train_data']
        print(f"dataset train_data size {data.shape}")
        return np.array(data, dtype=np.float32)

    def parse_target_file(self, target_path):
        mat = loadmat(target_path)
        target = mat['train_labels']
        print(f"dataset train_labels size {target.shape}")
        return np.array(target, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index, :]

        # 对每个通道单独进行标准化
        for channel_idx in range(item.shape[0]):
            channel_data = item[channel_idx, :]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            if std < 1e-6:
                std = 1e-6
            item[channel_idx, :] = (channel_data - mean) / std

        item = np.expand_dims(item, axis=0)
        target = self.target[index]
        if self.transform:
            item = self.transform(item)
        if self.target_transform:
            target = self.target_transform(target)
        return item, target
#
train_data_path = "/home/gxx/Documents/pythonProjects/datasets/dataset_SEED-IV/SEED-IV_train_data"
train_labels_path = "/home/gxx/Documents/pythonProjects/datasets/dataset_SEED-IV/SEED-IV_train_labels"
dataSet = EEGNetDataset(file_path=train_data_path, target_path=train_labels_path)
# len = len(dataSet)
# print(len)
# print(dataSet.__getitem__(4))
# print(dataSet.__getitem__(5))
# print(dataSet.__getitem__(6))