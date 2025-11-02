import matplotlib.pyplot as plt
import numpy as np
import os
import re
import torch
import scipy.io as sio
from torch.utils.data import TensorDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def extract_number(filename):
    # 提取文件名中的数字
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

# 递归地获取目录中的所有文件
# derectory: 目录名
# 返回文件名
def get_sorted_files(directory):
    try:
        # 获取目录中的所有文件和文件夹
        files_and_dirs = os.listdir(directory)

        # 过滤出文件
        files = [f for f in files_and_dirs if os.path.isfile(os.path.join(directory, f))]

        # 按文件名排序
        sorted_files = sorted(files, key=extract_number)

        return sorted_files
    except Exception as e:
        return str(e)

# 处理单个文件 返回训练集，测试集
def process_file(file_path, label):
    # seediv数据是mat的格式
    data = sio.loadmat(file_path)
    print("process " + file_path)
    frequency = 200
    trils = [str(i) for i in range(1, 25)]
    flag = 0
    train_set = []
    train_label = []
    test_set = []
    test_label = []
    label0 = 0
    label1 = 0
    label2 = 0
    label3 = 0
    search_str = "eeg"
    filtered_keys = sorted([k for k in data.keys() if search_str in k],  key=extract_number)
    for i in trils:
        data_temp = data[filtered_keys[flag]].transpose(1, 0)
        time = len(data_temp[:, 0]) // frequency
        data_time = []
        for j in range(time):
            data_time.append(data_temp[j*frequency:(j+1)*frequency, :])
        data_time = np.concatenate(data_time)
        data_time = data_time.reshape(-1, frequency, 62)
        label_temp = np.full(time, label[flag])
        if label[flag] == 0:
            label0 += 1
            if label0 <= 4:
                train_set.append(data_time)
                train_label.append(label_temp)
            else:
                test_set.append(data_time)
                test_label.append(label_temp)
        elif label[flag] == 1:
            label1 += 1
            if label1 <= 4:
                train_set.append(data_time)
                train_label.append(label_temp)
            else:
                test_set.append(data_time)
                test_label.append(label_temp)
        elif label[flag] == 2:
            label2 += 1
            if label2 <= 4:
                train_set.append(data_time)
                train_label.append(label_temp)
            else:
                test_set.append(data_time)
                test_label.append(label_temp)
        elif label[flag] == 3:
            label3 += 1
            if label3 <= 4:
                train_set.append(data_time)
                train_label.append(label_temp)
            else:
                test_set.append(data_time)
                test_label.append(label_temp)
        flag += 1
    train_set = np.concatenate(train_set)
    train_label = np.concatenate(train_label)
    test_set = np.concatenate(test_set)
    test_label = np.concatenate(test_label)

    return train_set, train_label, test_set, test_label


# 处理目录里的所有文件
# directory 目录名
def process_files_in_directory(directory, label):
    # 先获取目录中的所有文件
    sorted_files = get_sorted_files(directory)
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for filename in sorted_files:
        file_path = os.path.join(directory, filename)
        # 处理单个文件
        train_set, train_label, test_set, test_label = process_file(file_path, label)
        train_data.append(train_set)
        train_labels.append(train_label)
        test_data.append(test_set)
        test_labels.append(test_label)
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    test_data = np.concatenate(test_data)
    test_labels = np.concatenate(test_labels)
    return train_data, train_labels, test_data, test_labels

def load_SEEDIV():
    # 目录路径
    session1= "/home/gxx/Documents/pythonProjects/datasets/SEED_IV/eeg_raw_data/1"
    # session2= "/home/gxx/Documents/pythonProjects/datasets/SEED_IV/eeg_raw_data/2"
    # session3= "/home/gxx/Documents/pythonProjects/datasets/SEED_IV/eeg_raw_data/3"

    label = np.array([[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                          # [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                          # [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
                      ])

    train_data1, train_labels1, test_data1, test_labels1 = process_files_in_directory(session1, label[0])
    # train_data2, train_labels2, test_data2, test_labels2 = process_files_in_directory(session2, label[1])
    # train_data3, train_labels3, test_data3, test_labels3 = process_files_in_directory(session3, label[2])
    train_data_final = train_data1
    train_label_final = train_labels1
    test_data_final = test_data1
    test_label_final = test_labels1
    return train_data_final, train_label_final, test_data_final, test_label_final


if __name__ == '__main__':
    result_dir = "/home/gxx/Documents/pythonProjects/datasets/dataset_SEED-IV/"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    train_data_final, train_label_final, test_data_final, test_label_final = load_SEEDIV()
    print("final shape:", train_data_final.shape)
    sio.savemat(result_dir + "SEED-IV_train_data",
                {"train_data": train_data_final})
    sio.savemat(result_dir + "SEED-IV_train_labels",
                {"train_labels": train_label_final})
    sio.savemat(result_dir + "SEED-IV_test_data",
                {"test_data": test_data_final})
    sio.savemat(result_dir + "SEED-IV_test_labels",
                {"test_labels": test_label_final})
