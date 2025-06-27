#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# siamses+resnet，效果一般

import os
import numpy as np
import pywt
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from resnet import resnet_model

# 如果有GPU则使用GPU，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CSIDataset(Dataset):
    def __init__(self, path, transform=None):
        self.csiData, self.activities, self.labelNum = self.merge_csi_DataAndLabel(path)
        self.transform = transform

        if self.csiData.size == 0:
            print("csiData is empty. Please check the data loading process.")
            exit()

    def __len__(self):
        return len(self.csiData)

    def __getitem__(self, idx):
        # 根据idx获取CSI数据
        csiData = self.csiData[idx]
        amplitudeCSI = abs(csiData)
        # fftData = np.fft.fft(csiData)
        fftData = np.fft.fft(amplitudeCSI)

        wavename = 'db5'
        ca, cd = pywt.dwt(fftData, wavename)
        dwtCSI = pywt.idwt(None, cd, wavename, 'smooth')
        csi = torch.from_numpy(dwtCSI).float().unsqueeze(0)  # 1050*3*30*200
        # csi张量不参与反向传播
        csi.requires_grad = False
        # 得到一个1050*200*90的张量
        csi = csi.view(1, 200, 90)
        # 在第0维度上拼接，得到一个1050*200*90*1的张量
        csi = torch.cat([csi], dim=0)
        label1 = self.labelNum[idx].item()
        # label1 = torch.tensor(label1).long()  # 确保label1是长整型

        same_class = np.random.randint(2)
        indices_same = np.where(self.labelNum == label1)[0]
        indices_diff = np.where(self.labelNum != label1)[0]

        # 检查是否存在同类或异类样本
        if same_class and len(indices_same) > 0:
            idx2 = np.random.choice(indices_same)
        elif not same_class and len(indices_diff) > 0:
            idx2 = np.random.choice(indices_diff)
        else:
            # 如果既没有同类也没有异类样本，则抛出异常或返回特殊值
            raise ValueError(
                f"No valid sample found for {'same' if same_class else 'different'} class comparison at index {idx}.")

        csiData2 = self.csiData[idx2]
        amplitudeCSI2 = abs(csiData2)
        # fftData2 = np.fft.fft(csiData2)
        fftData2 = np.fft.fft(amplitudeCSI2)

        wavename2 = 'db5'
        ca2, cd2 = pywt.dwt(fftData2, wavename2)
        dwtCSI2 = pywt.idwt(None, cd2, wavename2, 'smooth')
        csi2 = torch.from_numpy(dwtCSI2).float().unsqueeze(0)  # 1050*3*30*200
        # csi张量不参与反向传播
        csi2.requires_grad = False
        # 得到一个1050*200*90的张量
        csi2 = csi2.view(1, 200, 90)
        # 在第0维度上拼接，得到一个1050*200*90*1的张量
        csi2 = torch.cat([csi2], dim=0)

        if self.transform:
            csi = self.transform(csi)
            csi2 = self.transform(csi2)
            # Transform csiData2 similarly if necessary

        return csi, csi2, np.array([same_class], dtype=np.float32)

    @staticmethod
    def merge_csi_DataAndLabel(path):
        listDir = []
        csiData = []
        labelList = ['lying', 'sitting', 'running', 'walking', 'pick up', 'wave hand', 'jump', 'squat', 'sit down',
                     'stand up', 'fall down']
        labelNum = []
        activity = []

        for root, dirs, files in os.walk(path, topdown=False):
            for name in dirs:
                dir_path = os.path.join(root, name)
                if name.isdigit():
                    listDir.append(int(name))
                whole_file = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith('.mat')]
                for w in whole_file:
                    data = loadmat(w)['csi']
                    csiData.append(data)
                    activity.append(name)
                    labelNum.append(listDir.index(int(name)))

        csiData = np.array(csiData, dtype=complex)
        labelNum = np.array(labelNum)
        return csiData, activity, labelNum


def create_dataloader(path, batch_size=16, num_workers=1, test_size=0.2):
    dataset = CSIDataset(path)

    # 假设CSIDataset有一个方法来获取数据的数量，例如__len__()
    dataset_length = len(dataset)

    # 计算测试集的大小
    test_size = int(dataset_length * test_size)

    # 分割数据集为训练集和测试集
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=40, shuffle=True)

    # 创建训练集和测试集的数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


# 定义CNN结构
class CNN_net(nn.Module):
    def __init__(self, num_classes):
        super(CNN_net, self).__init__()
        self.conv_block_a = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.5),  # 添加 Dropout
        )

        # Convolution Block B
        self.conv_block_b = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.5),  # 添加 Dropout
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_block_a(x)
        x = self.conv_block_b(x)
        x = self.flatten(x)
        return x


def get_resnet_feature_size(model, input_shape=(1, 1, 224, 224)):
    """
    动态计算ResNet模型输出特征向量的大小。

    参数:
    - model: ResNet模型实例。
    - input_shape: 虚拟输入的形状，格式为(batch_size, channels, height, width)。

    返回:
    - feature_size: 模型输出特征向量的大小。
    """
    # 创建一个虚拟输入张量。
    dummy_input = torch.rand(input_shape)
    # 计算特征向量大小。
    with torch.no_grad():
        _, features = model(dummy_input, is_feat=True)
    feature_size = features.size(1)
    return feature_size


class SimilarityNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimilarityNetwork, self).__init__()
        # 根据CNN的实际输出尺寸设置input_size
        self.fc1 = nn.Linear(input_size, 1024)  # input_size现在是281600
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)  # 用于分类的全连接层
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # nn.CrossEntropyLoss自带Softmax
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, num_classes):
        super(SiameseNetwork, self).__init__()
        # 实例化ResNet模型。
        self.feature_extraction = resnet_model(avg_pool=True, num_classes=num_classes)
        # 动态获取ResNet输出特征的大小。
        input_size = get_resnet_feature_size(self.feature_extraction)
        # 实例化SimilarityNetwork，根据ResNet输出调整的input_size。
        self.similarity_computation = SimilarityNetwork(input_size=input_size, num_classes=num_classes)

    def forward(self, x1, x2):
        F_x1 = self.feature_extraction(x1)
        F_x2 = self.feature_extraction(x2)
        distance = torch.abs(F_x1 - F_x2)
        similarity = self.similarity_computation(distance)
        return similarity


def train(model, train_loader, criterion, optimizer, num_epochs=20, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()

    # 引入学习率调度器（使用StepLR）
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for data in train_loader:
            inputs1, inputs2, labels = data
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            labels = labels.squeeze().long()
            optimizer.zero_grad()

            outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = correct_predictions / total_samples
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy * 100:.2f}%")

        # 更新学习率
        scheduler.step()

        # 可选：保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_out_name = 'models/siam_resnet_best_model' + '.pt'
            directory = os.path.dirname(model_out_name)
            # 如果不存在就创建
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(model.state_dict(), model_out_name)

def evaluate(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估过程中不计算梯度
        for data in test_loader:
            inputs1, inputs2, labels = data
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            labels = labels.squeeze().long()  # 确保labels是一维的长整型Tensor

            outputs = model(inputs1, inputs2)  # 获取模型输出
            _, predicted = torch.max(outputs, 1)  # 获取预测的类别索引

            total += labels.size(0)  # 累加总的样本数
            correct += (predicted == labels).sum().item()  # 累加正确预测的样本数

    accuracy = 100 * correct / total  # 计算准确率
    print(f"Accuracy: {accuracy:.4f}%")  # 打印准确率


def get_args():
    parser = argparse.ArgumentParser(description='Transformer-BLS')
    parser.add_argument('--sample', type=int, default=1, help='sample length on temporal side')
    parser.add_argument('--batch', type=int, default=16, help='batch size [default: 16]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [default: 0.001]')
    parser.add_argument('--epoch', type=int, default=10, help='number of epoch [default: 20]')
    parser.add_argument('--hlayers', type=int, default=6, help='horizontal transformer layers [default: 6]')
    parser.add_argument('--hheads', type=int, default=9, help='horizontal transformer head [default: 9]')
    parser.add_argument('--vlayers', type=int, default=1, help='vertical transformer layers [default: 1]')
    parser.add_argument('--vheads', type=int, default=200, help='vertical transformer head [default: 200]')
    parser.add_argument('--com_dim', type=int, default=50, help='compressor vertical transformer layers [default: 50]')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # 路径、数据加载及模型初始化
    # path = r'./Processed Data/bedroom/'
    path = r'./Processed Data/meetingroom/'
    labelList = ['lying', 'sitting', 'running', 'walking', 'pick up', 'wave hand', 'jump', 'squat', 'site down',
                 'stand up', 'fall down']  # 标签列表

    num_classes = len(labelList)  # 类别总数：11

    model = SiameseNetwork(num_classes).to(device='cuda')
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, test_loader = create_dataloader(path, batch_size=args.batch, num_workers=1)
    # test_loader = create_dataloader(path2, batch_size=args.batch, num_workers=1)

    #模型训练和推理测试
    train(model, train_loader, criterion, optimizer, num_epochs=args.epoch)
    model_out_name = 'models/model_siam_resnet' + '.pt'
    directory = os.path.dirname(model_out_name)
    # 如果不存在就创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), model_out_name)

    evaluate(model, test_loader)
