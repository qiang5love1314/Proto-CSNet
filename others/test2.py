#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 使用cvt模型进行预测

import multiprocessing as mp
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import time

from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split

import seaborn as sns
import datetime

import cls_cvt


# 读取mat数据
def read_mat(csi_directory_path, csi_action):
    datax = []
    datay = []

    csi_mats = os.listdir(csi_directory_path)
    for csi_mat in csi_mats:
        mat = loadmat(csi_directory_path + csi_mat)
        data = mat['csi']
        # datax为CSI数据元组，datay为对应的动作标签
        datax.extend([data])
        datay.extend([csi_action])
    return np.array(datax), np.array(datay)


# 读取所有csi
def read_csi(base_directory):
    datax = None
    datay = None
    '''
        # 创建进程池，创建多个进程去完成任务
        pool = mp.Pool(mp.cpu_count())
        # 遍历文件夹，将每个文件夹的路径和文件夹名传入read_mat函数
        results = [pool.apply(read_mat,args=(
                                  base_directory + '/' + directory + '/', directory,
                                  )) for directory in os.listdir(base_directory)]
        # 关闭进程池
        pool.close()
    '''
    # 优化
    pool = mp.Pool(mp.cpu_count())
    # 构建参数列表
    tasks = [(base_directory + '/' + directory + '/', directory)
             for directory in os.listdir(base_directory)]

    # 使用 pool.starmap 一次性启动所有任务
    results = pool.starmap(read_mat, tasks)
    # 关闭进程池并等待所有进程完成
    pool.close()
    pool.join()

    for result in results:
        if datax is None:
            # result[0]为CSI数据元组，result[1]为对应的动作标签
            datax = result[0]
            datay = result[1]
        else:
            # np.vstack将数组垂直堆叠在一起，np.concatenate将两个数组水平拼接在一起
            datax = np.vstack([datax, result[0]])
            datay = np.concatenate([datay, result[1]])
    return datax, datay


# 随机抽样
def extract_sample(n_way, n_support, n_query, datax, datay, test=False):
    sample = []
    if test:
        K = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
    else:
        K = np.random.choice(np.unique(datay), n_way, replace=False)
    for cls in K:
        datax_cls = datax[datay == cls]
        # Reshape each instance: from [1, 3, 30, 200] to [1, 90, 200]
        datax_cls = datax_cls.reshape(datax_cls.shape[0], 1, -1, 200)
        perm = np.random.permutation(datax_cls)
        sample_cls = perm[:(n_support + n_query)]
        sample.append(sample_cls)
    sample = np.array(sample)
    sample = torch.from_numpy(sample).float()
    return ({
        'csi_mats': sample,
        'n_way': n_way,
        'n_support': n_support,
        'n_query': n_query
    })


# 展平层，多维张量展成一维
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# 线性嵌入
class LinearEmbed(nn.Module):

    def __init__(self, dim_in=256 * 256 * 64, dim_out=16 * 64):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


def load_protonet_conv():
    model = cls_cvt.vision_transformer_model(num_classes=11)
    model.to(torch.device('cuda'))
    # 加载预训练模型
    # net = build_model(avg_pool=True, num_classes=11)
    # # 加载模型的权重
    # #     net.load_state_dict(torch.load(model_out_name))
    # net.load_state_dict(torch.load(model_out_name), strict=False)

    return ProtoNet(model)


# 计算欧式距离(或计算余弦相似度)
def euclidean_dist(x, y):
    # 获取张量x第一个维度的大小行数
    n = x.size(0)
    # 获取张量y第一个维度的大小行数
    m = y.size(0)
    # 获取张量x第二个维度的大小列数
    d = x.size(1)
    # 确保x和y的维度一致
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    # 计算两个张量x和y之间逐元素平方差的和，但只在最后一个维度上进行求和
    return torch.pow(x - y, 2).sum(2)


# 原型网络
class ProtoNet(nn.Module):
    def __init__(self, encoder):

        super(ProtoNet, self).__init__()
        if torch.cuda.is_available():
            self.encoder = encoder.cuda(0)
        else:
            self.encoder = encoder

    # 计算分类任务的损失、准确率和输出
    def set_forward_loss(self, sample):

        if torch.cuda.is_available():
            sample_images = sample['csi_mats'].cuda(0)
        else:
            sample_images = sample['csi_mats']

        n_way = sample['n_way']
        n_support = sample['n_support']
        n_query = sample['n_query']
        # 将 CSI 数据帧拆分为支持集和查询集
        x_support = sample_images[:, :n_support]
        x_query = sample_images[:, n_support:]
        # print("x_support:",x_support.size())
        # print("x_query:",x_query.size())

        # 目标索引 0——n_way-1
        target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if torch.cuda.is_available():
            target_inds = target_inds.cuda(0)
        else:
            target_inds = target_inds

        # 将支持集和查询集的CSI数据帧拼接成一个张量
        x = torch.cat([x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                       x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])], 0)
        # 对支持和查询集的CSI数据帧进行编码
        z = self.encoder.forward(x)
        z_dim = z.size(-1)  # usually 64
        # 获取每个类别的原型特征
        z_proto = z[:n_way * n_support].view(n_way, n_support, z_dim).mean(1)
        # 计算查询集的原型特征
        z_query = z[n_way * n_support:]

        # 计算query集和原型间的欧氏距离
        dists = euclidean_dist(z_query, z_proto)

        # 将距离通过softmax函数转换为概率分布
        log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
        # 根据目标索引和概率分布计算损失
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        # 通过调用max(2)方法，可以在第2个维度（类别维度）上找到每个查询样本的最大概率值和对应的类别索引。
        # 使用下划线 _ 来接收最大概率值，而将类别索引赋值给变量 y_hat 。
        _, y_hat = log_p_y.max(2)
        # 计算准确率，即预测类别与目标类别的匹配情况
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'y_hat': y_hat
            # ,'target':target
        }


# 训练
def train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size):
    # 使用StepLR学习率调度器，自动调整学习率，gamma=0.5 ：学习率的缩放因子
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    # 记录训练的epoch
    epoch = 0
    # 判断何时停止训练
    stop = False

    while epoch < max_epoch and not stop:

        running_loss = 0.0
        running_acc = 0.0

        # tnrange: 进度条库，用于在 Python 循环中显示进度信息
        for episode in tqdm(range(epoch_size), desc="Epoch {:d} train".format(epoch + 1)):
            # print(epoch_size)
            time.sleep(0.01)
            # 从训练集中提取样本
            sample = extract_sample(n_way, n_support, n_query, train_x, train_y, test=False)
            # 将优化器中所有参数的梯度置零
            optimizer.zero_grad()
            loss, output = model.set_forward_loss(sample)
            running_loss += output['loss']
            running_acc += output['acc']

            # 反向传播
            loss.backward()
            optimizer.step()
        # 计算epoch的平均损失和准确率
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size
        print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, epoch_loss, epoch_acc))

        epoch += 1
        # 调整学习率
        scheduler.step()


# 测试
def test(model, test_x, test_y, n_way, n_support, n_query, test_episode):
    # 记录分类结果的混淆矩阵
    conf_mat = torch.zeros(n_way, n_way)
    running_loss = 0.0
    running_acc = 0.0
    for episode in tqdm(range(test_episode)):
        time.sleep(0.01)
        sample = extract_sample(n_way, n_support, n_query, test_x, test_y, test=True)
        loss, output = model.set_forward_loss(sample)
        a = output['y_hat'].cpu().int()
        for cls in range(n_way):
            conf_mat[cls, :] = conf_mat[cls, :] + torch.bincount(a[cls, :], minlength=n_way)

        running_loss += output['loss']
        running_acc += output['acc']

    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode

    print('Test results -- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))

    # 检查是否存在/output文件夹，如果不存在则创建它
    output_folder = 'output/protonet_cnn/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 归一化混淆矩阵
    normalized_conf_mat = conf_mat / (test_episode * n_query)

    # 使用Seaborn绘制混淆矩阵的热力图
    #     class_labels = [f'Class {i}' for i in range(n_way)]
    class_labels = ['lying', 'sitting', 'running', 'walking', 'pick up', 'wave hand', 'jump', 'squat', 'sit down',
                    'stand up', 'fall down']
    df_conf_mat = pd.DataFrame(normalized_conf_mat.numpy(), columns=class_labels, index=class_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_conf_mat, annot=True, cmap='Blues', fmt=".2f")
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # 获取当前时间
    current_time = datetime.datetime.now()
    # 将当前时间转换为字符串形式，以便用于文件名
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    # 使用当前时间作为文件名
    output_filename = f"output_protonet_cnn_{formatted_time}.png"
    plt.savefig(os.path.join(output_folder, output_filename))  # 保存热力图为图像文件

    # 返回归一化的混淆矩阵和平均准确率
    return normalized_conf_mat, avg_acc


if __name__ == '__main__':
    path = r'./Processed Data/bedroom/'
    # path = r'./Processed Data/meetingroom/'
    trainx, trainy = read_csi(path)
    trainx = np.expand_dims(trainx, axis=1)
    train_x, test_x, train_y, test_y = train_test_split(trainx, trainy, test_size=0.2, random_state=40,
                                                        shuffle=True)  # test_size小于0.2报错？

    model = load_protonet_conv()

    # 设置初始学习率为0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    n_way = 11
    n_support = 4
    n_query = 3

    # train_x = trainx
    # train_y = trainy

    max_epoch = 5
    epoch_size = 100
    # 训练
    train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size)
    model_out_name = 'models/model_cvt_test' + '.pt'
    directory = os.path.dirname(model_out_name)
    # 如果不存在就创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), model_out_name)

    # model.load_state_dict(torch.load('./models/model_1_test.pt'))
    # path2 = r'./Processed Data/meetingroom/'
    # testx, testy = read_csi(path2)
    # testx = np.expand_dims(testx, axis=1)

    # n_way = 11
    # n_support = 5
    # n_query = 5

    # test_x = testx
    # test_y = testy

    test_episode = 100
    # 测试
    CF, acc = test(model, test_x, test_y, n_way, n_support, n_query, test_episode)
    result3 = {'CF': CF, 'acc': acc}
    print(result3, '\n')