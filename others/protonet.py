#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat
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


import matplotlib.pyplot as plt
import seaborn as sns
import datetime


import numpy as np
import os
from scipy.io import loadmat
import multiprocessing as mp

def read_mat(csi_directory_path, csi_action):
    datax = []
    datay = []

    csi_mats = os.listdir(csi_directory_path)
    for csi_mat in csi_mats:
        full_path = os.path.join(csi_directory_path, csi_mat)
        mat = loadmat(full_path)
        if 'PCA' in csi_directory_path:
            data = mat['cfm_data']
        else:
            data = mat['iq_data']
        datax.append(data)  # 直接使用append避免额外的括号
        datay.append(csi_action)  # 同上

    # 使用np.array包装是为了确保数据类型的一致性和后续处理的便利
    return np.array(datax, dtype=object), np.array(datay)

def read_csi(base_directory):
    datax = None
    datay = None

    pool = mp.Pool(mp.cpu_count())
    tasks = [(os.path.join(base_directory, directory), directory)
             for directory in os.listdir(base_directory)]

    results = pool.starmap(read_mat, tasks)
    pool.close()
    pool.join()

    for dx, dy in results:
        if datax is None:
            datax = dx
            datay = dy
        else:
            datax = np.vstack((datax, dx)) if dx.size else datax
            datay = np.concatenate((datay, dy)) if dy.size else datay

    return datax, datay

# 随机抽样
def extract_sample(n_way, n_support, n_query, datax, datay, test=False):
    sample = []
    # 测试集随机选取4个类别
    if test:
        K = np.array(['empty', 'jump', 'stand', 'walk'])
    else:
        K = np.random.choice(np.unique(datay), n_way, replace=False)
    # 随机抽取n_support + n_query个数据
    for cls in K:
        datax_cls = datax[datay == cls]
        perm = np.random.permutation(datax_cls)
        sample_cls = perm[:(n_support + n_query)]
        sample.append(sample_cls)
    sample = np.array(sample)
    sample = torch.from_numpy(sample).float()
    # sample = sample.permute(0,1,4,2,3)
    # sample = np.expand_dims(sample, axis= 0)
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


# 加载ProtoNet
def load_protonet_conv(**kwargs):
    # 输入维度
    x_dim = kwargs['x_dim']
    # 隐藏层维度
    hid_dim = kwargs['hid_dim']
    # 输出维度
    z_dim = kwargs['z_dim']

    # 卷积层
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    # 构建编码器，提取已有类别特征构建编码器
    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        # conv_block(hid_dim, hid_dim),
        # conv_block(hid_dim, hid_dim),
        # conv_block(hid_dim, hid_dim),
        # conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )
    return ProtoNet(encoder)


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
        # 将编码任务移动导GPU上计算，0为编号
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

        # 计算每个类别的准确率和召回率
        precisions = []
        recalls = []
        for i in range(n_way):
            true_positive = torch.eq(y_hat, i) & torch.eq(target_inds.squeeze(), i)
            false_positive = torch.eq(y_hat, i) & ~torch.eq(target_inds.squeeze(), i)
            false_negative = ~torch.eq(y_hat, i) & torch.eq(target_inds.squeeze(), i)

            precision = true_positive.sum().float() / (true_positive.sum() + false_positive.sum() + + 1e-10).float()
            recall = true_positive.sum().float() / (true_positive.sum() + false_negative.sum() + + 1e-10).float()

            precisions.append(precision)
            recalls.append(recall)

        # 处理由于除以零而导致的NAN值
        precisions = [p if not torch.isnan(p) else 0 for p in precisions]
        recalls = [r if not torch.isnan(r) else 0 for r in recalls]

        # 所有种类的平均值
        avg_precision = torch.mean(torch.tensor(precisions))
        avg_recall = torch.mean(torch.tensor(recalls))

        # 计算F1分数
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-10)  # 避免分母为零

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'precision': avg_precision.item(),
            'recall': avg_recall.item(),
            'f1_score': f1_score.item(),
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
        running_precision = 0.0
        running_recall = 0.0
        running_f1_score = 0.0

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
            running_precision += output['precision']
            running_recall += output['recall']
            running_f1_score += output['f1_score']

            # 反向传播
            loss.backward()
            optimizer.step()
        # 计算epoch的平均损失和准确率
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size
        epoch_precision = running_precision / epoch_size
        epoch_recall = running_recall / epoch_size
        epoch_f1_score = running_f1_score / epoch_size
        print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}  Precision:{:.4f} Recall:{:.4f}  F1 score{:.4f}'.
              format(epoch + 1, epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1_score))

        epoch += 1
        # 调整学习率
        scheduler.step()


# 测试
# def test(model, test_x, test_y, n_way, n_support, n_query, test_episode):
#     # 记录分类结果的混淆矩阵
#     conf_mat = torch.zeros(n_way, n_way)
#     running_loss = 0.0
#     running_acc = 0.0
#     running_precision = 0.0
#     running_recall = 0.0
#     running_f1_score = 0.0
#     for episode in tqdm(range(test_episode)):
#         time.sleep(0.01)
#         sample = extract_sample(n_way, n_support, n_query, test_x, test_y, test=True)
#         loss, output = model.set_forward_loss(sample)
#         a = output['y_hat'].cpu().int()
#         for cls in range(n_way):
#             conf_mat[cls, :] = conf_mat[cls, :] + torch.bincount(a[cls, :], minlength=n_way)
#
#         running_loss += output['loss']
#         running_acc += output['acc']
#         running_precision += output['precision']
#         running_recall += output['recall']
#         running_f1_score += output['f1_score']
#     avg_loss = running_loss / test_episode
#     avg_acc = running_acc / test_episode
#     avg_precision = running_precision / test_episode
#     avg_recall = running_recall / test_episode
#     avg_f1_score = running_f1_score / test_episode
#     print(
#         'Test results -- Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(avg_loss, avg_acc,
#                                                                                                       avg_precision,
#                                                                                                       avg_recall,
#                                                                                                       avg_f1_score))
#
#     # 检查是否存在/output文件夹，如果不存在则创建它
#     output_folder = 'output/protonet_cnn/'
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 归一化混淆矩阵
#     normalized_conf_mat = conf_mat / (test_episode * n_query)
#
#     # 使用Seaborn绘制混淆矩阵的热力图
#     #     class_labels = [f'Class {i}' for i in range(n_way)]
#     class_labels = ['empty', 'jump', 'stand', 'walk']
#     df_conf_mat = pd.DataFrame(normalized_conf_mat.numpy(), columns=class_labels, index=class_labels)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(df_conf_mat, annot=True, cmap='Blues', fmt=".2f")
#     plt.title('Normalized Confusion Matrix')
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#
#     # 获取当前时间
#     current_time = datetime.datetime.now()
#     # 将当前时间转换为字符串形式，以便用于文件名
#     formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
#     # 使用当前时间作为文件名
#     output_filename = f"output_protonet_cnn_{formatted_time}.png"
#     plt.savefig(os.path.join(output_folder, output_filename))  # 保存热力图为图像文件
#
#     # 返回归一化的混淆矩阵和平均准确率
#     return normalized_conf_mat, avg_acc


if __name__ == '__main__':
    data_folder = 'm1c1_PCA_test_80'
    train_env = 'A1'
    train_folder_name = 'few_shot_datasets/' + data_folder + '/test_A2'

    model_out_name = 'models/model_' + data_folder + '.pt'
    path = 'D:/pycharm/files/OurActivityDataset/OurActivityDataset/Processed Data/bedroom/'

    trainx, trainy = read_csi(path)
    print(trainx.shape)
    trainx = np.expand_dims(trainx, axis=1)
    # train_x, test_x, train_y, test_y = train_test_split(trainx, trainy, test_size = 0.2, random_state=40, shuffle=True)

    # model = load_protonet_conv(
    #     x_dim=(1, 512, 256),
    #     hid_dim=64,
    #     z_dim=64,
    # )
    # # 设置初始学习率为0.001
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    #
    # n_way = 4
    # n_support = 5
    # n_query = 2
    #
    # train_x = trainx
    # train_y = trainy
    #
    # max_epoch = 3
    # epoch_size = 20
    # # 训练
    # # train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size)
    #
    # #     test_env2 = 'A2'
    # #     test_folder_name = 'few_shot_datasets/' + data_folder + '/test_' + test_env2
    # #     testx, testy = read_csi(test_folder_name)
    # #     testx = np.expand_dims(testx, axis=1)
    #
    # #     result_out_name2 = 'results/result_A1_A2' + test_env2 + '_' + data_folder + '.pt'
    #
    # #     n_way = 4
    # #     n_support = 3
    # #     n_query = 2
    #
    # #     test_x = testx
    # #     test_y = testy
    #
    # #     test_episode = 100
    # #     print(data_folder + ': ' 'trained on ' + train_env + ', testing on ' + test_env2)
    # #     CF, acc = test(model, test_x, test_y, n_way, n_support, n_query, test_episode)
    # #     result2 = {'CF': CF, 'acc': acc}
    # #     print(result2,'\n')
    #
    # test_env3 = 'A3'
    # test_folder_name = 'few_shot_datasets/' + data_folder + '/test_' + test_env3
    # testx, testy = read_csi(test_folder_name)
    # testx = np.expand_dims(testx, axis=1)
    #
    # result_out_name3 = 'results/result_A1_A3' + test_env3 + '_' + data_folder + '.pt'
    #
    # n_way = 4
    # n_support = 5
    # n_query = 3
    #
    # test_x = testx
    # test_y = testy
    #
    # test_episode = 100
    # print(data_folder + ': ' 'trained on ' + train_env + ', testing on ' + test_env3)
    # # 测试
    # CF, acc = test(model, test_x, test_y, n_way, n_support, n_query, test_episode)
    # result3 = {'CF': CF, 'acc': acc}
    # print(result3, '\n')

    #     # 存储文件的目录
    #     directory1 = os.path.dirname(result_out_name)
    #     # 如果不存在就创建
    #     if not os.path.exists(directory1):
    #         os.makedirs(directory1)
    #     torch.save(result, result_out_name)

    #     # 存储文件的目录
    #     directory2 = os.path.dirname(model_out_name)
    #     # 如果不存在就创建
    #     if not os.path.exists(directory2):
    #         os.makedirs(directory2)
    #     torch.save(model.state_dict(), model_out_name)

    #     torch.save(result2, result_out_name2)
    # torch.save(result3, result_out_name3)

#     model_out_name2 = 'models/model_' + data_folder +'_A2'+'.pt'
#     model_out_name3 = 'models/model_' + data_folder +'_A3'+ '.pt'

#     torch.save(model.state_dict(), model_out_name2)
#     torch.save(model.state_dict(), model_out_name3)
