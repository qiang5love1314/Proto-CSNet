#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ResNet12网络

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

# 构建3*3卷积层
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# Squeeze-and-Excitation(SE)块：挤压和激励
# 挤压操作通过对每个通道求平均值来降低输入特征图的维度。
# 励操作使用全连接层学习每个通道的重要性。然后将 SE 块的输出与输入特征图相乘以生成最终输出
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化层，用于将输入特征图平均池化为一个值
        self.fc = nn.Sequential(  # 全连接序列模块
            nn.Linear(channel, channel // reduction),  # 线性层，用于将输入的通道数映射到减小后的通道数
            nn.ReLU(inplace=True),  # 激活函数，常用的非线性函数之一
            nn.Linear(channel // reduction, channel),  # 线性层，用于将减小后的通道数映射回原始的通道数
            nn.Sigmoid()  # 激活函数，将输出限制在0和1之间
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# 构建残差块
class BasicBlock(nn.Module):
    expansion = 1

    # 构建残差块,use_se表示是否使用SE层,
    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, use_se=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.conv3 = conv3x3(planes, planes)
        # self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        # 残差块的步幅
        self.stride = stride
        # dropout率
        self.drop_rate = drop_rate
        # 用于计算BN层的均值和方差的批次数。
        self.num_batches_tracked = 0
        # 是否使用SE层
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(planes, 4)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        return out


# 构建ResNet
class ResNet(nn.Module):

    def __init__(self, block, n_blocks, keep_prob=1.0, avg_pool=False,
                 num_classes=-1, use_se=False):
        super(ResNet, self).__init__()

        self.inplanes = 1
        self.use_se = use_se
        # 第一个残差层，包含n_blocks[0]个残差块，每个残差块的输出通道数为64，步幅为2
        self.layer1 = self._make_layer(block, n_blocks[0], 64, stride=2)
        # 第二个残差层，包含n_blocks[1]个残差块，每个残差块的输出通道数为160，步幅为2
        self.layer2 = self._make_layer(block, n_blocks[1], 160, stride=2)
        # 第三个残差层，包含n_blocks[2]个残差块，每个残差块的输出通道数为320，步幅为2
        self.layer3 = self._make_layer(block, n_blocks[2], 320, stride=2)
        # 第四个残差层，包含n_blocks[3]个残差块，每个残差块的输出通道数为640，步幅为2
        self.layer4 = self._make_layer(block, n_blocks[3], 640, stride=2)
        if avg_pool:
            # self.avgpool = nn.AvgPool2d(5, stride=1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        # dropout层
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)

        for m in self.modules():
            # 卷积层使用kaiming_normal_方法进行参数初始化，其中权重初始化方式为"fan_out"，非线性函数为"leaky_relu"
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            # 批量归一化层权重初始化为1，偏置初始化为0
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_classes = num_classes
        # 分类器
        if self.num_classes > 0:
            self.classifier = nn.Linear(640, self.num_classes)

    # 构建残差块
    def _make_layer(self, block, n_block, planes, stride=1):
        downsample = None
        # 创建下采样层
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        """
        # 创建残差层
            layers = []
            if n_block == 1:
                layer = block(self.inplanes, planes, stride, downsample, self.use_se)
            else:
                layer = block(self.inplanes, planes, stride, downsample, self.use_se)
            layers.append(layer)
            self.inplanes = planes * block.expansion

            for i in range(1, n_block):
                if i == n_block - 1:
                    layer = block(self.inplanes, planes, use_se=self.use_se)
                else:
                    layer = block(self.inplanes, planes, use_se=self.use_se)
                layers.append(layer)
        """
        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, self.use_se)
            layers.append(layer)
            self.inplanes = planes * block.expansion
        for i in range(1, n_block):
            layer = block(self.inplanes, planes, use_se=self.use_se)
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        x = self.layer1(x)
        f0 = x
        x = self.layer2(x)
        f1 = x
        x = self.layer3(x)
        f2 = x
        x = self.layer4(x)
        f3 = x
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feat = x
        if self.num_classes > 0:
            x = self.classifier(x)

        if is_feat:
            return [f0, f1, f2, f3, feat], x
        else:
            return x


# 构建模型
# keep_prob：在训练过程中进行dropout时保留的概率。avg_pool：是否在输出层使用平均池化。kwargs：其他可选参数
def resnet_model(keep_prob=1.0, avg_pool=False, **kwargs):
    model = ResNet(BasicBlock, [1, 1, 1, 1], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model

# 调用模型
# model = resnet_model(avg_pool=True, num_classes = 4)
