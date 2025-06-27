"idea：combine BLS with Transformer to recognize activities"
# 数据集：D:\pycharm\files\OurActivityDataset\OurActivityDataset\Processed Data\bedroom(\meetingroom)
import numpy as np
from scipy.io import loadmat, savemat
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import transformer_encoder
from Comparison import CSImodel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import KernelPCA
from tqdm import tqdm
import MyNewModel
import argparse
import torch
import time
from scipy import signal
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import torch.utils.data as Data
from torch import nn
import transformers
from transformers import (AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForSequenceClassification,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    is_tensorboard_available,)
from transformers.utils import check_min_version, get_full_repo_name
from transformers import Trainer, TrainingArguments
from datasets import Dataset

# 残差网络
class RN(nn.Module):
    def __init__(self):
        super(RN, self).__init__()
        # 线性层，对输入进行一系列线性变换和激活操作，从而提取输入数据的特征表示
        self.linear_stack = nn.Sequential(
            nn.Linear(90, 128),
            nn.Hardsigmoid(),
            nn.Linear(128, 90),
            nn.Hardsigmoid(),
        )
        # 线性层，对数据进一步特征提取
        self.linear_stack_2 = nn.Sequential(
            nn.Linear(90, 64),
            # 硬sigmoid加速计算
            nn.Hardsigmoid(),
            nn.Linear(64, 32),
            nn.Hardsigmoid(),
        )
        # 输出层
        self.output_layer = nn.Linear(32, 1)
        # 损失函数
        self.loss_f = nn.CrossEntropyLoss()

    # 前向传播，在train时输出loss和预测值
    def forward(self, x, labels, mode='train'):
        y = self.linear_stack(x)
        # 残差
        y = y + x
        y = self.linear_stack_2(y)
        y = self.output_layer(y)

        if mode is 'train':
            return {
                'loss': self.loss_f(y, labels),
                'predictions': y
            }
        return y

# 计算预测结果的准确率
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (labels == preds).sum()/len(labels)
    return {
        'accuracy': acc,
    }

# 合并数据集
def merge_csi_DataAndLabel(path):

    listDir = []
    csiData = []
    activity = []
    labelList = ['lying', 'sitting', 'running', 'walking', 'pick up', 'wave hand', 'jump', 'squat', 'site down', 'stand up', 'fall down']

    # 获取数据集子文件夹列表
    for root, dirs, files in os.walk(path, topdown=False):
        listDir = dirs
    listDir = sorted(list(map(int, listDir)))

    # 获取原始CSI数据
    for i in range(len(listDir)):
        subpath = path + str(listDir[i])
        whole_file = [os.path.join(subpath, file) for file in os.listdir(subpath)]
        # 对于每一个路径，将其打开之后，使用readlines获取全部内容
        for w in whole_file:
            data = loadmat(w)['csi']
            csiData.append(data)
    csiData = np.array(csiData, dtype=complex)

    # 构造标签
    labelNum = []
    for i in range(len(listDir)):
        if i != len(listDir)-1:
            activity.extend([labelList[i]] * 100)
            labelNum.extend([i] * 100)
        else:
            activity.extend([labelList[i]] * 50)
            labelNum.extend([i] * 50)
    labelNum = np.array(labelNum)

    return csiData, activity, labelNum

def data_loader(data):
    loader = Data.DataLoader(
        dataset=data,
        batch_size=200,
        shuffle=True,
        num_workers=1,
    )
    return loader

def get_args():
    parser = argparse.ArgumentParser(description='Transformer-BLS')
    parser.add_argument('--sample', type=int, default=1, help='sample length on temporal side')
    parser.add_argument('--batch', type=int, default=16, help='batch size [default: 16]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [default: 0.001]')
    parser.add_argument('--epoch', type=int, default=50, help='number of epoch [default: 20]')
    parser.add_argument('--hlayers', type=int, default=6, help='horizontal transformer layers [default: 6]')
    parser.add_argument('--hheads', type=int, default=9, help='horizontal transformer head [default: 9]')
    parser.add_argument('--vlayers', type=int, default=1, help='vertical transformer layers [default: 1]')
    parser.add_argument('--vheads', type=int, default=200, help='vertical transformer head [default: 200]')
    parser.add_argument('--com_dim', type=int, default=50, help='compressor vertical transformer layers [default: 50]')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    path1 = r'./Processed Data/bedroom/'
    path2 = r'./Processed Data/meetingroom/'
    csiData, csiLabel, labelNum = merge_csi_DataAndLabel(path1)   # every activity has 100 samples except fall down which only includes 50 samples.
    "csiData [1050, [3, 30, 200]], csiLabel [1050, 1]"
    amplitudeCSI = abs(csiData)

    # 复数形式下CSI的先快速傅里叶变化处理效果更好，然后再小波变换
    # fftData = np.fft.fft(csiData)
    fftData = np.fft.fft(amplitudeCSI)
    # b, a = signal.butter(5, 3 * 2 / 50, 'lowpass')    #过于平滑，数据特征消失
    # csiData = signal.filtfilt(b, a, csiData)
    wavename = 'db5'
    ca, cd = pywt.dwt(fftData, wavename)
    dwtCSI = pywt.idwt(None, cd, wavename, 'smooth')

    # AntennA, AntennB, AntennC = dwtCSI[0][0], dwtCSI[0][1], dwtCSI[0][2]
    # for i in range(30):
    #     plt.plot(AntennA[i])
    #     plt.plot(AntennB[i])
    #     plt.plot(AntennC[i])
    # plt.show()

    # 频谱图绘制，但感觉有点问题
    # f, t, zxx = signal.stft(amplitudeCSI[100][2][25])
    # plt.pcolormesh(t, f, np.abs(zxx))
    # plt.colorbar()
    # plt.show()

    csi = torch.from_numpy(dwtCSI).float().unsqueeze(0)  # 1050*3*30*200
    # csi张量不参与反向传播
    csi.requires_grad = False
    # 得到一个1050*200*90的张量
    csi = csi.view(1050, 200, 90)
    # 在第0维度上拼接，得到一个1050*200*90*1的张量
    csi = torch.cat([csi], dim=0)

    # 得到一个1050*200*90的张量
    # csi = torch.cat(list(csi), dim=0)
    # labelNum = torch.from_numpy(labelNum).int()
    # print(labelNum)

    # x_train, y_train, x_valid, y_valid = train_test_split(csi, labelNum, test_size=0.2, random_state=20)

    # 将csi变成torch张量
    labelNum = torch.tensor(labelNum)
    labelNum = torch.unsqueeze(labelNum, 1)
    # 构造数据集，将数据和标签拼接起来
    MIXdata = Data.TensorDataset(csi, labelNum)
    # 划分训练集和测试集
    train_dataset, test_dataset = torch.utils.data.random_split(MIXdata, [840, 210], random_state=40, shuffle=True)
    # 获取数据
    train_data = data_loader(train_dataset)
    trainDataset = Dataset.from_dict({'x':csi, 'labels':labelNum})
    test_dataset = data_loader(test_dataset)
    testDataset = Dataset.from_dict({'x': csi, 'labels': labelNum})

    args = get_args()

    # 创建TransformerM模型实例
    model = MyNewModel.TransformerM(args)

    if torch.cuda.is_available():
        model = model.cuda()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    n_epochs = args.epoch
    best = 0.0

    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        tr_acc = 0.
        total_num = 0
        print("\nEpoch{}/{}".format(epoch, n_epochs))
        print("-" * 10)
        #print("\n")
        steps = len(train_data)
        model.train()
        time_start = time.time()
        # 训练集
        for batch in tqdm(train_data):
            X_train, Y_train = batch
            Y_train = Y_train.unsqueeze(dim=1)
            Y_train = Y_train.long()
            X_train, Y_train = X_train.to(device), Y_train.to(device)

            outputs = model(X_train)
            pred = torch.max(outputs, 1)[1]
            # 变为1维
            Y_train=Y_train.view(-1)
            loss = criterion(outputs, Y_train)
            # 原有写法：
            # loss = criterion(outputs, Y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            running_correct = (pred.cpu() == Y_train.cpu()).sum()
            tr_acc += running_correct.item()
            total_num += len(batch[0])

        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        running_loss = 0.0
        running_correct = 0
        tr_acc = 0.
        total_num = 0
        print("\nStart validation")
        print("-" * 10)
        #print("\n")
        steps = len(train_data)
        model.eval()
        # conf_matrix = [[0 for _ in range(len(aclist))] for _ in range(len(aclist))]

        # 测试集
        time_start = time.time()
        for batch in tqdm(test_data):
            X_train, Y_train = batch
            # Y_train = Y_train.unsqueeze(dim=1)
            Y_train = Y_train.long()
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            outputs = model(X_train)
            pred = torch.max(outputs, 1)[1]
            running_correct = (pred.cpu() == Y_train.cpu()).sum()
            # conf_matrix = gen_conf_matrix(pred, Y_train, conf_matrix)
            tr_acc += running_correct.item()
            total_num += len(batch[0])
            # running_correct += torch.sum(pred == Y_train.data)
            acc = tr_acc/total_num
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        print("\nAccuracy is", tr_acc/total_num)
        if best < acc:
            best = acc
            # write_to_file(conf_matrix)
            torch.save(model, 'model\\model.pkl')
        print("\nBest is", best)



    # # 生成模型实例
    #
    # # 模型对象移动到指定的设备上进行计算，以提高计算效率和减少内存占用。
    # model = RN().to(device=device)
    # # model = model
    # training_args = TrainingArguments(
    #     output_dir='./results',           # output directory 结果输出地址
    #     num_train_epochs=100,             # total # of training epochs 训练总批次
    #     per_device_train_batch_size=100,  # batch size per device during training 训练批大小
    #     per_device_eval_batch_size=10,    # batch size for evaluation 评估批大小
    #     logging_dir='./logs/rn_log',    # directory for storing logs 日志存储位置
    #     learning_rate=1e-3,               # 学习率
    #     save_steps=False,
    # )
    #
    # trainer = Trainer(
    #     model=model,                      # the instantiated 🤗 Transformers model to be trained 需要训练的模型
    #     args=training_args,               # training arguments, defined above 训练参数
    #     train_dataset=trainDataset,            # training dataset 训练集
    #     eval_dataset=testDataset,         # evaluation dataset 测试集
    #     compute_metrics=compute_metrics   # 计算指标方法
    # )
    #
    # # trainer.train()
    # evaluate = trainer.evaluate()
    # print(evaluate.values())