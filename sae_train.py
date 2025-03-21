import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time

from tqdm import tqdm
from models.sae import *
# from models.apl_plus import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
num_epochs = 300

# load data
data_amp = sio.loadmat('data/train_floor_building_three.mat')
train_data_amp = data_amp['td']
train_data = train_data_amp
# data_amp = sio.loadmat('data/train_data_split_amp.mat')
# train_data_amp = data_amp['train_data']
# train_data = train_data_amp


train_activity_label = data_amp['train_floor_label']
train_location_label = data_amp['train_building_label']
train_label = np.concatenate((train_activity_label, train_location_label), 1)

num_train_instances = len(train_data)

train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
train_label = torch.from_numpy(train_label).type(torch.LongTensor)
# train_data = train_data.view(num_train_instances, 1, -1)
# train_label = train_label.view(num_train_instances, 2)


train_dataset = TensorDataset(train_data, train_label)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

data_amp = sio.loadmat('data/test_floor_building_three.mat')
test_data_amp = data_amp['test_data']
test_data = test_data_amp
# data_pha = sio.loadmat('data/test_data_split_pha.mat')
# test_data_pha = data_pha['test_data']
# test_data = np.concatenate((test_data_amp,test_data_pha), 1)


test_activity_label = data_amp['test_floor_label']
test_location_label = data_amp['test_building_label']
test_label = np.concatenate((test_activity_label, test_location_label), 1)

num_test_instances = len(test_data)
# print('----------')
# print(test_data.shape)
# print(len(test_data))
# print('----------')

test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
# print('----------')
# print(test_data.type)
# print('----------')
test_label = torch.from_numpy(test_label).type(torch.LongTensor)
test_label = test_label.cuda()
# test_data = test_data.view(num_test_instances, 1, -1)
# test_label = test_label.view(num_test_instances, 2)

test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)








EPOCH = 10
BATCH_SIZE = 64
LR = 0.005


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # 编码网络
        self.encoder = nn.Sequential(
            nn.Linear(520, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
        )
        # 解码网络
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 520),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# # 定义一个编码器对象
# autoencoder = AutoEncoder()
# autoencoder = autoencoder.cuda()
# optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
# loss_func = nn.MSELoss()
#
# for epoch in range(EPOCH):
#     for (samples, labels) in tqdm(train_data_loader):
#         samplesV = Variable(samples.to(device))
#         encoded_x, decoded_x = autoencoder(samplesV)
#         loss_sae = loss_func(decoded_x, samplesV)  # 这里如果写成b_x会更容易裂解
#         optimizer.zero_grad()  # clear gradients for this training step
#         loss_sae.backward()  # backpropagation, compute gradients
#         optimizer.step()  # apply gradients
#
#     # for p in autoencoder.parameters():
#     #     print(p.data, p.grad)
#     torch.save(autoencoder, './autoencoder.pth')
























aplnet = ResNet(block=BasicBlock, layers=[1, 1, 1, 1], inchannel=1)
#aplnet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], inchannel=1)
#aplnet = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], inchannel=1)
#

# aplnet = ResNet(block=Bottleneck, layers=[2, 3, 4, 6])

aplnet = aplnet.cuda()

#criterion = nn.CrossEntropyLoss(size_average=False).to(device)
criterion = nn.CrossEntropyLoss(reduction='sum').cuda()

optimizer = torch.optim.Adam(aplnet.parameters(), lr=0.05,weight_decay=0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,                                                                      #不懂
                                                 milestones=[10, 20, 30, 40, 60, 70, 80, 90, 100, 110, 120, 130,
                                                             140, 150, 160, 170, 180, 190, 200, 250, 300],
                                                 gamma=0.5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_loss_act = np.zeros([num_epochs, 1])
train_loss_loc = np.zeros([num_epochs, 1])
test_loss_act = np.zeros([num_epochs, 1])
test_loss_loc = np.zeros([num_epochs, 1])
train_acc_act = np.zeros([num_epochs, 1])
train_acc_loc = np.zeros([num_epochs, 1])
test_acc_act = np.zeros([num_epochs, 1])
test_acc_loc = np.zeros([num_epochs, 1])

for epoch in range(num_epochs):
    print('Epoch:', epoch)
    aplnet.train()                                                          #从116到156行看不懂
    #scheduler.step()
    # for i, (samples, labels) in enumerate(train_data_loader):
    loss_x = 0
    loss_y = 0
    for (samples, labels) in tqdm(train_data_loader):
        samplesV = Variable(samples.to(device))
        labels_act = labels[:, 0].squeeze()
        labels_loc = labels[:, 1].squeeze()
        labelsV_act = Variable(labels_act.to(device))
        labelsV_loc = Variable(labels_loc.to(device))

        # Forward + Backward + Optimize
        optimizer.zero_grad()                                               #每计算一个batch就清除历史损失函数
        predict_label_act, predict_label_loc,_,_,_,_,_,_,_ = aplnet(samplesV)     #把样本输入到网络并输出

        loss_act = criterion(predict_label_act, labelsV_act)                      #计算活动分类损失函数
        loss_loc = criterion(predict_label_loc, labelsV_loc)                      #计算位置分类损失函数

        loss = loss_act + loss_loc
        # loss = loss_loc
        # print(loss.item())
        loss.backward()                                                           #将损失函数反向传播
        optimizer.step()                                                          #进行参数的更新

        # loss = loss1+0.5*loss2+0.25*loss3+0.25*loss4
        # loss = loss1+loss2+loss3+loss4

        #打印损失
        loss_x += loss_act.item()                                                 #把loss_act累加到loss_x
        loss_y += loss_loc.item()

        # loss.backward()
        # optimizer.step()

    train_loss_act[epoch] = loss_x / num_train_instances
    train_loss_loc[epoch] = loss_y / num_train_instances

    aplnet.eval()                                                                 #执行一个字符串表达式，并返回表达式的值
    # loss_x = 0
    correct_train_act = 0
    correct_train_loc = 0
    for i, (samples, labels) in enumerate(train_data_loader):                     #enumerate()返回(samples, labels)和i
        with torch.no_grad():                                                     #强制之后的内容不进行计算图构建，不参与梯度反向传播等操作
            samplesV = Variable(samples.to(device))                               #Variable()是创建一个tensor变量
            labels = labels.squeeze()

            labels_act = labels[:, 0].squeeze()                                   #labels的shape是[128,2],labels[:, 0].squeeze()取第一列并转换为一维array
            labels_loc = labels[:, 1].squeeze()
            labelsV_act = Variable(labels_act.to(device))
            labelsV_loc = Variable(labels_loc.to(device))

            predict_label_act, predict_label_loc,_,_,_,_,_,_,_ = aplnet(samplesV)

            prediction = predict_label_loc.data.max(1)[1]                         #torch.max()[0]为只返回最大值的每个数；troch.max()[1]为返回最大值的每个索引
            #print(prediction)                                                    #torch.max(0)[]指的是每一列最大值，torch.max(1)[]指的是每一行最大值
            correct_train_loc += prediction.eq(labelsV_loc.data.long()).sum()     #torch.eq()表示比较prediction与labelsV_loc，不相等为0，相等为1，最后.sum()统计相等的个数

            prediction = predict_label_act.data.max(1)[1]
            correct_train_act += prediction.eq(labelsV_act.data.long()).sum()

            loss_act = criterion(predict_label_act, labelsV_act)
            loss_loc = criterion(predict_label_loc, labelsV_loc)
            # loss_x += loss.item()
            # scheduler.step()


    print("Activity Training accuracy:", (100 * float(correct_train_act) / num_train_instances))           #186到192行不懂
    print("Location Training accuracy:", (100 * float(correct_train_loc) / num_train_instances))

    # train_loss[epoch] = loss_x / num_train_instances
    train_acc_act[epoch] = 100 * float(correct_train_act) / num_train_instances
    train_acc_loc[epoch] = 100 * float(correct_train_loc) / num_train_instances


    trainacc_act = str(100 * float(correct_train_act) / num_train_instances)[0:6]
    trainacc_loc = str(100 * float(correct_train_loc) / num_train_instances)[0:6]

    loss_x = 0
    loss_y = 0
    correct_test_act = 0
    correct_test_loc = 0
    for i, (samples, labels) in enumerate(test_data_loader):
        with torch.no_grad():
            samplesV = Variable(samples.to(device))
            labels_act = labels[:, 0].squeeze()
            labels_loc = labels[:, 1].squeeze()
            labelsV_act = Variable(labels_act.to(device))
            labelsV_loc = Variable(labels_loc.to(device))

        predict_label_act, predict_label_loc,_,_,_,_,_,_,_ = aplnet(samplesV)
        prediction = predict_label_act.data.max(1)[1]
        correct_test_act += prediction.eq(labelsV_act.data.long()).sum()

        prediction = predict_label_loc.data.max(1)[1]
        correct_test_loc += prediction.eq(labelsV_loc.data.long()).sum()

        loss_act = criterion(predict_label_act, labelsV_act)
        loss_loc = criterion(predict_label_loc, labelsV_loc)
        loss_x += loss_act.item()
        loss_y += loss_loc.item()

    print("Activity Test accuracy:", (100 * float(correct_test_act) / num_test_instances))
    print("Location Test accuracy:", (100 * float(correct_test_loc) / num_test_instances))

    test_loss_act[epoch] = loss_x / num_test_instances
    test_acc_act[epoch] = 100 * float(correct_test_act) / num_test_instances

    test_loss_loc[epoch] = loss_y / num_test_instances
    test_acc_loc[epoch] = 100 * float(correct_test_loc) / num_test_instances

    testacc_act = str(100 * float(correct_test_act) / num_test_instances)[0:6]
    testacc_loc = str(100 * float(correct_test_loc) / num_test_instances)[0:6]

    if epoch == 0:
        temp_test = correct_test_act
        temp_train = correct_train_act
    elif correct_test_act > temp_test:
        torch.save(aplnet, 'weights/net1111epoch' + str(
            epoch) + 'Train' + trainacc_act + 'Test' + testacc_act + 'Train' + trainacc_loc + 'Test' + testacc_loc + '.pkl')

        temp_test = correct_test_act
        temp_train = correct_train_act


# for learning curves
sio.savemat(
    'result/net1111TrainLossAct_Train' + str(100 * float(temp_train) / num_train_instances)[
                                                                 0:6] + 'Test' + str(
        100 * float(temp_test) / num_test_instances)[0:6] + '.mat', {'train_loss': train_loss_act})
sio.savemat(
    'result/net1111TestLossACT_Train' + str(100 * float(temp_train) / num_train_instances)[
                                                                0:6] + 'Test' + str(
        100 * float(temp_test) / num_test_instances)[0:6] + '.mat', {'test_loss': test_loss_act})
sio.savemat(
    'result/net1111TrainLossLOC_Train' + str(100 * float(temp_train) / num_train_instances)[
                                 0:6] + 'Test' + str(
        100 * float(temp_test) / num_test_instances)[0:6] + '.mat', {'train_loss': train_loss_loc})
sio.savemat(
    'result/net1111TestLossLOC_Train' + str(100 * float(temp_train) / num_train_instances)[
                                 0:6] + 'Test' + str(
        100 * float(temp_test) / num_test_instances)[0:6] + '.mat', {'test_loss': test_loss_loc})

sio.savemat('result/net1111TrainAccuracyACT_Train' + str(
    100 * float(temp_train) / num_train_instances)[0:6] + 'Test' + str(100 * float(temp_test) / num_test_instances)[
                                                                   0:6] + '.mat', {'train_acc': train_acc_act})
sio.savemat('result/net1111TestAccuracyACT_Train' + str(
    100 * float(temp_train) / num_train_instances)[0:6] + 'Test' + str(100 * float(temp_test) / num_test_instances)[
                                                                   0:6] + '.mat', {'test_acc': test_acc_act})
print(str(100 * float(temp_test) / num_test_instances)[0:6])

sio.savemat('result/net1111TrainAccuracyLOC_Train' + str(
    100 * float(temp_train) / num_train_instances)[0:6] + 'Test' + str(100 * float(temp_test) / num_test_instances)[
                                                                   0:6] + '.mat', {'train_acc': train_acc_loc})
sio.savemat('result/net1111TestAccuracyLOC_Train' + str(
    100 * float(temp_train) / num_train_instances)[0:6] + 'Test' + str(100 * float(temp_test) / num_test_instances)[
                                                                   0:6] + '.mat', {'test_acc': test_acc_loc})

