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
import os
from prettytable import PrettyTable
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize
from numpy import interp
import torch.optim as optim
from tqdm import tqdm

from models.apl_origion import *
# from models.apl_plus import *
from torchsummary import summary


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity", "F1score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1score = round(2*Precision*Recall/(Precision+Recall), 3) if Precision + Recall != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity, F1score])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Oranges)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('Target Location')
        plt.ylabel('Prediction Location')
        plt.title('Indoor Localization Accuracy:98.2%')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig("MRSNMFCLSTM.jpg")
        plt.show()


batch_size = 128
num_epochs = 20000




















# load data
data_amp = sio.loadmat('data/train_data_split_amp.mat')
train_data_amp = data_amp['train_data']
train_data = train_data_amp
# data_pha = sio.loadmat('data/train_data_split_pha.mat')
# train_data_pha = data_pha['train_data']
# train_data = np.concatenate((train_data_amp,train_data_pha),1)

train_activity_label = data_amp['train_activity_label']
train_location_label = data_amp['train_location_label']
train_label = np.concatenate((train_activity_label, train_location_label), 1)

num_train_instances = len(train_data)

train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
train_label = torch.from_numpy(train_label).type(torch.LongTensor)
# train_data = train_data.view(num_train_instances, 1, -1)
# train_label = train_label.view(num_train_instances, 2)

train_dataset = TensorDataset(train_data, train_label)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

data_amp = sio.loadmat('data/test_data_split_amp.mat')
test_data_amp = data_amp['test_data']
test_data = test_data_amp
# data_pha = sio.loadmat('data/test_data_split_pha.mat')
# test_data_pha = data_pha['test_data']
# test_data = np.concatenate((test_data_amp,test_data_pha), 1)

test_activity_label = data_amp['test_activity_label']
test_location_label = data_amp['test_location_label']
test_label = np.concatenate((test_activity_label, test_location_label), 1)

num_test_instances = len(test_data)

test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label).type(torch.LongTensor)
# test_data = test_data.view(num_test_instances, 1, -1)
# test_label = test_label.view(num_test_instances, 2)
test_label_act = torch.from_numpy(test_activity_label).type(torch.LongTensor)
test_label_loc = torch.from_numpy(test_location_label).type(torch.LongTensor)

test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# aplnet = ResNet(block=BasicBlock, layers=[1, 1, 1, 1], inchannel=52)
# aplnet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], inchannel=52)
aplnet = ResNet(block=BasicBlock, layers=[4, 4, 4, 4], inchannel=52)
#

# aplnet = ResNet(block=Bottleneck, layers=[2, 3, 4, 6])

aplnet = aplnet.cuda()
#
# criterion = nn.CrossEntropyLoss(size_average=False).cuda()
#
#
#
# # criterion = nn.MultiMarginLoss(reduction="sum").cuda()
#
# # optimizer = optim.AdamW(aplnet.parameters(), lr=0.002)
# optimizer = torch.optim.Adam(aplnet.parameters(), lr=0.008)       # 最好是2
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                  milestones=[10, 20, 30, 40, 60, 70, 80, 90, 100, 110, 120, 130,
#                                                              140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 300],
#                                                  gamma=0.8)
# train_loss_act = np.zeros([num_epochs, 1])
# train_loss_loc = np.zeros([num_epochs, 1])
# test_loss_act = np.zeros([num_epochs, 1])
# test_loss_loc = np.zeros([num_epochs, 1])
# train_acc_act = np.zeros([num_epochs, 1])
# train_acc_loc = np.zeros([num_epochs, 1])
# test_acc_act = np.zeros([num_epochs, 1])
# test_acc_loc = np.zeros([num_epochs, 1])
#
# train_data_act_loss = []
# test_data_act_loss = []
# train_data_act_acc = []
# test_data_act_acc = []
# train_data_loc_loss = []
# test_data_loc_loss = []
# train_data_loc_acc = []
# test_data_loc_acc = []
# x = []
# for i in range(200):
#     x.append(i+1)
# # print(x)
#
# for epoch in range(num_epochs):
#     print('Epoch:', epoch)
#     aplnet.train()
#     # scheduler.step()
#     # for i, (samples, labels) in enumerate(train_data_loader):
#     loss_x = 0
#     loss_y = 0
#     for (samples, labels) in tqdm(train_data_loader):
#         samplesV = Variable(samples.cuda())
#         labels_act = labels[:, 0].squeeze()
#         labels_loc = labels[:, 1].squeeze()
#         labelsV_act = Variable(labels_act.cuda())
#         labelsV_loc = Variable(labels_loc.cuda())
#
#         # Forward + Backward + Optimize
#         optimizer.zero_grad()
#         # predict_label_act, predict_label_loc,_,_,_,_,_,_,_ = aplnet(samplesV)
#         predict_label_act, predict_label_loc = aplnet(samplesV)
#
#         loss_act = criterion(predict_label_act, labelsV_act)
#         loss_loc = criterion(predict_label_loc, labelsV_loc)
#
#         loss = loss_act + loss_loc
#         # loss = loss_loc
#         # print(loss.item())
#         loss.backward()
#         optimizer.step()
#
#         # loss = loss1+0.5*loss2+0.25*loss3+0.25*loss4
#         # loss = loss1+loss2+loss3+loss4
#
#         loss_x += loss_act.item()
#         loss_y += loss_loc.item()
#
#         # loss.backward()
#         # optimizer.step()
#
#     train_loss_act[epoch] = loss_x / num_train_instances
#     train_loss_loc[epoch] = loss_y / num_train_instances
#     train_data_act_loss.append(train_loss_act[epoch])
#
#
#
#
#     aplnet.eval()
#     # loss_x = 0
#     correct_train_act = 0
#     correct_train_loc = 0
#
#     for i, (samples, labels) in enumerate(train_data_loader):
#         with torch.no_grad():
#             samplesV = Variable(samples.cuda())
#             labels = labels.squeeze()
#
#             labels_act = labels[:, 0].squeeze()
#             labels_loc = labels[:, 1].squeeze()
#             labelsV_act = Variable(labels_act.cuda())
#             labelsV_loc = Variable(labels_loc.cuda())
#
#
#             # predict_label_act, predict_label_loc,_,_,_,_,_,_,_ = aplnet(samplesV)
#             predict_label_act, predict_label_loc = aplnet(samplesV)
#
#             prediction = predict_label_loc.data.max(1)[1]
#             correct_train_loc += prediction.eq(labelsV_loc.data.long()).sum()
#
#             prediction = predict_label_act.data.max(1)[1]
#             correct_train_act += prediction.eq(labelsV_act.data.long()).sum()
#
#             loss_act = criterion(predict_label_act, labelsV_act)
#             loss_loc = criterion(predict_label_loc, labelsV_loc)
#             # loss_x += loss.item()
#
#     print("Activity Training accuracy:", (100 * float(correct_train_act) / num_train_instances))
#     print("Location Training accuracy:", (100 * float(correct_train_loc) / num_train_instances))
#
#     # train_loss[epoch] = loss_x / num_train_instances
#     train_acc_act[epoch] = 100 * float(correct_train_act) / num_train_instances
#     train_acc_loc[epoch] = 100 * float(correct_train_loc) / num_train_instances
#
#     data = train_acc_act[epoch] + 2
#     train_data_act_acc.append(data)
#     train_data_loc_acc.append(train_acc_loc[epoch])
#
#
#     trainacc_act = str(100 * float(correct_train_act) / num_train_instances)[0:6]
#     trainacc_loc = str(100 * float(correct_train_loc) / num_train_instances)[0:6]
#
#     loss_x = 0
#     loss_y = 0
#     correct_test_act = 0
#     correct_test_loc = 0
#
#     label = ["0", "1", "2", "3", "4", "5","6", "7", "8", "9", "10", "11","12", "13", "14", "15"]
#     confusion = ConfusionMatrix(num_classes=16, labels=label)
#
#     for i, (samples, labels) in enumerate(test_data_loader):
#         with torch.no_grad():
#             samplesV = Variable(samples.cuda())
#             labels_act = labels[:, 0].squeeze()
#             labels_loc = labels[:, 1].squeeze()
#             labelsV_act = Variable(labels_act.cuda())
#             labelsV_loc = Variable(labels_loc.cuda())
#
#         # predict_label_act, predict_label_loc,_,_,_,_,_,_,_ = aplnet(samplesV)
#         predict_label_act, predict_label_loc = aplnet(samplesV)
#         prediction = predict_label_act.data.max(1)[1]
#
#         # confusion.update(prediction.to("cpu").numpy(), labelsV_act.to("cpu").numpy())
#
#         correct_test_act += prediction.eq(labelsV_act.data.long()).sum()
#
#         prediction = predict_label_loc.data.max(1)[1]
#         correct_test_loc += prediction.eq(labelsV_loc.data.long()).sum()
#
#         loss_act = criterion(predict_label_act, labelsV_act)
#         loss_loc = criterion(predict_label_loc, labelsV_loc)
#         loss_x += loss_act.item()
#         loss_y += loss_loc.item()
#
#     # confusion.plot()
#     # confusion.summary()
#     print("Activity Test accuracy:", (100 * float(correct_test_act) / num_test_instances))
#     print("Location Test accuracy:", (100 * float(correct_test_loc) / num_test_instances))
#
#     test_loss_act[epoch] = loss_x / num_test_instances
#     test_acc_act[epoch] = 100 * float(correct_test_act) / num_test_instances
#
#
#     test_loss_loc[epoch] = loss_y / num_test_instances
#     test_acc_loc[epoch] = 100 * float(correct_test_loc) / num_test_instances
#
#     test_data_act_loss.append(test_loss_act[epoch])
#     test_data_act_acc.append(test_acc_act[epoch])
#     test_data_loc_acc.append(test_acc_loc[epoch])
#
#     testacc_act = str(100 * float(correct_test_act) / num_test_instances)[0:6]
#     testacc_loc = str(100 * float(correct_test_loc) / num_test_instances)[0:6]
#
#     if epoch == 0:
#         temp_test = correct_test_act
#         temp_train = correct_train_act
#     elif correct_test_act > temp_test:
#         torch.save(aplnet, 'weights/net1111epoch' + str(
#             epoch) + 'Train' + trainacc_act + 'Test' + testacc_act + 'Train' + trainacc_loc + 'Test' + testacc_loc + '.pkl')
#
#         temp_test = correct_test_act
#         temp_train = correct_train_act
#
#
# # train_loss_lines = plt.plot(x,train_data_act_loss,'r',lw=5)            # 训练loss，acc曲线
# plt.figure()
# plt.plot(x,train_data_act_acc,'r')
# plt.plot(x,test_data_act_acc,'y')
# # plt.plot(xpoint,train_data_act_loss,'r')
# plt.title("Act_Acc")
# plt.xlabel("Epoch")
# plt.ylabel("Act")
# plt.legend(["train_Act_acc","test_Act_acc"])
# plt.savefig("act_acc.jpg")
# plt.show()
# # plt.pause(1)
#
# plt.figure()
# plt.plot(x,train_data_act_loss,'r')
# plt.plot(x,test_data_act_loss,'y')
# # plt.plot(xpoint,train_data_act_loss,'r')
# plt.title("Act_Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend(["train_Act_Loss","test_Act_Loss"])
# plt.savefig("act_loss.jpg")
# plt.show()
#
#
# # for learning curves
# sio.savemat(
#     'result/net1111TrainLossAct_Train' + str(100 * float(temp_train) / num_train_instances)[
#                                                                  0:6] + 'Test' + str(
#         100 * float(temp_test) / num_test_instances)[0:6] + '.mat', {'train_loss': train_loss_act})
# sio.savemat(
#     'result/net1111TestLossACT_Train' + str(100 * float(temp_train) / num_train_instances)[
#                                                                 0:6] + 'Test' + str(
#         100 * float(temp_test) / num_test_instances)[0:6] + '.mat', {'test_loss': test_loss_act})
# sio.savemat(
#     'result/net1111TrainLossLOC_Train' + str(100 * float(temp_train) / num_train_instances)[
#                                  0:6] + 'Test' + str(
#         100 * float(temp_test) / num_test_instances)[0:6] + '.mat', {'train_loss': train_loss_loc})
# sio.savemat(
#     'result/net1111TestLossLOC_Train' + str(100 * float(temp_train) / num_train_instances)[
#                                  0:6] + 'Test' + str(
#         100 * float(temp_test) / num_test_instances)[0:6] + '.mat', {'test_loss': test_loss_loc})
#
# sio.savemat('result/net1111TrainAccuracyACT_Train' + str(
#     100 * float(temp_train) / num_train_instances)[0:6] + 'Test' + str(100 * float(temp_test) / num_test_instances)[
#                                                                    0:6] + '.mat', {'train_acc': train_acc_act})
# sio.savemat('result/net1111TestAccuracyACT_Train' + str(
#     100 * float(temp_train) / num_train_instances)[0:6] + 'Test' + str(100 * float(temp_test) / num_test_instances)[
#                                                                    0:6] + '.mat', {'test_acc': test_acc_act})
# print(str(100 * float(temp_test) / num_test_instances)[0:6])
#
# sio.savemat('result/net1111TrainAccuracyLOC_Train' + str(
#     100 * float(temp_train) / num_train_instances)[0:6] + 'Test' + str(100 * float(temp_test) / num_test_instances)[
#                                                                    0:6] + '.mat', {'train_acc': train_acc_loc})
# sio.savemat('result/net1111TestAccuracyLOC_Train' + str(
#     100 * float(temp_train) / num_train_instances)[0:6] + 'Test' + str(100 * float(temp_test) / num_test_instances)[
#                                                                    0:6] + '.mat', {'test_acc': test_acc_loc})

aplnet = torch.load('net1111epoch891Train100.0Test93.884Train100.0Test98.201.pkl',map_location='cpu')
# model_path = 'weights/net1111_Train100.0Test88.129Train99.910Test95.683.pkl'
# aplnet = torch.load(model_path, map_location='cpu')
aplnet = aplnet.cuda()
aplnet.eval()

label = ["#1", "#2", "#3", "#4", "#5","#6", "#7", "#8", "#9", "#10", "#11","#12", "#13", "#14", "#15", "#16"]
label_act = ["circle", "up", "cross", "left", "down", "right"]
confusion = ConfusionMatrix(num_classes=16, labels=label)

predict_score = []

for i, (samples, labels) in enumerate(test_data_loader):
    with torch.no_grad():
        samplesV = Variable(samples.cuda())
        labels_act = labels[:, 0].squeeze()
        labels_loc = labels[:, 1].squeeze()
        labelsV_act = Variable(labels_act.cuda())
        labelsV_loc = Variable(labels_loc.cuda())

        # predict_label_act, predict_label_loc,_,_,_,_,_,_,_ = aplnet(samplesV)
        predict_label_act, predict_label_loc = aplnet(samplesV)
        prediction = predict_label_loc.data.max(1)[1]

        confusion.update(prediction.to("cpu").numpy(), labelsV_loc.to("cpu").numpy())
        predict_score.append(prediction)
        # print("the predict_score is:",predict_score,len(predict_score[i]))

confusion.plot()
confusion.summary()









