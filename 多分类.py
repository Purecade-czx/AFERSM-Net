import os

import torch
# 数据集相关包
import torchvision.datasets
from torchvision import transforms
# import torchvision.transforms   #对数据进行原始处理
from torchvision import datasets
from torch.utils.data import DataLoader

import torch.nn.functional as F  # 到时采用relu激活函数
import torch.optim as optim  # 优化器

# "1.数据导入"
batch_size = 64
transform = transforms.Compose([  # compose
    transforms.ToTensor(),  # 将图像转变为张量
    transforms.Normalize((0.1307), (0.3081))])  # 归一化处理，mnist数据集中0.1307均值，0.3081标准差

path = r'D:\文档\期刊专栏\使用WiFi指纹进行关节活动识别和室内定位\ARIL-master\data_mnist\raw'
print('---',os.listdir(path))
train_dataset = datasets.MNIST(root=path,
                               train=True, download=True, transform=transform)

# train_dataset  = datasets.MNIST(root = '../MNIST_data/mnist/',
#                                 train = True,download = True,transform = transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='./MNIST',
                              train=False, download=False, transform=transform)
# test_dataset = datasets.MNIST(root = '../MNIST_data/mnist/',train = False,download = True,transform = transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# "2.创建模型"
class Net(torch.nn.Module):  # 创建类
    def __init__(self):  # 构造函数
        super(Net, self).__init__()  # 继承父类
        self.l1 = torch.nn.Linear(784, 512)  # 构造对象，nn表示神经网络（neural network）
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 784)  # 这里-1是通过自动计算,自动算出N，N是样本数量的意思，（N,1,28,28） N*1*28*28/784,   784=28*28
        x = F.relu(self.l1(x))  # 加入非线性变换
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  # 最后一层不加入非线性变换


model = Net()

# "3.loss和优化器"
criterion = torch.nn.CrossEntropyLoss(size_average=True)  # 损失函数,交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 优化器,冲量=0.5，优化训练过程。【冲量，使其能够冲出鞍点】


# "4.设置训练周期"
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()  # 优化器使用前要进行清零

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()  # 反向传播
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d,111,%5d]  loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # test不需要梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # 取出这一行（10类中）中最大值的下标
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # predicted = label满足这个条件就进行相加
        print('Accuracy on test set:%d%%' % (100 * correct / total))


if __name__ == '__main__':

    for epoch in range(10):
        train(epoch)
        test()
