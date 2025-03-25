import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from models.layers import ConvBlock
from torch.autograd import Variable
from torch.nn import init



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,              #一维卷积
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):           #继承nn.Module父类
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out





class SEAttention1d(nn.Module):
    '''
    Modified from https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/SEAttention.py
    '''
    def __init__(self, channel, reduction):
        super(SEAttention1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b,c,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1)

        return x*y.expand_as(x)


class macnn_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=None, stride=1, reduction=16):
        super(macnn_block, self).__init__()

        if kernel_size is None:
            kernel_size = [3, 6, 12]

        self.reduction = reduction

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size[0], stride=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size[1], stride=1, padding='same')
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size[2], stride=1, padding='same')

        self.bn = nn.BatchNorm1d(out_channels*3)
        self.relu = nn.ReLU()

        # self.se = SEAttention1d(out_channels*3,reduction=reduction)

    def forward(self,x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x_con = torch.cat([x1,x2,x3], dim=1)

        out = self.bn(x_con)
        out = self.relu(out)

        # out_se = self.se(out)

        return out

class MACNN(nn.Module):

    def __init__(self, in_channels=52, channels=64, num_classes_act=6, num_classes_loc=16, block_num=None):
        super(MACNN, self).__init__()

        if block_num is None:
            block_num = [2, 2, 2]

        self.in_channel = in_channels
        self.num_classes_act = num_classes_act
        self.num_classes_loc = num_classes_loc
        self.channel = channels

        self.max_pool1 = nn.MaxPool1d(kernel_size=3,stride=2,padding=1)
        self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=2,padding=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_act = nn.Linear(self.channel*12, num_classes_act)
        self.fc_loc = nn.Linear(self.channel*12, num_classes_loc)

        self.layer1 = self._make_layer(macnn_block, block_num[0], self.channel)
        self.layer2 = self._make_layer(macnn_block, block_num[1], self.channel*2)
        self.layer3 = self._make_layer(macnn_block, block_num[2], self.channel*4)

    def _make_layer(self, block, block_num, channel, reduction=16):

        layers = []
        for i in range(block_num):
            layers.append(block(self.in_channel, channel, kernel_size=None,
                                stride=1, reduction=reduction))
            self.in_channel = 3*channel

        return nn.Sequential(*layers)

    def forward(self, x):

        out1 = self.layer1(x)
        out1 = self.max_pool1(out1)

        out2 = self.layer2(out1)
        out2 = self.max_pool2(out2)

        out3 = self.layer3(out2)
        out3 = self.avg_pool(out3)

        out = torch.flatten(out3, 1)
        out_act = self.fc_act(out)
        out_loc = self.fc_loc(out)


        return out_act, out_loc






class FCN(nn.Module):
    '''FCN'''
    def __init__(self, c_in, layers=[128, 256, 128], kss=[7, 5, 3], ksse=[5, 3, 1], clf=True):
        super(FCN, self).__init__()
        self.clf = clf  # 是否作为分类器

        self.convblock1 = ConvBlock(c_in, layers[0], ks=kss[0])
        self.convblock2 = ConvBlock(layers[0], layers[1], ks=kss[1])
        self.convblock3 = ConvBlock(layers[1], layers[2], ks=kss[2])
        self.convblock4 = ConvBlock(c_in, layers[0], ks=ksse[0])
        self.convblock5 = ConvBlock(layers[0], layers[1], ks=ksse[1])
        self.convblock6 = ConvBlock(layers[1], layers[2], ks=ksse[2])
        # self.gap = nn.AdaptiveAvgPool1d(1)
        # self.gap = nn.AdaptiveMaxPool1d(1)
        self.gap = nn.AvgPool1d(1)          # 最好
        # self.gap = nn.MaxPool1d(1)
        self.droup = nn.Dropout(p=0.2)
        # self.fc = nn.Linear(layers[-1], c_out)

    def forward(self, x):
        y = self.convblock1(x)
        # y = self.droup(y)
        # y = self.gap(y)
        y = self.convblock2(y)
        # y = self.gap(y)
        y = self.convblock3(y)
        y = self.gap(y)

        x = self.convblock4(x)
        # x = self.droup(x)
        # x = self.gap(x)
        x = self.convblock5(x)
        # x = self.gap(x)
        x = self.convblock6(x)
        x = self.gap(x)
        # print('the x is:',x.shape)
        # x = self.gap(x).squeeze(-1)
        # x = self.fc(x)
        # return F.softmax(x, dim=-1) if self.clf else x
        z = torch.cat((x,y),1)
        return z



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out




class ResNet(nn.Module):
    def __init__(self, block, layers,  inchannel=52, activity_num=6, location_num=16):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3,
                                 bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        self.fcn = FCN(inchannel)

        self.gru = nn.GRU(256, 10, activity_num, 2)


        self.ACTClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.act_fc = nn.Linear(512 * block.expansion, activity_num)

        self.LOCClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.loc_fc = nn.Linear(512 * block.expansion, location_num)
        self.loc_fc_f = nn.Linear(256, location_num)

        #
        # self.fc1 = nn.Linear(512 * block.expansion, )
        # self.fc2 = nn.Linear(512 * block.expansion, )
        # self.fc3 = nn.Linear(512 * block.expansion, )
        # self.fc4 = nn.Linear(512 * block.expansion, )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #
        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,L = y.size()
        return F.interpolate(x, size=L) + y


    def forward(self, x):
        # print('the x shape is:',x.shape)
        fcn_x = self.fcn(x)
        gru_x = fcn_x.permute(0, 2, 1)
        gru_x = self.gru(gru_x)




        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # print('the c4 shape is:',c4.shape)

        # c4 = torch.flatten(c4)
        # print('the flatten c4 shape is:', c4.shape)

        act = self.ACTClassifier(c4)
        # print('the act is:',act.shape)
        act = act.view(act.size(0), -1)
        act1 = self.act_fc(act)

        loc = self.LOCClassifier(c4)
        loc = loc.view(loc.size(0), -1)
        loc1 = self.loc_fc(loc)

        return act1, loc1, c1, c2, c3, c4, act, loc

