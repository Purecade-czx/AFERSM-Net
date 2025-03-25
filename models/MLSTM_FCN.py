import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from models.layers import ConvBlock



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MLSTMfcn(nn.Module):
    def __init__(self, *, num_classes_act, num_classes_loc, max_seq_len, num_features,
                 num_lstm_out=128, num_lstm_layers=1,
                 conv1_nf=128, conv2_nf=256, conv3_nf=128,
                 lstm_drop_p=0.8, fc_drop_p=0.3):
        super(MLSTMfcn, self).__init__()

        self.num_classes_act = num_classes_act
        self.num_classes_loc = num_classes_loc
        self.max_seq_len = max_seq_len
        self.num_features = num_features

        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers

        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        self.lstm = nn.LSTM(input_size=self.num_features,
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)

        self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
        self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
        self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)

        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)

        self.se1 = SELayer(self.conv1_nf)  # ex 128
        self.se2 = SELayer(self.conv2_nf)  # ex 256

        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.fc_drop_p)

        self.fc_act = nn.Linear(self.conv3_nf + self.num_lstm_out, self.num_classes_act)
        self.fc_loc = nn.Linear(self.conv3_nf + self.num_lstm_out, self.num_classes_loc)

    def forward(self, x, seq_lens):
        ''' input x should be in size [B,T,F], where
            B = Batch size
            T = Time samples
            F = features
        '''
        x1 = nn.utils.rnn.pack_padded_sequence(x, seq_lens,
                                               batch_first=True,
                                               enforce_sorted=False)
        x1, (ht, ct) = self.lstm(x1)
        x1, _ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True,
                                                 padding_value=0.0)
        x1 = x1[:, -1, :]

        x2 = x.transpose(2, 1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2, 2)

        x_all = torch.cat((x1, x2), dim=1)
        x_out_act = self.fc_act(x_all)
        x_out_act = F.log_softmax(x_out_act, dim=1)
        x_out_loc = self.fc_loc(x_all)
        x_out_loc = F.log_softmax(x_out_loc, dim=1)

        return x_out_act, x_out_loc




class FCN(nn.Module):
    '''FCN'''
    def __init__(self, c_in, layers=[128, 256, 128], kss=[8, 5, 3], ksse=[5, 3, 1], clf=True, reduction=16):
        super(FCN, self).__init__()
        self.clf = clf  # 是否作为分类器

        self.convblock1 = ConvBlock(c_in, layers[0], ks=kss[0])
        self.se = SELayer(layers[0], reduction)
        self.convblock2 = ConvBlock(layers[0], layers[1], ks=kss[1])
        self.se1 = SELayer(layers[1], reduction)
        self.convblock3 = ConvBlock(layers[1], layers[2], ks=kss[2])
        self.convblock4 = ConvBlock(c_in, layers[0], ks=ksse[0])
        self.convblock5 = ConvBlock(layers[0], layers[1], ks=ksse[1])
        self.convblock6 = ConvBlock(layers[1], layers[2], ks=ksse[2])
        # self.gap = nn.AdaptiveAvgPool1d(1)
        # self.gap = nn.AdaptiveMaxPool1d(1)
        self.gap = nn.AvgPool1d(1)          # 最好
        # self.gap = nn.MaxPool1d(1)
        self.droup = nn.Dropout(p=0.2)
        self.residual = Shrinkage(c_in,gap_size=(1))
        self.relu = nn.ReLU()
        # self.fc = nn.Linear(layers[-1], c_out)

    def forward(self, x):
        y = self.convblock1(x)

        # y = self.droup(y)
        y = self.se(y)
        y = self.convblock2(y)
        y = self.se1(y)
        y = self.convblock3(y)
        y = self.gap(y)


        return y





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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        # residual function
        self.residual_function = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, planes * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(planes * BasicBlock.expansion),
            SELayer(planes,reduction),
            Shrinkage(planes, gap_size=(1))
        )
        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or inplanes != BasicBlock.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inplanes, planes * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)



class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = torch.mean(x, dim=1, keepdim=True)  #CS
        # average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x

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

class ResNet(nn.Module):

    def __init__(self, block, layers, inchannel=52, activity_num=6, location_num=16):
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

        self.lstm = LSTM(inchannel, 10, 2, activity_num)
        self.droup = nn.Dropout(p=0.2)
        # self.lstm = torch.nn.LSTM(inchannel,activity_num)

        self.fcn = FCN(inchannel)

        self.ACTClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.act_fc = nn.Linear(518 * block.expansion, activity_num)

        self.LOCClassifier = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm1d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.loc_fc = nn.Linear(512 * block.expansion, location_num)
        self.loc_fc_f = nn.Linear(256, location_num)

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
        _, _, L = y.size()
        return F.interpolate(x, size=L) + y

    def forward(self, x):
        fcn_x = self.droup(x)
        fcn_x = self.fcn(fcn_x)

        # lstm_x = self.droup(x)
        fcn_x = fcn_x.permute(0, 2, 1)
        fcn_x = torch.tensor(fcn_x)
        # fcn_x = fcn_x.view(fcn_x.size(0), -1)
        # print('lstm_x is:',lstm_x.shape)
        lstm_x = x.permute(0, 2, 1)
        lstm_x = self.lstm(lstm_x)
        lstm_x = torch.tensor(lstm_x)
        # lstm_x = lstm_x.view(lstm_x.size(0), -1)

        # fcn_x = self.droup(x)
        # fcn_x = self.fcn(fcn_x)
        # fcn_x = torch.tensor(fcn_x)
        # fcn_x = fcn_x.view(fcn_x.size(0), -1)

        # print('the x shape is:',x.shape)
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

        # act = self.ACTClassifier(c4)
        # print('the act is:',act.shape)
        # act = act.view(act.size(0), -1)
        # print('the act.view(act.size(0), -1) is:',act.shape)
        act = torch.cat((lstm_x, fcn_x), 1)
        # act = torch.cat((fcn_x,act),1)
        # print('the act is:',act.shape)

        act1 = self.act_fc(act)


        loc1 = self.loc_fc(act)

        # return act1, loc1, x, c1, c2, c3, c4, act, loc

        return act1, loc1, x, c1, c2, c4, act
        # return act1, loc1, x, c1, c2, c3, c4, act, loc