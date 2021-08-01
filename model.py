import sys
import torch
import torch.nn.functional as F
from torch import nn


# Activation function
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference=False):
        assert (x.data.dim() == 4)
        # _, _, tH, tW = target_size

        if inference:

            # B = x.data.size(0)
            # C = x.data.size(1)
            # H = x.data.size(2)
            # W = x.data.size(3)

            return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1). \
                expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3),
                       target_size[3] // x.size(3)). \
                contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
        else:
            return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')


class Conv_Bn_Act_layer(nn.Module):
    """
    This layer is composed of a Convolutional layer, a Batch normalization layer and an Activation function.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        # stack the layers here.
        self.conv = nn.ModuleList()

        # Convolutional layer = conv
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        # Batch normalization layer = bn
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        # Activation function = act
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        else:
            print("activate error {} {} {}".format(sys._getframe().f_code.co_filename,
                                                   sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class Res_layer(nn.Module):
    """
    Stacking residual layers which consists of two convolution layers
    Args:
        ch (int): the number of input and output channels.
        num_res_layer (int): the number of residual blocks.
        shortcut (bool): If True, residual addition is enabled.
    """

    def __init__(self, ch, num_res_layer=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.block_list = nn.ModuleList()
        for i in range(num_res_layer):
            one_block = nn.ModuleList()
            one_block.append(Conv_Bn_Act_layer(ch, ch, 1, 1, 'mish'))
            one_block.append(Conv_Bn_Act_layer(ch, ch, 3, 1, 'mish'))
            self.block_list.append(one_block)

    def forward(self, x):
        for block in self.block_list:
            y = x
            for one in block:
                y = one(y)
            x = x + y if self.shortcut else y
        return x


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Act_layer(3, 32, 3, 1, 'mish')
        self.conv2 = Conv_Bn_Act_layer(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Act_layer(64, 64, 1, 1, 'mish')
        self.conv4 = Conv_Bn_Act_layer(64, 64, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Act_layer(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Act_layer(32, 64, 3, 1, 'mish')
        self.conv7 = Conv_Bn_Act_layer(64, 64, 1, 1, 'mish')
        self.conv8 = Conv_Bn_Act_layer(128, 64, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x6 = x6 + x4
        x7 = self.conv7(x6)
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Act_layer(64, 128, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Act_layer(128, 64, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Act_layer(128, 64, 1, 1, 'mish')
        self.resblock = Res_layer(ch=64, num_res_layer=2)
        self.conv4 = Conv_Bn_Act_layer(64, 64, 1, 1, 'mish')
        # r -1 -10
        self.conv5 = Conv_Bn_Act_layer(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Act_layer(128, 256, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Act_layer(256, 128, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Act_layer(256, 128, 1, 1, 'mish')

        self.resblock = Res_layer(ch=128, num_res_layer=8)
        self.conv4 = Conv_Bn_Act_layer(128, 128, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Act_layer(256, 256, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Act_layer(256, 512, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Act_layer(512, 256, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Act_layer(512, 256, 1, 1, 'mish')

        self.resblock = Res_layer(ch=256, num_res_layer=8)
        self.conv4 = Conv_Bn_Act_layer(256, 256, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Act_layer(512, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Act_layer(512, 1024, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Act_layer(1024, 512, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Act_layer(1024, 512, 1, 1, 'mish')

        self.resblock = Res_layer(ch=512, num_res_layer=4)
        self.conv4 = Conv_Bn_Act_layer(512, 512, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Act_layer(1024, 1024, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class TransferClassify(nn.Module):
    """
    Last layer to get classification results.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50176, 12544)
        self.fc2 = nn.Linear(12544, 1568)
        self.fc3 = nn.Linear(1568, 5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.softmax(x, dim=1)
        x = self.fc3(x)

        return x


class Neck(nn.Module):
    def __init__(self, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Act_layer(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Act_layer(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Act_layer(1024, 512, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Act_layer(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Act_layer(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Act_layer(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Act_layer(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv_Bn_Act_layer(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv_Bn_Act_layer(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Act_layer(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Act_layer(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Act_layer(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Act_layer(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Act_layer(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv_Bn_Act_layer(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv_Bn_Act_layer(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Act_layer(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Act_layer(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Act_layer(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Act_layer(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3, inference=False):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        up = self.upsample1(x7, downsample4.size(), self.inference)
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size(), self.inference)
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class Backbone(nn.Module):
    def __init__(self, yolov4conv137weight=None, inference=False):
        super(Backbone, self).__init__()
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()
        self.end = TransferClassify()
        # neck
        self.neck = Neck(inference)

        # yolov4conv137
        if yolov4conv137weight:
            _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neck)
            pretrained_dict = torch.load(yolov4conv137weight)

            model_dict = _model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)

        self._model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.end)

    def forward(self, x):
        return self._model(x)


backbone = Backbone('/home/jhj/Desktop/JHJ/git/yolov4_backbone_pre-training/yolov4.conv.137.pth')

