import torch
import torch.nn as nn
from torchstat import stat
NUM_CLASSES = 101


class Inception1(nn.Module):
    def __init__(self, in_planes, beta, kernel_1_out):
        super(Inception1, self).__init__()
        # 1x1 conv -> 3x3 conv branch
        self.kernel_3_in = int(in_planes * beta)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, self.kernel_3_in, kernel_size=1),
            nn.Conv2d(self.kernel_3_in, self.kernel_3_in, kernel_size=3, padding=1),
            nn.Conv2d(self.kernel_3_in, kernel_1_out, kernel_size=1),
            nn.BatchNorm2d(kernel_1_out),
            nn.ReLU(True),
        )

    def forward(self, x):
        y = self.b1(x)
        return y


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer


def inception1_block(in_list, out_list, pooling_k, pooling_s):
    layers = [Inception1(in_list[i], 0.25, out_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)


class VGG16Inception1(nn.Module):
    def __init__(self):
        super(VGG16Inception1, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = inception1_block([128, 256, 256], [256, 256, 256], 2, 2)
        self.layer4 = inception1_block([256, 512, 512], [512, 512, 512], 1, 1)
        self.layer5 = inception1_block([512, 512, 512], [512, 512, 512], 1, 1)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # GlobalAveragePooling 256*1*1
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.conv_out = nn.Conv2d(in_channels=512, out_channels=NUM_CLASSES, kernel_size=1)  # 1*1*10卷积计算 → 10*1*1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.gap(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        x = x.view(x.size(0), -1)
        x = self.softmax(x)

        return x


def Vgg16inception1():
    return VGG16Inception1()