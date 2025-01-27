"""resnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

CLASSNUM = 101
INPUTSHAPE = 64

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes):
        super().__init__()
        
        self.in_channels = INPUTSHAPE

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, INPUTSHAPE, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(INPUTSHAPE),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, INPUTSHAPE, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

class SVD(nn.Module):
    def __init__(self, in_planes, beta, kernel_1_out):
        super(SVD, self).__init__()
        # 1x1 conv -> 3x3 conv branch
        self.kernel_3_in = int(in_planes * beta)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, self.kernel_3_in, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(self.kernel_3_in, kernel_1_out, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(kernel_1_out),
            nn.ReLU(True),
        )

    def forward(self, x):
        y = self.b1(x)
        return y

class ResNetSVD(nn.Module):
    def __init__(self, block, num_blocks, num_classes, divided_by=1):
        super(ResNetSVD, self).__init__()

        self.in_planes = INPUTSHAPE//divided_by

        self.conv1 = nn.Conv2d(3, INPUTSHAPE//divided_by, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(INPUTSHAPE//divided_by)
        self.layer1 = self._make_layer(block, INPUTSHAPE//divided_by, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//divided_by, num_blocks[1], stride=2)
        self.layer3 = svd_block([128, 256, 256], [256, 256, 256], 2, 2)
        self.layer4 = svd_block([256, 512, 512], [512, 512, 512], 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.size()) # 64
        out = self.conv1(x)
        # print(out.size()) # 64
        out = self.bn1(out)
        # print(out.size()) # 64
        out = self.layer1(out)
        # print(out.size()) # 64
        out = self.layer2(out)
        # print(out.size()) # 128
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        out = self.avg_pool(out)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.linear(out)
        # print(out.size())
        return out

class Fire(nn.Module):
    def __init__(self, in_planes, beta, out):
        super(Fire, self).__init__()
        self.kernel_in = int(in_planes * beta)
        self.branch = out // 2
        self.squeeze = nn.Conv2d(in_planes, self.kernel_in, kernel_size=1)
        self.expand1 = nn.Conv2d(self.kernel_in, self.branch, kernel_size=1)
        self.expand2 = nn.Conv2d(self.kernel_in, self.branch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.squeeze(x)
        y1 = self.expand1(x)
        y2 = self.expand2(x)
        y = torch.cat((y1, y2), dim=1)
        y = self.bn(y)
        y = self.relu(y)
        return y

class ResNetFire(nn.Module):
    def __init__(self, block, num_blocks, num_classes, divided_by=1):
        super(ResNetFire, self).__init__()

        self.in_planes = 64//divided_by

        self.conv1 = nn.Conv2d(3, 64//divided_by, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64//divided_by)
        self.layer1 = self._make_layer(block, 64//divided_by, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//divided_by, num_blocks[1], stride=2)
        
        self.layer3 = fire_block([128, 256, 256], [256, 256, 256], 2, 2)
        self.layer4 = fire_block([256, 512, 512], [512, 512, 512], 1, 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.size())
        out = self.conv1(x)
        # print(out.size())
        out = self.bn1(out)
        # print(out.size())
        out = self.layer1(out)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        out = self.avg_pool(out)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.linear(out)
        # print(out.size())
        return out

class ResNetDpConv(nn.Module):
    def __init__(self, block, num_blocks, num_classes, divided_by=1):
        super(ResNetDpConv, self).__init__()

        self.in_planes = 64//divided_by

        self.conv1 = nn.Conv2d(3, 64//divided_by, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64//divided_by)
        self.layer1 = self._make_layer(block, 64//divided_by, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//divided_by, num_blocks[1], stride=2)
        
        self.layer3 = deep_conv_layer(128, 256, 3, 1)
        self.layer4 = deep_conv_layer(256, 512, 2, 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

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

class ResNetInception1(nn.Module):
    def __init__(self, block, num_blocks, num_classes, divided_by=1):
        super(ResNetInception1, self).__init__()

        self.in_planes = INPUTSHAPE//divided_by

        self.conv1 = nn.Conv2d(3, INPUTSHAPE//divided_by, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(INPUTSHAPE//divided_by)
        self.layer1 = self._make_layer(block, INPUTSHAPE//divided_by, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//divided_by, num_blocks[1], stride=2)
        
        self.layer3 = inception1_block([128, 256, 256], [256, 256, 256], 2, 2)
        self.layer4 = inception1_block([256, 512, 512], [512, 512, 512], 1, 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class Inception2(nn.Module):
    def __init__(self, in_planes, out):
        super(Inception2, self).__init__()
        self.branch = out // 2
        self.son1 = nn.Conv2d(in_planes, self.branch, kernel_size=(1, 3), padding=(0, 1))
        self.son2 = nn.Conv2d(in_planes, self.branch, kernel_size=(3, 1), padding=(1, 0))
        self.bn = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        y1 = self.son1(x)
        y2 = self.son2(x)
        y = torch.cat((y1, y2), dim=1)
        y = self.bn(y)
        y = self.relu(y)
        return y

class ResNetInception2(nn.Module):
    def __init__(self, block, num_blocks, num_classes, divided_by=1):
        super(ResNetInception2, self).__init__()

        self.in_planes = INPUTSHAPE//divided_by

        self.conv1 = nn.Conv2d(3, INPUTSHAPE//divided_by, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(INPUTSHAPE//divided_by)
        self.layer1 = self._make_layer(block, INPUTSHAPE//divided_by, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//divided_by, num_blocks[1], stride=2)
        
        self.layer3 = inception2_block([128, 256, 256], [256, 256, 256], 2, 2)
        self.layer4 = inception2_block([256, 512, 512], [512, 512, 512], 1, 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class resnet50DpconvInception2(nn.Module):
    def __init__(self, block, num_blocks, num_classes, divided_by=1):
        super(resnet50DpconvInception2, self).__init__()

        self.in_planes = INPUTSHAPE//divided_by

        self.conv1 = nn.Conv2d(3, INPUTSHAPE//divided_by, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(INPUTSHAPE//divided_by)
        self.layer1 = self._make_layer(block, INPUTSHAPE//divided_by, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//divided_by, num_blocks[1], stride=2)
        
        self.layer3 = deep_conv_layer(512, 256, 3, 1)
        self.layer4 = inception2_block([256, 512, 512], [512, 512, 512], 1, 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def svd_block(in_list, out_list, pooling_k, pooling_s):
    layers = [SVD(in_list[i], 0.25, out_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)

def fire_block(in_list, out_list, pooling_k, pooling_s):
    layers = [Fire(in_list[i], 0.25, out_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)

def deep_conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_in, kernel_size=k_size, groups=chann_in, padding=p_size),
        nn.Conv2d(chann_in, chann_out, kernel_size=1),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def inception1_block(in_list, out_list, pooling_k, pooling_s):
    layers = [Inception1(in_list[i], 0.25, out_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)

def inception2_block(in_list, out_list, pooling_k, pooling_s):
    layers = [Inception2(in_list[i], out_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)



# resnet18 start

def resnet18inception2():
    """ return a ResNet 18 Deep inception2 object
    """
    return ResNetInception2(BasicBlock, [2, 2, 2, 2])

def resnet18inception1():
    """ return a ResNet 18 inception1 object
    """
    return ResNetInception1(BasicBlock, [2, 2, 2, 2])

def resnet18dpconv():
    """ return a ResNet 18 Deep Conv object
    """
    return ResNetDpConv(BasicBlock, [2, 2, 2, 2])

def resnet18fire():
    """ return a ResNet 18 FIRE object
    """
    return ResNetFire(BasicBlock, [2, 2, 2, 2])

def resnet18svd():
    """ return a ResNet 18 SVD object
    """
    return ResNetSVD(BasicBlock, [2, 2, 2, 2])

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

# resnet18 end 

# resnet34 start

def resnet34inception2(num_classes = CLASSNUM):
    """ return a ResNet 34 Inception2 object
    """
    return ResNetInception2(BasicBlock, [3, 4, 6, 3], num_classes = num_classes)

def resnet34inception1(num_classes = CLASSNUM):
    """ return a ResNet 34 Inception1 object
    """
    return ResNetInception1(BasicBlock, [3, 4, 6, 3], num_classes = num_classes)

def resnet34dpconv(num_classes = CLASSNUM):
    """ return a ResNet 34 DpConv object
    """
    return ResNetDpConv(BasicBlock, [3, 4, 6, 3], num_classes = num_classes)

def resnet34fire(num_classes = CLASSNUM):
    """ return a ResNet 34 Fire object
    """
    return ResNetFire(BasicBlock, [3, 4, 6, 3], num_classes = num_classes)

def resnet34svd(num_classes = CLASSNUM):
    """ return a ResNet 34 SVD object
    """
    return ResNetSVD(BasicBlock, [3, 4, 6, 3], num_classes = num_classes)

def resnet34(num_classes = CLASSNUM):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes = num_classes)

# resnet34 end and resnet50 start

def resnet50(num_classes = CLASSNUM):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes = num_classes)

def resnet50svd(num_classes = CLASSNUM):
    """ return a ResNetSVD 50 object
    """
    return ResNetSVD(BottleNeck, [3, 4, 6, 3], num_classes = num_classes)

def resnet50dpconv(num_classes = CLASSNUM):
    """ return a ResNetDpConv 50 object
    """
    return ResNetDpConv(BottleNeck, [3, 4, 6, 3], num_classes = num_classes)

def resnet50fire(num_classes = CLASSNUM):
    """ return a ResNetFire 50 object
    """
    return ResNetFire(BottleNeck, [3, 4, 6, 3], num_classes = num_classes)

def resnet50inception2(num_classes = CLASSNUM):
    """ return a ResNetInception2 50 Inception2 object
    """
    return ResNetInception2(BottleNeck, [3, 4, 6, 3], num_classes = num_classes)

def resnet50inception1(num_classes = CLASSNUM):
    """ return a ResNetInception1 50 Inception1 object
    """
    return ResNetInception1(BottleNeck, [3, 4, 6, 3], num_classes = num_classes)

def resnet50dpconvinception2(num_classes = CLASSNUM):
    """ return a resnet50DpconvInception2 50 Dpconv Inception2 object
    """
    return resnet50DpconvInception2(BottleNeck, [3, 4, 6, 3], num_classes = num_classes)

# resnet50 end

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])






inputsize = 64

def test():
    net = resnet34()
    x = torch.randn(2, 3, inputsize, inputsize)
    y = net(x)
    print(y.size())
    return y.size()

# test()

def model_user():

    model = resnet34()
    input = torch.randn(2, 3, inputsize, inputsize)

    return model, input