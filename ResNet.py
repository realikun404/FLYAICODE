# import torch
# from torch import nn
# import torch.nn.functional as F
#
# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self,in_channel,out_channel,stride=1,downsample=None):  # 下采样，虚线那条路
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=stride,padding=1,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channel)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1,
#                                bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channel)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None):
#         super(Bottleneck,self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1,
#                                bias=False) # squeeze channels
#         self.bn1 = nn.BatchNorm2d(out_channel)
#
#         self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride,
#                                padding=1,
#                                bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channel)
#
#         self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1,
#                                bias=False)  # unsqueeze channels
#         self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         print("how many times?")
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         print(out.shape)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
# class ResNet(nn.Module):
#
#     def __init__(self,block,block_num,num_classes=101, include_top=True):
#         """
#         desc:
#         block: 看是哪一个block前两个就用BasicBlock，后三个就用Bottleneck
#         block_num: 传入的是列表，详情见图
#         num_classes: 训练集的个数
#         include_top: 默认即可
#         """
#         super(ResNet, self).__init__()
#         self.include_top = include_top
#         self.in_channel = 64  # 这个不知道要不要改
#
#         self.conv1 = nn.Conv2d(3, self.in_channel,kernel_size=7,stride=2,padding=3,bias=False) # 3 是RGB
#
#         self.bn1 = nn.BatchNorm2d(self.in_channel)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
#         self.layer1 = self._make_layer(block,64,block_num[0])
#         self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)
#         if self.include_top :
#             self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # output size = (1,1)
#             self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
#
#     def _make_layer(self,block,channel,block_num,stride=1):
#         downsample = None
#         if stride != 1 or self.in_channel != channel * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channel,channel * block.expansion, kernel_size=1,stride=stride,bias=False),
#                 nn.BatchNorm2d(channel * block.expansion)
#             )
#
#         layers = []
#         layers.append(block(self.in_channel,channel,downsample=downsample,stride=stride))
#         self.in_channel = channel*block.expansion
#
#         for _ in range(1,block_num):
#             layers.append(block(self.in_channel,channel))
#
#         return nn.Sequential(*layers)
#
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         if self.include_top:
#             x = self.avgpool(x)
#             x = torch.flatten(x, 1)
#             x = self.fc(x)
#
#         return x
#
#
# def resnet34(num_classes=101,include_top=True):
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes,include_top=include_top)
#
#
# def resnet101(num_classes=101,include_top=True):
#     return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes,include_top=include_top)

import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

__all__ = ['ResNet50', 'ResNet101','ResNet152']

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=101, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet50():
    return ResNet([3, 4, 6, 3])

def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet152():
    return ResNet([3, 8, 36, 3])


def ResNet18():
    return ResNet([2, 2, 2, 2])


def ResNet34():
    return ResNet([3, 4, 6, 3])


if __name__=='__main__':
    #model = torchvision.models.resnet50()
    model = ResNet50()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
