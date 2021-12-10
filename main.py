# -*- coding: utf-8 -*-
import argparse
import os
# import psutil
import torch
from torch import nn, optim
import csv
import cv2
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
import numpy as np
# import psutil
import Mydataset
from path import MODEL_PATH
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler

import torchvision.models as models
from Mydataset import MyDataset

from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
from path import MODEL_PATH
import re
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms



'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=1, type=int, help="batch size")
args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

device = torch.device(device)


class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def __init__(self):
        pass
        # self.images = []
        # self.labels = []
        # self.image_width, self.image_height = 512, 512

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("Food2")

    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''
        # image_path = "data/input/Food2/"
        # label_path = "data/input/Food2/train.csv"

        # f_csv = []
        # with open(label_path) as f:
        #     f_csv = list(csv.reader(f))

        # 数据集label最小就是1
        # row = f_csv[1]
        # for row in f_csv[1:]:
        # one_hot_label = torch.zeros(1, 101)
        # 此处接受的是list
        # index = torch.LongTensor([int(row[1])]).view(-1, 1)
        # one_hot_label.scatter_(dim=1, index=index, value=1)
        # self.labels.append(one_hot_label)

        # for row in f_csv[1:]:
        #     # 处理图片
        #     image = cv2.imread(image_path + row[0])
        #     # cv2.imshow('image', image)
        #     # cv2.waitKey(0)
        #     image = cv2.resize(image, (self.image_width, self.image_height), 0, 0, cv2.INTER_LINEAR)
        #     # cv2.imshow('image', image)
        #     # cv2.waitKey(0)
        #     image = image.astype(np.float32)
        #     image = np.multiply(image, 1. / 255.)
        #     self.images.append(image)
        #
        #     # 处理标签
        #     one_hot_label = torch.zeros(1, 101)
        #     # 此处接受的是list
        #     index = torch.LongTensor([int(row[1])]).view(-1, 1)
        #     one_hot_label.scatter_(dim=1, index=index, value=1)
        #     one_hot_label = one_hot_label.numpy()
        #     self.labels.append(one_hot_label)

        # self.images = np.array(self.images).astype(np.float32)
        # self.labels = np.array(self.labels).astype(np.float32)
        # self.images = torch.from_numpy(self.images)
        # self.labels = torch.from_numpy(self.labels)

        # self.images = self.images.transpose(1, 3).contiguous()  # B C W H

        # train_transform = transform = transforms.Compose([  # [1]
        #     transforms.Resize(256),  # [2]
        #     transforms.CenterCrop(224),  # [3]
        #     transforms.ToTensor(),  # [4]
        #     transforms.Normalize(  # [5]
        #         mean=[0.485, 0.456, 0.406],  # [6]
        #         std=[0.229, 0.224, 0.225]  # [7]
        #     )])

        # train_transform = transforms.Compose([
        #     # transforms.Resize(256),  # [2]
        #     # transforms.CenterCrop(224),  # [3]
        #     transforms.RandomResizedCrop(224),  # 随机裁剪到224x224大小
        #     transforms.RandomHorizontalFlip(),  # 随机水平翻转
        #     transforms.RandomRotation(degrees=15),  # 随机旋转
        #     transforms.ToTensor(),  # 转化成Tensor
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
        # ])
        train_transform = transforms.Compose([
            # transforms.Resize(256),  # [2]
            # transforms.CenterCrop(224),  # [3]
            transforms.RandomResizedCrop(224),  # 随机裁剪到224x224大小
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomRotation(degrees=15),  # 随机旋转
            # transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),  # 转化成Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
        ])



        train_dataset = Mydataset.MyDataset(transform=train_transform)
        # _weight=train_dataset.getWeight()
        batch_size = 16
        # sampler = WeightedRandomSampler(_weight, len(train_dataset), replacement=True)
        # train_dataset.initial()

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  # sampler=sampler,
                                  shuffle=True,
                                  num_workers=0)
        # print(len(train_dataset))
        # for _, data in enumerate(train_loader):
        #     print(_, data[0].shape)
        #     break
        # print("很离谱，为什么Dataloader不能读取数据？？？？")

        # for i in range(20):
        #     print(train_dataset[i])
        # print(self.images)
        # print(self.labels)
        return train_loader

    def train(self):
        '''
        训练模型，必须实现此方法
        :return:
        '''
        train_loader = self.deal_with_data()
        # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

        # net = EfficientNet.from_pretrained('efficientnet-b5')
        # feature = net._fc.in_features
        # net._fc = nn.Linear(in_features=feature, out_features=101, bias=True)

        net1 = models.resnet101(pretrained=True)
        inchannel = net1.fc.in_features
        net1.fc = nn.Linear(inchannel, 101)

        net2 = models.resnet101(pretrained=True)
        inchannel = net2.fc.in_features
        net2.fc = nn.Linear(inchannel, 101)

        net3 = models.resnet101(pretrained=True)
        inchannel = net3.fc.in_features
        net3.fc = nn.Linear(inchannel, 101)


        # net1 = models.densenet121(pretrained=True)
        # num_ftrs = net1.classifier.in_features
        # net1.classifier = nn.Linear(num_ftrs, 101)
        #
        # # net1 = models.resnet101(pretrained=True)
        # # inchannel = net1.fc.in_features
        # # net1.fc = nn.Linear(inchannel, 101)
        #
        # net2=models.resnet50(pretrained=True)
        # inchannel = net2.fc.in_features
        # net2.fc = nn.Linear(inchannel, 101)
        #
        # net3=models.alexnet(pretrained=True)
        # num_fc = net3.classifier[6].in_features
        # net3.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=101)


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        mlps = [net1.to(device), net2.to(device), net3.to(device)]

        optimizer1 = optim.Adam(net1.parameters(), lr=0.00005)
        optimizer2 = optim.Adam(net2.parameters(), lr=0.00004)
        optimizer3 = optim.Adam(net3.parameters(), lr=0.00003)

        loss_function = nn.CrossEntropyLoss()

        # loss_function =nn.L1Loss(reduction='mean')
        # loss_function = nn.NLLLoss()
        # loss_function = nn.BCEWithLogitsLoss() # 二分类才能用的大哥
        # optimizer = optim.Adam(net.parameters(), lr=0.00001)
        # optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.5)
        # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)#学习率每7个epoch衰减为原来的1/10
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.15)
        running_loss = 0.0
        epoch=17
        epoch1 = epoch
        epoch2=epoch
        epoch3=epoch
        for index1 in range(epoch1):
            # train
            pass
            running_loss = 0.0
            loss=""
            print(loss)
            print(f"This is epoch{index1 + 1}:")
            # print("--------------\n---------\n---------")
            for step, data in enumerate(train_loader, start=0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                optimizer1.zero_grad()
                net1.train()
                out = net1(images)
                loss = loss_function(out, labels)
                loss.backward()
                optimizer1.step()
                rate = (step + 1) / len(train_loader)
                if rate>=0.99:
                    print(loss.item())
        for index2 in range(epoch2):
            # train

            running_loss = 0.0
            # loss=""
            # print(loss)
            print(f"This is epoch{index2 + 1}:")
            # print("--------------\n---------\n---------")
            for step, data in enumerate(train_loader, start=0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                optimizer2.zero_grad()
                net2.train()
                out = net2(images)
                loss = loss_function(out, labels)
                loss.backward()
                optimizer2.step()
                rate = (step + 1) / len(train_loader)
                if rate >= 0.99:
                    print(loss.item())

        for index3 in range(epoch3):
            # train

            running_loss = 0.0
            # loss=""
            # print(loss)
            print(f"This is epoch{index3 + 1}:")
            # print("--------------\n---------\n---------")
            for step, data in enumerate(train_loader, start=0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                optimizer3.zero_grad()
                net3.train()
                out = net3(images)
                loss = loss_function(out, labels)
                loss.backward()
                optimizer3.step()
                rate = (step + 1) / len(train_loader)
                if rate>=0.99:
                    print(loss.item())

        torch.save(net1, 'data/output/model/model1.pkl')
        torch.save(net2, 'data/output/model/model2.pkl')
        torch.save(net3, 'data/output/model/model3.pkl')


                # net.train()
                # images, labels = data
                # images, labels = images.to(device), labels.to(device)

                # for item in images:
                #     for item1 in item:
                #         for item2 in item1:
                #             print(item2)

                # print(images,labels)
                # labels = labels.view(-1, 101)
                # labels = labels.argmax(dim=1)

                # for test

                # optimizer.zero_grad()
                #
                # logits = net(images)

                # logits = net(images).view(-1, 1, 101)
                # print(logits,logits.shape)
                # logits = logits.view(-1, 101)
                # print(logits, labels)
                # logits = F.softmax(logits)
                #
                # loss = loss_function(logits, labels)
                #
                # loss.backward()

                # for name, parms in net.named_parameters():
                #     print('-->name:', name)
                #     print('-->para:', parms)
                #     print('-->grad_requirs:', parms.requires_grad)
                #     print('-->grad_value:', parms.grad)
                #     print("===")

                # optimizer.step()

                # print statistics
                # running_loss += loss.item()
                # print train process
                # rate = (step + 1) / len(train_loader)
                # a = "*" * int(rate * 50)
                # b = "." * int((1 - rate) * 50)

                # print()
                # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
                # del images, labels, logits, loss, rate, a, b
                # torch.cuda.empty_cache()
                # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
                # print()



if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main = Main()
    main.download_data()
    main.train()
