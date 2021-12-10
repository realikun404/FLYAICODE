import argparse
import os
import sys
import torch
from torch import nn, optim
import csv
import cv2
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
import numpy as np
from path import MODEL_PATH
from torch.utils.data import Dataset, DataLoader
import time
from torchvision import transforms
import torch.nn.functional as F
from torchvision import datasets
from PIL import Image
from ResNet import ResNet
from ResNet import ResNet18
import torch.nn.functional as F
from torchvision import transforms

MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
Torch_MODEL_NAME = "model.pkl"
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


# 判断gpu是否可用
# if torch.cuda.is_available():
#     device = 'cuda'
# else:
#     device = 'cpu'
# device = torch.device(device)





class Prediction(FlyAI):
    def __init__(self):
        super(Prediction, self).__init__()
        self.mlps=[]
        self.model1=""
        self.model2=""
        self.model3=""

    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''

        self.model1 = torch.load("data/output/model/model1.pkl")  # 加载示例
        self.model2 = torch.load("data/output/model/model2.pkl")
        self.model3 = torch.load("data/output/model/model3.pkl")
        self.mlps.append(self.model2)
        self.mlps.append(self.model1)
        self.mlps.append(self.model3)

    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input:  评估传入样例 {"image_path":".\/image\/Akarna_Dhanurasana35.jpg"}
        :return: 模型预测成功之后返回给系统样例 {"label":1}
        '''
        train_transform = transform = transforms.Compose([  # [1]
            # transforms.Resize(256),  # [2]
            # transforms.CenterCrop(224),  # [3]
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # [4]
            transforms.Normalize(  # [5]
                mean=[0.485, 0.456, 0.406],  # [6]
                std=[0.229, 0.224, 0.225]  # [7]
            )])

        # train_transform = transforms.Compose([
        #     # transforms.Resize(256),  # [2]
        #     # transforms.CenterCrop(224),  # [3]
        #     transforms.RandomResizedCrop(224),  # 随机裁剪到224x224大小
        #     transforms.RandomHorizontalFlip(),  # 随机水平翻转
        #     transforms.RandomRotation(degrees=15),  # 随机旋转
        #     transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2),
        #     transforms.ToTensor(),  # 转化成Tensor
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
        # ])

        # train_transform = transforms.Compose([
        #     transforms.RandomResizedCrop(224),  # 随机裁剪到224x224大小
        #     transforms.RandomHorizontalFlip(),  # 随机水平翻转
        #     transforms.RandomRotation(degrees=15),  # 随机旋转
        #     transforms.ToTensor(),  # 转化成Tensor
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
        # ])

        image = Image.open(image_path)
        image = train_transform(image)
        # print(image.shape)
        image = image.unsqueeze(dim=0)
        # print(image.shape)
        # images.append(image)

        # images = np.array(images).astype(np.float32)
        # print(images.shape)
        # images = torch.from_numpy(images)
        # images = images.transpose(1, 3).contiguous()

        image = image.to(device)

        # print(images)

        vote_correct = 0
        mlps_correct = [0 for i in range(len(self.mlps))]

        pre_labels=[]
        for i, mlp in enumerate(self.mlps,0):
            mlp.eval()
            out = mlp(image)
            _, prediction = torch.max(out, 1)
            pre_labels.append(prediction.item())

        maxlabel = max(pre_labels, key=pre_labels.count)


        # outputs = self.model(image)
        # # print(outputs,outputs.shape)
        # outputs = F.softmax(outputs)

        # index = outputs.argmax(dim=1)
        # print(index[0].item())
        # print(outputs)
        # print(index[0].item()==62)
        # print(index[0].item())
        # print(outputs)
        # print(image_path)
        return {"label": maxlabel}


if __name__ == "__main__":
    pre = Prediction()
    pre.load_model()
    image_path = "data/input/Food2/"
    label_path = "data/input/Food2/train.csv"
    index=0
    rightnum=0

    with open(label_path) as f:
        f_csv = list(csv.reader(f))

    for row in f_csv[1:]:
        # 处理图片
        index=index+1
        print("---")

        print("true label:",int(row[1]))
        if int(pre.predict(image_path + row[0])["label"])==int(row[1]):
            rightnum+=1
        print("{}/{},{}%".format(rightnum,index,float(rightnum/index*100)))
