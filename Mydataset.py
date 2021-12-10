import torch.utils.data as Data
import random
import csv
import cv2
import torch
import numpy as np
import os
# import psutil
from PIL import Image
from torchvision import transforms
class MyDataset(Data.Dataset):
    def __init__(self,transform=None):
        """
        file_path: the path to the dataset file
        nraws: each time put nraws sample into memory for shuffle
        shuffle: whether the data need to shuffle
        """
        self.transform = transform
        self.images = []
        self.labels = []
        self.image_width, self.image_height = 224, 224
        self.file_rows = 0
        # self.image_path = "data/input/Food2/"
        # self.label_path = "data/input/Food2/train.csv"
        self.image_path = "data/input/Food2/3/output/train/"
        self.label_path = "data/input/Food2/3/output/train/train.csv"
        f_csv = []
        with open(self.label_path) as f:
            f_csv = list(csv.reader(f))
            self.file_rows = len(f_csv) - 1 # 有一行是无效的
            # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

        # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        # put num_rows samples into memory
        print(self.file_rows)
        for row in f_csv[1:]:
            # 处理图片
            self.images.append(self.image_path + row[0]) #文件路径
            self.labels.append(int(row[1]))
            # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

        # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        # print(self.images, self.labels)

    def __len__(self):
        return self.file_rows

    def __getitem__(self, index):
        # print(index)
        image, label = self.images[index], self.labels[index]
        image = Image.open(image)
        # image = cv2.imread(image)
        # print(image.size)

        if self.transform is not None:
            image = self.transform(image)

        # print(image,image.shape)
        # image = cv2.imread(image)
        # image = cv2.resize(image, (self.image_width, self.image_height), 0, 0, cv2.INTER_LINEAR)
        # image = image.astype(np.float32)
        # image = np.multiply(image, 1. / 255.)
        #
        # one_hot_label = torch.zeros(1, 101)
        # index = torch.LongTensor([int(label)]).view(-1, 1)
        # one_hot_label.scatter_(dim=1, index=index, value=1)
        # one_hot_label = one_hot_label.numpy()
        #
        # # images, labels = self.images[index], self.labels[index]
        # # # __getitem__中读取文件，随取随用，避免内存占用过大
        # # # print(f"读取数据:{index}")  # 压根就没执行好吧
        # image = np.array(image).astype(np.float32)
        # one_hot_label = np.array(one_hot_label).astype(np.float32)
        # image = torch.from_numpy(image)
        # one_hot_label = torch.from_numpy(one_hot_label)
        # image = image.transpose(0, 2).contiguous()
        return image, label

    def getWeight(self):
        _datasetLabel=np.zeros((101,1))
        for item in self.labels:
            _datasetLabel[int(item)]=_datasetLabel[int(item)]+1

        _datasetLabel=torch.from_numpy(_datasetLabel)
        _datasetLabel=1/_datasetLabel
        mysample=[]
        for item in self.labels:
            mysample.append(_datasetLabel[int(item)])


        return _datasetLabel

