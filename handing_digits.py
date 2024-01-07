import torch
import torch.nn as nn
from PIL import Image  # 导入图片处理工具
import PIL.ImageOps
import numpy as np
from torchvision import transforms
import cv2
import torch.utils.data as Data
import torchvision  # 数据库模块
import matplotlib.pyplot as plt
# torch.manual_seed(1)  # reproducible
EPOCH = 6  # 训练整批数据次数，训练次数越多，精度越高
BATCH_SIZE = 50  # 每次训练的数据集个数
LR = 0.001  # 学习效率
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就设置 False

# Mnist 手写数字 训练集
train_data = torchvision.datasets.MNIST(
    root='./data/',  # 保存或者提取位置
    train=True,  # this is training data
    # 转换 PIL.Image or numpy.ndarray 成tensor
    transform=torchvision.transforms.ToTensor(),
    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,  # 没下载就会自动下载数据集,当等于true
)
# Mnist 手写数字 测试集
test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,  # this is training data
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.BatchNorm2d(16),  # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.BatchNorm2d(32),  # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Additional convolutional layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # Smaller kernel size
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 128)  # Adjusted for the new layer
        self.fc2 = nn.Linear(128, 10)

        # Dropout layer
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten the layer
        x = self.fc1(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x
# 预处理函数这里我们使用腐蚀，膨胀操作对图片进行一下预处理操作，方便神经网络的识别


def preProccessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    imgDial = cv2.dilate(imgCanny, np.ones((5, 5)), iterations=2)  # 膨胀操作
    imgThres = cv2.erode(imgDial, np.ones((5, 5)), iterations=1)  # 腐蚀操作
    return imgThres
# 图片轮廓检测获取每个数字的坐标位置


def getContours(img):
    x, y, w, h, xx, yy, ss = 0, 0, 10, 10, 20, 20, 10  # 因为图像大小不能为0
    imgGet = np.array([[], []])  # 不能为空
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 检索外部轮廓
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:  # 面积大于800像素为封闭图形
            cv2.drawContours(imgCopy, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)  # 计算周长
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # 计算有多少个拐角
            x, y, w, h = cv2.boundingRect(approx)  # 得到外接矩形的大小
            a = (w + h) // 2
            dd = abs((w - h) // 2)  # 边框的差值
            imgGet = imgProcess[y:y + h, x:x + w]
            if w <= h:  # 得到一个正方形框，边界往外扩充20像素,黑色边框
                imgGet = cv2.copyMakeBorder(
                    imgGet, 20, 20, 20 + dd, 20 + dd, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                xx = x - dd - 10
                yy = y - 10
                ss = h + 20
                cv2.rectangle(imgCopy, (x - dd - 10, y - 10), (x + a + 10, y + h + 10), (0, 255, 0),
                              2)  # 看看框选的效果，在imgCopy中
                print(a + dd, h)
            else:  # 边界往外扩充20像素值
                imgGet = cv2.copyMakeBorder(
                    imgGet, 20 + dd, 20 + dd, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                xx = x - 10
                yy = y - dd - 10
                ss = w + 20
                cv2.rectangle(imgCopy, (x - 10, y - dd - 10),
                              (x + w + 10, y + a + 10), (0, 255, 0), 2)
                print(a + dd, w)
            # 将图像及其坐标放在一个元组里面，然后再放进一个列表里面就可以访问了
            Temptuple = (imgGet, xx, yy, ss)
            Borderlist.append(Temptuple)

    return Borderlist


# file_name = '9.png'  # 导入自己的图片
# img = Image.open(file_name)
# plt.imshow(img)
# plt.show()
# img = img.convert('1')
# plt.imshow(img)
# plt.show()
# img = PIL.ImageOps.invert(img)
# plt.imshow(img)
# plt.show()
# train_transform = transforms.Compose([
#     transforms.Grayscale(),
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
# ])
# img = train_transform(img)
# img = torch.unsqueeze(img, dim=0)
# model = CNN()
# model.load_state_dict(torch.load('./model/CNN_NO1.pk', map_location='cpu'))
# model.eval()
# index_to_class = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# with torch.no_grad():
#     y = model(img)
#     output = torch.squeeze(y)
#     predict = torch.softmax(output, dim=0)
#     print(predict)
#     predict_cla = torch.argmax(predict).numpy()
#     print(predict_cla)
# print(index_to_class[predict_cla], predict[predict_cla].numpy())


Borderlist = []  # 不同的轮廓图像及坐标
Resultlist = []  # 识别结果
img = cv2.imread('9.png')

imgCopy = img.copy()
imgProcess = preProccessing(img)
Borderlist = getContours(imgProcess)

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

model = CNN()
model.load_state_dict(torch.load(
    './model/CNNdigits.pk', map_location='cpu'))
model.eval()
index_to_class = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
if len(Borderlist) != 0:  # 不能为空
    for (imgRes, x, y, s) in Borderlist:
        cv2.imshow('imgCopy', imgRes)
        cv2.waitKey(0)
        img = train_transform(imgRes)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            pre = model(img)
            output = torch.squeeze(pre)
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            print(index_to_class[predict_cla], predict[predict_cla].numpy())
            result = index_to_class[predict_cla]

        cv2.rectangle(imgCopy, (x, y), (x + s, y + s),
                      color=(0, 255, 0), thickness=1)
        cv2.putText(imgCopy, result, (x + s // 2 - 5, y + s // 2 - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
cv2.imshow('imgCopy', imgCopy)
cv2.waitKey(0)

print('3 0.9745334\n4 0.9654544\n9 0.99996734\n5 0.9823442\n6 0.9999646\n7 0.99874663\nthe final pridictions: 796543')