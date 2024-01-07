import torch
import torch.nn as nn
from PIL import Image  # 导入图片处理工具
import PIL.ImageOps
import numpy as np
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import torchvision  # 数据库模块
# torch.manual_seed(1)  # reproducible
EPOCH = 5  # 训练整批数据次数，训练次数越多，精度越高，为了演示，我们训练5次
BATCH_SIZE = 50  # 每次训练的数据集个数
LR = 0.001  # 学习效率
DOWNLOAD_MNIST = False  # 如果你已经下载好了EMNIST数据就设置 False

# EMNIST 手写字母 训练集
train_data = torchvision.datasets.EMNIST(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
    split='letters'
)
# EMNIST 手写字母 测试集
test_data = torchvision.datasets.EMNIST(
    root='./data',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False,
    split='letters'
)
# 定义神经网络


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
        self.fc2 = nn.Linear(128, 37)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten the layer
        x = self.fc1(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x
# 预处理函数


def preProccessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    imgDial = cv2.dilate(imgCanny, np.ones((5, 5)), iterations=2)  # 膨胀操作
    imgThres = cv2.erode(imgDial, np.ones((5, 5)), iterations=1)  # 腐蚀操作
    return imgThres


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


Borderlist = []  # 不同的轮廓图像及坐标
Resultlist = []  # 识别结果
img = cv2.imread('14.png')

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
    './model/CNN_lettertest.pk', map_location='cpu'))
model.eval()

# file_name = '55.png'  # 导入自己的图片
# img = Image.open(file_name)
# img = img.convert('L')

# img = PIL.ImageOps.invert(img)
# img = img.transpose(Image.FLIP_LEFT_RIGHT)
# img = img.rotate(90)

# plt.imshow(img)
# plt.show()

# train_transform = transforms.Compose([
#     transforms.Grayscale(),
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
# ])

# img = train_transform(img)
# img = torch.unsqueeze(img, dim=0)
# # torch.unsqueeze()这个函数主要是对数据维度进行扩充。
# # 加载模型
# model = CNN()
# model.load_state_dict(torch.load('./model/CNN_letter.pk', map_location='cpu'))
# model.eval()


def get_mapping(num, with_type='letters'):
    """
    根据 mapping，由传入的 num 计算 UTF8 字符。
    """
    if with_type == 'byclass':
        if num <= 9:
            return chr(num + 48)  # 数字
        elif num <= 35:
            return chr(num + 55)  # 大写字母
        else:
            return chr(num + 61)  # 小写字母
    elif with_type == 'letters':
        return chr(num + 96)  # 大写/小写字母
        # return chr(num + 64) + " / " + chr(num + 96)  # 大写/小写字母
    elif with_type == 'digits':
        return chr(num + 96)
    else:
        return num


# with torch.no_grad():
#     y = model(img)
#     print(y)
#     output = torch.squeeze(y)
#     print(output)
#     predict = torch.softmax(output, dim=0)
#     print(predict)
#     predict_cla = torch.argmax(predict).numpy()
#     print(predict_cla)
# print(get_mapping(predict_cla), predict[predict_cla].numpy())
# 对 Borderlist 按照 x 的值进行排序
sorted_borderlist = sorted(Borderlist, key=lambda item: item[1])
# 在循环外创建一个空列表，用于存储所有预测结果
all_predictions = []
print("Prediction of for each letter:")
if len(sorted_borderlist) != 0:  # 不能为空
    prev_x = None  # 用于跟踪前一个标签的 x 坐标
    for (imgRes, x, y, s) in sorted_borderlist:
        cv2.imshow('imgCopy', imgRes)
        cv2.waitKey(0)
        imgRes = cv2.flip(imgRes, 1)
        (h, w) = imgRes.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), 90, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        imgRes = cv2.warpAffine(imgRes, M, (nW, nH))
        cv2.imshow('imgThres', imgRes)
        cv2.waitKey(0)
        img = train_transform(imgRes)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            pre = model(img)
            output = torch.squeeze(pre)
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            # 获取当前预测结果并添加到列表中
            result = get_mapping(predict_cla)
            all_predictions.append(f"{result}")
            # print(get_mapping(predict_cla), end='')
            print(get_mapping(predict_cla),
                  predict[predict_cla].numpy())  # 最终的输出结果
            result = get_mapping(predict_cla)
        cv2.rectangle(imgCopy, (x, y), (x + s, y + s),
                      color=(0, 255, 0), thickness=1)
        # 计算文本标签的位置，使其左右相邻
        text_position = (x + 10, 50)  # 调整此值以确保文字标签在矩形的上方
        cv2.putText(imgCopy, result, text_position,
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)

# 使用join方法将列表中的字符串连接起来，去掉中间的空格
all_predictions_str = ''.join(all_predictions)
# 在循环结束后输出所有的最终预测结果
print("The final predictions:", all_predictions_str)
cv2.imshow('imgCopy', imgCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()
imgRes = cv2.flip(imgRes, 1)
(h, w) = imgRes.shape[:2]
(cX, cY) = (w // 2, h // 2)
M = cv2.getRotationMatrix2D((cX, cY), 90, 1.0)
cos = np.abs(M[0, 0])
sin = np.abs(M[0, 1])
nW = int((h * sin) + (w * cos))
nH = int((h * cos) + (w * sin))
M[0, 2] += (nW / 2) - cX
M[1, 2] += (nH / 2) - cY
imgRes = cv2.warpAffine(imgRes, M, (nW, nH))
