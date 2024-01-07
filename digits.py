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
from sklearn.model_selection import KFold
from torch.utils.data.dataset import random_split
k_folds = 5  # 或者您想要的折叠数
kfold = KFold(n_splits=k_folds, shuffle=True)
# torch.manual_seed(1)  # reproducible
EPOCH = 5  # 训练整批数据次数，训练次数越多，精度越高
BATCH_SIZE = 50  # 每次训练的数据集个数
LR = 0.001  # 学习效率
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就设置 False
# 早停初始化
best_val_loss = float('inf')
patience = 6  # 比如等待5个epochs
min_improvement = 0.01  # 比如至少需要1%的改进
patience_counter = 0
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
# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# 每一步 loader 释放50个数据用来学习
# 为了演示, 我们测试时提取2000个数据先
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_x = torch.unsqueeze(test_data.data, dim=1).type(
    torch.FloatTensor)[:2000] / 255.
test_y = test_data.targets[:2000]
# test_x = test_x.cuda() # 若有cuda环境，取消注释
# test_y = test_y.cuda() # 若有cuda环境，取消注释
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
        self.fc2 = nn.Linear(128, 10)

        # Dropout layer
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten the layer
        x = self.fc1(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x
def validate(model, data_loader, loss_func):
    model.eval()  # 设置为评估模式
    total_loss = 0
    with torch.no_grad():  # 在评估阶段不计算梯度
        for data, target in data_loader:
            outputs = model(data)
            loss = loss_func(outputs, target)
            total_loss += loss.item()
    average_loss = total_loss / len(data_loader)
    return average_loss


# 设置验证集的大小，例如验证集占总训练集的 20%
val_size = int(0.2 * len(train_data))
train_size = len(train_data) - val_size
# 随机分割数据集为训练集和验证集
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])
# 存储每次折叠的训练损失和测试准确性
all_train_loss_list = []
all_test_accuracy_list = []
for fold, (train_ids, test_ids) in enumerate(kfold.split(train_data)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # 抽取训练数据和测试数据
    train_subsampler = Data.SubsetRandomSampler(train_ids)
    test_subsampler = Data.SubsetRandomSampler(test_ids)

    # 创建数据加载器
    train_loader = Data.DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, sampler=train_subsampler)
    test_loader = Data.DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, sampler=test_subsampler)
    val_loader = Data.DataLoader(
        dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 初始化模型
    cnn = CNN()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    # 训练和验证过程
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # 每一步 loader 释放50个数据用来学习
            output = cnn(b_x)  # 输入一张图片进行神经网络训练
            loss = loss_func(output, b_y)  # 计算神经网络的预测值与实际的误差
            optimizer.zero_grad()  # 将所有优化的torch.Tensors的梯度设置为零
            loss.backward()  # 反向传播的梯度计算
            optimizer.step()  # 执行单个优化步骤
            if step % 50 == 0:  # 我们每50步来查看一下神经网络训练的结果
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = float((pred_y == test_y).sum()) / \
                    float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data,
                      '| test accuracy: %.2f' % accuracy)
                # 存储每次折叠的训练损失和测试准确性的列表
                train_loss_list = []
                test_accuracy_list = []
        val_loss = validate(cnn, val_loader, loss_func)
        print(f'Epoch {epoch}, Validation Loss: {val_loss}')
        # # 检查是否有改进
        # if val_loss < best_val_loss - min_improvement:
        #     best_val_loss = val_loss
        #     patience_counter = 0  # 重置计数器
        #     # 保存最佳模型
        #     torch.save(cnn.state_dict(), './model/CNN_digitsBest.pk')
        # else:
        #     patience_counter += 1

        # # 检查是否需要早停
        # if patience_counter >= patience:
        #     print("Stopping early at epoch", epoch)
        #     break

# 加载最佳模型
# cnn.load_state_dict(torch.load('./model/CNN_digitsBest.pk'))
# test 神经网络
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
# save CNN
# # # 仅保存CNN参数，速度较快
# torch.save(cnn.state_dict(), './model/CNNdigits.pk')
# 保存CNN整个结构
torch.save(cnn, './model/CNNdigits1.pk')
# 每个fold的性能评估
# ...
# 存储每次折叠的训练损失和测试准确性
all_train_loss_list = []
all_test_accuracy_list = []

# 训练和验证过程
for fold, (train_ids, test_ids) in enumerate(kfold.split(train_data)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # ... 其他代码 ...

    # 存储每次折叠的训练损失和测试准确性的列表
    train_loss_list = []
    test_accuracy_list = []

    # 训练和验证过程
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            # ... 其他代码 ...
            if step % 50 == 0:
                # ... 其他代码 ...

                # 记录训练损失和测试准确性
                train_loss_list.append(loss.item())
                test_accuracy_list.append(accuracy)

    # # 为每次折叠的训练损失绘制图表
    # plt.plot(train_loss_list, label=f'Fold {fold + 1} - Training Loss')
    # plt.title(f'Fold {fold + 1} - Training Loss')
    # plt.xlabel('Step')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    #
    # # 为每次折叠的测试准确性绘制图表
    # plt.plot(test_accuracy_list, label=f'Fold {fold + 1} - Test Accuracy')
    # plt.title(f'Fold {fold + 1} - Test Accuracy')
    # plt.xlabel('Step')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()
    #
    # # 存储每次折叠的训练损失和测试准确性
    # all_train_loss_list.append(train_loss_list)
    # all_test_accuracy_list.append(test_accuracy_list)

# ... 其他代码 ...

# 为所有折叠绘制训练损失的图表
for fold in range(k_folds):
    plt.plot(all_train_loss_list[fold], label=f'Fold {fold + 1} - Training Loss')

plt.title('Training Loss - All Folds')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 为所有折叠绘制测试准确性的图表
for fold in range(k_folds):
    plt.plot(all_test_accuracy_list[fold], label=f'Fold {fold + 1} - Test Accuracy')
plt.title('Test Accuracy - All Folds')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.legend()
plt.show()