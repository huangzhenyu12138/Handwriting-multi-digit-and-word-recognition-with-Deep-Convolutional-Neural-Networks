import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库模块
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
# torch.manual_seed(1)  # reproducible
EPOCH = 20  # 30  # 训练整批数据次数，训练次数越多，精度越高，为了演示，我们训练5次
BATCH_SIZE = 128  # 128# 每次训练的数据集个数
LR = 0.001  # 学习效率
DOWNLOAD_MNIST = False  # 如果你已经下载好了EMNIST数据就设置 False
k_folds = 10  # 10# 或者您想要的折叠数
kfold = KFold(n_splits=k_folds, shuffle=True)
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
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False,
    split='letters'
)
# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# 每一步 loader 释放50个数据用来学习
# 为了演示, 我们测试时提取2000个数据先
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_x = torch.unsqueeze(test_data.data, dim=1).type(
    torch.FloatTensor)[:5000] / 255.
test_y = test_data.targets[:5000]  # 5000
# test_x = test_x.cuda() # 若有cuda环境，取消注释
# test_y = test_y.cuda() # 若有cuda环境，取消注释


def get_mapping(num, with_type='letters'):
    """
    根据 mapping，由传入的 num 计算 UTF8 字符
    """
    if with_type == 'byclass':
        if num <= 9:
            return chr(num + 48)  # 数字
        elif num <= 35:
            return chr(num + 55)  # 大写字母
        else:
            return chr(num + 61)  # 小写字母

    elif with_type == 'letters':
        return chr(num + 64) + " / " + chr(num + 96)  # 大写/小写字母
    elif with_type == 'digits':
        return chr(num + 96)
    else:
        return num


figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    print(label)
    figure.add_subplot(rows, cols, i)
    plt.title(get_mapping(label))
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
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
    # val_loader = Data.DataLoader(
    #     dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 初始化模型
    cnn = CNN()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    # 存储每次折叠的训练损失和测试准确性的列表
    train_loss_list = []
    test_accuracy_list = []
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

    # # 为每次折叠的测试准确性绘制图表
    # plt.plot(test_accuracy_list, label=f'Fold {fold + 1} - Test Accuracy')
    # plt.title(f'Fold {fold + 1} - Test Accuracy')
    # plt.xlabel('Step')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()
    # # 存储每次折叠的训练损失和测试准确性
    # all_train_loss_list.append(train_loss_list)
    # all_test_accuracy_list.append(test_accuracy_list)
    # 存储每次折叠的训练损失和测试准确性
    all_train_loss_list.append(train_loss_list)
    all_test_accuracy_list.append(test_accuracy_list)

# test 神经网络
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.squeeze()
# 若有cuda环境，使用92行，注释90行
# pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
# save CNN
# 仅保存CNN参数，速度较快
torch.save(cnn.state_dict(), './model/CNN_lettertest.pk')
# 保存CNN整个结构
# torch.save(cnn(), './model/CNN.pkl')

# 为所有折叠绘制训练损失的图表
for fold in range(k_folds):
    plt.plot(all_train_loss_list[fold],
             label=f'Fold {fold + 1} - Training Loss')

plt.title('Training Loss - All Folds')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 为所有折叠绘制测试准确性的图表
for fold in range(k_folds):
    plt.plot(all_test_accuracy_list[fold],
             label=f'Fold {fold + 1} - Test Accuracy')

plt.title('Test Accuracy - All Folds')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
