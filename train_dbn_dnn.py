import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from model.DBN_model import DBN

# 准备Mnist训练数据集
train_data = torchvision.datasets.MNIST(root="./data",
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=False)

# 加载Mnist训练数据集的数据并切片
train_loader = DataLoader(dataset=train_data,
                          batch_size=1000,
                          shuffle=True)

# 创建网络模型
dbn = DBN()
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# cuda
if torch.cuda.is_available():
    dbn = dbn.cuda()
    criterion = criterion.cuda()

# 优化器
learning_rate = 0.005   # 学习率
optimizer = optim.Adam(dbn.parameters(), lr=learning_rate)

# 绘图x，y初始化
X = []  # 训练次数epoch
Y = []  # 损失值loss

# 训练数据并统计准确率
epoch = 0
while True:
    correct = 0     # 预测正确的数量
    total = 0       # 训练数据的数量
    for data in train_loader:

        # 数据预处理
        inputs, labels = data    # 获取数据和标签
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs = inputs.view(-1, 784)   # 展平张量inputs

        # 调整参数
        optimizer.zero_grad()    # 清空过往梯度
        outputs = dbn(inputs)    # 获得输出值
        loss = criterion(outputs, labels)
        loss.backward()          # 反向传播,计算当前梯度
        optimizer.step()         # 根据梯度更新网络参数

        # 统计epoch和accuracy
        _, predicted = torch.max(outputs.data, 1)       # 获取预测值
        total += labels.size(0)  # 统计总数
        correct += (predicted == labels).sum().item()   # 统计预测正确的数量
        X.append(epoch)          # 将训练次数epoch写入X轴
        Y.append(loss.item())    # 将损失值loss写入Y轴
        print(loss)              # 输出loss的tensor数据类型
        epoch += 1

    print('epoch: {0:d},accuracy: {1:f}'.format(epoch, correct / total))
    if (correct / total) >= 0.999:
        break
print("训练结束")

# 保存训练后的参数
torch.save(dbn.state_dict(), "dbn.pth")

# 绘图
plt.title("DBN-DNN Model")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(X, Y)
plt.show()
