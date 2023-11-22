import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from model.DBN_model import DBN
from model.RBM_model import RBM

# 准备Mnist训练数据集
train_data = torchvision.datasets.MNIST(root="./data",
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=False)

# 加载Mnist训练数据集的数据并切片
train_loader = DataLoader(dataset=train_data,
                          batch_size=100,
                          shuffle=True)

# 初始化三层RBM
rbm1 = RBM(784, 500)
rbm2 = RBM(500, 200)
rbm3 = RBM(200, 50)
# cuda
if torch.cuda.is_available():
    rbm1 = rbm1.cuda()
    rbm2 = rbm2.cuda()
    rbm3 = rbm3.cuda()

# rbm训练次数
epochs = 10

# 训练rbm1
for epoch in range(epochs):
    for data in train_loader:
        inputs, _ = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        inputs = inputs.view(-1, 784)
        inputs = inputs.bernoulli()
        loss = rbm1.contrastive_divergence(inputs, 0.0001)
        loss = abs(loss)
    print('RBM1: epoch: {0:d}/{1:d}, loss: {2:f}'.format(epoch+1, epochs, loss))
print(f"RBM1 trained.")


# 训练rbm2
for epoch in range(epochs):
    for data in train_loader:
        inputs, _ = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        inputs = inputs.view(-1, 784)
        inputs = inputs.bernoulli()
        _, output1 = rbm1.forward(inputs)       # rbm1的输出当作rbm2的输入
        loss = rbm2.contrastive_divergence(output1, 0.0001)
        loss = abs(loss)
    print('RBM2: epoch: {0:d}/{1:d}, loss: {2:f}'.format(epoch+1, epochs, loss))
print("RBM2 trained")

for epoch in range(epochs):
    for data in train_loader:
        inputs, _ = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        inputs = inputs.view(-1, 784)
        inputs = inputs.bernoulli()

        _, output1 = rbm1.forward(inputs)       # rbm1的输出当作rbm2的输入
        _, output2 = rbm2.forward(output1)      # rbm2的输出当作rbm3的输入
        loss = rbm3.contrastive_divergence(output2, 0.0001)
        loss = abs(loss)
    print('RBM3: epoch: {0:d}/{1:d}, loss: {2:f}'.format(epoch+1, epochs, loss))
print("RBM3 trained")

# 创建网络模型
dbn = DBN()

# 将三层训练后的rbm参数赋值给dbn
dbn.rbm1 = rbm1
dbn.rbm2 = rbm2
dbn.rbm3 = rbm3

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# cuda
if torch.cuda.is_available():
    dbn = dbn.cuda()
    criterion = criterion.cuda()

# 优化器
learning_rate = 0.001  # 学习率
optimizer = optim.Adam(dbn.parameters(), lr=learning_rate)

# 绘图x，y初始化
X = []  # 训练次数epoch
Y = []  # 损失值loss

# 训练数据并统计准确率
epoch = 0
count = 1
while epoch <= 500:
    correct = 0  # 预测正确的数量
    total = 0  # 训练数据的数量
    for data in train_loader:

        # 数据预处理
        inputs, labels = data  # 获取数据和标签
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs = inputs.view(-1, 784)  # 展平张量inputs

        # 调整参数
        optimizer.zero_grad()  # 清空过往梯度
        outputs = dbn(inputs)  # 获得输出值
        loss = criterion(outputs, labels)
        loss.backward()  # 反向传播,计算当前梯度
        optimizer.step()  # 根据梯度更新网络参数

        # 统计epoch和accuracy
        _, predicted = torch.max(outputs.data, 1)  # 获取预测值
        total += labels.size(0)     # 统计总数
        correct += (predicted == labels).sum().item()  # 统计预测正确的数量
        X.append(count)             # 将训练次数epoch写入X轴
        Y.append(loss.item())       # 将损失值loss写入Y轴
        count += 1
    epoch += 1
    print('DBM: epoch: {0:d},accuracy: {1:f}，loss: {2:f}'.format(epoch, correct / total, loss.item()))
    if (correct / total) >= 0.98:
        break
print("训练结束")

# 保存训练后的参数
torch.save(dbn.state_dict(), "dbn.pth")

# 绘图
plt.title("DBN-DNN Model")
plt.xlabel("count")
plt.ylabel("loss")
plt.plot(X, Y)
plt.show()
