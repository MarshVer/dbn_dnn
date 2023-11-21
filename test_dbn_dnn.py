import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from model.DBN_model import DBN

# 准备Mnist测试数据集的数据
test_data = torchvision.datasets.MNIST(root="./data",
                                       train=False,
                                       transform=transforms.ToTensor(),
                                       download=False)
# 加载Mnist测试数据集的数据并切片
test_loader = DataLoader(test_data, batch_size=1000)

# 创建DBN模型
dbn_test = DBN()

# 加载训练后的参数
dbn_test.load_state_dict(torch.load("dbn.pth"))

# 初始化总数和预测成功的数量
total = 0       # 测试数据的数量
correct = 0     # 预测正确的数量

# 开始测试
for data in test_loader:
    inputs, labels = data                           # 获取数据和标签
    inputs = inputs.view(-1, 784)                   # 展平张量inputs
    outputs = dbn_test(inputs)                      # 获得输出值
    _, predicted = torch.max(outputs.data, 1)       # 从输出值中获取预测值
    total += labels.size(0)                         # 统计总数
    correct += (predicted == labels).sum().item()   # 统计预测值正确的数量
    print('correct: {0:d}, total: {1:d}'.format(correct, total))
print('accuracy: {0:f}'.format(correct / total))
