# dbn_dnn
* Pytorch框架下实现dbn_dnn
* 该深度置信网络有三层RBM和一层输出层，RBM层依次为（784，500）（500，200）（200，50）；输出层为（50，10）

# 文件目录
## data
* MNIST数据集

# model
* DBN_model：DBN模型
* RBM——model：RBM模型（包括CD算法）

## dbn.pth
* 数据训练后的DBN模型参数
## test_dbn_dnn
* MNIST数据集测试正确率

## train_dbn_dnn
* dbn_dnn训练MNIST数据集
