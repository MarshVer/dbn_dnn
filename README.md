# dbn_dnn
* Pytorch框架下使用常用的MINIST数据集，实现DBN识别手写数字。
* DBN-DNN结构，其主要由三层RBM和一层BP组成。在这个例子中，使用了3个RBM：第一个的隐含层单元个数为500，第二个RBM的隐含层单元个数为200，最后一个为50。
* 代码使用了cuda，且保存了模型训练的参数到dbn.pth

# 文件目录
## data
* data文件夹是MINIST数据集
## model
* model文件夹是DBN和RBM模型
## train_dbn_dnn.py
* train_dbn_dnn.py是训练数据的python代码
## test_dbn_dnn.py
* est_dbn_dnn.py是册数数据的python
## dbn.pth
* dbn.pth是训练数据后保存的模型参数
