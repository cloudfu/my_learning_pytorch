import torch
from torch import nn
from torch.utils import data
from enum import Enum


def generate_data(example_count, w, b):
    # x.shape = (1000,2)
    x = torch.normal(0, 1, (example_count, len(w)))

    # y.shape = (1,1000)
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)

    # y.reshape = (1000,1)
    return x, y.reshape((-1, 1))


def load_batch_data(data_arrays, batch_size, is_shuffle=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,is_shuffle)


class linear_regression_line(object):

    def __init__(self):
        self.nn_model = ""
        self.loss_function = ""
        self.optimizer = ""
        self.learning_rate = 0.03
        self.example_data = ""
        self.data_iter = ""

        # 训练数据集 - 训练参数准备
        self.true_w = torch.tensor([-2.2, 3.14])
        self.true_b = 3
        batch_size = 10
        example_count = 1000

        # 生成训练数据
        self.features, self.labels = generate_data(example_count, self.true_w, self.true_b)
        self.data_iter = load_batch_data((self.features, self.labels), batch_size)

        # init parameters
        self.__init_nn__()
        self.__init_loss_function__()
        self.__init_optimizer__()

        print(self.nn_model)

    def __init_nn__(self):
        # 线性神经模型
        # linear.Linear(2, 1) 输入两个神经元，输出一个结果；
        self.nn_model = nn.Sequential(nn.Linear(2, 1))

    def __init_loss_function__(self):
        # 均方差损失函数
        self.loss_function = nn.MSELoss()

    def __init_optimizer__(self):
        # 定义优化器:小批量随机梯度下降算法
        self.optimizer = torch.optim.SGD(self.nn_model.parameters(), lr=self.learning_rate)

    def train(self):
        num_epochs = 3
        for epoch in range(num_epochs):
            for features_iter, labels_iter in self.data_iter:
                prediction_label = self.nn_model(features_iter)
                loss = self.loss_function(prediction_label, labels_iter)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 每轮批次训练完成看下损失率
            epoch_loss = self.loss_function(self.nn_model(self.features), self.labels)
            print(f'epoch {epoch + 1}, loss {epoch_loss:f}')

        # 训练完成打印变量预估值
        # 由于使用线性模型 input=2,output=1，所以有2个线性函数作为入参，对应2个weight
        # 但是为什么bias只有1个？？对应输出节点，bias
        w = self.nn_model[0].weight.data
        print('prediction w：', w.reshape(self.true_w.shape))
        b = self.nn_model[0].bias.data
        print('prediction b：', b)


test = linear_regression_line()
test.train()
