import torch
from torch import nn
from torch.utils import data


def load_batch_data(data_arrays, batch_size, is_shuffle=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, is_shuffle)


# 通过二元一次方程对于sin函数进行模拟
class PolynomialNet(nn.Module):
    def __init__(self):
        super(PolynomialNet, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.a * x ** 2 + self.b * x + self.c

    def toString(self):
        return f'y = {self.a.item()} x**2 + {self.b.item()} x + {self.c.item()}'


def __generate_data__(example_count, start, end):
    x = torch.linspace(start, end, example_count)
    y = torch.sin(x)
    return x, y


class linear_regression_parameters(object):
    def __init__(self):

        self.nn_model = ""
        self.loss_function = ""
        self.optimizer = ""
        self.learning_rate = 0.03

        self.example_data = ""
        self.data_iter = ""

        # 训练数据集 - 训练参数准备
        x_region = [-3.0, 1.1]
        batch_size = 10
        example_count = 1000

        # 生成训练数据
        self.features, self.labels = __generate_data__(example_count, x_region[0], x_region[1])
        self.data_iter = load_batch_data((self.features, self.labels), batch_size)

        # init parameters
        self.__init_nn__()
        self.__init_loss_function__()
        self.__init_optimizer__()
        print(self.nn_model)

    def __init_nn__(self):
        # 线性神经模型
        self.nn_model = PolynomialNet()

    def __init_loss_function__(self):
        # 均方差损失函数
        self.loss_function = nn.MSELoss()

    def __init_optimizer__(self):
        # 定义优化器:小批量随机梯度下降算法
        self.optimizer = torch.optim.SGD(self.nn_model.parameters(), lr=self.learning_rate)

    def train(self):
        # 开始进行训练
        num_epochs = 3
        for epoch in range(num_epochs):
            for features_iter, labels_iter in self.data_iter:
                prod_label = self.nn_model(features_iter)
                loss = self.loss_function(prod_label, labels_iter)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 每轮批次训练完成看下损失率
            epoch_loss = self.loss_function(self.nn_model(self.features), self.labels)
            print(f'epoch {epoch + 1}, loss {epoch_loss:f}')

        print(self.nn_model.toString())


test = linear_regression_parameters()
test.train()
