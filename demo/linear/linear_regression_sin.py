import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_output):
        super(NeuralNet, self).__init__()

        # 设置一层隐藏层，包括 n_hidden 神经元
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.hidden4 = torch.nn.Linear(n_hidden3, n_hidden4)
        self.out = torch.nn.Linear(n_hidden4, n_output)

    def forward(self, x):
        ###############################
        # 训练数据在各个神经层之间的维度变化
        ###############################
        # 训练数据：可以作为横向数据矩阵：(10,1)
        # [[a],[b],[c],[d],[...]]

        # 第一层为10神经元，可以作为竖向矩阵:(1,10)
        # weight:[[a,b,c,d,e,f...]]
        # bias:[[a,b,c,d,e,f...]]
        # 简单理解：
        #   行：训练数据  列：神经元个数

        ###############################
        # weight、bias和神经元之间的关系
        ###############################
        # 每个神经元配置 一个weight和bias，可以通过self.hidden2.weight[i]进行查看

        # 训练数据 x.shape:(3,1)
        # 后续根据每层的神经元对于二维数据（列）进行变更/扩展

        # hidden1 = torch.linear.Linear(1, 6)
        # hidden1.weight.shape:(6,1)
        # hidden1.bias.shape:(6)
        x = F.relu(self.hidden1(x))
        # x.shape:(3,6)

        # hidden2 = torch.linear.Linear(6, 9)
        # hidden2.weight.shape:(9,6)
        # hidden2.bias.shape:(9)
        x = F.relu(self.hidden2(x))
        # x.shape:(3,9)

        # hidden3 = torch.linear.Linear(9, 12)
        # hidden3.weight.shape:(12,9)
        # hidden3.bias.shape:(12)
        x = F.relu(self.hidden3(x))
        # x.shape:(3,12)

        x = F.relu(self.hidden4(x))
        # x.shape:(3,16)

        x = self.out(x)
        # x.shape：(3,1)
        return x


def load_batch_data(data_arrays, batch_size, is_shuffle=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, is_shuffle)


def __generate_data__(example_count, start, end):
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())
    return x, y


class linear_regression_sin(object):
    def __init__(self):
        self.nn_model = ""
        self.loss_function = ""
        self.optimizer = ""
        self.learning_rate = 0.03
        self.example_data = ""
        self.data_iter = ""

        # 训练数据集 - 训练参数准备
        x_region = [-3.0, 1.1]
        batch_size = 3
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
        self.nn_model = NeuralNet(n_feature=1, n_hidden1=6, n_hidden2=9, n_hidden3=12, n_hidden4=16, n_output=1)

    def __init_loss_function__(self):
        # 均方差损失函数
        self.loss_function = nn.MSELoss()

    def __init_optimizer__(self):
        # 定义优化器:小批量随机梯度下降算法
        self.optimizer = torch.optim.SGD(self.nn_model.parameters(), lr=self.learning_rate)

    def train(self):
        # 开始进行训练
        num_epochs = 200
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

        self.nn_model.hidden1.weight[1]
        self.nn_model.hidden2.weight[1]
        # w = self.nn_model.hidden
        # print('prediction w：', w.reshape(self.true_w.shape))
        # b = self.nn_model[0].bias.data
        # print('prediction b：', b)


test = linear_regression_sin()
test.train()
