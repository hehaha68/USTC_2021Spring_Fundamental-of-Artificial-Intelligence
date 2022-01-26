import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MLP:

    def __init__(self):
        self.W1 = np.ones([4, 5])
        self.W2 = np.ones([4, 4])
        self.W3 = np.ones([3, 4])
        self.alpha = 0.01  # 学习率
        self.epsilon = 10e-6  # 阈值
        self.epoch = 1000  # 最大迭代次数
        self.h1 = np.zeros([4, 100])
        self.h2 = np.zeros([4, 100])

    def Sigmoid(self, x):
        return 1 / (np.exp(-x) + 1)

    def Diff_sigmoid(self, x):
        y = self.Sigmoid(x)
        return y * (1 - y)

    def Softmax(self, input):
        exp = np.exp(input)
        return exp / np.sum(exp, axis=1).reshape(-1, 1)

    def CrossEntropy(self, out, label):  # label属于one-hot向量(nx3)，out是输出神经元的向量(nx3)，返回交叉熵
        log = np.log(out)
        return np.trace(-log @ label.T) / len(label)

    def FP(self, W1, W2, W3, input):  # W1:(4x5) W2:(4x4) W3:(3x4)
        self.h1 = f.Sigmoid(W1 @ input.T)  # h1每列代表一行x的隐层
        self.h2 = f.Sigmoid(W2 @ self.h1)  # h2每列代表h1的隐层
        return f.Softmax((W3 @ self.h2).T)  # 输出nx3

    def One_hot(self, label):  # label:(nx1),返回nx3的one_hot矩阵
        n = len(label)
        t = np.zeros((n, 3))
        for i in range(n):
            t[i, label[i]] = 1
        return t

    def GD(self, d1, d2, d3):
        self.W1 = self.W1 - self.alpha * d1
        self.W2 = self.W2 - self.alpha * d2
        self.W3 = self.W3 - self.alpha * d3

    def BP(self, train, w1, w2, w3, label):  # label:(3xn)
        yloss = (self.FP(w1, w2, w3, train).T - label) / len(train)
        d3 = yloss @ self.h2.T  # d3:(3x4)
        t = (w3.T @ yloss) * self.Diff_sigmoid(w2 @ self.h1)  # t:(4xn)
        d2 = t @ self.h1.T  # d2:(4x4)
        tt = self.Diff_sigmoid(w1 @ train.T)
        d1 = (t * tt) @ train  # d1:(4x5)
        return d1, d2, d3

    def fit(self, train, label):  # train:(nx5),label:(nx1)
        Loss = []
        for i in range(self.epoch):
            loss = self.CrossEntropy(self.FP(self.W1, self.W2, self.W3, train), self.One_hot(label))
            Loss.append(loss)
            if len(Loss) < 2:
                pass
            elif abs(Loss[i] - Loss[i - 1]) < self.epsilon:
                break
            d1, d2, d3 = self.BP(train, self.W1, self.W2, self.W3, self.One_hot(label).T)
            self.GD(d1, d2, d3)
        print('myd1:', d1.tolist())     #显示最后一次的梯度并进行对比
        print('myd2:', d2.tolist())
        print('myd3:', d3.tolist())
        print('myloss:', Loss[len(Loss) - 1])
        print('训练了', i + 1, '次收敛')
        self.plot(Loss)     #绘图
        return Loss

    # 绘图
    def plot(self, L):
        x = range(1, len(L) + 1)
        plt.title('损失函数随训练次数的变化')
        plt.xlabel('训练次数')
        plt.ylabel('损失函数')
        plt.plot(x, L)
        plt.show()

    def compare(self, x, y):
        train = torch.from_numpy(x)
        y = y.flatten()
        label = torch.from_numpy(y)
        w1 = torch.from_numpy(np.ones([4, 5]))
        w2 = torch.from_numpy(np.ones([4, 4]))
        w3 = torch.from_numpy(np.ones([3, 4]))
        w1 = torch.autograd.Variable(w1, requires_grad=True)
        w2 = torch.autograd.Variable(w2, requires_grad=True)
        w3 = torch.autograd.Variable(w3, requires_grad=True)
        sig = nn.Sigmoid()
        ce = nn.CrossEntropyLoss()
        Loss = []
        for i in range(self.epoch):
            h1 = sig(torch.mm(w1, train.T))
            h2 = sig(torch.mm(w2, h1))
            y_hat = torch.mm(w3, h2).T
            loss = ce(y_hat, label.long())
            Loss.append(float(loss))
            if len(Loss) < 2:
                pass
            elif abs(Loss[i] - Loss[i - 1]) < self.epsilon:
                break
            loss.backward()
            d1 = w1.grad
            d2 = w2.grad
            d3 = w3.grad
            w1 = torch.autograd.Variable(w1 - 1 * self.alpha * d1, requires_grad=True)
            w2 = torch.autograd.Variable(w2 - 1 * self.alpha * d2, requires_grad=True)
            w3 = torch.autograd.Variable(w3 - 1 * self.alpha * d3, requires_grad=True)
        print('torch d1:', d1.tolist()) #显示最后一次的梯度并进行对比
        print('torch d2:', d2.tolist())
        print('torch d3:', d3.tolist())
        print('torch_loss:', Loss[len(Loss) - 1])
        print('训练了', i + 1, '次收敛')
        self.plot(Loss)     #绘图
        return Loss

f=MLP()
train_data = np.random.random((100,5)) * 100
label = np.random.randint(3, size=(100,1))
f.fit(train_data,label)
f.compare(train_data,label)
