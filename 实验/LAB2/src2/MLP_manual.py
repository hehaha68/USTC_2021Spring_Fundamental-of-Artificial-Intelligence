import torch
import numpy as np
import matplotlib.pyplot as plt


# plt.rcParams['font.sans-serif'] = ['SimHei']

def makedata(datanum, featurenum, labelnum):  # 生成随机训练集
    data = np.random.random((datanum, featurenum)) * 100
    label = np.random.randint(labelnum, size=(datanum, 1))
    return data, label


class MultiLayerPerceptron:

    def __init__(self):
        self.w1 = np.ones([4, 5])
        self.w2 = np.ones([4, 4])
        self.w3 = np.ones([3, 4])   # 权值w
        self.b1 = np.ones([4, 1])
        self.b2 = np.ones([4, 1])
        self.b3 = np.ones([3, 1])   # 偏置b
        self.h1 = None  # h1
        self.h2 = None  # h2
        self.h3 = None  # h3
        self.L = []  # 损失函数表
        self.alpha = 0.01  # 学习率
        self.epochs = 1000  # 最大训练次数
        self.epsilon = 1e-8  # 结束条件

    def Sigmoid(self, data):  # 激活函数Sigmoid
        x = data.ravel()  # 展平
        y = []
        for i in range(len(x)):
            if x[i] >= 0:
                y.append(1.0 / (1 + np.exp(-x[i])))
            else:   # 防止计算溢出
                y.append(np.exp(x[i]) / (np.exp(x[i]) + 1))
        return np.array(y).reshape(data.shape)

    def Softmax(self, data):  # 激活函数Softmax
        y = np.exp(data)
        return y / np.sum(y, axis=1).reshape(-1, 1)

    def DSigmoid(self, data):   # Sigmoid导数
        x = self.Sigmoid(data)
        return x * (1 - x)

    def CrossEntropy(self, y, label):  # 交叉熵
        datanum, _ = np.shape(label)
        loss = np.dot(-np.log(y), label.T)
        Loss = np.trace(loss) / datanum
        return Loss

    def forword(self, data, w1, w2, w3):    # 前向传播
        self.h1 = self.Sigmoid(np.dot(w1, data.T) + self.b1)
        self.h2 = self.Sigmoid(np.dot(w2, self.h1) + self.b2)
        self.h3 = self.Softmax((np.dot(w3, self.h2) + self.b3).T)
        return self.h3

    def BackPropagation(self, data, score, label):  # 逆向传播求梯度
        datanum, _ = np.shape(label)
        # w3和b3的梯度
        temp0 = (score - label) / datanum
        j_b3 = np.dot(temp0.T, np.ones([datanum, 1]))
        j_w3 = np.dot(temp0.T, self.h2.T)
        # w2和b2的梯度
        temp1 = np.dot(self.w3.T, temp0.T)
        temp2 = self.DSigmoid(np.dot(self.w2, self.h1) + self.b2)
        j_b2 = np.dot(temp1 * temp2, np.ones([datanum, 1]))
        j_w2 = np.dot(temp1 * temp2, self.h1.T)
        # w1和b1的梯度
        temp3 = temp1 * temp2
        temp4 = self.DSigmoid(np.dot(self.w1, data.T) + self.b1)
        j_b1 = np.dot(temp3 * temp4, np.ones([datanum, 1]))
        j_w1 = np.dot(temp3 * temp4, data)
        return j_w1, j_w2, j_w3, j_b1, j_b2, j_b3

    def GradientDescent(self, j_w1, j_w2, j_w3, j_b1, j_b2, j_b3):  # 梯度下降
        alpha = self.alpha
        self.w1 = self.w1 - alpha * j_w1
        self.w2 = self.w2 - alpha * j_w2
        self.w3 = self.w3 - alpha * j_w3
        self.b1 = self.b1 - alpha * j_b1
        self.b2 = self.b2 - alpha * j_b2
        self.b3 = self.b3 - alpha * j_b3

    def fit(self, data, y):     # 训练
        label = np.zeros([100, 3])
        for i in range(100):
            label[i][int(y[i][0])] = 1  # label转换为One-Hot
        self.L = []

        for i in range(self.epochs):
            score = self.forword(data, self.w1, self.w2, self.w3)   # 前向求值
            self.L.append(self.CrossEntropy(score, label))   # 计算损失函数（交叉熵）
            if len(self.L) > 1 and abs(self.L[i] - self.L[i - 1]) < self.epsilon:   # 终止条件
                print("训练次数", i + 1, "损失函数", self.L[i])
                break
            j_w1, j_w2, j_w3, j_b1, j_b2, j_b3 = self.BackPropagation(data, score, label)   # 逆向求梯度
            if i == 0:
                print("手动")
                print("j_w1:",j_w1,"\nj_w2:",j_w2,"\nj_w3:",j_w3)
                print("j_b1:", j_b1, "\nj_b2:", j_b2, "\nj_b3:", j_b3)
            self.GradientDescent(j_w1, j_w2, j_w3, j_b1, j_b2, j_b3)    # 梯度下降
            if (i + 1) % 100 == 0:
                print("训练次数", i + 1, "损失函数", self.L[i])

    def LossPlot(self, title):  # 绘制损失训练曲线
        num = len(self.L)
        plt.figure()
        plt.title(title)
        plt.xlabel("Train_num")
        plt.ylabel("Loss_value")
        plt.plot(range(num), self.L)
        plt.show()

    def usetorch(self, data, label):    # torch自动求导训练
        label = label.flatten()
        data, label = torch.from_numpy(data), torch.from_numpy(label).long()
        alpha = self.alpha
        self.L = []
        w1 = torch.autograd.Variable(torch.from_numpy(np.ones([4, 5])), requires_grad=True)
        w2 = torch.autograd.Variable(torch.from_numpy(np.ones([4, 4])), requires_grad=True)
        w3 = torch.autograd.Variable(torch.from_numpy(np.ones([3, 4])), requires_grad=True)
        b1 = torch.autograd.Variable(torch.from_numpy(np.ones([4, 1])), requires_grad=True)
        b2 = torch.autograd.Variable(torch.from_numpy(np.ones([4, 1])), requires_grad=True)
        b3 = torch.autograd.Variable(torch.from_numpy(np.ones([3, 1])), requires_grad=True)
        sigmoid = torch.nn.Sigmoid()
        ce = torch.nn.CrossEntropyLoss()  # 包含softmax

        for i in range(self.epochs):
            h1 = sigmoid(torch.mm(w1, data.T) + b1)
            h2 = sigmoid(torch.mm(w2, h1) + b2)
            h3 = (torch.mm(w3, h2) + b3).T  # 前向求值
            Loss = ce(h3, label)   # 计算损失函数（交叉熵）
            self.L.append(Loss)

            if len(self.L) > 1 and abs(self.L[i] - self.L[i - 1]) < self.epsilon:   # 终止条件
                print("训练次数", i + 1, "损失函数", self.L[i].detach().numpy())
                break

            if i != 0:  # 梯度清零
                w1.grad.data.zero_()
                w2.grad.data.zero_()
                w3.grad.data.zero_()
                b1.grad.data.zero_()
                b2.grad.data.zero_()
                b3.grad.data.zero_()

            Loss.backward()     # 求梯度，梯度下降优化
            if i == 0:
                print("自动")
                print("j_w1:", w1.grad, "\nj_w2:", w2.grad, "\nj_w3:", w3.grad)
                print("j_b1:", b1.grad, "\nj_b2:", b2.grad, "\nj_b3:", b3.grad)
            w1.data = w1.data - alpha * w1.grad
            w2.data = w2.data - alpha * w2.grad
            w3.data = w3.data - alpha * w3.grad
            b1.data = b1.data - alpha * b1.grad
            b2.data = b2.data - alpha * b2.grad
            b3.data = b3.data - alpha * b3.grad

            if (i + 1) % 100 == 0:
                print("训练次数", i + 1, "损失函数", self.L[i].detach().numpy())


def main():
    data, label = makedata(100, 5, 3)
    MLP = MultiLayerPerceptron()
    MLP.fit(data, label)
    MLP.LossPlot(title='Manual-Loss Curve')
    MLP.usetorch(data, label)
    MLP.LossPlot(title='Auto-Loss Curve')


main()
