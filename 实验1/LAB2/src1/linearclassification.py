from process_data import load_and_process_data
from evaluation import get_macro_F1, get_micro_F1, get_acc
import numpy as np


# 实现线性回归的类
class LinearClassification:
    '''参数初始化
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''

    def __init__(self, lr=0.05, Lambda=0.001, epochs=1000):
        self.lr = lr
        self.Lambda = Lambda
        self.epochs = epochs

    def fit(self, train_features, train_labels):

        def dj(w_now, x, y):  # 梯度
            Y = np.dot(x, w_now)
            N, num = np.shape(x)
            dJ = np.zeros([num, 1])
            for i in range(num):
                dJ[i, 0] = 2 * np.dot((Y - y).T, x[:, i]) / N + 2 * self.Lambda * w_now[i]
            return dJ

        num = train_features.shape[1]
        w = np.random.random((num, 1))
        # 梯度下降训练
        for i in range(self.epochs):
            gradient = dj(w, train_features, train_labels)
            w = w - self.lr * gradient
        self.w = w

    def predict(self, test_features):
        y = np.dot(test_features, self.w)
        for i in range(y.shape[0]):
            # 结果进行舍入处理
            if y[i] <= 1:
                y[i] = 1
            elif y[i] >= 3:
                y[i] = 3
            elif abs(y[i] - int(y[i])) < 0.5:
                y[i] = int(y[i])
            else:
                y[i] = int(y[i]) + 1
        return y


def main():
    # 加载训练集和测试集
    train_data, train_label, test_data, test_label = load_and_process_data()
    lR = LinearClassification()
    lR.fit(train_data, train_label)  # 训练模型
    pred = lR.predict(test_data)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: " + str(get_acc(test_label, pred)))
    print("macro-F1: " + str(get_macro_F1(test_label, pred)))
    print("micro-F1: " + str(get_micro_F1(test_label, pred)))


main()
