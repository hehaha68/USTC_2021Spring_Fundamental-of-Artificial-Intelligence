import numpy as np
import cvxopt  # 用于求解线性规划
from process_data import load_and_process_data
from evaluation import get_micro_F1, get_macro_F1, get_acc


# 根据指定类别main_class生成1/-1标签
def svm_label(labels, main_class):
    new_label = []
    for i in range(len(labels)):
        if labels[i] == main_class:
            new_label.append(1)
        else:
            new_label.append(-1)
    return np.array(new_label)


# 实现线性回归
class SupportVectorMachine:

    def __init__(self, kernel, C, Epsilon):
        self.kernel = kernel
        self.C = C
        self.Epsilon = Epsilon

    '''KERNEL用于计算两个样本x1,x2的核函数'''

    def KERNEL(self, x1, x2, kernel='Gauss', d=2, sigma=1):
        # d是多项式核的次数,sigma为Gauss核的参数
        K = 0
        if kernel == 'Gauss':
            K = np.exp(-(np.sum((x1 - x2) ** 2)) / (2 * sigma ** 2))
        elif kernel == 'Linear':
            K = np.dot(x1, x2)
        elif kernel == 'Poly':
            K = np.dot(x1, x2) ** d
        else:
            print('No support for this kernel')
        return K

    '''
    根据训练数据train_data,train_label（均为np数组）求解svm,并对test_data进行预测,返回预测分数，即svm使用符号函数sign之前的值
    train_data的shape=(train_num,train_dim),train_label的shape=(train_num,) train_num为训练数据的数目，train_dim为样本维度
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    '''

    def fit(self, train_data, train_label, test_data):

        # 创建核矩阵
        def KernelMatrix(data, kernel, fun_KERNEL):
            num = data.shape[0]
            K_Matrix = np.zeros((num, num))
            for i in range(num):
                for j in range(num):
                    K_Matrix[i, j] = fun_KERNEL(data[i], data[j], kernel)
            return K_Matrix

        # 计算偏置b
        def fit_bias(a, vlabel, v, K_Matrix, ind):
            b = 0
            for i in range(len(a)):
                b += vlabel[i]
                b -= np.sum(a * vlabel * K_Matrix[ind[i], v])
            b /= len(a)
            return b

        # 计算权重w
        def fit_weight(kernel, data, a, v, vlabel):
            if kernel == 'Linear':
                w = np.zeros(data)
                for i in range(len(a)):
                    w += a[i] * vlabel[i] * v[i]
            else:
                w = None
            return w

        # 求Lagrange乘子
        def fit_Lagrange(data, vlabel, C, K_Matrix):
            num = data.shape[0]
            P = cvxopt.matrix(np.outer(vlabel, vlabel) * K_Matrix)
            q = cvxopt.matrix(np.ones(num) * -1)
            A = cvxopt.matrix(vlabel, (1, num), 'd')
            b = cvxopt.matrix(0.0)

            if C is None:
                G = cvxopt.matrix(np.diag(-np.ones(num)))
                h = cvxopt.matrix(np.zeros(num))
            else:
                G = cvxopt.matrix(np.vstack([np.diag(-np.ones(num)), np.identity(num)]))
                h = cvxopt.matrix(np.hstack([np.zeros(num), np.ones(num) * C]))

            solution = cvxopt.solvers.qp(P, q, G, h, A, b)  # 解模型
            a = np.ravel(solution['x'])
            return a

        self.a = None  # 乘子
        self.b = 0  # 偏置
        self.w = []  # 权值
        self.v = []  # 向量
        self.vlabel = []  # 向量标签

        # 训练参数
        _, N = train_data.shape
        K_Matrix = KernelMatrix(train_data, self.kernel, self.KERNEL)
        a = fit_Lagrange(train_data, train_label, self.C, K_Matrix)
        v = a > self.Epsilon
        ind = np.arange(len(a))[v]  # 向量下标
        self.a = a[v]
        self.v = train_data[v]  # 向量
        self.vlabel = train_label[v]
        self.b = fit_bias(self.a, self.vlabel, v, K_Matrix, ind)
        self.w = fit_weight(self.kernel, N, self.a, self.v, self.vlabel)

        # 预测
        if self.w is not None:
            y = np.dot(test_data, self.w) + self.b
            y = y.reshape(test_data.shape[0], 1)
            return y
        else:
            pre = np.zeros(len(test_data))
            for i in range(len(test_data)):
                s = 0
                for a, vl, v in zip(self.a, self.vlabel, self.v):
                    s += a * vl * self.KERNEL(test_data[i], v, self.kernel)
                pre[i] = s
            y = pre + self.b
            y = y.reshape(test_data.shape[0], 1)
            return y


def main():
    # 加载训练集和测试集
    Train_data, Train_label, Test_data, Test_label = load_and_process_data()
    Train_label = [label[0] for label in Train_label]
    Test_label = [label[0] for label in Test_label]
    train_data = np.array(Train_data)
    test_data = np.array(Test_data)
    test_label = np.array(Test_label).reshape(-1, 1)
    # 类别个数
    num_class = len(set(Train_label))
    # kernel为核函数类型，可能的类型有'Linear'/'Poly'/'Gauss'
    # C为软间隔参数；
    # Epsilon为拉格朗日乘子阈值，低于此阈值时将该乘子设置为0
    kernel = 'Linear'
    C = 1
    Epsilon = 10e-5
    # 生成SVM分类器
    SVM = SupportVectorMachine(kernel, C, Epsilon)

    predictions = []
    # one-vs-all方法训练num_class个二分类器
    for k in range(1, num_class + 1):
        # 将第k类样本label置为1，其余类别置为-1
        train_label = svm_label(Train_label, k)
        # 训练模型，并得到测试集上的预测结果
        prediction = SVM.fit(train_data, train_label, test_data)
        predictions.append(prediction)
    predictions = np.array(predictions)
    # one-vs-all, 最终分类结果选择最大score对应的类别
    pred = np.argmax(predictions, axis=0) + 1

    # 计算准确率Acc及多分类的F1-score
    print("Acc: " + str(get_acc(test_label, pred)))
    print("macro-F1: " + str(get_macro_F1(test_label, pred)))
    print("micro-F1: " + str(get_micro_F1(test_label, pred)))


main()
