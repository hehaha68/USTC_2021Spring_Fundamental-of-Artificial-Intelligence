import numpy as np
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1, get_macro_F1, get_acc, get_binary_TP_FP_FN


class NaiveBayes:
    '''参数初始化
    Pc: P(c) 每个类别c的概率分布
    Pxc: P(c|x) 每个特征的条件概率
    '''

    def __init__(self):
        self.Pc = {}
        self.Pxc = {}

    '''
    通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
    建议全部取log，避免相乘为0
    '''

    def fit(self, traindata, trainlabel, featuretype):

        num = traindata.shape[0]
        N = traindata.shape[1]
        self.Gau = {}

        # 先验概率计算
        label_result = Counter(trainlabel[:, 0])  # 统计
        label_num = len(label_result)
        for i in range(label_num):
            k = list(label_result.keys())[i]
            self.Pc[k] = np.log((label_result[k] + 1) / (num + label_num))

        # 条件概率计算
        for i in range(N):
            # 离散属性
            if featuretype[i] == 0:
                data1 = {}
                for j in range(label_num):
                    label = list(label_result.keys())[j]
                    data2 = []

                    for k in range(num):
                        if trainlabel[:, 0][k] == label:
                            data2.append(traindata[:, i][k])

                    data_result = Counter(data2)
                    data_num = len(data_result)

                    for k in range(data_num):
                        t = list(data_result.keys())[k]
                        data_result[t] = np.log((data_result[t] + 1) / (label_result[label] + data_num))

                    data1[label] = data_result
                self.Pxc[i] = data1
            # 连续属性
            else:
                data1 = {}
                for j in range(label_num):
                    label = list(label_result.keys())[j]
                    data2 = []

                    for k in range(num):
                        if trainlabel[:, 0][k] == label:
                            data2.append(traindata[:, i][k])

                    u = np.mean(data2)  # 均值
                    v = np.var(data2)  # 方差

                    data1[label] = [u, v]
                self.Gau[i] = data1

    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''

    def predict(self, features, featuretype):
        Pc = self.Pc
        Pxc = self.Pxc
        Gau = self.Gau
        num = features.shape[0]
        N = features.shape[1]
        print(self.Pc)
        result = []
        for k in range(num):
            h = {}
            for j in range(len(Pc)):
                t1 = list(Pc.keys())[j]
                time = Pc[t1]
                for i in range(N):
                    if featuretype[i] == 0:  # 离散属性
                        time = time * Pxc[i][t1][features[k][i]]
                    else:   # 连续属性
                        f1 = -np.power(features[k][i] - Gau[i][t1][0], 2) / (2 * Gau[i][t1][1])
                        f2 = 1 / np.sqrt(2 * np.pi * Gau[i][t1][1])
                        f3 = f2 * np.power(np.e, f1)
                        time = time * f3
                h[t1] = time

            result.append([sorted(h.items(), key=lambda x: x[1], reverse=True)[0][0]])
        return np.array(result)


def main():
    # 加载训练集和测试集
    train_data, train_label, test_data, test_label = load_and_process_data()
    feature_type = [0, 1, 1, 1, 1, 1, 1, 1]  # 表示特征的数据类型，0表示离散型，1表示连续型

    Nayes = NaiveBayes()
    Nayes.fit(train_data, train_label, feature_type)  # 在训练集上计算先验概率和条件概率

    pred = Nayes.predict(test_data, feature_type)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    #print("Acc: " + str(get_acc(test_label, pred)))
    #print("macro-F1: " + str(get_macro_F1(test_label, pred)))
    #print("micro-F1: " + str(get_micro_F1(test_label, pred)))


main()
