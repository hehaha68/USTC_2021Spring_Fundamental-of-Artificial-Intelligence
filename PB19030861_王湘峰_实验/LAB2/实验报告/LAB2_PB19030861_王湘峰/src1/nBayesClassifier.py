import numpy as np
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1, get_macro_F1, get_acc


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
        # 计算先验概率分布p(c)
        static1 = Counter(trainlabel[:, 0])
        for i in static1:
            static1[i] = (static1[i] + 1) / (traindata.shape[0] + 3)
        self.Pc = static1
        # 计算条件概率分布p(x|c)
        self.P1c = {}  # 第一个特征是离散的，单独拿出来
        for i in self.Pc:
            for j in range(3):
                filter1 = traindata[:, 0] == j + 1
                filter2 = trainlabel == i
                sum1 = np.sum(filter2)  #sum1代表Dc
                sum2 = 0                #sum2代表Dc,xi
                for t in range(traindata.shape[0]):
                    if filter1[t] and filter2[t]:
                        sum2 = sum2 + 1
                self.P1c[(j + 1, i)] = (sum2 + 1) / (sum1 + 3)  #拉普拉斯平滑处理

        #对于连续的属性，假设其符合正态分布，现计算均值和方差
        for i in range(traindata.shape[1] - 1):
            for j in self.Pc:
                filt = trainlabel[:, 0] == j
                temp = traindata[:, i + 1]
                temp = temp[filt]
                miu = np.mean(temp)
                sigema = np.var(temp)
                self.Pxc[(i + 1, j)] = (miu, sigema)
                #在Pxc中，(i,j):(miu,sigema) 意味着当label=j时，第i个属性的均值为miu，方差为sigema

    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''

    def predict(self, features, featuretype):
        result = []
        for i in range(features.shape[0]):
            prob = []   #prob代表取不同label时的概率
            for j in [1, 2, 3]:
                log = 0
                log += math.log(self.Pc[j]) + math.log(self.P1c[(features[i, 0], j)]) #先把先验P(c)和第一个属性的概率相乘
                for k in range(features.shape[1] - 1):  #将离散的属性对应的条件概率相乘
                    miu = self.Pxc[(k + 1, j)][0]
                    sigema = self.Pxc[(k + 1, j)][1]
                    t = math.exp(-(features[i, k + 1] - miu) ** 2 / 2 / sigema) / math.sqrt(2 * math.pi * sigema)
                    log += math.log(t)
                prob.append(log)
            if prob[0] > prob[1] and prob[0] > prob[2]:
                result.append(1)
            elif prob[1] > prob[0] and prob[1] > prob[2]:
                result.append(2)
            else:
                result.append(3)
        return np.array(result).reshape(features.shape[0], 1)

def main():
    # 加载训练集和测试集
    train_data,train_label,test_data,test_label=load_and_process_data()
    feature_type=[0,1,1,1,1,1,1,1] #表示特征的数据类型，0表示离散型，1表示连续型
    Nayes=NaiveBayes()
    Nayes.fit(train_data,train_label,feature_type) # 在训练集上计算先验概率和条件概率
    pred=Nayes.predict(test_data,feature_type)  # 得到测试集上的预测结果
    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))

main()