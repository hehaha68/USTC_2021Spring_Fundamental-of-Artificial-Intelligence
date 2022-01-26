from process_data import load_and_process_data
from evaluation import get_macro_F1,get_micro_F1,get_acc
import numpy as np


# 实现线性回归的类
class LinearClassification:

    '''参数初始化 
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''
    def __init__(self,lr=0.05,Lambda= 0.001,epochs = 1000):
        self.lr=lr
        self.Lambda=Lambda
        self.epochs =epochs


    '''根据训练数据train_features,train_labels计算梯度更新参数W'''
    def fit(self,train_features,train_labels):
        self.omega = np.ones([8, 1])
        #获得矩阵的行数和列数
        attrnum = train_features.shape[1]
        colnum = train_features.shape[0]
        #初始化omega和梯度
        self.omega = np.ones([attrnum, 1])
        dloss = np.zeros([attrnum, 1])
        #梯度下降法
        for n in range(self.epochs):
            y_hat = np.dot(train_features, self.omega)  #y_hat表示预测的y，用于计算与真实值的误差以便求梯度
            for i in range(attrnum):
                dloss[i] = 2 * np.dot(train_features[:, i].T, y_hat - train_labels)/colnum + 2 * self.Lambda * self.omega[i]
            #以lr的学习率更新omega
            self.omega = self.omega - self.lr * dloss

    '''根据训练好的参数对测试数据test_features进行预测，返回预测结果
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目'''
    def predict(self,test_features):
        result = np.dot(test_features, self.omega)
        #由于预测集是整数，所以进行四舍五入
        for i in range(result.shape[0]):
            if result[i] > 2.5:
                result[i] = 3
            elif result[i] < 1.5:
                result[i] = 1
            else:
                result[i] = 2
        return result

def main():
    # 加载训练集和测试集
    train_data,train_label,test_data,test_label=load_and_process_data()
    lR = LinearClassification()
    lR.fit(train_data,train_label) # 训练模型
    pred=lR.predict(test_data) # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))

main()
