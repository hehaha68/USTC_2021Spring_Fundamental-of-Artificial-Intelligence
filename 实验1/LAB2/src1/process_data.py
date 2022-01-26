import numpy as np

# 数据标准化
def standardization(data):
    mean=np.sum(data,axis=0)/(data.shape[0])
    std=np.std(data, axis=0)
    return (data-mean)/std

# 对数据加载并处理处理
def load_and_process_data():
    train_feature=[]
    train_label=[]
    test_feature=[]
    test_label=[]
    # 处理训练集数据
    with open('./data/train_new.data')as f:
        lines=f.readlines()
        print("train_num: "+str(len(lines)))
        for line in lines:
            feature=[]
            if line==None:
                continue
            # 数据集的第一列数据为性别[M,F,],将其对应为数字1,2,3
            line=line.rstrip('\n').split(',')
            if line[0]=='M':
                feature.append(int(1))
            elif line[0]=='F':
                feature.append(int(2))
            else:
                feature.append(int(3))
            for i in range(1,8):
                feature.append(float(line[i]))         
            train_label.append(int(line[8]))
            train_feature.append(feature)     
    
    # 处理测试集数据
    with open('./data/test_new.data')as f:
        lines=f.readlines()
        print("test_num: "+str(len(lines)))
        for line in lines:
            feature=[]
            if line==None:
                continue
            line=line.rstrip('\n').split(',')

            if line[0]=='M':
                feature.append(int(1))
            elif line[0]=='F':
                feature.append(int(2))
            else:
                feature.append(int(3))
            for i in range(1,8):
                feature.append(float(line[i]))      
            test_label.append(int(line[8]))
            test_feature.append(feature)
   
    train_feature=np.array(train_feature).astype(float)
    print("train_feature's shape:"+str(train_feature.shape))
    train_label=np.array(train_label).astype(int).reshape(-1,1)

    test_feature=np.array(test_feature,dtype=np.float32)
    print("test_feature's shape:"+str(test_feature.shape))
    test_label=np.array(test_label).astype(int).reshape(-1,1)

    return train_feature,train_label,test_feature,test_label 
