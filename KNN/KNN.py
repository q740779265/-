import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import sqrt
import math
from collections import Counter

def distance(vector1,vector2):
    dis = 0
    for i in range(len(vector1)):
        dis += pow((vector1[i]-vector2[i]), 2)
    return sqrt(dis)

class KNNmodel():
    def __init__(self, x_train, y_train):
        self.saved_x = x_train
        self.saved_y = y_train

    def nearest_k(self, x_test, k):
        dis_array =[]
        for i in range(len(x_train)):
            dis = distance(x_train[i], x_test)
            dis_array.append(dis)
        dis_sort = np.argsort(dis_array)
        near_k_label = [y_train[i] for i in dis_sort[:k]]
        return near_k_label

    def prediction(self, X_test, Y_test ,k):
        test_nums = len(X_test)
        acc_nums = 0
        pres = []
        for i, x_test in enumerate(X_test):
            near_k = self.nearest_k(x_test, k)
            # 使用counter函数记录near_k中重复元素的个数
            top = Counter(near_k)
            # 使用counter.most_common函数找出重复最多的元素
            pre = top.most_common(1)[0][0]
            pres.append(pre)

            if pre == Y_test[i]:
                acc_nums += 1
        pres = np.array(pres)
        print("真实值", Y_test)
        print("预测值", pres)
        print("正确率", acc_nums/test_nums)


# 加载数据集
iris = load_iris()
# 训练集的特征值x_train 测试集的特征值x_test 训练集的目标值y_train 测试集的目标值y_test，指定测试集所占的比例为20%
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

model = KNNmodel(x_train, y_train)
model.prediction(x_test, y_test, 4)
