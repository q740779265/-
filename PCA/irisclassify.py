from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
# 加载数据集
iris = load_iris()
data = iris.data
label = iris.target
# PCA操作
pca = PCA(n_components=2)   # 实例化模型  n_components 可以是"mle"让机器自己选超参数 也可以是百分比,表示需要的信息量
pca.fit(data)               # 拟合数据,获得特征向量
data = pca.transform(data)  # 对原数据进行线性变换达到降维效果
# data = PCA(2).fit_transform(data)  也可以一步到位
plt.figure()
plt.scatter(data[label == 0, 0], data[label == 0, 1], c="red", label=iris.target_names[0])
plt.scatter(data[label == 1, 0], data[label == 1, 1], c="yellow", label=iris.target_names[1])
plt.scatter(data[label == 2, 0], data[label == 2, 1], c="black", label=iris.target_names[2])
plt.legend()  # 显示图例
plt.show()
