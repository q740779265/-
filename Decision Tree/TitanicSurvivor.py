import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV  # 网格搜索调参
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score  # 交叉验证

# 数据预处理  主要包括导入文件 处理缺失值 删除无关特征 将非数值特征转化为数值特征
data = pd.read_csv("C:/Users/lenovo/Desktop/DL/0 New/Decision Tree/train.csv")  # 读入csv文件 注意双引号
data.drop(['Cabin', 'Name', 'Ticket'], inplace=True, axis=1)   # 删除无关输入特征, inplace表示是否替换原表, axis=1表示对列操作
data['Age'] = data['Age'].fillna(data["Age"].mean())    # 将某一列的缺失值填充为平均值,data["XXX"]表示选中名字为XXX的某一列,data["XXX"].mean()是某一列的均值
data = data.dropna()  # 若某一行的数据有缺失，直接删除该行
label_Embarked = data['Embarked'].unique().tolist()  # unique函数用于提取特征的种类数，tolist函数将结果转化为list列表
data['Embarked'] = data['Embarked'].apply(lambda x: label_Embarked.index(x))  # 利用apply和lambda对某一列进行批量操作,这里的lambda类似于auto x
label_Sex = data['Sex'].unique().tolist()      # 同上
data['Sex'] = data['Sex'].apply(lambda x: label_Sex.index(x))

# 取出输入特征和标签
X = data.loc[:, data.columns != "Survived"]    # loc[行,列]表示选中目标列,loc是按标签索引,iloc[]是按值索引
Y = data.loc[:, "Survived"]

# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)
for x in [X_train, X_test, Y_train, Y_test]:
    x.reset_index(drop=True, inplace=True)        # 重设行号

# 训练模型
clf = DecisionTreeClassifier(criterion="gini",
                             random_state=30,
                             splitter="best",
                             max_depth=4,
                             min_samples_split=15,
                             min_samples_leaf=4)
clf = clf.fit(X_train, Y_train)
print("普通训练准确度：", clf.score(X_test, Y_test))

print("交叉验证训练准确度：", cross_val_score(clf, X, Y, cv=10).mean())
parameters = {"criterion": ("gini", "entropy"),
              "splitter": ("best", "random"),
              "max_depth": [*range(1, 10)],
              "min_samples_leaf": [*range(1, 50, 5)],
              "min_impurity_decrease": [*np.linspace(0, 0.5, 10)]}
GS = GridSearchCV(clf, parameters, cv=10).fit(X, Y)
print("网格搜索训练准确度：", GS.best_score_)
