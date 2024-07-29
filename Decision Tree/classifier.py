from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz

wine = load_wine()
X = wine.data
Y = wine.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.3)

# 分类树
clf = tree.DecisionTreeClassifier(criterion="entropy",   # 有gini和entropy两个值
                                  random_state=0,       # 随机种子
                                  splitter="best",     # 划分随机模式
                                  max_depth=5,             # 最大深度
                                  min_samples_split=10,  # 分支节点的数据量最小值
                                  min_samples_leaf=5)   # 叶子结点数据量最小值
clf = clf.fit(X_train, Y_train)    # 训练分类树
print(clf.score(X_test, Y_test))    # 验证分类树

# 绘图
tree = tree.export_graphviz(clf, feature_names=wine.feature_names, class_names=wine.target_names)     # 利用sklearn建立一颗树
graph = graphviz.Source(tree)           # 保存为graphviz格式
graph.render("g", view=True)    # 保存为png

