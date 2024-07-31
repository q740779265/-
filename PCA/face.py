from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

faces = fetch_lfw_people(min_faces_per_person=60)
X = faces.data
pca = PCA(150).fit(X)
X_dr = pca.transform(X)
V = pca.components_
X_inverse = pca.inverse_transform(X_dr)
# 可视化特征提取过程
# subplot_kw用于设置轴,注意subplots和subplot的区别 fig返回的是整个画布 axes返回的时候数组，里面保存着每一张子图信息
fig, axes = plt.subplots(3, 8,  figsize=(8, 4), subplot_kw={"xticks": [], "yticks": []})
for i, ax in enumerate(axes.flat):
    ax.imshow(V[i, :].reshape(62, 47), cmap="gray")  # imshow用于填入图片信息
# 显示原图
fig, axes = plt.subplots(3, 8,  figsize=(8, 4), subplot_kw={"xticks": [], "yticks": []})
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.images[i, :].reshape(62, 47), cmap="gray")  # imshow用于填入图片信息
# 显示压缩后还原的图片
fig, axes = plt.subplots(3, 8,  figsize=(8, 4), subplot_kw={"xticks": [], "yticks": []})
for i, ax in enumerate(axes.flat):
    ax.imshow(X_inverse[i, :].reshape(62, 47), cmap="gray")  # imshow用于填入图片信息

plt.show()


