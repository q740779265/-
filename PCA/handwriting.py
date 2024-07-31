from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
# 显示原图
fig, axes = plt.subplots(3, 8,  figsize=(8, 4), subplot_kw={"xticks": [], "yticks": []})
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i, :], cmap="gray")  # imshow用于填入图片信息
# 添加噪音
noisy = np.random.normal(digits.data, 2)
# 显示添加噪声的图片
fig, axes = plt.subplots(3, 8,  figsize=(8, 4), subplot_kw={"xticks": [], "yticks": []})
for i, ax in enumerate(axes.flat):
    ax.imshow(noisy[i, :].reshape(8, 8), cmap="gray")  # imshow用于填入图片信息
# PCA降噪
pca = PCA(0.5, svd_solver="full").fit(noisy)
X_dr = pca.transform(noisy)
X_inverse = pca.inverse_transform(X_dr)
# 显示PCA降噪的图片
fig, axes = plt.subplots(3, 8,  figsize=(8, 4), subplot_kw={"xticks": [], "yticks": []})
for i, ax in enumerate(axes.flat):
    ax.imshow(X_inverse[i, :].reshape(8, 8), cmap="gray")  # imshow用于填入图片信息


plt.show()