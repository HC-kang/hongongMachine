import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/hongongMachine')

import numpy as np
import pandas as pd

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(300, 10000)
print(fruits_2d)

from sklearn.decomposition import PCA
pca = PCA(n_components = 50)
pca.fit(fruits_2d)

print(pca.components_.shape)

draw_fruits(pca.components_.reshape(-1, 100, 100))

print(fruits_2d.shape)

fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)

fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print()

print(np.sum(pca.explained_variance_ratio_))

plt.plot(pca.explained_variance_ratio_)
plt.show()


#####
# 다른 알고리즘과 함께 사용하기
###

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

target = np.array([0]*100 + [1]*100 + [2]*100)

from sklearn.model_selection import cross_validate
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))
        # 0.9966666666666667
print(np.mean(scores['fit_time']))
        # 0.49727940559387207

scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
        # 1.0
print(np.mean(scores['fit_time']))
        # 0.026801538467407227


pca = PCA(n_components = 0.5)
pca.fit(fruits_2d)

print(pca.n_components_)
        # 2

fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
        # 0.9933333333333334
print(np.mean(scores['fit_time']))
        # 0.04217162132263184



from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, random_state = 42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts = True))
        # (array([0, 1, 2], dtype=int32), array([110,  99,  91]))

import matplotlib.pyplot as plt
for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print()

for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:, 0], data[:, 1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()