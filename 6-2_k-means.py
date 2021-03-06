import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/hongongMachine')

import numpy as np
import pandas as pd
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 10000)
        # 그림의 100x100 성분을 하나로 평탄화
fruits.shape
fruits_2d

from sklearn.cluster import KMeans

km = KMeans(n_clusters = 3, random_state = 42)
km.fit(fruits_2d)       # 타겟 없음.

print(km.labels_)

print(np.unique(km.labels_, return_counts=True))
        # (array([0, 1, 2, 3, 4, 5], dtype=int32), 
        # array([110,  32,  91,  16,  35,  16]))
        # return_counts = 기본값은 False. True로 하면 해당 unique 값의
        # 개수까지 알려줌. 

import matplotlib.pyplot as plt
def draw_fruits(arr, ratio=1):
    n = len(arr)    # n은 샘플의 개수
    # 한 줄에 10개씩 이미지 그리기. 샘플 개수는 10으로 나눠서 전체 행 개수 계산
    rows = int(np.ceil(n/10))
    # 행이 1개이면 열의 개수는 샘플 개수임. 그렇지 않으면 10개??
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize = (cols*ratio, rows*ratio), squeeze = False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:
                axs[i, j].imshow(arr[i*10 + j], cmap = 'gray_r')
            axs[i, j].axis('off')
    plt.show()

draw_fruits(fruits[km.labels_==0])
draw_fruits(fruits[0:40])


#####
# 클러스터 중심
###

draw_fruits(km.cluster_centers_.reshape(-1, 100,100), ratio =3)

print(km.transform(fruits_2d[100:101]))

print(km.predict(fruits_2d[100:101]))

print(km.n_iter_) 

fruit

#####
# 최적의 k 찾기
###

inertia = []
for k in range(2, 10):
    km = KMeans(n_clusters = k, random_state = 42)
    km.fit(fruits_2d[200:300])
    inertia.append(km.inertia_)
plt.plot(range(2, 10), inertia)
plt.show()

plt.imshow(fruits[100], cmap = 'gray_r')