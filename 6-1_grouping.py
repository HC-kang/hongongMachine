import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/hongongMachine')

fruits = np.load('fruits_300.npy')
fruits

print(fruits.shape)

print(fruits[0, 0, :])

plt.imshow(fruits[0], cmap='gray')
plt.show

plt.imshow(fruits[0], cmap = 'gray_r')
plt.show

fit, axs = plt.subplots(1,2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()

apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 10000)
banana = fruits[200:300].reshape(-1, 10000)
print(apple.shape)

print(apple.mean(axis=1))

plt.hist(np.mean(apple, axis =1),alpha=0.8)
plt.hist(np.mean(pineapple, axis = 1), alpha = 0.8)
plt.hist(np.mean(banana, axis = 1), alpha = 0.8)

plt.legend(['apple', 'pineapple', 'banan'])
plt.show

fig, axs = plt.subplots(1, 3, figsize = (20, 5))
axs[0].bar(range(10000), np.mean(apple, axis = 0))
axs[1].bar(range(10000), np.mean(pineapple, axis = 0))
axs[2].bar(range(10000), np.mean(banana, axis = 0))
plt.show()


apple_mean = np.mean(apple, axis = 0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis = 0).reshape(100, 100)
banana_mean = np.mean(banana, axis = 0).reshape(100, 100)
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()


abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis = (1, 2))
print(abs_mean.shape)

# 사과
apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize = (10, 10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap = 'gray_r')
        axs[i, j].axis('off')
plt.show()

# 빠인애플
pineapple_index = np.argsort(abs_mean)[100:200]
fig, axs = plt.subplots(10, 10, figsize = (10, 10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[pineapple_index[i*10 + j]], cmap = 'gray_r')
        axs[i, j].axis('off')
plt.show()

# 빠나나
banana_index = np.argsort(abs_mean)[200:300]
fig, axs = plt.subplots(10, 10, figsize = (10, 10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[banana_index[i*10 + j]], cmap = 'gray_r')
        axs[i, j].axis('off')
plt.show()

aa = np.array([[1,1,1,1,1,2,1,1,1,1,1,1], [1,1,1,1,1,2,2,2,2,2,2,2],[1,1,1,1,1,2,1,1,1,1,1,1], [1,1,1,1,1,2,2,2,2,2,2,2]])
bb = np.mean(aa, axis = 1)
bb

import pandas as pd
dd = pd.DataFrame(aa)
dd