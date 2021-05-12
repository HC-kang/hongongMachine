import pandas as pd
import numpy as np

# 파일 불러오기, object 생성
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()

# 유니크값 고유값 뽑아오기
print(pd.unique(fish['Species']))

# fish DataFrame 에서 필요한 5열을 numpy로 저장
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()

print(fish_input[:5])

# target 만들기
fish_target = fish['Species'].to_numpy()

print(fish_target[:5])

# sklearn으로 트레인, 타깃세트 분리
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state = 42
)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input) # 학습 데이터
test_scaled = ss.transform(test_input)   # 테스트 데이터


# k-nearest 확률 예측
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

# sklearn 에서는 키값들의 순서가 바뀌니 확인 필요
print(kn.classes_)

print(kn.predict(test_scaled[:5]))
print(test_target[:5])
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))

print(kn.classes_)

distances, indexs = kn.kneighbors(test_scaled[3:4])
print(train_target[indexs])

##########################
# Logistic Regression
###
import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

## 불리언 인덱싱
char_arr = np.array(['A', 'B', 'C','D', 'E'])
print(char_arr[[True, False, True, False, False]])
print(char_arr)

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict_proba(train_bream_smelt[:5]))

print(lr.classes_)

print(lr.coef_, lr.intercept_)

decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

from scipy.special import expit
print(expit(decisions))

## 다중 분류 시행하기
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.predict(test_scaled[:5]))

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))

print(lr.classes_)

print(lr.coef_.shape, lr.intercept_.shape)


decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))


# 사이파이 소프트맥스
from scipy.special import softmax
proba = softmax(decision, axis = 1)
print(np.round(proba, decimals = 3))


z = np.arange(-5, 5, 0.1)
soma = softmax(z)
plt.plot(z, soma)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()