import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size = 0.2, random_state=42
)

sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42
)

print(sub_input.shape, val_input.shape)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))

from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores)

import numpy as np
print(np.mean(scores['test_score']))

from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input, train_target, cv = StratifiedKFold())
print(np.mean(scores['test_score']))

splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))


from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease':[0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params,n_jobs=-1)
gs.fit(train_input, train_target)
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(gs.best_params_)

print(gs.cv_results_['mean_test_score'])
gs.cv_results_

best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])

params = {'min_impurity_decrease':np.arange(0.0001, 0.001, 0.0001),
          'max_depth' : range(5, 20, 1),
          'min_samples_split' : range(2, 100, 10)
          }

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

print(gs.best_params_)

print(np.max(gs.cv_results_['mean_test_score']))


from scipy.stats import uniform, randint
rgen = randint(0, 10)
rgen.rvs(10)

np.unique(rgen.rvs(1000), return_counts=True)

params = {
    'min_impurity_decrease' : uniform(0.0001, 0.001),
    'max_depth' : randint(20, 50),
    'min_samples_split' : randint(2, 25),
    'min_samples_leaf' : randint(1, 25),
}

from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state = 42)
gs.fit(train_input, train_target)

print(gs.best_params_)

print(np.max(gs.cv_results_['mean_test_score']))

dt = gs.best_estimator_
print(dt.score(test_input, test_target))

#######################
# 0512 수업, 5-2 교차 검증과 그리드 서치
import pandas as pd
import numpy as np

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
        # 자료 불러오고 독립변수 data와 종속변수 target으로 분할.

from sklearn.model_selection import train_test_split
        # train 과 test 데이터로 나누기 위한 train_test_split 함수 불러오기

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size = 0.2, random_state = 42)
        # 1차적으로 train, test 셋 나눠주기, test_size는 설정 안하면 0.25 / random_state는 seed를 고정하기 위함.

sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size = 0.2, random_state = 42)
        # train 셋을 검증셋과 트레인셋으로 다시 한 번 8:2로 분할.

print(sub_input.shape, val_input.shape)
        # (4157, 3) (1040, 3)
        # 분할이 제대로 되었는지 shape 함수를 통해 확인

from sklearn.tree import DecisionTreeClassifier
        # sklearn의 의사결정나무 분류기 불러오기

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
        # 훈련셋을 통한 학습

print(dt.score(sub_input, sub_target))
        # 0.977
print(dt.score(val_input, val_target))
        # 0.864

#####
# 교차검증
###
from sklearn.model_selection import cross_validate
        # 교차검증 함수 불러오기
scores = cross_validate(dt, train_input, train_target)
print(scores)
        # cross_validate 함수는 cs = None인경우 기본적으로 5-fold cross validation을 수행
        # 결과적으로 'fit_time', 'score_time', 'test_score'의 세 가지 키를 가진 딕셔너리를 반환함 

print(np.mean(scores['test_score']))
        # 0.8697
        # 'test_score' 필드의 평균값을 확인

from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input, train_target, cv = StratifiedKFold())
print(np.mean(scores['test_score']))
        # 0.855300214703487

splitter = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
scores = cross_validate(dt, train_input, train_target, cv = splitter)
print(np.mean(scores['test_score']))
        # 0.8574181117533719
##################
# # 2회차
# import numpy as np
# print(np.mean(scores['test_score']))

# from sklearn.model_selection import StratifiedKFold
# scores = cross_validate(dt, train_input, train_target, cv = StratifiedKFold())
# print(np.mean(scores['test_scores']))

# splitter = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
# scores = cross_validate(dt, train_input, train_target, cv = splitter)
# print(np.mean(scores['test_score']))

#####
# 그리드 서치
###
from sklearn.model_selection import GridSearchCV
        # GridSearchCV 함수 불러오기
params = {'min_impurity_decrease' : [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
        # GridSearchCV에 적용할 parameter 지정해주기
gs = GridSearchCV(DecisionTreeClassifier(random_state = 42), params, n_jobs = -1)
        # n_jobs 는 병렬작업 수 지정. -1이면 가용한 모든 연산을 동시에 진행하라는 뜻.
        # cv를 지정 안했으니 기본값 5로 들어감.  
        # 최상의 평균검증 점수가 나오는 매개변수 조합을 찾아서 gs에 넣어두기.

gs.fit(train_input, train_target)
        # 5종류의 params 를 활용해서 train set 학습시키기

dt = gs.best_estimator_
print(dt.score(train_input, train_target))
        # 가장 우수한 성능을 가진 param의 점수 0.962

print(gs.best_params_)
        # 최고성능 param은 0.0001

print(gs.cv_results_['mean_test_score'])
        # [0.86819297 0.86453617 0.86492226 0.86780891 0.86761605]

best_index = np.argmax(gs.cv_results_['mean_test_score'])
        # np.argmax 로 array gs.cv_results_ 의 mean_test_score 필드 내 최고값의 인덱스값 출력.

print(gs.cv_results_['params'][best_index])
        # 출력된 인덱스값을 가지고 value 값 출력

params = {'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001),
          'max_depth' : range(5, 20, 1),
          'min_samples_split' : range(2, 100, 10)}
        # 기존 params에서 max_depth 와 min sample split을 추가하되 범위를 주고 지정함.

gs = GridSearchCV(DecisionTreeClassifier(random_state = 42), params, n_jobs = -1)
        # params 외에 다른 조건을 위와 동일하게 세팅하여 인스턴스 생성

gs.fit(train_input, train_target)
        # params 에 대해 생각해보면, min_impurity_decrease 9종, max_depth 15종, 
        # min_samples_split 10종으로 9x15x10 = 1,350 가지 경우의수를 뽑아줌.
        # 여기에 추가적으로 cv = None이므로 기본 5회 적용되어 총 6750회 모델 검증


print(gs.best_params_)
        # {'max_depth': 14, 'min_impurity_decrease': 0.0004, 'min_samples_split': 12}

print(np.max(gs.cv_results_['mean_test_score']))
        # 0.868


##################################

from scipy.stats import uniform, randint
        # scipy의 stats 서브 패키지에 있는 uniform과 randint는 모두 균등분포에서 샘플링한다고 함.
        # randint : 정수, uniform : 실수
rgen = randint(0, 10)
rgen
rgen.rvs(10)
        # 1~10까지 정수 10개 생성

np.unique(rgen.rvs(1000), return_counts = True)

ugen = uniform(0, 1)
ugen.rvs(10)
        # 1~10까지 실수 10개 생성

params = {
    'min_impurity_decrease' : uniform(0.0001, 0.001),
    'max_depth' : randint(20, 50),
    'min_samples_leaf' : randint(2, 25),
    'min_samples_leaf' : randint(1, 25)
}
        # 이번에는 params를 수동으로 작성하지 않고 uniform 과 randint 함수를 통해 생성함.
        # parameter 의 종류는 해당 함수의 공식문서를 확인하자.


from sklearn.model_selection import RandomizedSearchCV
        # 위의 그리드 서치와 비슷하게, 랜덤 서치 함수를 불러와준다.

gs = RandomizedSearchCV(DecisionTreeClassifier(random_state = 42), params, n_iter = 100, n_jobs = -1, random_state = 42)
        # n_iter 는 지정된 params 의 범위 내에서 랜덤하게 시행하는 횟수이고, 기본값은 10회. 여기서는 100회로 지정.
        # 숫자가 클수록 성능은 잘 나올수 있겠지만, 소요시간에 늘어남.

gs.fit(train_input, train_target)

print(gs.best_params_)
        #{'max_depth': 36, 말단 노드까지의 층 수
        # 'min_impurity_decrease': 0.00041435598107632666,  - 최소 순수도 감소폭
        # 'min_samples_leaf': 4} - 말단 노드에 있어야할 최소한의 샘플 수

print(np.max(gs.cv_results_['mean_test_score']))
        # 0.8697336566224921

dt = gs.best_estimator_

print(dt.score(test_input, test_target))
        # 0.8592307692307692


# 확인 문제s

gs = RandomizedSearchCV(DecisionTreeClassifier(splitter = 'random', random_state = 42), params, n_iter = 100, n_jobs = -1, random_state=42)
gs.fit(train_input, train_target)

print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score']))

dt = gs.best_estimator_
print(dt.score(test_input, test_target))

