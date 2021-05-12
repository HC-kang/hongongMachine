import numpy as np
from numpy.core.numeric import cross
import pandas as pd
from sklearn.model_selection import train_test_split
        # 관련, 필요한 라이브러리 불러오기

wine = pd.read_csv('https://bit.ly/wine-date')
        # 파일 불러오기

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
        # data는 행렬로 만들어주기 위해 

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size = 0.2, random_state=42)
        # 세트 분할

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs = -1, random_state = 42)
scores = cross_validate(rf, train_input, train_target, return_train_score = True, n_jobs=-1)
print(scores)
        # return_train_score = True 로 인해 scores에 train_score가 포함되어 출력됨.

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        # 0.9973541965122431 0.8905151032797809
        # 테스트세트가 10%가량 낮음,, 과적합. 

rf.fit(train_input, train_target)
print(rf.feature_importances_)
        # [0.23167441 0.50039841 0.26792718]

rf = RandomForestClassifier(oob_score = True, n_jobs = -1, random_state = 42)
        # oob_score : out of bag - 부트스트랩 샘플에 포함되지 않은 샘플.
        # 일정의 검정세트로 작동 
rf.fit(train_input, train_target)
print(rf.oob_score_)
        # 0.8934000384837406



#####
# 엑스트라 트리 - 부트스트랩 샘플을 쓰지 않음.
###
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_jobs = -1, random_state = 42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        # 0.9974503966084433 0.8887848893166506

et.fit(train_input, train_target)
print(et.feature_importances_)
        # [0.20183568 0.52242907 0.27573525]


#####
# 그레디언트 부스팅 - 중요. 엄청.
###

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs = -1)
        # n_estimators = None = 100 = 기본값.
        # learning rate 기본값 : 0.1 

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        # 0.8881086892152563 0.8720430147331015

gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        # 0.9464595437171814 0.8780082549788999

gb.fit(train_input, train_target)
print(gb.feature_importances_)
        # [0.15872278 0.68010884 0.16116839]
        # 랜덤 포레스트보다 당도라는 특성에 좀 더 집중.

#####
# 히스토그램 기반 부스팅
###
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(random_state = 42)
scores = cross_validate(hgb, train_input, train_target, return_train_score = True, n_jobs = -1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        # 0.9321723946453317 0.8801241948619236


from sklearn.inspection import permutation_importance

hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats=10, random_state = 42, n_jobs=-1)
print(result.importances_mean)
        # [0.08876275 0.23438522 0.08027708]
        # n_repeat = None = 5 :  
print(result.importances)
print(result.importances_std)



result = permutation_importance(hgb, test_input, test_target, n_repeats = 10, random_state = 42, n_jobs = -1)
print(result.importances_mean)
        # [0.05969231 0.20238462 0.049     ]

hgb.score(test_input, test_target)
        # 0.8723076923076923
print(result.importances)
print(result.importances_std)


#####
# XHBoost
###

from xgboost import XGBClassifier
xgb = XGBClassifier(tree_method='hist', random_state = 42)
