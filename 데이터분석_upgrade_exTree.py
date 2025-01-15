import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, classification_report, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint
from sklearn.metrics import RocCurveDisplay

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드
try:
    df = pd.read_csv('병합된_연도별_감염병_및_의료인력_현황.csv')
    print("데이터 로드 완료")
    print(df.head())
except FileNotFoundError:
    print("오류: 파일을 찾을 수 없습니다.")
    exit()

# 2. 데이터 전처리
df = df.drop('연도', axis=1)
independent_variables = [
    '면허미보유_medsol', '간호_medsol', '약_medsol', '방사선_medsol', '임상병리_medsol',
    '치과위생_medsol', '물리치료_medsol', '치과가공_medsol', '응급구조_medsol', '간호조무_medsol',
    '통합_medsol', '간호조무사_mednco', '임상병리사_mednco', '방사선사_mednco',
    '물리치료사_mednco', '치위생사_mednco', '응급구조사_mednco', '간호사_medciv',
    '간호조무사_medciv', '약사_medciv', '임상병리사_medciv', '방사선사_medciv',
    '물리치료사_medciv', '치위생사_medciv', '치기공사_medciv', '응급구조사_medciv'
]
dependent_variables = [
    'A형 간염', '결 핵', '렙토스피라증', '말라리아', '매 독', '세균성 이질', '수 두',
    '수막구균성 수막염', '신증후군 출혈열', '유행성 이하선염', '장티푸스', '쯔쯔가무시증',
    '파상풍', '폐 렴', '홍 역'
]

# 결측치 확인 및 처리
print("\n결측치 확인:")
print(df.isnull().sum())
df = df.dropna()  # 결측치가 있는 행 제거

X = df[independent_variables]
y = df[dependent_variables]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KFold를 사용한 교차 검증
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # ExtraTreesClassifier 모델 학습 및 평가
    print("\n------ ExtraTreesClassifier 모델 학습 ------")

    # 1. 기본 ExtraTreesClassifier 모델 학습 및 평가
    et = ExtraTreesClassifier(n_jobs = -1, random_state=42)
    scores = cross_validate(et, X_train, y_train, return_train_score=True,
                            n_jobs = -1, cv = 10)
    print("기본 ExtraTreesClassifier 교차 검증 결과:")
    print(scores)
    print(f"기본 ExtraTreesClassifier 평균 train score: {np.mean(scores['train_score'])}, 평균 test score: {np.mean(scores['test_score'])}")

    # 2. n_estimators 조정 ExtraTreesClassifier 모델 학습 및 평가
    et_tuned = ExtraTreesClassifier(n_estimators=200, n_jobs = -1, random_state=42)
    scores_tuned = cross_validate(et_tuned, X_train, y_train, return_train_score=True,
                            n_jobs = -1, cv = 10)
    print("\nn_estimators 조정 ExtraTreesClassifier 교차 검증 결과:")
    print(scores_tuned)
    print(f"n_estimators 조정 ExtraTreesClassifier 평균 train score: {np.mean(scores_tuned['train_score'])}, 평균 test score: {np.mean(scores_tuned['test_score'])}")

    et_tuned.fit(X_train, y_train)

    f_imp_v = et_tuned.feature_importances_
    f_imp_v = pd.Series(f_imp_v, index = X.columns)
    f_top = f_imp_v.sort_values(ascending = False)

    plt.figure(figsize = (6, 3))
    sns.barplot(x = f_top, y = f_top.index)
    plt.title('Top feature importance')
    plt.show()
    
    y_train_pred = et_tuned.predict(X_train)
    y_test_pred = et_tuned.predict(X_test)
    print(f"y_train_pred[:5]: {y_train_pred[:5]}")
    print(f"y_train.values[:5]: {y_train.values[:5]}")
    print(f"y_test_pred[:5]: {y_test_pred[:5]}")
    print(f"y_test.values[:5]: {y_test.values[:5]}")

    # 정확도 (분류 모델이므로 정확도 사용)
    print(f"훈련 데이터 정확도: {np.mean(y_train == y_train_pred)}")
    print(f"테스트 데이터 정확도: {np.mean(y_test == y_test_pred)}")

    # 성능평가 (분류 모델)
    print("\nConfusion Matrix (train):")
    print(confusion_matrix(y_train, y_train_pred))
    print("Confusion Matrix (test):")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report (train):")
    print(classification_report(y_train, y_train_pred, zero_division=0))
    print("Classification Report (test):")
    print(classification_report(y_test, y_test_pred, zero_division=0))
    # 3. RandomizedSearchCV를 사용한 하이퍼파라미터 튜닝
    params = {'min_impurity_decrease': uniform(0.0001, 0.0011),
            'max_depth': randint(2, 12),
            'min_samples_split': randint(2, 52),
            'min_samples_leaf': randint(1, 26)}
    
    rs = RandomizedSearchCV(ExtraTreesClassifier(n_jobs = -1, random_state=42),
                            params, n_iter = 100, n_jobs = -1, random_state=42, cv = 2)
    rs.fit(X_train, y_train)
    
    best_model = rs.best_estimator_
    print(f"\n최적 모델:\n{best_model}")
    print(f"최적 모델 훈련 데이터 점수: {best_model.score(X_train, y_train)}")
    print(f"최적 모델 테스트 데이터 점수: {best_model.score(X_test, y_test)}")
    
    print(f"\n최적 모델의 하이퍼파라미터: \n{rs.best_params_}")
    
    print(f"교차 검증 결과 params:\n {rs.cv_results_['params'][:5]}")
    print(f"교차 검증 결과 mean_test_score 개수: {len(rs.cv_results_['mean_test_score'])}")
    
    print(f"최고 교차 검증 점수: {np.max(rs.cv_results_['mean_test_score'])}")
    
    f_imp_v = best_model.feature_importances_
    f_imp_v = pd.Series(f_imp_v, index = X.columns)
    f_top = f_imp_v.sort_values(ascending = False)
    
    plt.figure(figsize = (6, 3))
    sns.barplot(x = f_top, y = f_top.index)
    plt.title('Top feature importance')
    plt.show()
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    print(f"최적 모델 y_train_pred[:5]: {y_train_pred[:5]}")
    print(f"최적 모델 y_train.values[:5]: {y_train.values[:5]}")
    print(f"최적 모델 y_test_pred[:5]: {y_test_pred[:5]}")
    print(f"최적 모델 y_test.values[:5]: {y_test.values[:5]}")
    
    # 정확도 (분류 모델이므로 정확도 사용)
    print(f"최적 모델 훈련 데이터 정확도: {np.mean(y_train == y_train_pred)}")
    print(f"최적 모델 테스트 데이터 정확도: {np.mean(y_test == y_test_pred)}")
    
    # 성능평가 (분류 모델)
    print("\n최적 모델 Confusion Matrix (train):")
    print(confusion_matrix(y_train, y_train_pred))
    print("최적 모델 Confusion Matrix (test):")
    print(confusion_matrix(y_test, y_test_pred))
    print("\n최적 모델 Classification Report (train):")
    print(classification_report(y_train, y_train_pred, zero_division=0))
    print("최적 모델 Classification Report (test):")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    
    
    # ROC 커브 및 AUC 계산 (각 질병별 반복)
    for i, disease in enumerate(dependent_variables):
        try:
            y_true = y_test[disease]
            y_pred_proba = best_model.predict_proba(X_test)
            if y_pred_proba.shape[1] == 2:  # 이진 분류
              y_pred_proba1 = y_pred_proba[:, 1]
              fpr, tpr, cut = roc_curve(y_true, y_pred_proba1)
              fig = plt.figure(figsize=(5,4))
              ax = fig.add_subplot()
              RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
              ax.plot([0, 1], [0, 1], color='red')
              ax.set_title(f"ROC Curve for {disease}")
              ax.legend()
              plt.show()
              roc_auc = auc(fpr, tpr)
              print(f"AUC Score for {disease}: {roc_auc}")

            else:  # 다중 분류
                print(f"{disease}는 다중 분류이므로 ROC 커브 및 AUC를 계산하지 않습니다.")
                print(f"predict_proba shape: {y_pred_proba.shape}")
        except Exception as e:
            print(f"{disease}에 대한 ROC 커브 및 AUC 계산 중 오류 발생: {e}")


'''
from sklearn.model_selection import cross_validate
from sklearn.ensemble import ExtraTreesClassifier
# DecisionTreeClassifier(splitter = 'random') 사용
et = ExtraTreesClassifier(n_jobs = -1, random_state=42)
scores = cross_validate(et, X_train, y_train, return_train_score=True,
                        n_jobs = -1, cv = 10)
print(scores)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

from sklearn.model_selection import cross_validate
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_estimators = 200, n_jobs = -1, random_state=42)
scores = cross_validate(et, X_train, y_train, return_train_score=True,
                        n_jobs = -1, cv = 10)
print(scores)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

et.fit(X_train, y_train)

et.feature_importances_

f_imp_v = et.feature_importances_

f_imp_v = pd.Series(f_imp_v, index = X.columns)

f_imp_v.sort_values(ascending = False)

f_top = f_imp_v.sort_values(ascending = False)
plt.figure(figsize = (6, 3))
sns.barplot(x = f_top, y = f_top.index)
plt.title('Top feature importance')
plt.show()

y_train_pred = et.predict(X_train)
y_test_pred = et.predict(X_test)
print(y_train_pred[:5])
print(y_train.values[:5])
print(y_test_pred[:5])
print(y_test.values[:5])

# 정확도
print(np.mean(y_train == y_train_pred))
print(np.mean(y_test == y_test_pred))

# 성능평가
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train, y_train_pred))
print(confusion_matrix(y_test, y_test_pred))

from sklearn.metrics import classification_report
print(classification_report(y_train, y_train_pred))
print(classification_report(y_test, y_test_pred))

from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
# 하이퍼파라미터 값 설정
params = {'min_impurity_decrease': uniform(0.0001, 0.0011),
          'max_depth': randint(2, 12),
          'min_samples_split': randint(2, 52),
          'min_samples_leaf': randint(1, 26)}

rs = RandomizedSearchCV(ExtraTreesClassifier(n_jobs = -1, random_state=42),
                        params, n_iter = 100, n_jobs = -1, random_state=42)

rs.fit(X_train, y_train)

# 최적 모델
best_model = rs.best_estimator_
best_model

# 정확도
print(best_model.score(X_train, y_train))
print(best_model.score(X_test, y_test))

# 최적 모델의 매개변수
rs.best_params_

rs.cv_results_['params']

len(rs.cv_results_['mean_test_score'])

np.max(rs.cv_results_['mean_test_score'])

best_model.feature_importances_

f_imp_v = best_model.feature_importances_

f_imp_v = pd.Series(f_imp_v, index = X.columns)

f_imp_v.sort_values(ascending = False)

f_top = f_imp_v.sort_values(ascending = False)
plt.figure(figsize = (6, 3))
sns.barplot(x = f_top, y = f_top.index)
plt.title('Top feature importance')
plt.show()

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)
print(y_train_pred[:5])
print(y_train.values[:5])
print(y_test_pred[:5])
print(y_test.values[:5])

# 정확도
print(np.mean(y_train == y_train_pred))
print(np.mean(y_test == y_test_pred))

# 성능평가
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train, y_train_pred))
print(confusion_matrix(y_test, y_test_pred))

from sklearn.metrics import classification_report
print(classification_report(y_train, y_train_pred))
print(classification_report(y_test, y_test_pred))

# ROC - 1 기준
y_test_proba1 = best_model.predict_proba(X_test)[:,1]

# 더미변수
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

print(y_test)
y_test_new = label_encoder.fit_transform(y_test)
print(y_test_new)

# FPR, TPR, Cutoff
from sklearn.metrics import roc_curve
fpr, tpr, cut = roc_curve(y_test_new, y_test_proba1)

# ROC Curve
from sklearn.metrics import RocCurveDisplay
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot()
RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=ax)
ax.plot([0, 1], [0, 1], color='red')
ax.legend()
plt.show()

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test_new, y_test_proba1))

from sklearn.metrics import roc_auc_score, auc
print(roc_auc_score(y_test, y_test_proba1))
print(auc(fpr, tpr))
'''