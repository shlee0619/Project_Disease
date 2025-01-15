import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

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
r2_scores = []

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 3. 모델 정의 (랜덤 포레스트)
    rf = RandomForestRegressor(oob_score=True, n_jobs = -1, random_state=42)

    # 4. 모델 학습 및 평가
    print("\n------ Random Forest 모델 학습 ------")

    # 랜덤 포레스트 모델 학습 (전체 종속변수에 대해 학습)
    rf.fit(X_train, y_train)
    # 특성 중요도 출력
    f_imp_v = rf.feature_importances_
    f_imp_v = pd.Series(f_imp_v, index = X.columns)
    f_imp_v_sorted = f_imp_v.sort_values(ascending = False)
    print(f"\n특성 중요도:\n{f_imp_v_sorted}")

    # 성능 평가
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    print(f"OOB Score: {rf.oob_score_}")
    r2_train = rf.score(X_train, y_train)
    print(f'최적의 모델 점수(train): {r2_train}')
    r2_test = rf.score(X_test, y_test)
    print(f'최적의 모델 점수(test): {r2_test}')
    r2_scores.append(r2_test)


    from scipy.stats import uniform, randint
    from sklearn.model_selection import RandomizedSearchCV
    # 하이퍼파라미터 값 설정
    params = {'min_impurity_decrease': uniform(0.0001, 0.0011),
            'max_depth': randint(2, 12),
            'min_samples_split': randint(2, 52),
            'min_samples_leaf': randint(1, 26)}

    rs = RandomizedSearchCV(RandomForestRegressor(n_jobs = -1, random_state=42),
                            params, n_iter = 100, n_jobs = -1, random_state=42, cv=2)  # cv 파라미터 변경

    rs.fit(X_train, y_train) # X_train -> X_train_scaled 변경

    # 최적 모델
    best_model = rs.best_estimator_
    best_model

    # 정확도
    print(best_model.score(X_train, y_train)) # X_train -> X_train_scaled 변경
    print(best_model.score(X_test, y_test)) # X_test -> X_test_scaled 변경

    len(rs.cv_results_['mean_test_score'])

    np.max(rs.cv_results_['mean_test_score'])
    y_train_pred = best_model.predict(X_train) # X_train -> X_train_scaled 변경
    y_test_pred = best_model.predict(X_test) # X_test -> X_test_scaled 변경
    print(y_train_pred[:5])
    print(y_train.values[:5])
    print(y_test_pred[:5])
    print(y_test.values[:5])

    # 정확도
    print(np.mean(y_train == y_train_pred))
    print(np.mean(y_test == y_test_pred))

    

print(f"\n평균 테스트 R2: {np.mean(r2_scores)}")