import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint

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
    
    # 각 질병별 모델 학습 및 평가
    for disease in dependent_variables:
        print(f"\n------ GradientBoostingRegressor 모델 학습 ({disease}) ------")
        
        # 1. 기본 GradientBoostingRegressor 모델 학습 및 평가
        gb = GradientBoostingRegressor(random_state=42)
        gb.fit(X_train, y_train[disease])
        print(f"기본 GradientBoostingRegressor 결과 ({disease}):")
        print(f"  train score: {gb.score(X_train, y_train[disease])}, test score: {gb.score(X_test, y_test[disease])}")
        
        # 2. subsample 조정 GradientBoostingRegressor 모델 학습 및 평가
        gb_subsample = GradientBoostingRegressor(subsample=0.1, random_state=42)
        gb_subsample.fit(X_train, y_train[disease])
        print(f"\nsubsample 조정 GradientBoostingRegressor 결과 ({disease}):")
        print(f"  train score: {gb_subsample.score(X_train, y_train[disease])}, test score: {gb_subsample.score(X_test, y_test[disease])}")
        
        # 3. n_estimators 조정 GradientBoostingRegressor 모델 학습 및 평가
        gb_n_estimators = GradientBoostingRegressor(n_estimators=200, random_state=42)
        gb_n_estimators.fit(X_train, y_train[disease])
        print(f"\nn_estimators 조정 GradientBoostingRegressor 결과 ({disease}):")
        print(f"  train score: {gb_n_estimators.score(X_train, y_train[disease])}, test score: {gb_n_estimators.score(X_test, y_test[disease])}")

        # 4. learning_rate 조정 GradientBoostingRegressor 모델 학습 및 평가
        gb_learning_rate = GradientBoostingRegressor(learning_rate=0.2, random_state=42)
        gb_learning_rate.fit(X_train, y_train[disease])
        print(f"\nlearning_rate 조정 GradientBoostingRegressor 결과 ({disease}):")
        print(f"  train score: {gb_learning_rate.score(X_train, y_train[disease])}, test score: {gb_learning_rate.score(X_test, y_test[disease])}")
    
    
        f_imp_v = gb_learning_rate.feature_importances_
        f_imp_v = pd.Series(f_imp_v, index = X.columns)
        f_top = f_imp_v.sort_values(ascending = False)
        
        plt.figure(figsize = (6, 3))
        sns.barplot(x = f_top, y = f_top.index)
        plt.title(f'Top feature importance ({disease})')
        plt.show()
        
        y_train_pred = gb_learning_rate.predict(X_train)
        y_test_pred = gb_learning_rate.predict(X_test)
        print(f"y_train_pred[:5]: {y_train_pred[:5]}")
        print(f"y_train.values[:5]: {y_train[disease].values[:5]}")
        print(f"y_test_pred[:5]: {y_test_pred[:5]}")
        print(f"y_test.values[:5]: {y_test[disease].values[:5]}")

        # 성능 평가 (회귀 모델)
        mse = mean_squared_error(y_test[disease], y_test_pred)
        r2 = r2_score(y_test[disease], y_test_pred)
        mae = mean_absolute_error(y_test[disease], y_test_pred)
        print(f"  MSE: {mse:.3f}, R2: {r2:.3f}, MAE: {mae:.3f}")