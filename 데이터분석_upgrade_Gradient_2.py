import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

# 모델 성능 시각화 함수
def plot_all_diseases_performance(results, dependent_variables):
    """모든 질병에 대한 모델 성능(MSE, R2, MAE)을 막대 그래프로 시각화합니다."""
    
    num_diseases = len(dependent_variables)
    x = np.arange(num_diseases)
    width = 0.2
    
    plt.figure(figsize=(20, 10))

    mse_values = [results[disease]['mse'] for disease in dependent_variables]
    r2_values = [results[disease]['r2'] for disease in dependent_variables]
    mae_values = [results[disease]['mae'] for disease in dependent_variables]
    
    plt.bar(x - width, mse_values, width, label='MSE')
    plt.bar(x, r2_values, width, label='R2')
    plt.bar(x + width, mae_values, width, label='MAE')
    
    plt.xticks(x, dependent_variables, rotation=45, ha='right')
    plt.xlabel('감염병')
    plt.ylabel('평가 지표')
    plt.title('모든 질병에 대한 GradientBoostingRegressor 모델 성능')
    plt.legend()
    plt.tight_layout()
    plt.show()

# KFold를 사용한 교차 검증
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 각 질병별 모델 학습 및 평가 결과 저장
    all_results = {}

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
    
        
        y_test_pred = gb_learning_rate.predict(X_test)

        # 성능 평가 (회귀 모델)
        mse = mean_squared_error(y_test[disease], y_test_pred)
        r2 = r2_score(y_test[disease], y_test_pred)
        mae = mean_absolute_error(y_test[disease], y_test_pred)
        print(f"  MSE: {mse:.3f}, R2: {r2:.3f}, MAE: {mae:.3f}")

        all_results[disease] = {'mse': mse, 'r2': r2, 'mae': mae}
    # 모델 성능 시각화
    plot_all_diseases_performance(all_results, dependent_variables)