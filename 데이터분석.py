import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
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
df = df.dropna() #결측치가 있는 행 제거

X_train, X_test, y_train, y_test = train_test_split(
    df[independent_variables], df[dependent_variables], test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    print(f"\n------ {model_name} 모델 학습 ------")
    mse_list = []
    r2_list = []
    y_pred_list = []
    
    for disease in dependent_variables:
        # 특정 감염병에 대한 y 데이터 추출
        y_train_disease = y_train[disease]
        y_test_disease = y_test[disease]
        
        # 모델 학습 및 예측
        model.fit(X_train, y_train_disease)
        y_pred_disease = model.predict(X_test)
        
        # 평가
        mse = mean_squared_error(y_test_disease, y_pred_disease)
        r2 = r2_score(y_test_disease, y_pred_disease)
        
        mse_list.append(mse)
        r2_list.append(r2)
        y_pred_list.append(y_pred_disease)
        
        print(f"  {disease} - MSE: {mse:.3f}, R2: {r2:.3f}")
    
    return mse_list, r2_list, y_pred_list

# 3. 모델 학습 및 평가 (랜덤 포레스트)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_mse, rf_r2, rf_pred = train_and_evaluate_model(rf_model, "Random Forest", X_train_scaled, y_train, X_test_scaled, y_test)

# 4. 모델 학습 및 평가 (그래디언트 부스팅)
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbr_mse, gbr_r2, gbr_pred = train_and_evaluate_model(gbr_model, "Gradient Boosting", X_train_scaled, y_train, X_test_scaled, y_test)


# 5. 결과 시각화 (모델 성능 비교)
x = np.arange(len(dependent_variables))  # x 축 값 설정
width = 0.35  # 막대 폭 설정

plt.figure(figsize=(18, 6))
plt.bar(x - width/2, rf_mse, width, label='Random Forest MSE', color = 'skyblue')
plt.bar(x + width/2, gbr_mse, width, label='Gradient Boosting MSE', color = 'lightcoral')
plt.xticks(x, dependent_variables, rotation=45, ha='right')
plt.xlabel('감염병')
plt.ylabel('MSE')
plt.title('모델별 MSE 비교')
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(18, 6))
plt.bar(x - width/2, rf_r2, width, label='Random Forest R2', color = 'skyblue')
plt.bar(x + width/2, gbr_r2, width, label='Gradient Boosting R2', color = 'lightcoral')
plt.xticks(x, dependent_variables, rotation=45, ha='right')
plt.xlabel('감염병')
plt.ylabel('R2')
plt.title('모델별 R2 비교')
plt.legend()
plt.tight_layout()
plt.show()


# # 6. 결과 시각화 (각 감염병별 예측 결과 비교). 각 감염병 별로 보여주는 방식.
# for i, disease in enumerate(dependent_variables):
#   plt.figure(figsize=(10,6))
#   plt.scatter(y_test[disease], rf_pred[i], color='skyblue', label='Random Forest', s = 20)
#   plt.scatter(y_test[disease], gbr_pred[i], color='lightcoral', label='Gradient Boosting', s = 20)
#   plt.plot(y_test[disease], y_test[disease], color='red', linestyle='-', linewidth=1)  # y=x 그래프 추가
#   plt.title(f"{disease} 예측 결과 비교")
#   plt.xlabel("실제 환자 수")
#   plt.ylabel("예측 환자 수")
#   plt.legend()
#   plt.tight_layout()
#   plt.show()

# # 6. 결과 시각화 (각 감염병별 예측 결과 비교) , 한화면에 보여주는 방식.
# num_diseases = len(dependent_variables)
# fig, axes = plt.subplots(4, 4, figsize=(20, 16)) # 4x4 격자 형태로 생성
# axes = axes.flatten() # 2차원 axes를 1차원으로 펼침

# for i, disease in enumerate(dependent_variables):
#     ax = axes[i]
#     ax.scatter(y_test[disease], rf_pred[i], color='skyblue', label='Random Forest', s=20)
#     ax.scatter(y_test[disease], gbr_pred[i], color='lightcoral', label='Gradient Boosting', s=20)
#     ax.plot(y_test[disease], y_test[disease], color='red', linestyle='-', linewidth=1)
#     ax.set_title(f"{disease} 예측 결과 비교")
#     ax.set_xlabel("실제 환자 수")
#     ax.set_ylabel("예측 환자 수")
#     ax.legend()

# # 사용하지 않는 서브플롯 제거
# for j in range(num_diseases, len(axes)):
#     fig.delaxes(axes[j])

# plt.tight_layout()
# plt.show()