import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error


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
df = df.set_index('연도')
dependent_variables = [
    'A형 간염', '결 핵', '렙토스피라증', '말라리아', '매 독', '세균성 이질', '수 두',
    '수막구균성 수막염', '신증후군 출혈열', '유행성 이하선염', '장티푸스', '쯔쯔가무시증',
    '파상풍', '폐 렴', '홍 역'
]

results = {}

for disease in dependent_variables:
    print(f"\n------ {disease} 분석 시작 ------")
    
    data = df[disease].replace(0, 1) # log 변환 에러 방지
    # 로그 변환 (0을 1로 대체)
    data_log = np.log1p(data)
    
    # 데이터 분할 (시계열 순서 유지)
    train_data = data_log[:-2]  # 2016 ~ 2018
    val_data = data_log[-2:-1]  # 2019
    test_data = data_log[-1:]    # 2020

    # 정규화
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
    val_scaled = scaler.transform(val_data.values.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_data.values.reshape(-1, 1)).flatten()


    # 최적 ARIMA 파라미터 탐색 (검증 데이터 사용)
    # if len(train_scaled) > 2: # 이부분 삭제
    #     auto_model = auto_arima(train_scaled, seasonal=False, suppress_warnings=True, d=None, max_d=1) # d 값 탐색 범위를 제한함
    #     arima_order = auto_model.order
    #     print(f"최적 ARIMA 파라미터 (p,d,q): {arima_order}")

    # ARIMA 모델 학습 (훈련 데이터만 사용)
    arima_order = (1, 1, 1) # (1,1,1) 로 고정
    print(f"사용되는 ARIMA 파라미터 (p,d,q): {arima_order}")
    arima_model = ARIMA(train_scaled, order=arima_order)
    arima_fit = arima_model.fit()


    # 예측 (차분 역변환 적용)
    y_val_arima_pred = arima_fit.predict(start=len(train_scaled), end = len(train_scaled) + len(val_scaled) - 1)
    y_test_arima_pred = arima_fit.predict(start=len(train_scaled) + len(val_scaled), end = len(train_scaled) + len(val_scaled) + len(test_scaled) - 1)


    # 스케일링 역변환
    y_val_arima_pred = scaler.inverse_transform(y_val_arima_pred.reshape(-1, 1)).flatten()
    y_test_arima_pred = scaler.inverse_transform(y_test_arima_pred.reshape(-1, 1)).flatten()

    y_val_true = np.expm1(val_data).values
    y_test_true = np.expm1(test_data).values
    
    # 평가
    val_mse_arima = mean_squared_error(y_val_true, y_val_arima_pred)
    val_mae_arima = mean_absolute_error(y_val_true, y_val_arima_pred)
    test_mse_arima = mean_squared_error(y_test_true, y_test_arima_pred)
    test_mae_arima = mean_absolute_error(y_test_true, y_test_arima_pred)
    val_mape_arima = mean_absolute_percentage_error(y_val_true, y_val_arima_pred)
    test_mape_arima = mean_absolute_percentage_error(y_test_true, y_test_arima_pred)


    results[disease] = {
    "Validation MSE (ARIMA)": val_mse_arima,
    "Validation MAE (ARIMA)": val_mae_arima,
     "Validation MAPE (ARIMA)": val_mape_arima,
    "Test MSE (ARIMA)": test_mse_arima,
    "Test MAE (ARIMA)": test_mae_arima,
     "Test MAPE (ARIMA)": test_mape_arima,
    "Validation True": y_val_true,
    "Validation Predicted": y_val_arima_pred,
    "Test True": y_test_true,
    "Test Predicted": y_test_arima_pred
    }
    
    # 예측 결과 시각화
    plt.figure(figsize=(10, 6))
    
    # 검증 데이터 시각화
    plt.plot(val_data.index, y_val_true, label='Validation Actual', marker='o', color='blue')
    plt.plot(val_data.index, y_val_arima_pred, label='Validation Predicted (ARIMA)', marker='x', color='cyan')
    
    # 테스트 데이터 시각화
    plt.plot(test_data.index, y_test_true, label='Test Actual', marker='o', linestyle='--', color='orange')
    plt.plot(test_data.index, y_test_arima_pred, label='Test Predicted (ARIMA)', marker='x', linestyle='--', color='red')

    plt.title(f'ARIMA Model: Actual vs Predicted for {disease}')
    plt.xlabel('Time Steps (Index)')
    plt.ylabel('Infectious Disease Cases')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# 모든 질병에 대한 결과 출력
print("\n------ ARIMA 모델 결과 ------")
for disease, result in results.items():
  if 'Validation MSE (ARIMA)' in result:
    print(f"\n--- {disease} ---")
    print(f"   Validation MSE (ARIMA): {result['Validation MSE (ARIMA)']:.3f}")
    print(f"   Validation MAE (ARIMA): {result['Validation MAE (ARIMA)']:.3f}")
    print(f"   Validation MAPE (ARIMA): {result['Validation MAPE (ARIMA)']:.3f}")
    print(f"   Test MSE (ARIMA): {result['Test MSE (ARIMA)']:.3f}")
    print(f"   Test MAE (ARIMA): {result['Test MAE (ARIMA)']:.3f}")
    print(f"   Test MAPE (ARIMA): {result['Test MAPE (ARIMA)']:.3f}")