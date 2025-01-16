import pandas as pd

# 파일 경로 설정
file_path = '병합된_연도별_감염병_및_의료인력_현황.csv'

# 데이터 불러오기
df = pd.read_csv(file_path)
from sklearn.preprocessing import MinMaxScaler

# 독립변수와 종속변수 분리
X = df.drop(columns=['연도'])  # 독립변수
y = df['연도']  # 종속변수

# 데이터 스케일링 (정규화)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 스케일링 결과를 DataFrame으로 변환
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 데이터 시계열 분할 (2016-2018: 훈련, 2019: 검증, 2020: 테스트)
train_df = df[df['연도'] < 2019]
val_df = df[df['연도'] == 2019]
test_df = df[df['연도'] == 2020]

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# 데이터 분리
X_train = train_df.drop(columns=['연도'])
y_train = train_df['연도']
X_val = val_df.drop(columns=['연도'])
y_val = val_df['연도']
X_test = test_df.drop(columns=['연도'])
y_test = test_df['연도']

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 트리 기반 모델: XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)

# 예측
y_val_xgb_pred = xgb_model.predict(X_val)
y_test_xgb_pred = xgb_model.predict(X_test)

# 평가
val_mse_xgb = mean_squared_error(y_val, y_val_xgb_pred)
val_mae_xgb = mean_absolute_error(y_val, y_val_xgb_pred)
test_mse_xgb = mean_squared_error(y_test, y_test_xgb_pred)
test_mae_xgb = mean_absolute_error(y_test, y_test_xgb_pred)

xgb_results = {
    "Validation MSE (XGBoost)": val_mse_xgb,
    "Validation MAE (XGBoost)": val_mae_xgb,
    "Test MSE (XGBoost)": test_mse_xgb,
    "Test MAE (XGBoost)": test_mae_xgb,
}
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Convert data to CNN input shape (samples, timesteps, features)
X_train_cnn = np.expand_dims(X_train.values, axis=2)  # (samples, timesteps, features)
X_val_cnn = np.expand_dims(X_val.values, axis=2)
X_test_cnn = np.expand_dims(X_test.values, axis=2)

# Define the CNN model
cnn_model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.3),
    Dense(1)  # Single output
])

# Compile the model
cnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
cnn_history = cnn_model.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=50,
    batch_size=4,
    callbacks=[early_stopping],
    verbose=1
)

# Predictions
y_val_cnn_pred = cnn_model.predict(X_val_cnn)
y_test_cnn_pred = cnn_model.predict(X_test_cnn)

# Evaluation
val_mse_cnn = mean_squared_error(y_val, y_val_cnn_pred)
val_mae_cnn = mean_absolute_error(y_val, y_val_cnn_pred)
test_mse_cnn = mean_squared_error(y_test, y_test_cnn_pred)
test_mae_cnn = mean_absolute_error(y_test, y_test_cnn_pred)

# Results
cnn_results = {
    "Validation MSE (CNN)": val_mse_cnn,
    "Validation MAE (CNN)": val_mae_cnn,
    "Test MSE (CNN)": test_mse_cnn,
    "Test MAE (CNN)": test_mae_cnn,
}

print("CNN Results:", cnn_results)


'''
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ARIMA 모델 적용을 위해 연도별 평균값으로 간단히 예시
# 종속변수(y_train, y_val, y_test)를 기준으로 ARIMA 모델 적용
arima_order = (1, 1, 1)  # (p, d, q) 설정

# ARIMA 모델 학습 (훈련 데이터만 사용)
arima_model = ARIMA(y_train, order=arima_order)
arima_fit = arima_model.fit()

# 검증 및 테스트 데이터에 대한 예측
y_val_arima_pred = arima_fit.forecast(steps=len(y_val))
y_test_arima_pred = arima_fit.forecast(steps=len(y_test))

# 평가
val_mse_arima = mean_squared_error(y_val, y_val_arima_pred)
val_mae_arima = mean_absolute_error(y_val, y_val_arima_pred)
test_mse_arima = mean_squared_error(y_test, y_test_arima_pred)
test_mae_arima = mean_absolute_error(y_test, y_test_arima_pred)

arima_results = {
    "Validation MSE (ARIMA)": val_mse_arima,
    "Validation MAE (ARIMA)": val_mae_arima,
    "Test MSE (ARIMA)": test_mse_arima,
    "Test MAE (ARIMA)": test_mae_arima,
}

print("ARIMA results:", arima_results)

# import matplotlib.pyplot as plt

# # 실제값과 예측값 시각화
# plt.figure(figsize=(10, 6))
# plt.plot(y_val.index, y_val, label='Validation Actual', marker='o')
# plt.plot(y_val.index, y_val_arima_pred, label='Validation Predicted (ARIMA)', marker='x')
# plt.plot(y_test.index, y_test, label='Test Actual', marker='o', linestyle='--')
# plt.plot(y_test.index, y_test_arima_pred, label='Test Predicted (ARIMA)', marker='x', linestyle='--')

# plt.title('ARIMA Model: Actual vs Predicted')
# plt.xlabel('Index (Time Steps)')
# plt.ylabel('Values')
# plt.legend()
# plt.grid()
# plt.show()
import matplotlib.pyplot as plt

# ARIMA 예측값과 실제값 비교 시각화
plt.figure(figsize=(10, 6))

# 검증 데이터 시각화
plt.plot(y_val.index, y_val, label='Validation Actual', marker='o', color='blue')
plt.plot(y_val.index, y_val_arima_pred, label='Validation Predicted (ARIMA)', marker='x', color='cyan')

# 테스트 데이터 시각화
plt.plot(y_test.index, y_test, label='Test Actual', marker='o', linestyle='--', color='orange')
plt.plot(y_test.index, y_test_arima_pred, label='Test Predicted (ARIMA)', marker='x', linestyle='--', color='red')

# 그래프 꾸미기
plt.title('ARIMA Model: Actual vs Predicted for Infectious Disease Cases')
plt.xlabel('Time Steps (Index)')
plt.ylabel('Infectious Disease Cases')
plt.legend()
plt.grid()

# 이미지 출력
plt.tight_layout()
plt.show()
