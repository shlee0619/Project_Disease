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

# 머신러닝 모델: 랜덤 포레스트
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# 예측
y_val_pred = rf_model.predict(X_val)
y_test_pred = rf_model.predict(X_test)

# 평가
val_mse = mean_squared_error(y_val, y_val_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

{
    "Validation MSE": val_mse,
    "Validation MAE": val_mae,
    "Test MSE": test_mse,
    "Test MAE": test_mae,
}
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# LSTM 입력 형태에 맞게 데이터 변환 (samples, timesteps, features)
X_train_lstm = np.expand_dims(X_train.values, axis=1)  # (samples, timesteps, features)
X_val_lstm = np.expand_dims(X_val.values, axis=1)
X_test_lstm = np.expand_dims(X_test.values, axis=1)

# LSTM 모델 설계
lstm_model = Sequential([
    LSTM(50, activation='tanh', return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2),
    LSTM(50, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(1)  # 예측값 1개 출력
])

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 조기 종료 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 훈련
lstm_history = lstm_model.fit(
    X_train_lstm, y_train,
    validation_data=(X_val_lstm, y_val),
    epochs=50,
    batch_size=4,
    callbacks=[early_stopping],
    verbose=1
)

# LSTM 예측
y_val_lstm_pred = lstm_model.predict(X_val_lstm)
y_test_lstm_pred = lstm_model.predict(X_test_lstm)

# 평가
val_mse_lstm = mean_squared_error(y_val, y_val_lstm_pred)
val_mae_lstm = mean_absolute_error(y_val, y_val_lstm_pred)
test_mse_lstm = mean_squared_error(y_test, y_test_lstm_pred)
test_mae_lstm = mean_absolute_error(y_test, y_test_lstm_pred)

{
    "Validation MSE (LSTM)": val_mse_lstm,
    "Validation MAE (LSTM)": val_mae_lstm,
    "Test MSE (LSTM)": test_mse_lstm,
    "Test MAE (LSTM)": test_mae_lstm,
}

'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Assuming X_train, X_val, X_test, y_train, y_val, y_test are already defined
# Convert input to LSTM shape (samples, timesteps, features)
X_train_lstm = np.expand_dims(X_train.values, axis=1)  # (samples, timesteps, features)
X_val_lstm = np.expand_dims(X_val.values, axis=1)
X_test_lstm = np.expand_dims(X_test.values, axis=1)

# LSTM model definition
lstm_model = Sequential([
    LSTM(50, activation='tanh', return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2),
    LSTM(50, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(1)  # Single output
])

# Compile the model
lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model training
lstm_history = lstm_model.fit(
    X_train_lstm, y_train,
    validation_data=(X_val_lstm, y_val),
    epochs=50,
    batch_size=4,
    callbacks=[early_stopping],
    verbose=1
)

# Predictions
y_val_lstm_pred = lstm_model.predict(X_val_lstm)
y_test_lstm_pred = lstm_model.predict(X_test_lstm)

# Evaluation
val_mse_lstm = mean_squared_error(y_val, y_val_lstm_pred)
val_mae_lstm = mean_absolute_error(y_val, y_val_lstm_pred)
test_mse_lstm = mean_squared_error(y_test, y_test_lstm_pred)
test_mae_lstm = mean_absolute_error(y_test, y_test_lstm_pred)

# Results
lstm_results = {
    "Validation MSE (LSTM)": val_mse_lstm,
    "Validation MAE (LSTM)": val_mae_lstm,
    "Test MSE (LSTM)": test_mse_lstm,
    "Test MAE (LSTM)": test_mae_lstm,
}

print("LSTM Results:", lstm_results)
