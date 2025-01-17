# ===============================
# 1. 라이브러리 임포트
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 사이킷런
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# 시계열 모델들
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

from prophet import Prophet

from xgboost import XGBRegressor

# 딥러닝 (LSTM, GRU)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ===============================
# 2. 데이터 불러오기
# ===============================
# (예시) pandas.read_excel() 사용
# 파일 경로/이름은 상황에 맞게 수정해 주세요.
df = pd.read_csv('modified_infectious_disease_data.csv')  

# 일단 데이터가 잘 불러와졌는지 확인
print(df.head())
print(df.info())

# ===============================
# 3. 독립변수(X), 종속변수(y) 분리
# ===============================
# 문제에서 독립변수 컬럼들:
# ['최고기온 (C)','최저기온 (C)','일 평균기온 (C)','일 평균 풍속 (m/s)',
#  '일강수량 (mm)','최심 신적설 (cm)','일 평균 상대습도 (%)','일교차 (C)']

X = df[['최고기온 (C)',
        '최저기온 (C)',
        '일 평균기온 (C)',
        '일 평균 풍속 (m/s)',
        '일강수량 (mm)',
        '최심 신적설 (cm)',
        '일 평균 상대습도 (%)',
        '일교차 (C)']].copy()

# 종속변수 2개: 호흡기_new, 매개성 (숫자형으로 가정)
y = df[['호흡기_new', '매개성']].copy()

# ===============================
# 4. 학습/테스트 데이터 분할
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test  shape:", X_test.shape)
print("y_test  shape:", y_test.shape)

# ===============================
# 5. 전처리(스케일링) + 모델 구성
# ===============================
# 예: StandardScaler + RandomForestRegressor + MultiOutputRegressor

# (1) 스케일러 핏
scaler = StandardScaler()
scaler.fit(X_train)

# (2) 스케일 적용
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# (3) 모델 선언 (랜덤포레스트를 예시로 사용)
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# (4) 다중출력 회귀 래퍼(MultiOutputRegressor)
multi_rf = MultiOutputRegressor(rf)

# ===============================
# 6. 모델 훈련
# ===============================
multi_rf.fit(X_train_scaled, y_train)

# ===============================
# 7. 모델 예측 및 평가
# ===============================
y_pred = multi_rf.predict(X_test_scaled)

# y_test, y_pred를 비교해보자
print("예측 결과 일부:\n", y_pred[:5])
print("실제 값 일부:\n", y_test[:5].values)

# ===============================
# 8. 간단한 성능 지표(MSE, MAE 등)
# ===============================
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse_0 = mean_squared_error(y_test.iloc[:,0], y_pred[:,0])
mse_1 = mean_squared_error(y_test.iloc[:,1], y_pred[:,1])
mae_0 = mean_absolute_error(y_test.iloc[:,0], y_pred[:,0])
mae_1 = mean_absolute_error(y_test.iloc[:,1], y_pred[:,1])

print(f"[호흡기_new] MSE: {mse_0:.3f}, MAE: {mae_0:.3f}")
print(f"[매개성]     MSE: {mse_1:.3f}, MAE: {mae_1:.3f}")

# ===============================
# (추가) 교차검증 예시
# ===============================
# 교차검증은 '시계열' 특징이 강하다면 TimeSeriesSplit 으로 해야 하지만,
# 단순 예시로 K-Fold cross_val_score 를 보여드립니다.

# 파이프라인 없이 간단하게 작성하면:
X_scaled = scaler.fit_transform(X)  # 전체 데이터 스케일링
scores_mse = cross_val_score(multi_rf, X_scaled, y,
                             scoring='neg_mean_squared_error',
                             cv=5)  # 5-Fold
scores_rmse = np.sqrt(-scores_mse)

print("5-Fold RMSE:", np.round(scores_rmse, 3))
print("RMSE 평균:", np.mean(scores_rmse).round(3))


#----------------------------------
# (1) 날짜/분기 인덱스 만들기
#----------------------------------
# 예: '연도' 열, '분기구분' 열을 이용해 2015-2분기 → 2015-Q2 → 임의 날짜 2015-04-01
# 실제로는 더 정밀하게 '연도-분기'를 datetime으로 변환해도 됨
def quarter_to_month(row):
    # 예: 1분기 -> 01월로, 2분기 -> 04월로, 3분기 -> 07월로, 4분기 -> 10월로 매핑
    mapping = {1: '01', 2: '04', 3: '07', 4: '10'}
    year = row['연도']
    q    = row['분기구분']
    month = mapping.get(q, '01')
    # 날짜는 임의로 1일로 설정
    return pd.to_datetime(f"{year}-{month}-01")

df['date'] = df.apply(quarter_to_month, axis=1)
df = df.sort_values('date').reset_index(drop=True)

#----------------------------------
# (2) 독립변수 / 종속변수 선택
#----------------------------------
# 독립변수(기상 정보 등)
features = [
    '최고기온 (C)', '최저기온 (C)', '일 평균기온 (C)',
    '일 평균 풍속 (m/s)', '일강수량 (mm)',
    '최심 신적설 (cm)', '일 평균 상대습도 (%)', '일교차 (C)'
]
# 종속변수: 예시로 '호흡기_new'만 사용
target = '매개성'

# (참고) '매개성'을 예측하려면 target = '매개성' 으로 교체하거나, 각각 별도 코드로 처리

#----------------------------------
# (3) 시계열 DF 만들기
#----------------------------------
ts_df = df[['date'] + features + [target]].copy()
ts_df = ts_df.set_index('date')  # 시계열 인덱스로 설정

from pmdarima import auto_arima

# 2-1. 데이터 분할 (train, test)
#     예: 마지막 4개(4분기)를 테스트로
train_size = len(ts_df) - 4

df_train = ts_df.iloc[:train_size]
df_test = ts_df.iloc[train_size:]

X_train = df_train[features]
y_train = df_train[target]
X_test  = df_test[features]
y_test  = df_test[target]

# 2-2. Auto ARIMA 모델 피팅
model_arima = pm.auto_arima(
    y_train,
    exogenous=X_train,      # 다중 독립변수
    seasonal=False,         # 분기별/연간 계절성을 넣으려면 True 후 m=4 or 1년=4분기
    m=4,                    # 분기 단위 계절성 고려하려면 m=4
    trace=True,             # 중간 과정 출력
    error_action='ignore',
    suppress_warnings=True
)

print(model_arima.summary())

# 2-3. 예측
forecast = model_arima.predict(n_periods=len(X_test), exogenous=X_test)
# 실제값 vs 예측값 비교
df_test['pred_arima'] = forecast

print(df_test[[target, 'pred_arima']])

# 2-4. 간단한 성능평가(MSE, MAE 등)
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse_arima = mean_squared_error(y_test, forecast)
mae_arima = mean_absolute_error(y_test, forecast)
print("ARIMA MSE:", mse_arima)
print("ARIMA MAE:", mae_arima)

# Prophet은 "ds", "y"라는 고정된 컬럼명 사용
prophet_df = df_train.reset_index()[['date', target] + features].copy()
prophet_df.rename(columns={'date':'ds', target:'y'}, inplace=True)

# 모델 선언
model_prophet = Prophet(seasonality_mode='additive', yearly_seasonality=True)
# 외부 변수(회귀자) 추가
for col in features:
    model_prophet.add_regressor(col)

# fit
model_prophet.fit(prophet_df)

# 예측용 DF (테스트 구간)
prophet_future = df_test.reset_index()[['date'] + features].copy()
prophet_future.rename(columns={'date':'ds'}, inplace=True)

# 예측
forecast_prophet = model_prophet.predict(prophet_future)
df_test['pred_prophet'] = forecast_prophet['yhat'].values

# 평가
mse_prophet = mean_squared_error(df_test[target], df_test['pred_prophet'])
mae_prophet = mean_absolute_error(df_test[target], df_test['pred_prophet'])
print("Prophet MSE:", mse_prophet)
print("Prophet MAE:", mae_prophet)

import numpy as np

def make_sequence_features(x_data, y_data, window_size=4):
    """
    x_data: (샘플 수, 독립변수 개수)
    y_data: (샘플 수, )
    window_size: 과거 몇 개 분기를 볼지
    """
    X_seq, Y_seq = [], []
    for i in range(len(x_data) - window_size):
        X_seq.append(x_data[i : i + window_size])
        Y_seq.append(y_data[i + window_size])  # window 다음 시점의 y
    return np.array(X_seq), np.array(Y_seq)

# 데이터 전체를 넘파이 배열로
all_X = ts_df[features].values
all_y = ts_df[target].values

window_size = 4  # 4분기(1년) 정보를 보고 다음 분기 예측

X_seq, y_seq = make_sequence_features(all_X, all_y, window_size=window_size)
# X_seq.shape => (샘플수, window_size, feature개수)
# y_seq.shape => (샘플수, )

# train/test 분할 (예: 마지막 4개 샘플을 테스트)
train_len = len(X_seq) - 4
X_train_lstm = X_seq[:train_len]
y_train_lstm = y_seq[:train_len]
X_test_lstm  = X_seq[train_len:]
y_test_lstm  = y_seq[train_len:]

print(X_train_lstm.shape, y_train_lstm.shape)
print(X_test_lstm.shape, y_test_lstm.shape)

model_lstm = keras.Sequential()
model_lstm.add(layers.Input(shape=(window_size, len(features))))  # (타임스텝=4, 특성=8)
model_lstm.add(layers.LSTM(64, activation='tanh', return_sequences=False))
model_lstm.add(layers.Dense(1))  # 회귀니까 출력1

model_lstm.compile(optimizer='adam', loss='mse')

model_lstm.summary()

history_lstm = model_lstm.fit(
    X_train_lstm, y_train_lstm,
    epochs=100, batch_size=4,
    validation_split=0.2,
    verbose=0
)

# 예측
y_pred_lstm = model_lstm.predict(X_test_lstm).flatten()

mse_lstm = mean_squared_error(y_test_lstm, y_pred_lstm)
mae_lstm = mean_absolute_error(y_test_lstm, y_pred_lstm)
print("LSTM MSE:", mse_lstm)
print("LSTM MAE:", mae_lstm)

model_gru = keras.Sequential()
model_gru.add(layers.Input(shape=(window_size, len(features))))
model_gru.add(layers.GRU(64, activation='tanh', return_sequences=False))
model_gru.add(layers.Dense(1))

model_gru.compile(optimizer='adam', loss='mse')

history_gru = model_gru.fit(
    X_train_lstm, y_train_lstm,  # 동일 데이터
    epochs=100, batch_size=4,
    validation_split=0.2,
    verbose=0
)

y_pred_gru = model_gru.predict(X_test_lstm).flatten()

mse_gru = mean_squared_error(y_test_lstm, y_pred_gru)
mae_gru = mean_absolute_error(y_test_lstm, y_pred_gru)
print("GRU MSE:", mse_gru)
print("GRU MAE:", mae_gru)

model_xgb = XGBRegressor(n_estimators=100, random_state=42)

model_xgb.fit(X_train_lstm.reshape(X_train_lstm.shape[0], -1),  # (샘플수, window_size*features)
              y_train_lstm)

y_pred_xgb = model_xgb.predict(X_test_lstm.reshape(X_test_lstm.shape[0], -1))

mse_xgb = mean_squared_error(y_test_lstm, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test_lstm, y_pred_xgb)
print("XGBoost MSE:", mse_xgb)
print("XGBoost MAE:", mae_xgb)

print(f"ARIMA   : MSE={mse_arima:.3f}, MAE={mae_arima:.3f}")
print(f"Prophet : MSE={mse_prophet:.3f}, MAE={mae_prophet:.3f}")
print(f"LSTM    : MSE={mse_lstm:.3f}, MAE={mae_lstm:.3f}")
print(f"GRU     : MSE={mse_gru:.3f}, MAE={mae_gru:.3f}")
print(f"XGBoost : MSE={mse_xgb:.3f}, MAE={mae_xgb:.3f}")

# (예) 엑셀 파일에서 읽어온다고 가정
# 실제 파일 경로 / sheet_name 등은 상황에 맞춰 변경하세요.
# df_raw = pd.read_excel('your_data.xlsx', engine='openpyxl')

# 여기서는 이미 df에 데이터가 있다고 가정하겠습니다.
# 사용자가 주신 표(연도, 분기구분, 호흡기_new, etc.)가 df에 들어있다고 가정.

# 사용자 데이터프레임. 아래 단계부터 적용해 주세요.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from xgboost import XGBRegressor

# 분기→월 매핑 함수
def quarter_to_date(row):
    mapping = {1: '01', 2: '04', 3: '07', 4: '10'}
    year = row['연도']
    q    = row['분기구분']
    # 분기에 따라 월 결정
    month = mapping.get(q, '01')  
    return pd.to_datetime(f"{year}-{month}-01")

# 날짜 컬럼 생성
df['date'] = df.apply(quarter_to_date, axis=1)

# 날짜 기준으로 정렬
df = df.sort_values('date').reset_index(drop=True)

# 시계열 인덱스로 설정
df = df.set_index('date')
df = df.sort_index()

# 혹시 확인
print(df.head())
print(df.index)

features = [
    '최고기온 (C)', '최저기온 (C)', '일 평균기온 (C)',
    '일 평균 풍속 (m/s)', '일강수량 (mm)',
    '최심 신적설 (cm)', '일 평균 상대습도 (%)', '일교차 (C)'
]

# 여기서는 '호흡기_new'를 타깃으로 예시
# 만약 '매개성' 예측을 원하시면 y_col을 바꿔주세요.
y_col = '호흡기_new'

X = df[features]
y = df[y_col]

print("X shape:", X.shape)
print("y shape:", y.shape)
def walk_forward_validation(X, y, model, train_size=16):
    """
    X, y : 시계열 정렬된 데이터 (인덱스는 날짜)
    model: fit/predict 가능한 회귀 모델 객체 (예: XGBRegressor)
    train_size: 초기 훈련 구간 크기
    
    returns:
        df_pred (DataFrame): 인덱스=X.index, ['actual','pred'] 컬럼
    """
    # 결과 저장할 DF
    df_pred = pd.DataFrame(index=X.index, columns=['actual','pred'])
    df_pred['actual'] = y  # 실제값 복사

    # 초기 훈련 세트 (앞부분 train_size개)
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]

    # main loop
    for i in range(train_size, len(X)):
        # 1) 모델 훈련
        model.fit(X_train, y_train)

        # 2) 다음 시점(i) 예측
        X_test = X.iloc[[i]]  # i번째(한 샘플)
        y_hat = model.predict(X_test)[0]

        # 3) 예측값 저장
        df_pred.iloc[i, df_pred.columns.get_loc('pred')] = y_hat

        # 4) 훈련 세트 확장 (i+1까지)
        X_train = X.iloc[:i+1]
        y_train = y.iloc[:i+1]

    return df_pred
# XGBoost 회귀 모델
model_xgb = XGBRegressor(n_estimators=50, random_state=42)

# Walk-Forward Validation
df_result = walk_forward_validation(X, y, model_xgb, train_size=16)

# 예측값 vs 실제값이 존재하는 구간만 평가
mask_eval = df_result['pred'].notnull()
y_true = df_result.loc[mask_eval, 'actual']
y_pred = df_result.loc[mask_eval, 'pred']

rmse = sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
print(f"[XGB Walk-Forward] RMSE={rmse:.3f}, MAE={mae:.3f}")

plt.figure(figsize=(10,5))
plt.plot(df_result.index, df_result['actual'], label='Actual', marker='o')
plt.plot(df_result.index, df_result['pred'],   label='Predicted', marker='x')
plt.title('Walk-Forward Validation (XGBoost)')
plt.xlabel('Date')
plt.ylabel(y_col)
plt.legend()
plt.grid(True)
plt.show()


###############################################
# 1. 라이브러리 임포트
###############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# ARIMA
import pmdarima as pm

# Prophet
from prophet import Prophet

# LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

###############################################
# 2. 사용자 데이터 불러오기 + 날짜(분기) 인덱스 생성
###############################################

# 예시: Excel 파일에서 읽어온다고 가정 (sheet_name, 파일명 등 상황에 맞춰 수정)
# df_raw = pd.read_excel('your_file.xlsx', engine='openpyxl')

# 여기서는 이미 df_raw 라고 가정.
# df_raw에는 아래 컬럼이 포함되어 있어야 합니다:
# ['연도', '분기구분', '최고기온 (C)', '최저기온 (C)', '일 평균기온 (C)',
#  '일 평균 풍속 (m/s)', '일강수량 (mm)', '최심 신적설 (cm)',
#  '일 평균 상대습도 (%)', '일교차 (C)', '호흡기_new', ...]




# # (1) 연도+분기 → 날짜 변환
# def quarter_to_month(row):
#     mapping = {1:'01', 2:'04', 3:'07', 4:'10'}
#     year = row['연도']
#     q    = row['분기구분']
#     month = mapping.get(q, '01')
#     return pd.to_datetime(f"{year}-{month}-01")

# df['date'] = df.apply(quarter_to_month, axis=1)

# # (2) 날짜 정렬 및 인덱스화
# df = df.sort_values('date').reset_index(drop=True)
# df = df.set_index('date')
# df = df.sort_index()

###############################################
# 3. 독립변수(X), 종속변수(y) 설정
###############################################
# 예시: 기상 정보 8개를 독립변수, '호흡기_new'를 종속변수로
x_cols = [
    '최고기온 (C)', '최저기온 (C)', '일 평균기온 (C)',
    '일 평균 풍속 (m/s)', '일강수량 (mm)',
    '최심 신적설 (cm)', '일 평균 상대습도 (%)', '일교차 (C)'
]
y_col = '호흡기_new'  # or '매개성' 등

# X, y 분리
X = df[x_cols]
y = df[y_col]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Index (dates):", df.index)

###############################################
# 4. Walk-Forward Validation 공통 함수(ARIMA, Prophet 용)
###############################################
def walk_forward_validation_arbitrary(
    df,             # 시계열 (인덱스: date)
    train_size,
    train_func,     # 모델 훈련 콜백
    predict_func,   # 모델 예측 콜백
    x_cols=None,    
    y_col='y',
):
    """
    df: 시계열 DataFrame (시간순 정렬), index=date
    train_size: 초기 훈련 구간 크기
    train_func(X_train, y_train) -> model
    predict_func(model, X_test) -> yhat
    x_cols: exogenous(독립변수) 컬럼 리스트
    y_col: 종속변수 컬럼명

    returns:
        df_result: DataFrame(index=df.index, columns=['actual','pred'])
                   Walk-Forward 예측값 저장
    """
    n = len(df)
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col].values  # 실제값

    data_y = df[y_col].values
    data_X = None
    if x_cols is not None:
        data_X = df[x_cols].values

    # Walk-Forward
    for i in range(train_size, n):
        # 훈련 구간: 0 ~ i-1
        y_train = data_y[:i]
        X_train = None
        if data_X is not None:
            X_train = data_X[:i]

        # 모델 훈련
        model = train_func(X_train, y_train)

        # 예측 대상: i
        X_test = None
        if data_X is not None:
            X_test = data_X[i:i+1]  # (1, num_features) or None

        yhat = predict_func(model, X_test)
        df_result.iloc[i, df_result.columns.get_loc('pred')] = yhat

    return df_result

###############################################
# 5. ARIMA (pmdarima) 예시
###############################################
def train_arima(X_train, y_train):
    if X_train is None:
        model = pm.auto_arima(y_train, seasonal=False, error_action='ignore', suppress_warnings=True)
    else:
        model = pm.auto_arima(y_train, exogenous=X_train, seasonal=False, error_action='ignore', suppress_warnings=True)
    return model

def predict_arima(model, X_test):
    if X_test is None:
        forecast = model.predict(n_periods=1)
    else:
        forecast = model.predict(n_periods=1, exogenous=X_test)
    return forecast[0]

train_size_arima = 10  # 초기 훈련 샘플 수 (적절히 조정)
df_result_arima = walk_forward_validation_arbitrary(
    df, train_size=train_size_arima,
    train_func=train_arima,
    predict_func=predict_arima,
    x_cols=x_cols,
    y_col=y_col
)

mask_arima = df_result_arima['pred'].notnull()
y_true_arima = df_result_arima.loc[mask_arima, 'actual']
y_pred_arima = df_result_arima.loc[mask_arima, 'pred']

rmse_arima = sqrt(mean_squared_error(y_true_arima, y_pred_arima))
mae_arima  = mean_absolute_error(y_true_arima, y_pred_arima)
print(f"[ARIMA Walk-Forward] RMSE={rmse_arima:.3f}, MAE={mae_arima:.3f}")

plt.figure(figsize=(8,4))
plt.plot(df_result_arima.index, df_result_arima['actual'], label='Actual', marker='o')
plt.plot(df_result_arima.index, df_result_arima['pred'],   label='ARIMA Pred', marker='x')
plt.title("ARIMA Walk-Forward")
plt.xlabel("Date")
plt.ylabel(y_col)
plt.legend()
plt.grid(True)
plt.show()

###############################################
# 6. Prophet 예시
###############################################
def train_prophet(X_train, y_train):
    # 과거 시점 갯수
    i = len(y_train)
    # 인덱스 중 0~i-1 부분
    ds_idx = df.index[:i]  # datetime index

    prophet_df = pd.DataFrame({
        'ds': ds_idx,
        'y': y_train
    })
    model = Prophet(seasonality_mode='additive')

    if X_train is not None:
        # add_regressor
        for col in x_cols:
            model.add_regressor(col)
        # regressor 컬럼도 prophet_df에 붙이기
        for idx, col in enumerate(x_cols):
            prophet_df[col] = X_train[:, idx]

    model.fit(prophet_df)
    return model

def predict_prophet(model, X_test):
    # 임시로 last_ds + 30일
    # 실제론 walk-forward에서 정확한 예측일자를 주는 게 이상적
    last_ds = model.history['ds'].max()
    next_ds = last_ds + pd.Timedelta(days=30)

    prophet_future = pd.DataFrame({'ds':[next_ds]})
    if X_test is not None:
        for idx, col in enumerate(x_cols):
            prophet_future[col] = X_test[0, idx]

    forecast = model.predict(prophet_future)
    return forecast['yhat'].values[0]

train_size_prophet = 10
df_result_prophet = walk_forward_validation_arbitrary(
    df, train_size=train_size_prophet,
    train_func=train_prophet,
    predict_func=predict_prophet,
    x_cols=x_cols,
    y_col=y_col
)

mask_prophet = df_result_prophet['pred'].notnull()
y_true_prophet = df_result_prophet.loc[mask_prophet, 'actual']
y_pred_prophet = df_result_prophet.loc[mask_prophet, 'pred']

rmse_prophet = sqrt(mean_squared_error(y_true_prophet, y_pred_prophet))
mae_prophet  = mean_absolute_error(y_true_prophet, y_pred_prophet)
print(f"[Prophet Walk-Forward] RMSE={rmse_prophet:.3f}, MAE={mae_prophet:.3f}")

plt.figure(figsize=(8,4))
plt.plot(df_result_prophet.index, df_result_prophet['actual'], label='Actual', marker='o')
plt.plot(df_result_prophet.index, df_result_prophet['pred'],   label='Prophet Pred', marker='x')
plt.title("Prophet Walk-Forward")
plt.xlabel("Date")
plt.ylabel(y_col)
plt.legend()
plt.grid(True)
plt.show()

###############################################
# 7. LSTM 예시 (전용 Walk-Forward 함수)
###############################################
def make_lstm_dataset(X_data, y_data, window_size=4):
    X_seq, y_seq = [], []
    for i in range(len(X_data)-window_size):
        X_seq.append(X_data[i : i+window_size, :])
        y_seq.append(y_data[i+window_size])
    return np.array(X_seq), np.array(y_seq)

WINDOW_SIZE = 4  # 과거 4분기(1년) 본다고 가정

def build_lstm_model(num_features):
    model = keras.Sequential()
    model.add(layers.Input(shape=(WINDOW_SIZE, num_features)))
    model.add(layers.LSTM(32, activation='tanh'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(X_train, y_train):
    # X_train: shape (i, num_features)
    # y_train: shape (i,)
    if len(X_train) <= WINDOW_SIZE:
        # 데이터가 너무 적으면 모델만 만들어 반환
        model = build_lstm_model(num_features=X_train.shape[1])
        return model

    X_seq, y_seq = make_lstm_dataset(X_train, y_train, window_size=WINDOW_SIZE)

    model = build_lstm_model(num_features=X_train.shape[1])
    model.fit(
        X_seq, y_seq,
        epochs=30, batch_size=2, verbose=0
    )
    return model

def walk_forward_validation_lstm(df, train_size, x_cols, y_col='y'):
    """
    LSTM용 Walk-Forward:
    매 시점 i에서 0~i-1까지로 모델 학습,
    i 시점 예측 시, 직전 window_size 샘플을 한 시퀀스로 사용
    """
    n = len(df)
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col]

    X_all = df[x_cols].values
    y_all = df[y_col].values

    for i in range(train_size, n):
        # 훈련 구간: 0 ~ i-1
        X_train = X_all[:i]
        y_train = y_all[:i]

        # LSTM 모델 훈련
        model = train_lstm(X_train, y_train)

        # 예측 시점: i
        if i < WINDOW_SIZE:
            # window 크기보다 작은 경우는 예측 불가
            df_result.iloc[i, df_result.columns.get_loc('pred')] = np.nan
        else:
            # 직전 WINDOW_SIZE
            X_seq = X_all[i-WINDOW_SIZE:i]
            X_seq = X_seq.reshape(1, WINDOW_SIZE, len(x_cols))
            yhat = model.predict(X_seq)[0,0]
            df_result.iloc[i, df_result.columns.get_loc('pred')] = yhat

    return df_result

train_size_lstm = 10
df_result_lstm = walk_forward_validation_lstm(df, train_size_lstm, x_cols, y_col)
mask_lstm = df_result_lstm['pred'].notnull()
y_true_lstm = df_result_lstm.loc[mask_lstm, 'actual']
y_pred_lstm = df_result_lstm.loc[mask_lstm, 'pred']

rmse_lstm = sqrt(mean_squared_error(y_true_lstm, y_pred_lstm))
mae_lstm  = mean_absolute_error(y_true_lstm, y_pred_lstm)
print(f"[LSTM Walk-Forward] RMSE={rmse_lstm:.3f}, MAE={mae_lstm:.3f}")

plt.figure(figsize=(8,4))
plt.plot(df_result_lstm.index, df_result_lstm['actual'], label='Actual', marker='o')
plt.plot(df_result_lstm.index, df_result_lstm['pred'],   label='LSTM Pred', marker='x')
plt.title("LSTM Walk-Forward")
plt.xlabel("Date")
plt.ylabel(y_col)
plt.legend()
plt.grid(True)
plt.show()
