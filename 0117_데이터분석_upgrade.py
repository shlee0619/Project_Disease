###########################################
# 1. 라이브러리 및 공통 함수 임포트
###########################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pmdarima as pm
from prophet import Prophet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# 한글 폰트 설정
import matplotlib.font_manager as fm
font_name = fm.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
plt.rc('font', family=font_name)

###########################################
# 2. 데이터 로드 및 전처리
###########################################
def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    사용자께서 준비한 CSV 혹은 Excel 파일 경로를 입력받아,
    1) CSV/Excel 로드
    2) '연도' + '분기구분' → datetime 변환
    3) 날짜 정렬 및 인덱스화
    4) 반환
    """
    # 파일 읽기 (CSV or Excel 상황에 맞게)
    df = pd.read_csv(filepath)  # 필요하다면 pd.read_excel로 변경
    print("== Raw Data ==")
    print(df.head())
    
    # 분기→월 매핑 함수
    def quarter_to_date(row):
        mapping = {1:'01', 2:'04', 3:'07', 4:'10'}
        year = row['연도']
        q    = row['분기구분']
        month = mapping.get(q, '01')
        return pd.to_datetime(f"{year}-{month}-01")

    # 날짜 컬럼 생성
    df['date'] = df.apply(quarter_to_date, axis=1)

    # 날짜 기준으로 정렬
    df = df.sort_values('date').reset_index(drop=True)

    # 시계열 인덱스 설정
    df = df.set_index('date')
    df = df.sort_index()

    print("\n== Preprocessed Data ==")
    print(df.head())
    print("Index (date):", df.index)

    return df

###########################################
# 3. 모델별 Walk-Forward 함수
###########################################

# 3-1) 공통 Walk-Forward (ARIMA, Prophet 등)
def walk_forward_validation_arbitrary(
    df: pd.DataFrame,
    train_size: int,
    train_func,     # 모델 훈련 함수
    predict_func,   # 모델 예측 함수
    x_cols: list,   # 독립변수 컬럼 목록
    y_col: str
) -> pd.DataFrame:
    """
    일반적인 (ARIMA, Prophet 등) 모델에 대해
    - i 시점까지 훈련 → (i+1) 시점 1개 예측 → 실제값 반영 → 다음 스텝
    - 다만 Prophet은 날짜 매핑 등 추가 처리가 필요할 수 있음(콜백함수 내)
    
    returns: df_result (['actual','pred']) with same index as df
    """
    n = len(df)
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col].values

    data_X = df[x_cols].values
    data_y = df[y_col].values

    for i in range(train_size, n):
        # 훈련 구간: 0 ~ i-1
        X_train = data_X[:i]
        y_train = data_y[:i]

        # 모델 생성+훈련
        model = train_func(X_train, y_train, df.index[:i])  # 날짜 인덱스도 넘겨줄 수 있음

        # 예측 구간: i
        X_test = data_X[i:i+1]  # (1, num_features)
        yhat = predict_func(model, X_test, df.index[i])     # 시점 i의 날짜도 넘길 수 있음

        df_result.iloc[i, df_result.columns.get_loc('pred')] = yhat

    return df_result


# 3-2) LSTM 전용 (슬라이딩 윈도우)
def make_lstm_dataset(X_data, y_data, window_size=4):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - window_size):
        X_seq.append(X_data[i:i+window_size, :])
        y_seq.append(y_data[i+window_size])
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(num_features: int, window_size=4):
    model = keras.Sequential()
    model.add(layers.Input(shape=(window_size, num_features)))
    model.add(layers.LSTM(32, activation='tanh'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def walk_forward_validation_lstm(
    df: pd.DataFrame,
    train_size: int,
    x_cols: list,
    y_col: str,
    window_size: int = 4,
    epochs: int = 30,
    batch_size: int = 2
) -> pd.DataFrame:
    """
    LSTM용 Walk-Forward:
    - 매 시점 i에서, 0 ~ (i-1)까지의 데이터로 모델 재학습
    - 시점 i 예측 시, 직전 window_size 샘플을 하나의 시퀀스로 넣어 추론
    - epochs, batch_size는 간단히 지정
    """
    n = len(df)
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col].values

    X_all = df[x_cols].values
    y_all = df[y_col].values

    def train_lstm_model(X_train, y_train):
        # window 변환
        if len(X_train) <= window_size:
            # 데이터 너무 적으면 그냥 모델 리턴
            return build_lstm_model(num_features=X_train.shape[1], window_size=window_size)

        X_seq, y_seq = make_lstm_dataset(X_train, y_train, window_size=window_size)
        model_temp = build_lstm_model(num_features=X_train.shape[1], window_size=window_size)
        model_temp.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, verbose=0)
        return model_temp

    for i in range(train_size, n):
        # 훈련: 0 ~ i-1
        X_train = X_all[:i]
        y_train = y_all[:i]

        model = train_lstm_model(X_train, y_train)

        if i < window_size:
            df_result.iloc[i, df_result.columns.get_loc('pred')] = np.nan
        else:
            # 직전 window_size 샘플
            X_seq = X_all[i-window_size:i]
            X_seq = X_seq.reshape(1, window_size, len(x_cols))
            yhat = model.predict(X_seq)[0,0]
            df_result.iloc[i, df_result.columns.get_loc('pred')] = yhat

    return df_result


###############################################
# 4. 모델별 train/predict 콜백 함수 (ARIMA, Prophet)
###############################################

# ARIMA (pmdarima) -----------------------------
def train_arima(X_train, y_train, train_dates=None):
    """
    X_train: shape (i, num_features)
    y_train: shape (i,)
    train_dates: 시점 정보(옵션)
    """
    # 단변량 vs 다변량 구분
    if X_train.shape[1] == 0:
        # 만약 독립변수 없는 경우(단변량) -> X_train=None
        model = pm.auto_arima(y_train, seasonal=False, error_action='ignore', suppress_warnings=True)
    else:
        model = pm.auto_arima(y_train, exogenous=X_train, seasonal=False, error_action='ignore', suppress_warnings=True)
    return model

def predict_arima(model, X_test, pred_date=None):
    if X_test.shape[1] == 0:
        forecast = model.predict(n_periods=1)
    else:
        forecast = model.predict(n_periods=1, exogenous=X_test)
    return forecast[0]

# Prophet -------------------------------------
def train_prophet(X_train, y_train, train_dates=None):
    """
    X_train: shape (i, num_features)
    y_train: shape (i,)
    train_dates: Index of length i
    """
    i = len(y_train)
    if train_dates is None:
        # 임시 날짜
        ds_idx = pd.date_range('2000-01-01', periods=i, freq='MS')
    else:
        ds_idx = train_dates  # 실제 날짜 인덱스

    prophet_df = pd.DataFrame({'ds': ds_idx, 'y': y_train})
    model = Prophet(seasonality_mode='additive')

    # x_cols 개수 > 0 인 경우 -> add_regressor
    num_features = X_train.shape[1]
    if num_features > 0:
        for f_idx in range(num_features):
            col_name = f"reg_{f_idx}"
            model.add_regressor(col_name)
            prophet_df[col_name] = X_train[:, f_idx]

    model.fit(prophet_df)
    return model

def predict_prophet(model, X_test, pred_date=None):
    """
    X_test: shape(1, num_features)
    pred_date: 예측할 날짜 (시점 i), 없으면 임시로 last_ds + 30일
    """
    last_ds = model.history['ds'].max()
    if pred_date is None:
        next_ds = last_ds + pd.Timedelta(days=30)
    else:
        # pred_date가 실제 날짜라면, 한 번에 예측해 볼 수 있음
        # Prophet은 통상 future_df 작성 -> predict
        # 여기서는 단일 시점만 예시
        next_ds = pred_date

    prophet_future = pd.DataFrame({'ds':[next_ds]})

    num_features = X_test.shape[1]
    if num_features > 0:
        for f_idx in range(num_features):
            col_name = f"reg_{f_idx}"
            prophet_future[col_name] = X_test[0, f_idx]

    forecast = model.predict(prophet_future)
    return forecast['yhat'].values[0]


###########################################
# 5. 메인 실행부
###########################################
def run_all_models(filepath, target_list):
    """
    1) 데이터 로딩 + 전처리
    2) 모델별(ARIMA, Prophet, LSTM)로 Walk-Forward
    3) RMSE/MAE 결과 출력 + 그래프
    4) target_list: ['호흡기_new', '매개성'] 등
    """

    # 1) 데이터
    df = load_and_preprocess_data(filepath)

    # 독립변수 8개
    x_cols = [
        '최고기온 (C)', '최저기온 (C)', '일 평균기온 (C)',
        '일 평균 풍속 (m/s)', '일강수량 (mm)',
        '최심 신적설 (cm)', '일 평균 상대습도 (%)', '일교차 (C)'
    ]

    # 결과 저장용
    results = {}

    for tgt in target_list:
        print(f"\n======== Target: {tgt} ========")
        # df가 해당 컬럼(tgt)을 가진다고 가정
        # 시계열 df (X + y)
        # 필요하다면 df.dropna() 등 처리
        sub_df = df[x_cols + [tgt]].copy()

        # (A) ARIMA Walk-Forward
        train_size_arima = 10  # 초기 훈련 크기 (상황 따라 조정)
        df_result_arima = walk_forward_validation_arbitrary(
            sub_df,
            train_size=train_size_arima,
            train_func=train_arima,
            predict_func=predict_arima,
            x_cols=x_cols,
            y_col=tgt
        )
        # 평가
        mask_arima = df_result_arima['pred'].notnull()
        y_true_arima = df_result_arima.loc[mask_arima, 'actual']
        y_pred_arima = df_result_arima.loc[mask_arima, 'pred']
        rmse_arima = sqrt(mean_squared_error(y_true_arima, y_pred_arima))
        mae_arima  = mean_absolute_error(y_true_arima, y_pred_arima)

        # 시각화
        plt.figure(figsize=(6,3))
        plt.plot(df_result_arima.index, df_result_arima['actual'], marker='o', label='Actual')
        plt.plot(df_result_arima.index, df_result_arima['pred'], marker='x', label='ARIMA')
        plt.title(f"ARIMA - {tgt}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # (B) Prophet Walk-Forward
        train_size_prophet = 10
        df_result_prophet = walk_forward_validation_arbitrary(
            sub_df,
            train_size=train_size_prophet,
            train_func=train_prophet,
            predict_func=predict_prophet,
            x_cols=x_cols,
            y_col=tgt
        )
        mask_prophet = df_result_prophet['pred'].notnull()
        y_true_prophet = df_result_prophet.loc[mask_prophet, 'actual']
        y_pred_prophet = df_result_prophet.loc[mask_prophet, 'pred']
        rmse_prophet = sqrt(mean_squared_error(y_true_prophet, y_pred_prophet))
        mae_prophet  = mean_absolute_error(y_true_prophet, y_pred_prophet)

        plt.figure(figsize=(6,3))
        plt.plot(df_result_prophet.index, df_result_prophet['actual'], marker='o', label='Actual')
        plt.plot(df_result_prophet.index, df_result_prophet['pred'], marker='x', label='Prophet')
        plt.title(f"Prophet - {tgt}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # (C) LSTM Walk-Forward
        train_size_lstm = 10
        df_result_lstm = walk_forward_validation_lstm(
            df=sub_df,
            train_size=train_size_lstm,
            x_cols=x_cols,
            y_col=tgt,
            window_size=4,
            epochs=30,
            batch_size=2
        )
        mask_lstm = df_result_lstm['pred'].notnull()
        y_true_lstm = df_result_lstm.loc[mask_lstm, 'actual']
        y_pred_lstm = df_result_lstm.loc[mask_lstm, 'pred']
        rmse_lstm = sqrt(mean_squared_error(y_true_lstm, y_pred_lstm))
        mae_lstm  = mean_absolute_error(y_true_lstm, y_pred_lstm)

        plt.figure(figsize=(6,3))
        plt.plot(df_result_lstm.index, df_result_lstm['actual'], marker='o', label='Actual')
        plt.plot(df_result_lstm.index, df_result_lstm['pred'], marker='x', label='LSTM')
        plt.title(f"LSTM - {tgt}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 결과 저장
        results[tgt] = {
            'ARIMA':    (rmse_arima, mae_arima),
            'Prophet':  (rmse_prophet, mae_prophet),
            'LSTM':     (rmse_lstm, mae_lstm)
        }

    # 결과 요약
    print("\n======= Summary =======")
    for tgt, model_res in results.items():
        print(f"\nTarget: {tgt}")
        for model_name, (r, m) in model_res.items():
            print(f"{model_name:8s}: RMSE={r:.3f}, MAE={m:.3f}")


###########################################
# 6. 실행 예시
###########################################
if __name__ == "__main__":
    """
    예시 사용법:
    1) CSV 파일 경로: "modified_infectious_disease_data.csv"
    2) 타깃 리스트: ['호흡기_new', '매개성']
    """
    file_path = "modified_infectious_disease_data.csv"
    target_vars = ["호흡기_new", "매개성"]

    run_all_models(file_path, target_vars)
