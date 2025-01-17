###############################################
# 1. 라이브러리 및 한글 폰트 설정
###############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pmdarima as pm
from prophet import Prophet
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 한글 폰트 설정 (Windows 예시)
import matplotlib.font_manager as fm
font_name = fm.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
plt.rc('font', family=font_name)

###############################################
# 2. 데이터 로드 및 전처리
###############################################
def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    1) CSV 로드
    2) '연도' + '분기구분' → date
    3) 시계열 정렬 + 인덱싱
    4) 간단한 결측/이상치 처리
    5) 반환
    """
    # (A) CSV 읽기
    df = pd.read_csv(filepath)
    print("== Raw Data Preview ==")
    print(df.head())
    

    # (B) 날짜 인덱스 설정
    df = df.sort_values('date').reset_index(drop=True)
    df = df.set_index('date')
    df = df.sort_index()

    # (C) 결측치 처리 + 이상치 클리핑
    feat_cols = [
        '최고기온 (C)', '최저기온 (C)', '일 평균기온 (C)',
        '일 평균 풍속 (m/s)', '일강수량 (mm)',
        '최심 신적설 (cm)', '일 평균 상대습도 (%)', '일교차 (C)'
    ]
    # 결측치: 중간값 대체
    for col in feat_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # 이상치 클리핑 (±3sigma)
    for col in feat_cols:
        if col in df.columns:
            mean_ = df[col].mean()
            std_  = df[col].std()
            lower = mean_ - 3*std_
            upper = mean_ + 3*std_
            df[col] = df[col].clip(lower=lower, upper=upper)

    print("\n== Preprocessed Data Preview ==")
    print(df.head())
    print("Index (date):", df.index)
    return df

###############################################
# 3. 피처 엔지니어링 (lag, roll)
###############################################
def create_feature_engineering(df: pd.DataFrame,
                               lag_features: dict = None,
                               roll_features: dict = None) -> pd.DataFrame:
    df_ext = df.copy()

    # 시차(lag)
    if lag_features:
        for col, lags in lag_features.items():
            if col not in df_ext.columns:
                continue
            for lag in lags:
                new_col = f"{col}_lag{lag}"
                df_ext[new_col] = df_ext[col].shift(lag)

    # 이동평균(roll)
    if roll_features:
        for col, windows in roll_features.items():
            if col not in df_ext.columns:
                continue
            for w in windows:
                new_col = f"{col}_roll{w}"
                df_ext[new_col] = df_ext[col].rolling(w).mean()

    # 초기 구간 결측
    df_ext.dropna(inplace=True)
    return df_ext

###############################################
# 4. Walk-Forward 함수 (단일 스텝)
###############################################
def walk_forward_validation_arbitrary(
    df: pd.DataFrame,
    train_size: int,
    train_func,
    predict_func,
    x_cols: list,
    y_col: str
) -> pd.DataFrame:

    n = len(df)
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col].values

    X_data = df[x_cols].values
    y_data = df[y_col].values

    for i in range(train_size, n):
        X_train = X_data[:i]
        y_train = y_data[:i]

        # 모델 훈련
        model = train_func(X_train, y_train, df.index[:i])

        # 예측
        X_test = X_data[i:i+1]
        yhat = predict_func(model, X_test, df.index[i])
        df_result.iloc[i, df_result.columns.get_loc('pred')] = yhat

    return df_result

###############################################
# 4-1. 다중 스텝 예측 (Recursive)
###############################################
def walk_forward_validation_multi_step(
    df: pd.DataFrame,
    train_size: int,
    steps_ahead: int,
    train_func,
    predict_func,
    x_cols: list,
    y_col: str
) -> pd.DataFrame:
    n = len(df)
    df_result = pd.DataFrame(index=df.index)
    df_result['actual'] = df[y_col].values

    X_data = df[x_cols].values
    y_data = df[y_col].values

    for k in range(1, steps_ahead+1):
        df_result[f'pred_{k}'] = np.nan

    for i in range(train_size, n):
        X_train = X_data[:i]
        y_train = y_data[:i]
        model = train_func(X_train, y_train, df.index[:i])

        X_cur = X_data[i].copy()
        for k in range(1, steps_ahead+1):
            idx_k = i + k
            if idx_k >= n:
                break
            yhat_k = predict_func(model, X_cur.reshape(1, -1), df.index[idx_k])
            df_result.iloc[idx_k, df_result.columns.get_loc(f'pred_{k}')] = yhat_k
            # (필요시 X_cur 갱신)

    return df_result

###############################################
# 5. LSTM / GRU (슬라이딩 윈도우)
###############################################
def make_lstm_dataset(X_data, y_data, window_size=4):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - window_size):
        X_seq.append(X_data[i:i+window_size, :])
        y_seq.append(y_data[i + window_size])
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

    n = len(df)
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col].values

    X_all = df[x_cols].values
    y_all = df[y_col].values

    def train_lstm_model(X_train, y_train):
        if len(X_train) <= window_size:
            return build_lstm_model(num_features=X_train.shape[1], window_size=window_size)
        X_seq, y_seq = make_lstm_dataset(X_train, y_train, window_size=window_size)
        model_temp = build_lstm_model(num_features=X_train.shape[1], window_size=window_size)
        model_temp.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, verbose=0)
        return model_temp

    for i in range(train_size, n):
        X_train = X_all[:i]
        y_train = y_all[:i]
        model = train_lstm_model(X_train, y_train)

        if i < window_size:
            df_result.iloc[i, df_result.columns.get_loc('pred')] = np.nan
        else:
            X_seq = X_all[i-window_size:i]
            X_seq = X_seq.reshape(1, window_size, len(x_cols))
            yhat = model.predict(X_seq)[0,0]
            df_result.iloc[i, df_result.columns.get_loc('pred')] = yhat

    return df_result

def build_gru_model(num_features: int, window_size=4):
    model = keras.Sequential()
    model.add(layers.Input(shape=(window_size, num_features)))
    model.add(layers.GRU(32, activation='tanh'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def walk_forward_validation_gru(
    df: pd.DataFrame,
    train_size: int,
    x_cols: list,
    y_col: str,
    window_size: int = 4,
    epochs: int = 30,
    batch_size: int = 2
) -> pd.DataFrame:

    n = len(df)
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col].values

    X_all = df[x_cols].values
    y_all = df[y_col].values

    def train_gru_model(X_train, y_train):
        if len(X_train) <= window_size:
            return build_gru_model(num_features=X_train.shape[1], window_size=window_size)
        X_seq, y_seq = make_lstm_dataset(X_train, y_train, window_size=window_size)
        model_ = build_gru_model(num_features=X_train.shape[1], window_size=window_size)
        model_.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, verbose=0)
        return model_

    for i in range(train_size, n):
        X_train = X_all[:i]
        y_train = y_all[:i]
        model = train_gru_model(X_train, y_train)

        if i < window_size:
            df_result.iloc[i, df_result.columns.get_loc('pred')] = np.nan
        else:
            X_seq = X_all[i-window_size:i]
            X_seq = X_seq.reshape(1, window_size, len(x_cols))
            yhat = model.predict(X_seq)[0,0]
            df_result.iloc[i, df_result.columns.get_loc('pred')] = yhat

    return df_result

###############################################
# 6. 모델별 train/predict 콜백 (ARIMA, Prophet, XGB)
###############################################
def train_arima(X_train, y_train, train_dates=None):
    if X_train.shape[1] == 0:  # 단변량
        model = pm.auto_arima(y_train, seasonal=False, error_action='ignore', suppress_warnings=True)
    else:  # 다변량
        model = pm.auto_arima(y_train, exogenous=X_train, seasonal=False, error_action='ignore', suppress_warnings=True)
    return model

def predict_arima(model, X_test, pred_date=None):
    if X_test.shape[1] == 0:
        fcst = model.predict(n_periods=1)
    else:
        fcst = model.predict(n_periods=1, exogenous=X_test)
    return fcst[0]

def train_prophet(X_train, y_train, train_dates=None):
    # Prophet은 df에 [ds, y, + regressor들] 형태 필요
    i = len(y_train)
    if train_dates is None:
        ds_idx = pd.date_range('2000-01-01', periods=i, freq='MS')
    else:
        ds_idx = train_dates

    df_p = pd.DataFrame({'ds': ds_idx, 'y': y_train})

    # 모델 선언
    model = Prophet(seasonality_mode='additive')

    # 숫자형 regressor만 add_regressor
    num_features = X_train.shape[1]
    if num_features > 0:
        for f_idx in range(num_features):
            col_name = f"reg_{f_idx}"
            model.add_regressor(col_name)
            # regressor 칼럼 붙이기
            df_p[col_name] = X_train[:, f_idx]

    model.fit(df_p)
    return model

def predict_prophet(model, X_test, pred_date=None):
    # 예측 시점
    last_ds = model.history['ds'].max()
    if pred_date is None:
        next_ds = last_ds + pd.Timedelta(days=30)
    else:
        next_ds = pred_date

    df_f = pd.DataFrame({'ds':[next_ds]})

    num_features = X_test.shape[1]
    if num_features > 0:
        for f_idx in range(num_features):
            col_name = f"reg_{f_idx}"
            df_f[col_name] = X_test[0, f_idx]

    fcst = model.predict(df_f)
    return fcst['yhat'].values[0]


def train_xgboost(X_train, y_train, train_dates=None):
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_xgboost(model, X_test, pred_date=None):
    return model.predict(X_test)[0]

###############################################
# 7. 메인 실행
###############################################
def run_all_models(
    filepath, 
    target_list,
    use_feature_engineering=True,
    lag_features=None,
    roll_features=None
):


    # (1) 데이터 로드
    df = load_and_preprocess_data(filepath)

    # (2) 피처 엔지니어링(옵션)
    if use_feature_engineering:
        print("\n--- [Feature Engineering] lag/roll ---")
        df = create_feature_engineering(df, lag_features, roll_features)
        print(df.head())

    # 독립변수 후보
    x_cols = [
        c for c in df.columns
        if c not in ['연도','분기구분','호흡기','매개성']
    ]

    results = {}

    for tgt in target_list:
        if tgt not in df.columns:
            print(f"[WARNING] {tgt} not found in df.columns. Skip.")
            continue

        print(f"\n============== [Target: {tgt}] ==============")
        sub_df = df[x_cols + [tgt]].copy()

        # ---------- A. ARIMA ----------
        df_arima = walk_forward_validation_arbitrary(
            sub_df, train_size=10,
            train_func=train_arima,
            predict_func=predict_arima,
            x_cols=x_cols, y_col=tgt
        )
        msk_a = df_arima['pred'].notnull()
        y_true_a = df_arima.loc[msk_a,'actual']
        y_pred_a = df_arima.loc[msk_a,'pred']
        rmse_arima = sqrt(mean_squared_error(y_true_a, y_pred_a))
        mae_arima  = mean_absolute_error(y_true_a, y_pred_a)
        plt.figure(figsize=(6,3))
        plt.plot(df_arima.index, df_arima['actual'],'-o', label='Actual')
        plt.plot(df_arima.index, df_arima['pred'],'-x', label='ARIMA')
        plt.title(f"[ARIMA] {tgt}")
        plt.legend()
        plt.show()

        # ---------- B. Prophet ----------
        df_prophet = walk_forward_validation_arbitrary(
            sub_df, train_size=10,
            train_func=train_prophet,
            predict_func=predict_prophet,
            x_cols=x_cols, y_col=tgt
        )
        msk_p = df_prophet['pred'].notnull()
        y_true_p = df_prophet.loc[msk_p,'actual']
        y_pred_p = df_prophet.loc[msk_p,'pred']
        rmse_prophet = sqrt(mean_squared_error(y_true_p, y_pred_p))
        mae_prophet  = mean_absolute_error(y_true_p, y_pred_p)
        plt.figure(figsize=(6,3))
        plt.plot(df_prophet.index, df_prophet['actual'],'-o', label='Actual')
        plt.plot(df_prophet.index, df_prophet['pred'],'-x', label='Prophet')
        plt.title(f"[Prophet] {tgt}")
        plt.legend()
        plt.show()

        # ---------- C. XGBoost ----------
        df_xgb = walk_forward_validation_arbitrary(
            sub_df, train_size=10,
            train_func=train_xgboost,
            predict_func=predict_xgboost,
            x_cols=x_cols, y_col=tgt
        )
        msk_x = df_xgb['pred'].notnull()
        y_true_x = df_xgb.loc[msk_x,'actual']
        y_pred_x = df_xgb.loc[msk_x,'pred']
        rmse_xgb = sqrt(mean_squared_error(y_true_x, y_pred_x))
        mae_xgb  = mean_absolute_error(y_true_x, y_pred_x)
        plt.figure(figsize=(6,3))
        plt.plot(df_xgb.index, df_xgb['actual'],'-o', label='Actual')
        plt.plot(df_xgb.index, df_xgb['pred'],'-x', label='XGB')
        plt.title(f"[XGBoost] {tgt}")
        plt.legend()
        plt.show()

        # ---------- D. LSTM ----------
        df_lstm = walk_forward_validation_lstm(
            sub_df, train_size=10,
            x_cols=x_cols, y_col=tgt,
            window_size=4, epochs=20, batch_size=2
        )
        msk_l = df_lstm['pred'].notnull()
        y_true_l = df_lstm.loc[msk_l,'actual']
        y_pred_l = df_lstm.loc[msk_l,'pred']
        rmse_lstm = sqrt(mean_squared_error(y_true_l, y_pred_l))
        mae_lstm  = mean_absolute_error(y_true_l, y_pred_l)
        plt.figure(figsize=(6,3))
        plt.plot(df_lstm.index, df_lstm['actual'],'-o', label='Actual')
        plt.plot(df_lstm.index, df_lstm['pred'],'-x', label='LSTM')
        plt.title(f"[LSTM] {tgt}")
        plt.legend()
        plt.show()

        # ---------- E. GRU ----------
        df_gru = walk_forward_validation_gru(
            sub_df, train_size=10,
            x_cols=x_cols, y_col=tgt,
            window_size=4, epochs=20, batch_size=2
        )
        msk_g = df_gru['pred'].notnull()
        y_true_g = df_gru.loc[msk_g,'actual']
        y_pred_g = df_gru.loc[msk_g,'pred']
        rmse_gru = sqrt(mean_squared_error(y_true_g, y_pred_g))
        mae_gru  = mean_absolute_error(y_true_g, y_pred_g)
        plt.figure(figsize=(6,3))
        plt.plot(df_gru.index, df_gru['actual'],'-o', label='Actual')
        plt.plot(df_gru.index, df_gru['pred'],'-x', label='GRU')
        plt.title(f"[GRU] {tgt}")
        plt.legend()
        plt.show()

        # (선택) 멀티스텝 예측 (예: 2분기 ahead)
        # df_multi = walk_forward_validation_multi_step(
        #     df=sub_df,
        #     train_size=10,
        #     steps_ahead=2,
        #     train_func=train_xgboost,
        #     predict_func=predict_xgboost,
        #     x_cols=x_cols,
        #     y_col=tgt
        # )
        # # 'pred_1','pred_2' 열이 생김

        # 결과 저장
        results[tgt] = {
            'ARIMA':   (rmse_arima,   mae_arima),
            'Prophet': (rmse_prophet, mae_prophet),
            'XGB':     (rmse_xgb,     mae_xgb),
            'LSTM':    (rmse_lstm,    mae_lstm),
            'GRU':     (rmse_gru,     mae_gru),
        }

    # 결과 요약
    print("\n======= [Summary] RMSE / MAE =======")
    for tgt, model_res in results.items():
        print(f"\n[Target: {tgt}]")
        for model_name, (rmse_, mae_) in model_res.items():
            print(f"{model_name:8s} | RMSE={rmse_:.3f}, MAE={mae_:.3f}")


###############################################
# 8. 실행 예시
###############################################
if __name__ == "__main__":
    file_path = "modified_infectious_disease_data_copy.csv"
    target_list = ["호흡기", "매개성"]

    # (옵션) 시차/이동평균 피처
    lag_feats = {
        '호흡기': [1],  # 예: 1시차
        '매개성': [1],      # 예: 1시차
    }
    roll_feats = {
        '호흡기': [2],  # 예: 2분기 이동평균
    }

    run_all_models(
        filepath=file_path,
        target_list=target_list,
        use_feature_engineering=True,
        lag_features=lag_feats,
        roll_features=roll_feats
    )
