###############################################
# 1. 라이브러리 및 한글 폰트 설정
###############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

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
    CSV 로드 -> date 인덱스 설정 -> 결측치/이상치 처리
    사용자가 수작업으로 'date' 만들었으므로
    CSV에는 이미 'date' 컬럼이 YYYY-MM-DD 형태로 들어있다고 가정
    """
    df = pd.read_csv(filepath)
    print("== Raw Data Preview ==")
    print(df.head())

    # 1) date 인덱스
    df = df.sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    # 2) 결측치 처리 + 이상치 클리핑
    feat_cols = [
        '최고기온 (C)', '최저기온 (C)', '일 평균기온 (C)',
        '일 평균 풍속 (m/s)', '일강수량 (mm)',
        '최심 신적설 (cm)', '일 평균 상대습도 (%)', '일교차 (C)'
    ]
    # 결측치: 중간값 대체
    for col in feat_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # ±3sigma 클리핑
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
    # lag
    if lag_features:
        for col, lags in lag_features.items():
            if col not in df_ext.columns:
                continue
            for lag in lags:
                new_col = f"{col}_lag{lag}"
                df_ext[new_col] = df_ext[col].shift(lag)
    # roll
    if roll_features:
        for col, windows in roll_features.items():
            if col not in df_ext.columns:
                continue
            for w in windows:
                new_col = f"{col}_roll{w}"
                df_ext[new_col] = df_ext[col].rolling(w).mean()
    # dropna
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

        model = train_func(X_train, y_train)  # 날짜 인덱스 필요하면 추가
        X_test = X_data[i:i+1]
        yhat = predict_func(model, X_test)
        df_result.iloc[i, df_result.columns.get_loc('pred')] = yhat

    return df_result

###############################################
# 4-1. 다중 스텝 예측 (Recursive)
###############################################
# (생략, 기존과 동일)

###############################################
# 5. LSTM / GRU (슬라이딩 윈도우)
###############################################
def make_lstm_dataset(X_data, y_data, window_size=4):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - window_size):
        X_seq.append(X_data[i:i+window_size, :])
        y_seq.append(y_data[i + window_size])
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(num_features: int, window_size=4, 
                     lstm_units=32):
    model = keras.Sequential()
    model.add(layers.Input(shape=(window_size, num_features)))
    model.add(layers.LSTM(lstm_units, activation='tanh'))
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
    batch_size: int = 2,
    lstm_units: int = 32  # << 하이퍼파라미터
) -> pd.DataFrame:
    # (생략, 기존과 동일) + lstm_units 인자 추가
    n = len(df)
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col].values

    X_all = df[x_cols].values
    y_all = df[y_col].values

    def train_lstm_model(X_train, y_train):
        if len(X_train) <= window_size:
            return build_lstm_model(X_train.shape[1], window_size, lstm_units)
        X_seq, y_seq = make_lstm_dataset(X_train, y_train, window_size=window_size)
        model_temp = build_lstm_model(X_train.shape[1], window_size, lstm_units)
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

def build_gru_model(num_features: int, window_size=4, 
                    gru_units=32):
    model = keras.Sequential()
    model.add(layers.Input(shape=(window_size, num_features)))
    model.add(layers.GRU(gru_units, activation='tanh'))
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
    batch_size: int = 2,
    gru_units: int = 32  # << 하이퍼파라미터
) -> pd.DataFrame:
    # (생략, 기존과 동일) + gru_units 인자 추가
    n = len(df)
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col].values

    X_all = df[x_cols].values
    y_all = df[y_col].values

    def make_lstm_dataset(X_train, y_train, window_size=4):
        # 중복 정의 but for clarity 
        X_seq, y_seq = [], []
        for i in range(len(X_train) - window_size):
            X_seq.append(X_train[i:i+window_size, :])
            y_seq.append(y_train[i + window_size])
        return np.array(X_seq), np.array(y_seq)

    def train_gru_model(X_train, y_train):
        if len(X_train) <= window_size:
            return build_gru_model(X_train.shape[1], window_size, gru_units)
        X_seq, y_seq = make_lstm_dataset(X_train, y_train, window_size=window_size)
        model_ = build_gru_model(X_train.shape[1], window_size, gru_units)
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

### (A) ARIMA 하이퍼파라미터 튜닝 예시 ###
def train_arima(X_train, y_train, train_dates=None):
    """
    예: p, d, q 범위를 수동 탐색하며
    AIC가 최소인 (p,d,q)를 찾는다
    """
    best_aic = np.inf
    best_order = None
    best_model = None

    p_values = [0,1,2]
    d_values = [0,1]
    q_values = [0,1,2]

    # 다변량 vs 단변량
    exog = X_train if X_train.shape[1] != 0 else None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model_ = pm.ARIMA(order=(p,d,q))
                    fit_ = model_.fit(y_train, exog=exog)
                    aic_ = fit_.aic()
                    if aic_ < best_aic:
                        best_aic = aic_
                        best_order = (p,d,q)
                        best_model = fit_
                except:
                    pass

    print(f"[ARIMA Tuning] best_order={best_order}, AIC={best_aic:.2f}")
    return best_model

def predict_arima(model, X_test, pred_date=None):
    exog_test = X_test if X_test.shape[1] != 0 else None
    fcst = model.predict(n_periods=1, exogenous=exog_test)
    return fcst[0]


### (B) Prophet 하이퍼파라미터 튜닝 예시 ###
def train_prophet(X_train, y_train, train_dates=None):
    """
    예: seasonality_mode in ['additive','multiplicative']
        yearly_seasonality in [True,False]
    간단히 2x2=4가지 조합 중 RMSE가 최소인 모델 고르기
    """
    from sklearn.metrics import mean_squared_error

    if train_dates is None:
        train_dates = pd.date_range('2000-01-01', periods=len(y_train), freq='MS')

    # regressor
    num_features = X_train.shape[1]

    combos = [
        ('additive', True),
        ('additive', False),
        ('multiplicative', True),
        ('multiplicative', False)
    ]
    best_rmse = np.inf
    best_combo = None
    best_model = None

    for mode_, yearly_ in combos:
        df_p = pd.DataFrame({'ds': train_dates, 'y': y_train})
        model_ = Prophet(seasonality_mode=mode_, yearly_seasonality=yearly_)
        if num_features>0:
            for f_idx in range(num_features):
                col_name = f"reg_{f_idx}"
                model_.add_regressor(col_name)
                df_p[col_name] = X_train[:, f_idx]

        model_.fit(df_p)

        # 간단 검증: 예측 training셋 (in-sample error)
        fcst = model_.predict(df_p)
        rmse_ = sqrt(mean_squared_error(df_p['y'], fcst['yhat']))
        if rmse_ < best_rmse:
            best_rmse = rmse_
            best_combo = (mode_,yearly_)
            best_model = model_

    print(f"[Prophet Tuning] best_combo={best_combo}, RMSE={best_rmse:.2f}")
    return best_model

def predict_prophet(model, X_test, pred_date=None):
    last_ds = model.history['ds'].max()
    next_ds = last_ds + pd.Timedelta(days=30) if pred_date is None else pred_date
    df_f = pd.DataFrame({'ds':[next_ds]})
    num_features = X_test.shape[1]
    if num_features > 0:
        for f_idx in range(num_features):
            col_name = f"reg_{f_idx}"
            df_f[col_name] = X_test[0, f_idx]
    fcst = model.predict(df_f)
    return fcst['yhat'].values[0]


### (C) XGBoost 하이퍼파라미터 튜닝 예시 ###
def train_xgboost(X_train, y_train, train_dates=None):
    """
    GridSearchCV 예시 (TimeSeriesSplit)
    """
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

    param_grid = {
        'n_estimators': [50,100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3,5]
    }
    xgb = XGBRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=3)  # 단순 예시
    gsearch = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=tscv,
        verbose=0
    )
    gsearch.fit(X_train, y_train)
    print(f"[XGB Tuning] best_params={gsearch.best_params_}, best_score={gsearch.best_score_:.2f}")
    return gsearch.best_estimator_

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
    df = load_and_preprocess_data(filepath)

    if use_feature_engineering:
        print("\n--- [Feature Engineering] lag/roll ---")
        df = create_feature_engineering(df, lag_features, roll_features)
        print(df.head())

    # 독립변수: (호흡기,매개성) 제외
    x_cols = [
        c for c in df.columns
        if c not in ['호흡기','매개성'] + target_list
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
        plt.title(f"[XGB] {tgt}")
        plt.legend()
        plt.show()

        # ---------- D. LSTM (추가 하이퍼파라미터: lstm_units)
        df_lstm = walk_forward_validation_lstm(
            df=sub_df, train_size=10,
            x_cols=x_cols, y_col=tgt,
            window_size=4, epochs=20, batch_size=2, lstm_units=32
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

        # ---------- E. GRU (추가 하이퍼파라미터: gru_units)
        df_gru = walk_forward_validation_gru(
            df=sub_df, train_size=10,
            x_cols=x_cols, y_col=tgt,
            window_size=4, epochs=20, batch_size=2, gru_units=32
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
    target_list = ["호흡기_new","매개성"]

    lag_feats = {
        '호흡기_new': [1],
        '매개성': [1]
    }
    roll_feats = {
        '호흡기_new': [2]
    }

    run_all_models(
        filepath=file_path,
        target_list=target_list,
        use_feature_engineering=True,
        lag_features=lag_feats,
        roll_features=roll_feats
    )
