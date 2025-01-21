###############################################
# 0. 라이브러리 import + 한글 폰트 설정
###############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import optuna                 # Optuna: 하이퍼파라미터 자동 탐색 라이브러리
import keras_tuner as kt      # KerasTuner: Keras 기반 모델 하이퍼파라미터 탐색 라이브러리
import pmdarima as pm         # pmdarima: ARIMA 모델 구현 라이브러리

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 텐서플로 GRU, LSTM 등 RNN 기반 모델
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# XGBoost (회귀용)
from xgboost import XGBRegressor

# Prophet (시계열 예측용)
from prophet import Prophet

# 한글 폰트 설정 (Windows 기준)
import matplotlib.font_manager as fm
font_name = fm.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
plt.rc('font', family=font_name)


###############################################
# 1. 데이터 로드 + 전처리
###############################################
def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    1) CSV 파일을 읽어온다.
    2) date 컬럼을 datetime으로 변환하고, 이를 인덱스로 설정한다.
    3) 특정 열(feat_cols)에 대해 결측치가 있을 경우, 중간값(median)으로 채운다.
    4) 동일 열에 대해 평균±3표준편차를 벗어나는 이상치(outlier)를 clip하여 제거한다.
    5) 최종 전처리된 DataFrame을 반환한다.
    """
    df = pd.read_csv(filepath)
    print("== Raw Data Preview ==")
    print(df.head())

    # (1) date 기준 정렬 + 인덱스 설정
    df = df.sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])      # 문자열을 datetime 객체로 변환
    df = df.set_index('date').sort_index()       # date를 인덱스로 설정

    # (2) 특정 열에 대해 결측치 처리 및 이상치 처리
    feat_cols = [
        '최고기온 (C)', '최저기온 (C)', '일 평균기온 (C)',
        '일 평균 풍속 (m/s)', '일강수량 (mm)',
        '최심 신적설 (cm)', '일 평균 상대습도 (%)', '일교차 (C)',
        'PM2.5(μg/m³)', 'PM10(μg/m³)', '아황산가스(ppm)',
        '오존(ppm)', '이산화질소(ppm)', '일산화질소(ppm)'
    ]
    for col in feat_cols:
        # 만약 결측치가 있다면 중간값으로 대체
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in feat_cols:
        if col in df.columns:
            # 평균±3표준편차 밖의 값들을 clip 처리
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
# 2. 피처 엔지니어링 (lag, roll)
###############################################
def create_feature_engineering(df, lag_features=None, roll_features=None):
    """
    - lag_features: { '컬럼명': [1,2,3...] } 형태로, 해당 컬럼을 몇 시점(lag) 이전 값을 추가할지 지정.
      예) {'호흡기_new':[1,2]} -> 호흡기_new_lag1, 호흡기_new_lag2 라는 열 생성
    - roll_features: { '컬럼명': [window1, window2,...] } 형태로, rolling window(이동평균) 적용.
      예) {'호흡기_new':[3]} -> 호흡기_new_roll3 라는 열 생성(3일 이동평균)
    1) df를 복사한 뒤, lag, roll 특징을 만든다.
    2) 최종적으로 생긴 NaN 행은 dropna()로 제거한다.
    """
    df_ext = df.copy()

    # (1) Lag features 생성
    if lag_features:
        for col, lags in lag_features.items():
            if col not in df_ext.columns:
                continue
            for lag in lags:
                df_ext[f"{col}_lag{lag}"] = df_ext[col].shift(lag)

    # (2) Rolling features 생성
    if roll_features:
        for col, windows in roll_features.items():
            if col not in df_ext.columns:
                continue
            for w in windows:
                df_ext[f"{col}_roll{w}"] = df_ext[col].rolling(w).mean()

    # (3) 이 과정에서 생긴 NaN 제거
    df_ext.dropna(inplace=True)
    return df_ext


###############################################
# 3. Walk-Forward (ARIMA, XGB 등 2D 입력용)
###############################################
def walk_forward_validation_arbitrary(
    df: pd.DataFrame,
    train_size: int,
    train_func,
    predict_func,
    x_cols: list,
    y_col: str
) -> pd.DataFrame:
    """
    다차원(2D) 입력 X와 1D 타겟 y가 있을 때, Walk-Forward Validation을 수행한다.
    - df: 전체 데이터
    - train_size: 처음 몇 개 행을 훈련으로 사용할지 (고정 길이)
    - train_func: X_train, y_train을 입력받아 모델을 훈련하여 반환하는 함수
    - predict_func: 훈련된 모델, X_test(1개 시점) -> 예측값 yhat
    - x_cols: 피처로 사용할 열 목록
    - y_col: 예측하려는 타겟 열
    * i번째 루프에서, 0~i-1까지를 학습, i번째(1-step ahead)를 예측한다.
    """
    n = len(df)
    # 예측 결과를 저장할 df_result 생성
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col].values

    # 넘파이 배열로 변환
    X_data = df[x_cols].values
    y_data = df[y_col].values

    # train_size부터 끝까지(1-step씩) 이동하며 예측
    for i in range(train_size, n):
        X_train = X_data[:i]
        y_train = y_data[:i]

        # (1) 모델 훈련
        model = train_func(X_train, y_train)

        # (2) 예측
        X_test = X_data[i:i+1]  # i번째 행(1개)
        if np.isnan(X_test).any():
            X_test = np.nan_to_num(X_test, nan=0.0)

        yhat = predict_func(model, X_test)
        df_result.iloc[i, df_result.columns.get_loc('pred')] = yhat

    return df_result


###############################################
# 4. RNN(LSTM, GRU)용 Dataset 생성 및 Walk-Forward
###############################################
def make_lstm_dataset(X_data, y_data, window_size=4):
    """
    LSTM/GRU 모델은 (batch_size, time_steps, features) 형태의 3D 입력이 필요.
    - window_size: 시퀀스 길이(과거 몇 개 시점)
    예) window_size=4일 때
        i=0~3 입력 -> i=4 예측
        i=1~4 입력 -> i=5 예측 ...
    """
    X_seq, y_seq = [], []
    for i in range(len(X_data) - window_size):
        # 연속된 window_size만큼 X
        X_seq.append(X_data[i:i+window_size, :])
        # 그 다음 시점의 y를 타겟
        y_seq.append(y_data[i+window_size])
    return np.array(X_seq), np.array(y_seq)


def walk_forward_validation_lstm(
    df: pd.DataFrame,
    train_size: int,
    x_cols: list,
    y_col: str,
    train_func,
    predict_func,
    window_size=4
) -> pd.DataFrame:
    """
    LSTM을 이용한 Walk-Forward Validation.
    - i 시점에서 모델 훈련 시, 0~i-1 데이터를 사용
    - i < window_size인 경우에는 예측 불가(혹은 window_size만큼 부족) → NaN
    - i >= window_size가 되면, (i-window_size ~ i-1) 구간 데이터를 입력으로 1-step 예측
    """
    n = len(df)
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col].values

    X_all = df[x_cols].values
    y_all = df[y_col].values

    # i를 10부터 n-1까지 루프
    for i in range(train_size, n):
        X_train = X_all[:i]
        y_train = y_all[:i]

        # (1) 모델 훈련
        model = train_func(X_train, y_train)

        # (2) 예측: 시퀀스가 window_size 이상이어야 한다
        if i < window_size:
            df_result.iloc[i, df_result.columns.get_loc('pred')] = np.nan
        else:
            X_seq = X_all[i-window_size:i]  # i-window_size~i-1
            X_seq = X_seq.reshape(1, window_size, X_seq.shape[1])  # (1, window_size, num_features)
            yhat = predict_func(model, X_seq)
            df_result.iloc[i, df_result.columns.get_loc('pred')] = yhat

    return df_result


def walk_forward_validation_gru(
    df: pd.DataFrame,
    train_size: int,
    x_cols: list,
    y_col: str,
    train_func,
    predict_func,
    window_size=4
) -> pd.DataFrame:
    """
    GRU 버전도 LSTM과 동일한 구조로 Walk-Forward를 수행.
    """
    n = len(df)
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col].values

    X_all = df[x_cols].values
    y_all = df[y_col].values

    for i in range(train_size, n):
        X_train = X_all[:i]
        y_train = y_all[:i]

        # (1) 모델 훈련
        model = train_func(X_train, y_train)

        # (2) 예측
        if i < window_size:
            df_result.iloc[i, df_result.columns.get_loc('pred')] = np.nan
        else:
            X_seq = X_all[i-window_size:i]
            X_seq = X_seq.reshape(1, window_size, X_seq.shape[1])
            yhat = predict_func(model, X_seq)
            df_result.iloc[i, df_result.columns.get_loc('pred')] = yhat

    return df_result


###############################################
# 5. ARIMA/XGB + Optuna 하이퍼 파라미터 튜닝
###############################################
def objective_arima(trial, X_train, y_train):
    """
    ARIMA의 (p, d, q) 파라미터를 Optuna로 탐색하기 위한 Objective 함수.
    - trial.suggest_int()를 이용해 p, d, q 범위를 지정.
    - pmdarima.ARIMA(order=(p,d,q))를 사용하여 모델 훈련 후 AIC 값을 반환.
    - AIC가 낮을수록 좋은 모델로 간주.
    """
    p = trial.suggest_int('p', 0, 3)
    d = trial.suggest_int('d', 0, 2)
    q = trial.suggest_int('q', 0, 3)
    exog = X_train if X_train.shape[1] > 0 else None

    try:
        model_ = pm.ARIMA(order=(p, d, q))
        fit_ = model_.fit(y_train, exogenous=exog)
        return fit_.aic()
    except:
        # 모델 훈련 실패 시, 큰 값(1e9) 반환 -> 해당 파라미터는 배제
        return 1e9


def train_arima_optuna(X_train, y_train, n_trials=3):
    """
    ARIMA 파라미터 (p, d, q)를 Optuna로 탐색하여
    가장 낮은 AIC를 주는 모델을 최종 반환.
    """
    exog = X_train if X_train.shape[1] > 0 else None

    def _obj(trial):
        return objective_arima(trial, X_train, y_train)

    study = optuna.create_study(direction='minimize')
    study.optimize(_obj, n_trials=n_trials)
    best_ = study.best_trial

    # best_ 파라미터 추출
    p_ = best_.params['p']
    d_ = best_.params['d']
    q_ = best_.params['q']
    print(f"[ARIMA Optuna] best order=({p_},{d_},{q_}), AIC={best_.value}")

    # 최적 파라미터로 다시 모델 적합
    model_ = pm.ARIMA(order=(p_, d_, q_))
    fit_ = model_.fit(y_train, exogenous=exog)
    return fit_


def predict_arima_optuna(model, X_test):
    """
    ARIMA 모델로 1-step ahead 예측을 수행 (return_conf_int=False)
    -> 신뢰구간 계산 비활성화
    """
    exog_test = X_test if X_test.shape[1] > 0 else None
    if exog_test is not None and np.isnan(exog_test).any():
        exog_test = np.nan_to_num(exog_test, nan=0.0)

    fcst = model.predict(
        n_periods=1,
        exogenous=exog_test,
        return_conf_int=False
    )
    return fcst[0]


def objective_xgb(trial, X_train, y_train):
    """
    XGBRegressor의 주요 파라미터를 Optuna로 탐색하기 위한 objective 함수.
    - n_estimators, learning_rate, max_depth 등을 제시된 범위에서 선택.
    - 모델을 훈련 후, RMSE를 반환.
    """
    n_estimators_ = trial.suggest_int('n_estimators', 50, 200, step=50)
    lr_ = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
    max_depth_ = trial.suggest_int('max_depth', 2, 6)

    xgb_ = XGBRegressor(
        n_estimators=n_estimators_,
        learning_rate=lr_,
        max_depth=max_depth_,
        random_state=42
    )
    xgb_.fit(X_train, y_train)
    y_pred = xgb_.predict(X_train)
    return sqrt(mean_squared_error(y_train, y_pred))


def train_xgb_optuna(X_train, y_train, n_trials=3):
    """
    XGBRegressor 파라미터를 Optuna로 n_trials만큼 탐색,
    가장 낮은 RMSE를 주는 모델을 최종 반환.
    """
    def _obj(trial):
        return objective_xgb(trial, X_train, y_train)

    study = optuna.create_study(direction='minimize')
    study.optimize(_obj, n_trials=n_trials)
    best_ = study.best_trial

    xgb_final = XGBRegressor(
        n_estimators=best_.params['n_estimators'],
        learning_rate=best_.params['learning_rate'],
        max_depth=best_.params['max_depth'],
        random_state=42
    )
    xgb_final.fit(X_train, y_train)

    print(f"[XGB Optuna] best params={best_.params}, best rmse={best_.value}")
    return xgb_final


def predict_xgb_optuna(model, X_test):
    """
    XGB 회귀 모델로 1개 샘플(X_test)에 대한 예측값을 반환.
    """
    return model.predict(X_test)[0]


###############################################
# 6. Prophet 전용 Optuna 튜닝 + Walk-Forward
###############################################
def train_prophet_optuna(
    df: pd.DataFrame,
    exogenous_cols: list = None,
    param_space: dict = None,
    n_trials: int = 3
) -> callable:
    """
    Prophet의 주요 파라미터(seasonality_mode, yearly_seasonality, changepoint_prior_scale 등)를
    Optuna로 탐색하기 위한 함수를 만들어 반환한다.
    - walk-forward에서 각 시점(i)마다, tune_func(dates_train, y_train, exog_train)을 호출해
      Prophet 모델을 최적 파라미터로 훈련할 수 있다.
    """
    from sklearn.metrics import mean_squared_error

    # (1) 기본 파라미터 범위 설정(필요 시 수정 가능)
    if param_space is None:
        param_space = {
            'seasonality_mode': ['additive', 'multiplicative'],
            'yearly_seasonality': [True, False],
            'changepoint_prior_scale': [0.01, 0.1, 0.5, 1.0]
        }

    df_temp = df.copy()

    def tune_func(dates_train, y_train, exog_train=None):
        """
        walk-forward 내에서 호출되는 함수.
        dates_train, y_train, exog_train을 사용해 Prophet 파라미터를 Optuna로 튜닝 후,
        최적 모델을 훈련해 반환.
        """
        def local_objective(trial):
            seasonality_mode_ = trial.suggest_categorical(
                'seasonality_mode',
                param_space['seasonality_mode']
            )
            yearly_seasonality_ = trial.suggest_categorical(
                'yearly_seasonality',
                param_space['yearly_seasonality']
            )
            # changepoint_prior_scale이 float 범위인지, categorical인지에 따라 분기
            if len(param_space['changepoint_prior_scale']) > 2:
                changepoint_prior_ = trial.suggest_float(
                    'changepoint_prior_scale',
                    min(param_space['changepoint_prior_scale']),
                    max(param_space['changepoint_prior_scale']),
                    step=0.01
                )
            else:
                changepoint_prior_ = trial.suggest_categorical(
                    'changepoint_prior_scale',
                    param_space['changepoint_prior_scale']
                )

            model_ = Prophet(
                seasonality_mode=seasonality_mode_,
                yearly_seasonality=yearly_seasonality_,
                changepoint_prior_scale=changepoint_prior_
            )

            # Prophet 훈련용 DataFrame 준비 (ds, y)
            df_p = pd.DataFrame({'ds': dates_train, 'y': y_train})
            # exogenous_cols가 있으면, 해당 열을 add_regressor()로 추가
            if exogenous_cols and exog_train is not None:
                for idx, c in enumerate(exogenous_cols):
                    df_p[c] = exog_train[:, idx]
                    model_.add_regressor(c)

            model_.fit(df_p)

            # 훈련 데이터에 대해 예측 후 RMSE 계산
            forecast_ = model_.predict(df_p[['ds'] + (exogenous_cols if exogenous_cols else [])])
            rmse_ = sqrt(mean_squared_error(df_p['y'], forecast_['yhat']))
            return rmse_

        # (2) Optuna Study 생성 & 탐색
        study = optuna.create_study(direction='minimize')
        study.optimize(local_objective, n_trials=n_trials)
        best_ = study.best_trial
        print(f"[Prophet Optuna] best params={best_.params}, best RMSE={best_.value}")

        # (3) 최적 파라미터로 모델 생성
        best_model = Prophet(
            seasonality_mode=best_.params['seasonality_mode'],
            yearly_seasonality=best_.params['yearly_seasonality'],
            changepoint_prior_scale=best_.params['changepoint_prior_scale']
        )

        # (4) 최적 파라미터 모델에 exogenous 등록 + fit
        df_p = pd.DataFrame({'ds': dates_train, 'y': y_train})
        if exogenous_cols and exog_train is not None:
            for idx, c in enumerate(exogenous_cols):
                df_p[c] = exog_train[:, idx]
                best_model.add_regressor(c)

        best_model.fit(df_p)
        return best_model

    # tune_func 자체를 반환
    return tune_func


def predict_prophet_optuna(model, next_date, exogenous=None):
    """
    Prophet 모델로 특정 다음 시점(next_date)에 대한 예측값 yhat을 반환.
    exogenous(추가 회귀자)가 있으면 해당 값을 future_df에 추가.
    """
    future_df = pd.DataFrame({'ds': [next_date]})
    if exogenous is not None:
        # 모델에 등록된 extra_regressors 순서대로 값 입력
        for idx, c in enumerate(model.extra_regressors.keys()):
            future_df[c] = exogenous[0, idx]

    fcst = model.predict(future_df)
    return fcst['yhat'].values[0]


def walk_forward_validation_prophet_optuna(
    df: pd.DataFrame,
    train_size: int,
    y_col: str,
    prophet_tune_func,
    exog_cols: list = None
) -> pd.DataFrame:
    """
    Prophet에서도 walk-forward 방식으로 1-step ahead 예측을 수행.
    - i번째 시점에서 0~i-1 데이터를 사용해 Optuna로 Prophet 모델 튜닝 & 훈련
    - i 시점에 대한 예측
    """
    n = len(df)
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col].values

    # 시계열 인덱스, 타겟, 외생변수 준비
    date_idx = df.index
    y_all    = df[y_col].values
    exog_all = df[exog_cols].values if exog_cols else None

    for i in range(train_size, n):
        # (1) 0~i-1까지 학습
        date_train = date_idx[:i]
        y_train    = y_all[:i]
        exog_train = exog_all[:i] if exog_all is not None else None

        # (2) Prophet 모델 튜닝 & 훈련
        model = prophet_tune_func(date_train, y_train, exog_train)

        # (3) i 시점 예측
        next_d = date_idx[i]
        exog_i = exog_all[i:i+1] if exog_all is not None else None
        yhat = predict_prophet_optuna(model, next_d, exog_i)
        df_result.iloc[i, df_result.columns.get_loc('pred')] = yhat

    return df_result


###############################################
# 7. (수정본) LSTM + KerasTuner
###############################################
def build_lstm_model(num_features, window_size=4, lstm_units=32):
    """
    LSTM 신경망 구조를 정의하는 함수.
    - num_features: 입력 피처 개수
    - window_size: 타임스텝 길이
    - lstm_units: LSTM 셀의 유닛 수
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(window_size, num_features)))
    model.add(layers.LSTM(lstm_units, activation='tanh'))
    model.add(layers.Dense(1))  # 최종 출력 1개
    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm_ktuner(
    X_train, y_train,
    window_size=4,
    max_trials=5,
    batch_size_candidates=[4, 8, 16],
    epochs_candidates=[5, 10, 20],
    patience=2
):
    """
    LSTM 모델의 하이퍼파라미터를 KerasTuner(RandomSearch)로 탐색.
    - X_train, y_train: 2D 입력(실제로는 make_lstm_dataset으로 3D seq로 변환)
    - window_size: 시퀀스 길이
    - max_trials: RandomSearch 시도 횟수
    - batch_size_candidates, epochs_candidates: fit 시의 후보
    - patience: EarlyStopping patience
    """
    # (1) 학습 데이터(행 수)가 window_size 이하라면 fallback
    if len(X_train) <= window_size:
        fallback_model = build_lstm_model(
            num_features=X_train.shape[1],
            window_size=window_size,
            lstm_units=32
        )
        # 임시로 1번만 fit
        fallback_model.fit(X_train[:1], y_train[:1], epochs=1, verbose=0)
        return fallback_model

    # (2) LSTM 시퀀스 데이터셋 생성
    X_seq, y_seq = make_lstm_dataset(X_train, y_train, window_size)

    # (3) KerasTuner용 "모델 빌드" 함수 정의
    def build_lstm_model_with_hp(hp):
        units = hp.Int('units', min_value=16, max_value=128, step=16)
        drop_ = hp.Float('dropout', 0.0, 0.5, step=0.1)
        lr_   = hp.Choice('learning_rate', [1e-3, 1e-4, 5e-4, 5e-3])

        model = keras.Sequential()
        model.add(layers.Input(shape=(window_size, X_train.shape[1])))
        model.add(layers.LSTM(units, activation='tanh'))
        if drop_ > 0:
            model.add(layers.Dropout(drop_))
        model.add(layers.Dense(1))
        model.compile(
            optimizer=keras.optimizers.Adam(lr_),
            loss='mse'
        )
        return model

    # (4) RandomSearch Tuner 설정
    tuner = kt.RandomSearch(
        hypermodel=build_lstm_model_with_hp,
        objective='val_loss',
        max_trials=max_trials,
        overwrite=True,
        directory='C:/temp/ktuner_lstm_output',
        project_name='lstm_tune_wf'
    )

    # (5) EarlyStopping 콜백
    es = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    # (6) KerasTuner 탐색(여기서는 epochs, batch_size를 고정)
    best_epoch = max(epochs_candidates)
    best_batch = min(batch_size_candidates)

    tuner.search(
        X_seq, y_seq,
        validation_split=0.2,
        epochs=best_epoch,
        batch_size=best_batch,
        callbacks=[es],
        verbose=0
    )

    # (7) 최적 하이퍼파라미터 확인
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"[LSTM Tuning] best -> units={best_hp.get('units')}, "
          f"drop={best_hp.get('dropout')}, "
          f"lr={best_hp.get('learning_rate')}")

    # (8) 최적 모델로 재훈련
    final_model = tuner.hypermodel.build(best_hp)
    final_model.fit(
        X_seq, y_seq,
        validation_split=0.2,
        epochs=best_epoch,
        batch_size=best_batch,
        callbacks=[es],
        verbose=0
    )
    return final_model


def predict_lstm_ktuner(model, X_seq):
    """
    LSTM 모델에 (1, window_size, num_features) 입력을 주어 예측.
    예측값이 shape (1,1)이므로 [0,0] 추출.
    """
    return model.predict(X_seq)[0, 0]


###############################################
# (추가) GRU + KerasTuner
###############################################
def build_gru_model(num_features, window_size=4, gru_units=32):
    """
    GRU 기본 구조를 정의하는 함수 (LSTM과 유사, 레이어만 GRU)
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(window_size, num_features)))
    model.add(layers.GRU(gru_units, activation='tanh'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_gru_ktuner(
    X_train, y_train,
    window_size=4,
    max_trials=5,
    batch_size_candidates=[4, 8, 16],
    epochs_candidates=[5, 10, 20],
    patience=2
):
    """
    GRU 모델의 하이퍼파라미터를 KerasTuner로 탐색.
    LSTM과 동일한 방식이지만, build_gru_model_with_hp 안에서 GRU 레이어 사용.
    """
    # (1) 데이터 부족 시 fallback
    if len(X_train) <= window_size:
        fallback_model = build_gru_model(
            num_features=X_train.shape[1],
            window_size=window_size,
            gru_units=32
        )
        fallback_model.fit(X_train[:1], y_train[:1], epochs=1, verbose=0)
        return fallback_model

    # (2) make_lstm_dataset 재활용
    X_seq, y_seq = make_lstm_dataset(X_train, y_train, window_size)

    # (3) GRU 하이퍼파라미터 서치 함수
    def build_gru_model_with_hp(hp):
        units = hp.Int('units', min_value=16, max_value=128, step=16)
        drop_ = hp.Float('dropout', 0.0, 0.5, step=0.1)
        lr_   = hp.Choice('learning_rate', [1e-3, 1e-4, 5e-4, 5e-3])

        model = keras.Sequential()
        model.add(layers.Input(shape=(window_size, X_train.shape[1])))
        model.add(layers.GRU(units, activation='tanh'))
        if drop_ > 0:
            model.add(layers.Dropout(drop_))
        model.add(layers.Dense(1))

        model.compile(
            optimizer=keras.optimizers.Adam(lr_),
            loss='mse'
        )
        return model

    # KerasTuner: RandomSearch
    tuner = kt.RandomSearch(
        hypermodel=build_gru_model_with_hp,
        objective='val_loss',
        max_trials=max_trials,
        overwrite=True,
        directory='C:/temp/ktuner_gru_output',
        project_name='gru_tune_wf'
    )

    # EarlyStopping
    es = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    best_epoch = max(epochs_candidates)
    best_batch = min(batch_size_candidates)

    # 하이퍼파라미터 탐색
    tuner.search(
        X_seq, y_seq,
        validation_split=0.2,
        epochs=best_epoch,
        batch_size=best_batch,
        callbacks=[es],
        verbose=0
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"[GRU Tuning] best -> units={best_hp.get('units')}, "
          f"drop={best_hp.get('dropout')}, "
          f"lr={best_hp.get('learning_rate')}")

    final_model = tuner.hypermodel.build(best_hp)
    final_model.fit(
        X_seq, y_seq,
        validation_split=0.2,
        epochs=best_epoch,
        batch_size=best_batch,
        callbacks=[es],
        verbose=0
    )
    return final_model


def predict_gru_ktuner(model, X_seq):
    """
    GRU 모델 입력 (1, window_size, num_features) -> 예측값 1개.
    """
    return model.predict(X_seq)[0, 0]


###############################################
# 8. 결과 시각화 및 종합 실행 함수
###############################################
def plot_pred_vs_actual(df_result: pd.DataFrame, model_name: str, y_col: str):
    """
    예측 결과(df_result)에서 actual, pred 컬럼을 시각화.
    - model_name: 차트 제목에 들어갈 모델명
    - y_col: 예측한 타겟 컬럼명
    """
    plt.figure(figsize=(8,4))
    plt.plot(df_result.index, df_result['actual'], label='Actual', marker='o')
    plt.plot(df_result.index, df_result['pred'],   label=f'{model_name} Pred', marker='x')
    plt.title(f"{model_name} - {y_col}")
    plt.xlabel("Date")
    plt.ylabel(y_col)
    plt.legend()
    plt.grid(True)
    plt.show()


def run_all_models(
    filepath,
    target_list,
    use_feature_engineering=True,
    lag_features=None,
    roll_features=None,
    window_size=4
):
    """
    1) CSV 로드 & 전처리
    2) 피처 엔지니어링
    3) Prophet Optuna 튜너 생성
    4) 각 타겟 컬럼에 대해 ARIMA, XGB, Prophet, LSTM, GRU 모델을 walk-forward로 훈련 & 예측
    5) RMSE, MAE 계산 & 시각화
    6) 결과 요약 출력
    """
    # (1) 데이터 로드 + 전처리
    df = load_and_preprocess_data(filepath)

    # (2) 피처 엔지니어링
    if use_feature_engineering:
        df = create_feature_engineering(df, lag_features, roll_features)
        print("\n[Check NaN after Feature Engineering]\n", df.isnull().sum())

    # (2-1) 추가적으로 다시 한 번 dropna
    df.dropna(inplace=True)
    print("[Final Check] 전체 NaN 개수:\n", df.isnull().sum())
    print("최종 데이터 shape =", df.shape)

    # (3) 입력(X), 타겟(y) 구분
    x_cols = [c for c in df.columns if c not in target_list]
    results = {}

    # Prophet용 Optuna 튜너(필요 시 exogenous_cols 지정 가능)
    prophet_tune_func = train_prophet_optuna(
        df, exogenous_cols=None,
        param_space=None,
        n_trials=3
    )

    # (4) 타겟별 모델 실행
    for tgt in target_list:
        if tgt not in df.columns:
            print(f"[WARNING] {tgt} not in columns -> skip")
            continue

        print(f"\n======= Target: {tgt} =======")
        sub_df = df[x_cols + [tgt]].copy()  # 모델별로 사용할 서브셋

        # ---- ARIMA ----
        df_arima = walk_forward_validation_arbitrary(
            sub_df,
            train_size=10,
            train_func=lambda Xtr, ytr: train_arima_optuna(Xtr, ytr, n_trials=3),
            predict_func=predict_arima_optuna,
            x_cols=x_cols,
            y_col=tgt
        )
        mask_arima = df_arima['pred'].notnull()
        rmse_arima = sqrt(mean_squared_error(df_arima.loc[mask_arima, 'actual'],
                                             df_arima.loc[mask_arima, 'pred']))
        mae_arima  = mean_absolute_error(df_arima.loc[mask_arima, 'actual'],
                                         df_arima.loc[mask_arima, 'pred'])
        plot_pred_vs_actual(df_arima, "ARIMA(Optuna)", tgt)

        # ---- XGB ----
        df_xgb = walk_forward_validation_arbitrary(
            sub_df,
            train_size=10,
            train_func=lambda Xtr, ytr: train_xgb_optuna(Xtr, ytr, n_trials=3),
            predict_func=predict_xgb_optuna,
            x_cols=x_cols,
            y_col=tgt
        )
        mask_xgb = df_xgb['pred'].notnull()
        rmse_xgb = sqrt(mean_squared_error(df_xgb.loc[mask_xgb, 'actual'],
                                           df_xgb.loc[mask_xgb, 'pred']))
        mae_xgb  = mean_absolute_error(df_xgb.loc[mask_xgb, 'actual'],
                                       df_xgb.loc[mask_xgb, 'pred'])
        plot_pred_vs_actual(df_xgb, "XGB(Optuna)", tgt)

        # ---- Prophet ----
        df_prophet = walk_forward_validation_prophet_optuna(
            df=sub_df,
            train_size=10,
            y_col=tgt,
            prophet_tune_func=prophet_tune_func,
            exog_cols=None
        )
        mask_ppt = df_prophet['pred'].notnull()
        rmse_ppt = sqrt(mean_squared_error(df_prophet.loc[mask_ppt, 'actual'],
                                           df_prophet.loc[mask_ppt, 'pred']))
        mae_ppt  = mean_absolute_error(df_prophet.loc[mask_ppt, 'actual'],
                                       df_prophet.loc[mask_ppt, 'pred'])
        plot_pred_vs_actual(df_prophet, "Prophet(Optuna)", tgt)

        # ---- LSTM ----
        df_lstm = walk_forward_validation_lstm(
            df=sub_df,
            train_size=10,
            x_cols=x_cols,
            y_col=tgt,
            train_func=lambda Xtr, ytr: train_lstm_ktuner(
                Xtr, ytr,
                window_size=window_size,
                max_trials=2,
                batch_size_candidates=[4,8,16],
                epochs_candidates=[5,10],
                patience=2
            ),
            predict_func=predict_lstm_ktuner,
            window_size=window_size
        )
        mask_lstm = df_lstm['pred'].notnull()
        rmse_lstm = sqrt(mean_squared_error(df_lstm.loc[mask_lstm, 'actual'],
                                            df_lstm.loc[mask_lstm, 'pred']))
        mae_lstm  = mean_absolute_error(df_lstm.loc[mask_lstm, 'actual'],
                                        df_lstm.loc[mask_lstm, 'pred'])
        plot_pred_vs_actual(df_lstm, "LSTM(KerasTuner)", tgt)

        # ---- GRU ----
        df_gru = walk_forward_validation_gru(
            df=sub_df,
            train_size=10,
            x_cols=x_cols,
            y_col=tgt,
            train_func=lambda Xtr, ytr: train_gru_ktuner(
                Xtr, ytr,
                window_size=window_size,
                max_trials=2,
                batch_size_candidates=[4,8,16],
                epochs_candidates=[5,10],
                patience=2
            ),
            predict_func=predict_gru_ktuner,
            window_size=window_size
        )
        mask_gru = df_gru['pred'].notnull()
        rmse_gru = sqrt(mean_squared_error(df_gru.loc[mask_gru, 'actual'],
                                           df_gru.loc[mask_gru, 'pred']))
        mae_gru  = mean_absolute_error(df_gru.loc[mask_gru, 'actual'],
                                       df_gru.loc[mask_gru, 'pred'])
        plot_pred_vs_actual(df_gru, "GRU(KerasTuner)", tgt)

        # 모델별 결과 요약 저장
        results[tgt] = {
            'ARIMA(Optuna)': (rmse_arima, mae_arima),
            'XGB(Optuna)'  : (rmse_xgb, mae_xgb),
            'Prophet(Optuna)': (rmse_ppt, mae_ppt),
            'LSTM(KerasTuner)': (rmse_lstm, mae_lstm),
            'GRU(KerasTuner)':  (rmse_gru, mae_gru)
        }

    # (5) 결과 요약 출력
    print("\n======= [Summary: RMSE / MAE] =======")
    for tgt, model_res in results.items():
        print(f"\n[Target: {tgt}]")
        for mname, (r, m) in model_res.items():
            print(f"{mname:20s} | RMSE={r:.3f}, MAE={m:.3f}")


###############################################
# 9. 실행 예시 (if __name__=="__main__":)
###############################################
if __name__=="__main__":
    # 예시 CSV 파일 경로
    file_path = "modified_infectious_disease_data_copy.csv"
    # 예측할 타겟 열 리스트
    targets   = ["호흡기_new", "매개성"]

    # Lag, Rolling 설정 예시
    lag_feats = {
        '호흡기_new': [1],
        '매개성': [1]
    }
    roll_feats = {
        '호흡기_new': [2]
    }

    # run_all_models 함수 실행
    run_all_models(
        filepath=file_path,
        target_list=targets,
        use_feature_engineering=True,
        lag_features=lag_feats,
        roll_features=roll_feats,
        window_size=4
    )
