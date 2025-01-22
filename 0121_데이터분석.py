###################################################
# 개선된 Python 코드 (주석 포함)
###################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
import keras_tuner as kt
import pmdarima as pm

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 텐서플로 GRU, LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# XGBoost
from xgboost import XGBRegressor

# Prophet
from prophet import Prophet

# (가정) Windows 환경에서만 필요할 수 있는 폰트 설정
import matplotlib.font_manager as fm
font_name = fm.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
plt.rc('font', family=font_name)


###################################################
# [0] 데이터 로드 및 전처리 함수
###################################################
def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    CSV 파일을 불러온 뒤,
    1) 'date' 컬럼을 datetime으로 변환하여 인덱스로 설정
    2) 일부 열에 대한 결측치 처리(예: 중앙값)
    3) 이상치 처리(평균 ± 3*표준편차 범위 벗어나면 clip)
    4) sort_index로 시계열 정렬

    Parameters
    ----------
    filepath : str
        CSV 파일 경로

    Returns
    -------
    df : pd.DataFrame
        전처리 완료된 DataFrame
    """
    df = pd.read_csv(filepath)
    print("== Raw Data Preview ==")
    print(df.head())

    # 날짜형 인덱스
    df = df.sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    # 예시로 특정 컬럼들에 대해 결측치/이상치 처리
    feat_cols = [
        '최고기온 (C)', '최저기온 (C)', '일 평균기온 (C)',
        '일 평균 풍속 (m/s)', '일강수량 (mm)',
        '최심 신적설 (cm)', '일 평균 상대습도 (%)', '일교차 (C)',
        'PM2.5(μg/m³)', 'PM10(μg/m³)', '아황산가스(ppm)',
        '오존(ppm)', '이산화질소(ppm)', '일산화질소(ppm)'
    ]
    # 1) 결측치 처리
    for col in feat_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    # 2) 이상치 처리 (평균 ± 3*표준편차)
    for col in feat_cols:
        if col in df.columns:
            mean_ = df[col].mean()
            std_  = df[col].std()
            lower = mean_ - 3*std_
            upper = mean_ + 3*std_
            df[col] = df[col].clip(lower=lower, upper=upper)

    print("\n== Preprocessed Data Preview ==")
    print(df.head())
    return df


###################################################
# [1] 피처 엔지니어링 (lag, roll)
###################################################
def create_feature_engineering(df: pd.DataFrame,
                               lag_features: dict = None,
                               roll_features: dict = None) -> pd.DataFrame:
    """
    주어진 DataFrame에 lag, roll(이동평균) 등의 파생 피처를 생성

    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터
    lag_features : dict
        예: { '컬럼명': [1,2,3] } 형태
    roll_features : dict
        예: { '컬럼명': [3,7] } 형태 (3,7일 이동평균 등)

    Returns
    -------
    df_ext : pd.DataFrame
        lag, roll 피처가 추가된 결과
    """
    df_ext = df.copy()

    # Lag: 이전 시점들(1,2,3...) 추가
    if lag_features:
        for col, lags in lag_features.items():
            if col not in df_ext.columns:
                continue
            for lag in lags:
                df_ext[f"{col}_lag{lag}"] = df_ext[col].shift(lag)

    # Rolling Mean: 이동평균
    if roll_features:
        for col, windows in roll_features.items():
            if col not in df_ext.columns:
                continue
            for w in windows:
                df_ext[f"{col}_roll{w}"] = df_ext[col].rolling(w).mean()

    # 결과적으로 생긴 NaN 제거
    df_ext.dropna(inplace=True)
    return df_ext


###################################################
# [2] Walk-Forward - (ARIMA/XGB 등 2D)
###################################################
def walk_forward_validation_arbitrary(
    df: pd.DataFrame,
    train_size: int,
    train_func,
    predict_func,
    x_cols: list,
    y_col: str
) -> pd.DataFrame:
    """
    임의 2D 모델(ARIMA, XGB 등)에 대해 walk-forward로 1스텝씩 예측

    Parameters
    ----------
    df : pd.DataFrame
        전체 데이터
    train_size : int
        초기 훈련 크기
    train_func : function(X_train, y_train) -> model
    predict_func : function(model, X_test) -> float(예측값 1개)
    x_cols : list
        설명변수(피처) 컬럼
    y_col : str
        타깃 컬럼명

    Returns
    -------
    df_result : pd.DataFrame
        실제값(actual)과 예측값(pred)이 기록된 DataFrame
    """
    n = len(df)
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col].values

    X_data = df[x_cols].values
    y_data = df[y_col].values

    for i in range(train_size, n):
        # 0~i-1까지 train
        X_train = X_data[:i]
        y_train = y_data[:i]

        model = train_func(X_train, y_train)

        # i 시점 예측
        X_test = X_data[i:i+1]
        if np.isnan(X_test).any():
            X_test = np.nan_to_num(X_test, nan=0.0)

        yhat = predict_func(model, X_test)
        df_result.iloc[i, df_result.columns.get_loc('pred')] = yhat

    return df_result


###################################################
# [3] 시계열 + LSTM, GRU (연속된 window 데이터 필요)
###################################################
def make_lstm_dataset(X_data, y_data, window_size=4):
    """
    시계열 데이터를 (window_size, feature)로 잘라 시퀀스화
    """
    X_seq, y_seq = [], []
    for i in range(len(X_data) - window_size):
        X_seq.append(X_data[i:i+window_size,:])
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
    LSTM용 walk-forward 1스텝 예측

    Parameters
    ----------
    df : pd.DataFrame
    train_size : int
    x_cols : list
    y_col : str
    train_func : function(X_train, y_train) -> LSTM model
    predict_func : function(model, X_seq) -> float(1개 예측)
    window_size : int

    Returns
    -------
    df_result : pd.DataFrame
        actual, pred
    """
    n = len(df)
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col].values

    X_all = df[x_cols].values
    y_all = df[y_col].values

    for i in range(train_size, n):
        # 훈련 구간: 0~i-1
        X_train = X_all[:i]
        y_train = y_all[:i]
        model = train_func(X_train, y_train)

        # window_size보다 작으면 시퀀스 구성 불가 -> skip
        if i < window_size:
            df_result.iloc[i, df_result.columns.get_loc('pred')] = np.nan
        else:
            # i시점 예측
            X_seq = X_all[i-window_size:i]
            X_seq = X_seq.reshape(1, window_size, X_seq.shape[1])
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
    GRU용 walk-forward 1스텝 예측
    """
    n = len(df)
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col].values

    X_all = df[x_cols].values
    y_all = df[y_col].values

    for i in range(train_size, n):
        X_train = X_all[:i]
        y_train = y_all[:i]
        model = train_func(X_train, y_train)

        if i < window_size:
            df_result.iloc[i, df_result.columns.get_loc('pred')] = np.nan
        else:
            X_seq = X_all[i-window_size:i]
            X_seq = X_seq.reshape(1, window_size, X_seq.shape[1])
            yhat = predict_func(model, X_seq)
            df_result.iloc[i, df_result.columns.get_loc('pred')] = yhat

    return df_result


###################################################
# [4] ARIMA/XGB (Optuna)
###################################################
def objective_arima(trial, X_train, y_train):
    p = trial.suggest_int('p', 0, 3)
    d = trial.suggest_int('d', 0, 2)
    q = trial.suggest_int('q', 0, 3)
    exog = X_train if X_train.shape[1] > 0 else None

    try:
        model_ = pm.ARIMA(order=(p, d, q))
        fit_ = model_.fit(y_train, exogenous=exog)
        return fit_.aic()
    except:
        return 1e9


def train_arima_optuna(X_train, y_train, n_trials=3):
    exog = X_train if X_train.shape[1] > 0 else None

    def _obj(trial):
        return objective_arima(trial, X_train, y_train)

    study = optuna.create_study(direction='minimize')
    study.optimize(_obj, n_trials=n_trials)
    best_ = study.best_trial

    p_ = best_.params['p']
    d_ = best_.params['d']
    q_ = best_.params['q']
    print(f"[ARIMA Optuna] best order=({p_},{d_},{q_}), AIC={best_.value}")

    model_ = pm.ARIMA(order=(p_, d_, q_))
    fit_ = model_.fit(y_train, exogenous=exog)
    return fit_


def predict_arima_optuna(model, X_test):
    exog_test = X_test if X_test.shape[1] > 0 else None
    if exog_test is not None and np.isnan(exog_test).any():
        exog_test = np.nan_to_num(exog_test, nan=0.0)

    fcst = model.predict(n_periods=1, exogenous=exog_test, return_conf_int=False)
    return fcst[0]


def objective_xgb(trial, X_train, y_train):
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
    return model.predict(X_test)[0]


###################################################
# [5] Prophet (Optuna) + Walk-Forward
###################################################
def train_prophet_optuna(
    df: pd.DataFrame,
    exogenous_cols: list = None,
    param_space: dict = None,
    n_trials: int = 3
):
    """
    Prophet용 Optuna 튜너 함수를 반환한다.
    Walk-forward 시점마다 tune_func(date_train, y_train, exog_train) 형태로 호출
    """

    from sklearn.metrics import mean_squared_error

    if param_space is None:
        param_space = {
            'seasonality_mode': ['additive', 'multiplicative'],
            'yearly_seasonality': [True, False],
            'changepoint_prior_scale': [0.01, 0.1, 0.5, 1.0]
        }

    def tune_func(dates_train, y_train, exog_train=None):
        def local_objective(trial):
            seasonality_mode_ = trial.suggest_categorical(
                'seasonality_mode',
                param_space['seasonality_mode']
            )
            yearly_seasonality_ = trial.suggest_categorical(
                'yearly_seasonality',
                param_space['yearly_seasonality']
            )
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
            df_p = pd.DataFrame({'ds': dates_train, 'y': y_train})
            if exogenous_cols and exog_train is not None:
                for idx, c in enumerate(exogenous_cols):
                    df_p[c] = exog_train[:, idx]
                    model_.add_regressor(c)

            model_.fit(df_p)
            forecast_ = model_.predict(df_p[['ds'] + (exogenous_cols if exogenous_cols else [])])
            rmse_ = sqrt(mean_squared_error(df_p['y'], forecast_['yhat']))
            return rmse_

        study = optuna.create_study(direction='minimize')
        study.optimize(local_objective, n_trials=n_trials)
        best_ = study.best_trial
        print(f"[Prophet Optuna] best params={best_.params}, best RMSE={best_.value}")

        # 최적 모델 구축
        best_model = Prophet(
            seasonality_mode=best_.params['seasonality_mode'],
            yearly_seasonality=best_.params['yearly_seasonality'],
            changepoint_prior_scale=best_.params['changepoint_prior_scale']
        )
        df_p = pd.DataFrame({'ds': dates_train, 'y': y_train})
        if exogenous_cols and exog_train is not None:
            for idx, c in enumerate(exogenous_cols):
                df_p[c] = exog_train[:, idx]
                best_model.add_regressor(c)

        best_model.fit(df_p)
        return best_model

    return tune_func


def predict_prophet_optuna(model, next_date, exogenous=None):
    future_df = pd.DataFrame({'ds': [next_date]})
    if exogenous is not None:
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
    Prophet(Optuna)로 walk-forward 1스텝 예측
    """
    n = len(df)
    df_result = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_result['actual'] = df[y_col].values

    date_idx = df.index
    y_all    = df[y_col].values
    exog_all = df[exog_cols].values if exog_cols else None

    for i in range(train_size, n):
        date_train = date_idx[:i]
        y_train    = y_all[:i]
        exog_train = exog_all[:i] if exog_all is not None else None

        model = prophet_tune_func(date_train, y_train, exog_train)

        next_d = date_idx[i]
        exog_i = exog_all[i:i+1] if exog_all is not None else None
        yhat = predict_prophet_optuna(model, next_d, exog_i)
        df_result.iloc[i, df_result.columns.get_loc('pred')] = yhat

    return df_result


###################################################
# [6] LSTM/GRU (KerasTuner) 개선 (유연화)
###################################################
def build_lstm_model(num_features, window_size=4, lstm_units=32):
    """
    간단한 LSTM 기본 구조
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(window_size, num_features)))
    model.add(layers.LSTM(lstm_units, activation='tanh'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def build_gru_model(num_features, window_size=4, gru_units=32):
    """
    간단한 GRU 기본 구조
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(window_size, num_features)))
    model.add(layers.GRU(gru_units, activation='tanh'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_rnn_ktuner(
    X_train: np.ndarray,
    y_train: np.ndarray,
    window_size: int = 4,
    max_trials: int = 5,
    model_type: str = "LSTM",
    batch_size_candidates: list = [4, 8, 16],
    epochs_candidates: list = [5, 10, 20],
    patience: int = 2
) -> keras.Model:
    """
    LSTM 혹은 GRU를 선택하여 KerasTuner로 하이퍼파라미터 튜닝 후 모델을 반환.
    model_type이 "LSTM"이면 LSTM, "GRU"이면 GRU를 빌드.

    파라미터:
    - window_size: 슬라이딩 윈도우 크기
    - model_type: "LSTM" 또는 "GRU"
    - patience: EarlyStopping patience

    Returns
    -------
    final_model : keras.Model
        최적 하이퍼파라미터로 학습된 모델
    """

    # (1) window_size보다 작으면 fallback
    if len(X_train) <= window_size:
        if model_type.upper() == "GRU":
            fallback_model = build_gru_model(
                num_features=X_train.shape[1],
                window_size=window_size,
                gru_units=32
            )
        else:
            fallback_model = build_lstm_model(
                num_features=X_train.shape[1],
                window_size=window_size,
                lstm_units=32
            )
        fallback_model.fit(X_train[:1], y_train[:1], epochs=1, verbose=0)
        return fallback_model

    # (2) 시퀀스화
    X_seq, y_seq = make_lstm_dataset(X_train, y_train, window_size)

    # (3) 하이퍼파라미터 탐색 함수
    def build_rnn_with_hp(hp):
        units = hp.Int('units', min_value=16, max_value=128, step=16)
        drop_ = hp.Float('dropout', 0.0, 0.5, step=0.1)
        lr_   = hp.Choice('learning_rate', [1e-3, 1e-4, 5e-4, 5e-3])

        model = keras.Sequential()
        model.add(layers.Input(shape=(window_size, X_train.shape[1])))

        if model_type.upper() == "GRU":
            model.add(layers.GRU(units, activation='tanh'))
        else:
            model.add(layers.LSTM(units, activation='tanh'))

        if drop_ > 0:
            model.add(layers.Dropout(drop_))
        model.add(layers.Dense(1))

        model.compile(optimizer=keras.optimizers.Adam(lr_), loss='mse')
        return model

    # (4) Tuner
    tuner = kt.RandomSearch(
        hypermodel=build_rnn_with_hp,
        objective='val_loss',
        max_trials=max_trials,
        overwrite=True,
        directory=f'C:/temp/ktuner_{model_type.lower()}_output',
        project_name=f'{model_type.lower()}_tune_wf'
    )

    es = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

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

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"[{model_type} Tuning] best -> units={best_hp.get('units')}, "
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


def predict_rnn_ktuner(model, X_seq):
    """
    RNN(KerasTuner로 학습된 LSTM/GRU) 예측값 반환
    """
    yhat = model.predict(X_seq)
    return yhat[0, 0]


###################################################
# [7] 결과 시각화
###################################################
def plot_pred_vs_actual(df_result: pd.DataFrame, model_name: str, y_col: str):
    """
    실제값 vs 예측값 시각화
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


###################################################
# [8] 메인 실행: 개선된 파이프라인
###################################################
def run_all_models(
    filepath: str,
    target_list: list,
    use_feature_engineering: bool = True,
    lag_features: dict = None,
    roll_features: dict = None,
    window_size: int = 4
):
    """
    CSV 파일 경로, 예측 대상 컬럼을 받아서
    1) 데이터 전처리
    2) (선택) 피처 엔지니어링(lag, roll)
    3) Walk-forward - ARIMA, XGB, Prophet, LSTM, GRU 모델 순차 적용
    4) RMSE, MAE 비교 및 결과 시각화

    모델 내부 하이퍼파라미터:
    - ARIMA, XGB, Prophet: Optuna 사용 (n_trials=3)
    - LSTM, GRU: KerasTuner 사용 (max_trials=2, epochs_candidates=[5,10] 등 예시)
    """
    # 1) 데이터 로드
    df = load_and_preprocess_data(filepath)

    # 2) 피처 엔지니어링 (선택)
    if use_feature_engineering:
        df = create_feature_engineering(df, lag_features, roll_features)
        print("\n[Check NaN after Feature Engineering]\n", df.isnull().sum())
    df.dropna(inplace=True)
    print("[Final Check] 전체 NaN 개수:\n", df.isnull().sum())
    print("최종 데이터 shape =", df.shape)

    # 입력/타겟 구분
    x_cols = [c for c in df.columns if c not in target_list]

    # Prophet용 튜너 함수 (Exogenous 없이 예시)
    prophet_tune_func = train_prophet_optuna(
        df, exogenous_cols=None, param_space=None, n_trials=3
    )

    # 결과 저장용
    results = {}

    # 3) 타겟별 모델 적용
    for tgt in target_list:
        if tgt not in df.columns:
            print(f"[WARNING] '{tgt}' not in columns -> skip")
            continue
        print(f"\n======= Target: {tgt} =======")

        sub_df = df[x_cols + [tgt]].copy()

        # (1) ARIMA(Optuna)
        df_arima = walk_forward_validation_arbitrary(
            sub_df,
            train_size=10,  # 예시
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

        # (2) XGB(Optuna)
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

        # (3) Prophet(Optuna)
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

        # (4) LSTM(KerasTuner)
        df_lstm = walk_forward_validation_lstm(
            df=sub_df,
            train_size=10,
            x_cols=x_cols,
            y_col=tgt,
            window_size=window_size,
            train_func=lambda Xtr, ytr: train_rnn_ktuner(
                Xtr, ytr,
                window_size=window_size,
                max_trials=2,          # 하이퍼 파라미터 탐색 크기
                model_type="LSTM",     # LSTM
                batch_size_candidates=[4,8,16],
                epochs_candidates=[5,10],
                patience=2
            ),
            predict_func=predict_rnn_ktuner
        )
        mask_lstm = df_lstm['pred'].notnull()
        rmse_lstm = sqrt(mean_squared_error(df_lstm.loc[mask_lstm, 'actual'],
                                            df_lstm.loc[mask_lstm, 'pred']))
        mae_lstm  = mean_absolute_error(df_lstm.loc[mask_lstm, 'actual'],
                                        df_lstm.loc[mask_lstm, 'pred'])
        plot_pred_vs_actual(df_lstm, "LSTM(KerasTuner)", tgt)

        # (5) GRU(KerasTuner)
        df_gru = walk_forward_validation_gru(
            df=sub_df,
            train_size=10,
            x_cols=x_cols,
            y_col=tgt,
            window_size=window_size,
            train_func=lambda Xtr, ytr: train_rnn_ktuner(
                Xtr, ytr,
                window_size=window_size,
                max_trials=2,         # 하이퍼 파라미터 탐색 크기
                model_type="GRU",     # GRU
                batch_size_candidates=[4,8,16],
                epochs_candidates=[5,10],
                patience=2
            ),
            predict_func=predict_rnn_ktuner
        )
        mask_gru = df_gru['pred'].notnull()
        rmse_gru = sqrt(mean_squared_error(df_gru.loc[mask_gru, 'actual'],
                                           df_gru.loc[mask_gru, 'pred']))
        mae_gru  = mean_absolute_error(df_gru.loc[mask_gru, 'actual'],
                                       df_gru.loc[mask_gru, 'pred'])
        plot_pred_vs_actual(df_gru, "GRU(KerasTuner)", tgt)

        # 결과 요약 저장
        results[tgt] = {
            'ARIMA(Optuna)': (rmse_arima, mae_arima),
            'XGB(Optuna)'  : (rmse_xgb, mae_xgb),
            'Prophet(Optuna)': (rmse_ppt, mae_ppt),
            'LSTM(KerasTuner)': (rmse_lstm, mae_lstm),
            'GRU(KerasTuner)':  (rmse_gru, mae_gru)
        }

    # (6) 전체 결과 출력
    print("\n======= [Summary: RMSE / MAE] =======")
    for tgt, model_res in results.items():
        print(f"\n[Target: {tgt}]")
        for mname, (r, m) in model_res.items():
            print(f"{mname:20s} | RMSE={r:.3f}, MAE={m:.3f}")


###################################################
# [9] 실행 예시 (if __name__=="__main__":)
###################################################
if __name__=="__main__":
    """
    아래 main 구문은 실제 실행 예시입니다.
    - 파일 경로(file_path)와 예측 대상 컬럼(targets)을 지정 후,
    - lag_features, roll_features를 원하는 대로 구성할 수 있습니다.
    - window_size = 4 (시계열 RNN 모델용)
    """

    file_path = "modified_infectious_disease_data_copy.csv"
    targets   = ["호흡기_new", "매개성"]

    # lag, roll 피처 설정 예시
    lag_feats = {
        '호흡기_new': [1],
        '매개성': [1]
    }
    roll_feats = {
        '호흡기_new': [2]
    }

    run_all_models(
        filepath=file_path,
        target_list=targets,
        use_feature_engineering=True,
        lag_features=lag_feats,
        roll_features=roll_feats,
        window_size=4
    )
