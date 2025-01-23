###################################################
# 0. 라이브러리 Import (필요한 도구 불러오기)
###################################################
import numpy as np                      # 숫자, 배열 계산용
import pandas as pd                     # 표(데이터프레임) 처리용
import matplotlib.pyplot as plt         # 그래프 시각화
import seaborn as sns                   # 그래프 꾸미기
from math import sqrt                   # 제곱근 (RMSE 계산할 때 사용)
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 시계열 + AutoML 관련 라이브러리
import optuna                           # AutoML (하이퍼파라미터 자동 탐색)
import keras_tuner as kt               # 딥러닝 하이퍼파라미터 탐색
import pmdarima as pm                   # ARIMA 모델 라이브러리

# 텐서플로(딥러닝)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 시계열 예측용 라이브러리(Prophet)
from prophet import Prophet

###################################################
# 1. 데이터 로드 및 전처리
###################################################
def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    [역할]
      - CSV 파일에서 데이터를 불러온 뒤,
      - 'Date'라는 날짜 열을 'datetime' 형식으로 변환,
      - 결측치(빈 칸)은 '중간값'으로 간단 대체,
      - 그리고 'Date' 열을 '인덱스'로 설정합니다.

    [입력]
      - filepath: 읽을 CSV 파일 경로(문자열)

    [출력]
      - 전처리 완료된 pandas DataFrame
    """
    # (1) CSV 읽기 (인코딩이 'ANSI'인 파일을 가정)
    df = pd.read_csv(filepath, encoding='ANSI')

    # (2) 'Date' 열을 실제 날짜(datetime)로 변환
    df['Date'] = pd.to_datetime(df['Date'])

    # (3) 날짜 기준으로 정렬
    df.sort_values('Date', inplace=True)

    # (4) 인덱스를 날짜로 설정
    df.set_index('Date', inplace=True)

    # (5) 결측치(빈 칸)가 있을 경우, 중간값(median)으로 채움
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df


def create_feature_engineering(
    df: pd.DataFrame,
    lag_features=None,
    roll_features=None,
    dropna=True
) -> pd.DataFrame:
    """
    [역할]
      - lag(시차) 변수, rolling(이동평균/이동표준편차) 변수를 생성해줌.
      - 예: lag_features={'Cases':[1,2]} 이면, 'Cases_lag1', 'Cases_lag2' 열을 추가

    [입력]
      - df: 원본 데이터
      - lag_features: { '컬럼명': [1,2,3], ... }
      - roll_features: { '컬럼명': [3,5], ... }
      - dropna: 만들어진 lag, rolling 변수가 NaN이면 행삭제할지 여부

    [출력]
      - 새로운 열들이 추가된 데이터프레임
    """
    # 원본 df를 복사 (원본 훼손 방지)
    df_ = df.copy()

    # (1) Lag(시차) 변수
    if lag_features:
        for col, lags in lag_features.items():
            if col not in df_.columns:
                # 만약 df에 col이 실제 없다면 skip
                continue
            for lag in lags:
                # 예: lag=1 → 과거 1일치
                df_[f"{col}_lag{lag}"] = df_[col].shift(lag)

    # (2) Rolling(이동평균, 이동표준편차)
    if roll_features:
        for col, wins in roll_features.items():
            if col not in df_.columns:
                continue
            for w in wins:
                # 이동평균
                df_[f"{col}_rmean{w}"] = df_[col].rolling(w).mean()
                # 이동표준편차
                df_[f"{col}_rstd{w}"]  = df_[col].rolling(w).std()

    # (3) dropna=True라면, NaN(결측행)을 통째로 버림
    if dropna:
        df_.dropna(inplace=True)

    return df_


###################################################
# 2. Walk-forward - ARIMA / XGB / Prophet / LSTM / GRU
###################################################
def walk_forward_validation_arbitrary(
    df: pd.DataFrame,
    train_size: int,
    train_func,
    predict_func,
    x_cols: list,
    y_col: str,
    min_required: int = 10
) -> pd.DataFrame:
    """
    [역할]
      - ARIMA, XGB 등 2D(2차원) 입력을 다루는 모델에서
        'Walk-forward' 검증을 수행하는 함수.
      - i 시점(행)을 예측할 때, 0~(i-1)행까지 학습, i행 예측

    [파라미터]
      - train_size: 초반 몇 행을 '학습 데이터' 최소 확보로 사용할지
      - train_func(X_train, y_train) → 모델 반환
      - predict_func(모델, X_test)   → 예측값 1개 반환
      - x_cols: 입력(독립변수) 열 이름 리스트
      - y_col: 타겟(종속변수) 열 이름
      - min_required: ARIMA 등이 학습하려면 최소 샘플 수 필요
                     (이보다 적으면 오류나므로 fallback)

    [출력]
      - df_res( actual, pred ) : 실제값과 예측값을 담은 DataFrame
    """
    n = len(df)
    # 결과 df_res: 인덱스 그대로, 'actual','pred' 열 만들어 놓음
    df_res = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_res['actual'] = df[y_col].values

    # X_data, y_data (행렬 형태)
    X_data = df[x_cols].values
    y_data = df[y_col].values

    # i를 train_size ~ n-1까지 순회
    for i in range(train_size, n):
        # (1) 학습 구간
        X_train = X_data[:i]
        y_train = y_data[:i]

        # (2) 만약 데이터가 너무 작으면 -> fallback(평균 예측)
        if len(y_train) < min_required:
            df_res.iloc[i, df_res.columns.get_loc('pred')] = np.mean(y_train)
            continue

        # (3) 학습
        try:
            model = train_func(X_train, y_train)
        except:
            # 혹시 train_func 내부에서 오류가 뜨면 fallback
            df_res.iloc[i, df_res.columns.get_loc('pred')] = np.mean(y_train)
            continue

        # (4) 예측
        X_test = X_data[i:i+1]
        try:
            yhat = predict_func(model, X_test)
        except:
            # 예측 오류 발생 시 fallback
            yhat = np.mean(y_train)

        # (5) 결과 기록
        df_res.iloc[i, df_res.columns.get_loc('pred')] = yhat

    return df_res


def walk_forward_validation_prophet(
    df: pd.DataFrame,
    train_size: int,
    y_col: str
) -> pd.DataFrame:
    """
    [역할]
      - Prophet 전용 Walk-forward 검증
      - Prophet은 (ds, y) 형태 입력이 필요하므로,
        인덱스(date), y를 df_p로 만들어 모델 훈련 후
        i 시점 예측

    [주의]
      - Prophet은 행이 매우 적으면 경고가 뜨거나 성능이 떨어질 수 있음
    """
    n = len(df)
    df_res = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_res['actual'] = df[y_col].values

    idx_ = df.index        # 날짜 인덱스
    y_   = df[y_col].values

    # i: train_size부터 n-1까지 반복
    for i in range(train_size, n):
        date_train = idx_[:i]   # 0 ~ i-1
        y_train    = y_[:i]

        try:
            # (1) Prophet 모델 생성, fit
            model_ = Prophet()
            df_p = pd.DataFrame({'ds': date_train, 'y': y_train})
            model_.fit(df_p)

            # (2) i 시점 예측
            next_d = idx_[i]   # 예측하려는 날짜(단일)
            df_ = pd.DataFrame({'ds': [next_d]})
            fcst = model_.predict(df_)
            yhat = fcst['yhat'].values[0]
        except:
            # 오류 시 fallback
            yhat = np.mean(y_train)

        # 결과 저장
        df_res.iloc[i, df_res.columns.get_loc('pred')] = yhat

    return df_res


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
    [역할]
      - LSTM 모델의 Walk-forward 검증
      - LSTM은 시계열 윈도우(window_size) 만큼 쌓아둔 3D 입력이 필요
      - 이 함수는 매 시점 i 마다 0~(i-1)행까지로 LSTM을 학습,
        i 행을 1-step 예측
    """
    n = len(df)
    df_res = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_res['actual'] = df[y_col].values

    # 전체 x, y array
    X_all = df[x_cols].values
    y_all = df[y_col].values

    for i in range(train_size, n):
        # (1) 학습데이터: 0 ~ i-1
        X_train = X_all[:i]
        y_train = y_all[:i]

        # (2) 모델 학습
        try:
            model = train_func(X_train, y_train)
        except:
            # LSTM 학습 오류 -> fallback
            df_res.iloc[i, df_res.columns.get_loc('pred')] = np.mean(y_train)
            continue

        # (3) 예측
        if i < window_size:
            # 윈도우 사이즈보다 작으면 예측 불가 -> NaN
            df_res.iloc[i, df_res.columns.get_loc('pred')] = np.nan
        else:
            x_seq = X_all[i-window_size:i]
            x_seq = x_seq.reshape(1, window_size, x_seq.shape[1])
            try:
                yhat  = predict_func(model, x_seq)
            except:
                yhat = np.mean(y_train)
            df_res.iloc[i, df_res.columns.get_loc('pred')] = yhat

    return df_res


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
    [역할]
      - GRU 모델의 Walk-forward 검증 (LSTM과 유사)
    """
    n = len(df)
    df_res = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_res['actual'] = df[y_col].values

    X_all = df[x_cols].values
    y_all = df[y_col].values

    for i in range(train_size, n):
        X_train = X_all[:i]
        y_train = y_all[:i]
        try:
            model = train_func(X_train, y_train)
        except:
            df_res.iloc[i, df_res.columns.get_loc('pred')] = np.mean(y_train)
            continue

        if i < window_size:
            df_res.iloc[i, df_res.columns.get_loc('pred')] = np.nan
        else:
            x_seq = X_all[i-window_size:i]
            x_seq = x_seq.reshape(1, window_size, x_seq.shape[1])
            try:
                yhat  = predict_func(model, x_seq)
            except:
                yhat = np.mean(y_train)
            df_res.iloc[i, df_res.columns.get_loc('pred')] = yhat

    return df_res


###################################################
# 3. ARIMA(Optuna) / XGB(Optuna)
###################################################
def objective_arima(trial, X_train, y_train):
    """
    [역할]
      - ARIMA의 (p, d, q) 하이퍼파라미터를 Optuna로 탐색
      - 에러 가능성 낮추기 위해 범위를 (0~2) 정도로 제한
      - 모델 성능 지표로 AIC를 사용
    """
    p = trial.suggest_int('p', 0, 2)
    d = trial.suggest_int('d', 0, 1)
    q = trial.suggest_int('q', 0, 2)
    try:
        exog = X_train if X_train.shape[1] > 0 else None
        model_ = pm.ARIMA(order=(p,d,q))
        fit_   = model_.fit(y_train, exogenous=exog)
        return fit_.aic()  # AIC 값이 낮을수록 좋음
    except:
        return 1e9

def train_arima_optuna(X_train, y_train, n_trials=3):
    """
    [역할]
      - objective_arima 함수를 통해
        ARIMA 최적 (p,d,q) 찾고, 최종 모델 훈련
    """
    def _obj(trial):
        return objective_arima(trial, X_train, y_train)

    # Optuna 스터디 (목표: AIC 최소화)
    study = optuna.create_study(direction='minimize')
    study.optimize(_obj, n_trials=n_trials)
    best_ = study.best_trial

    # 최적 하이퍼파라미터
    p_ = best_.params['p']
    d_ = best_.params['d']
    q_ = best_.params['q']

    exog = X_train if X_train.shape[1] > 0 else None
    model_ = pm.ARIMA(order=(p_, d_, q_))
    fit_   = model_.fit(y_train, exogenous=exog)
    return fit_

def predict_arima_optuna(model, X_test):
    """
    [역할]
      - ARIMA 모델로 1스텝 예측
      - exogenous(X_test)가 있다면 넣어주고,
        혹시 NaN 있으면 0으로 치환
    """
    exog = X_test if X_test.shape[1]>0 else None
    if exog is not None and np.isnan(exog).any():
        exog = np.nan_to_num(exog, nan=0.0)
    fcst = model.predict(n_periods=1, exogenous=exog, return_conf_int=False)
    return fcst[0]


### XGB(Optuna)
from xgboost import XGBRegressor

def objective_xgb(trial, X_train, y_train):
    """
    [역할]
      - XGB 하이퍼파라미터(학습률, 트리개수 등) 자동 탐색
      - 성능 척도로 RMSE
    """
    n_  = trial.suggest_int('n_est', 50, 200, step=50)     # 트리 개수
    lr_ = trial.suggest_float('lr', 1e-3, 1e-1, log=True)  # 학습률 (로그 스케일)
    md_ = trial.suggest_int('md', 2, 6)                    # 트리 깊이

    xgb_ = XGBRegressor(n_estimators=n_, learning_rate=lr_, max_depth=md_, random_state=42)
    xgb_.fit(X_train, y_train)
    yhat_ = xgb_.predict(X_train)
    # 실제 y와 비교해 RMSE 계산
    return sqrt(mean_squared_error(y_train, yhat_))

def train_xgb_optuna(X_train, y_train, n_trials=3):
    """
    [역할]
      - objective_xgb로 n_trials번 탐색 → 최적 hyperparam으로 최종 모델 훈련
    """
    def _obj(trial):
        return objective_xgb(trial, X_train, y_train)
    study = optuna.create_study(direction='minimize')
    study.optimize(_obj, n_trials=n_trials)
    best_ = study.best_trial

    n_  = best_.params['n_est']
    lr_ = best_.params['lr']
    md_ = best_.params['md']
    xgb_final = XGBRegressor(n_estimators=n_, learning_rate=lr_, max_depth=md_, random_state=42)
    xgb_final.fit(X_train, y_train)
    return xgb_final

def predict_xgb_optuna(model, X_test):
    """
    [역할]
      - XGB 예측 함수 (1행 예측)
    """
    if np.isnan(X_test).any():
        X_test = np.nan_to_num(X_test, nan=0.0)
    return model.predict(X_test)[0]

###################################################
# 4. LSTM / GRU KerasTuner
###################################################
def make_lstm_dataset(X, y, window_size):
    """
    [역할]
      - (window_size) 길이만큼 시계열 슬라이딩
      - 예: window_size=4면, (X0..X3 -> y4), (X1..X4 -> y5) 등
    """
    X_seq, y_seq = [], []
    for i in range(len(X)-window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

import keras_tuner as kt

def train_lstm_ktuner(X_train, y_train,
                      window_size=4,
                      max_trials=2,
                      epochs=10,
                      batch_size=8):
    """
    [역할]
      - KerasTuner(RanodmSearch)로
        LSTM의 유닛 수, dropout, lr 등을 탐색
    """
    # (1) 혹시 trainset이 window보다 작으면 fallback
    if len(X_train) <= window_size:
        dummy = keras.Sequential([
            layers.Input(shape=(window_size,X_train.shape[1])),
            layers.LSTM(16),
            layers.Dense(1)
        ])
        dummy.compile(optimizer='adam', loss='mse')
        if len(X_train) > 0:
            dummy.fit(
                X_train[:1].reshape(1,window_size,-1),
                np.array([y_train[0]]),
                epochs=1, verbose=0
            )
        return dummy

    # (2) LSTM 시퀀스화
    X_seq, y_seq = make_lstm_dataset(X_train, y_train, window_size)

    # (3) 빌드 함수
    def build_lstm_with_hp(hp):
        units_ = hp.Int('units', 16, 64, step=16)
        drop_  = hp.Float('dropout', 0.0, 0.5, step=0.1)
        lr_    = hp.Choice('lr', [1e-3, 1e-4, 5e-4])

        model  = keras.Sequential()
        model.add(layers.Input(shape=(window_size,X_train.shape[1])))
        model.add(layers.LSTM(units_))
        if drop_>0:
            model.add(layers.Dropout(drop_))
        model.add(layers.Dense(1))
        model.compile(optimizer=keras.optimizers.Adam(lr_), loss='mse')
        return model

    # (4) RandomSearch Tuner
    tuner = kt.RandomSearch(
        build_lstm_with_hp,
        objective='val_loss',
        max_trials=max_trials,
        overwrite=True,
        directory=r"C:\temp\ktuner_lstm",
        project_name='lstm_wf'
    )

    # (5) EarlyStopping
    es = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

    # (6) tuner.search()
    tuner.search(X_seq, y_seq,
                 validation_split=0.2,
                 epochs=epochs,
                 batch_size=batch_size,
                 callbacks=[es],
                 verbose=0)

    best_hp = tuner.get_best_hyperparameters(1)[0]
    final_model = tuner.hypermodel.build(best_hp)
    final_model.fit(X_seq, y_seq,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[es],
                    verbose=0)
    return final_model

def predict_lstm_ktuner(model, X_seq):
    """
    [역할]
      - LSTM 모델 예측
      - X_seq shape = (1, window_size, #features)
    """
    if np.isnan(X_seq).any():
        X_seq = np.nan_to_num(X_seq, nan=0.0)
    p = model.predict(X_seq)
    return p[0,0]


def train_gru_ktuner(X_train, y_train,
                     window_size=4,
                     max_trials=2,
                     epochs=10,
                     batch_size=8):
    """
    [역할]
      - GRU 모델도 KerasTuner로 유닛, dropout, lr 탐색
      - LSTM과 구조 비슷, 레이어만 GRU
    """
    if len(X_train) <= window_size:
        dummy = keras.Sequential([
            layers.Input(shape=(window_size,X_train.shape[1])),
            layers.GRU(16),
            layers.Dense(1)
        ])
        dummy.compile(optimizer='adam', loss='mse')
        if len(X_train) > 0:
            dummy.fit(X_train[:1].reshape(1,window_size,-1),
                      np.array([y_train[0]]),
                      epochs=1, verbose=0)
        return dummy

    X_seq, y_seq = make_lstm_dataset(X_train, y_train, window_size)

    def build_gru_with_hp(hp):
        units_ = hp.Int('units', 16, 64, step=16)
        drop_  = hp.Float('dropout', 0.0, 0.5, step=0.1)
        lr_    = hp.Choice('lr', [1e-3, 1e-4, 5e-4])

        model  = keras.Sequential()
        model.add(layers.Input(shape=(window_size,X_train.shape[1])))
        model.add(layers.GRU(units_))
        if drop_>0:
            model.add(layers.Dropout(drop_))
        model.add(layers.Dense(1))
        model.compile(optimizer=keras.optimizers.Adam(lr_), loss='mse')
        return model

    tuner = kt.RandomSearch(
        build_gru_with_hp,
        objective='val_loss',
        max_trials=max_trials,
        overwrite=True,
        directory=r"C:\temp\ktuner_gru",
        project_name='gru_wf'
    )

    es = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

    tuner.search(X_seq, y_seq,
                 validation_split=0.2,
                 epochs=epochs,
                 batch_size=batch_size,
                 callbacks=[es],
                 verbose=0)
    best_hp = tuner.get_best_hyperparameters(1)[0]
    final_model = tuner.hypermodel.build(best_hp)
    final_model.fit(X_seq, y_seq,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[es],
                    verbose=0)
    return final_model

def predict_gru_ktuner(model, X_seq):
    """
    [역할]
      - GRU 모델 예측
    """
    if np.isnan(X_seq).any():
        X_seq = np.nan_to_num(X_seq, nan=0.0)
    p = model.predict(X_seq)
    return p[0,0]


###################################################
# 5. 결과 시각화
###################################################
def plot_pred_vs_actual(df_res, model_name, y_col):
    """
    [역할]
      - 실제값 vs 예측값 그래프
    """
    plt.figure(figsize=(8,4))
    plt.plot(df_res.index, df_res['actual'], marker='o', label='Actual')
    plt.plot(df_res.index, df_res['pred'],   marker='x', label=f'{model_name} Pred')
    plt.title(f"{model_name} - {y_col}")
    plt.xlabel("Date")
    plt.ylabel(y_col)
    plt.legend()
    plt.grid()
    plt.show()


###################################################
# 6. 종합 실행
###################################################
def run_all_models_infectious(
    file_path="modified_infectious_disease_data_copy.csv",
    target_list=["호흡기_new","매개성"],
    train_size=3,
    window_size=2,
    n_trials=2
):
    """
    [역할]
      - 위 모든 단계를 한번에 수행.
      - 1) CSV를 불러오고 전처리
      - 2) 피처엔지니어링(lag, roll)
      - 3) ARIMA / XGB / Prophet / LSTM / GRU 모델로
         Walk-forward 검증
      - 4) 예측 그래프 및 RMSE/MAE 출력

    [핵심 파라미터]
      - file_path: CSV 경로
      - target_list: 예측할 타겟 열 이름 리스트
      - train_size: walk-forward 최소 학습용 크기
      - window_size: LSTM, GRU 윈도우 크기
      - n_trials: Optuna / KerasTuner 탐색 횟수
    """
    # (1) 데이터 로드
    df = load_and_preprocess_data(file_path)
    print("[INFO] Data shape:", df.shape)

    # (2) 피처 엔지니어링 설정
    exclude_cols = list(set(target_list))  # 타겟 열 제외
    x_cols = [c for c in df.columns if c not in exclude_cols]

    # 예시: 모든 독립변수에 대해 lag=1, rolling=2 설정
    lagF = {}
    rollF= {}
    for c in x_cols:
        lagF[c] = [1]
        rollF[c]= [2]

    # 피처 엔지니어링 적용
    df_fe = create_feature_engineering(df, lagF, rollF, dropna=True)
    print("[INFO] After FE shape:", df_fe.shape)

    results = {}

    # (3) 각 타겟에 대해 반복
    for tgt in target_list:
        if tgt not in df_fe.columns:
            print(f"Skip {tgt}")
            continue

        print(f"\n===== Target={tgt} =====")
        sub_cols = [c for c in df_fe.columns if c != tgt]
        sub_df   = df_fe[sub_cols + [tgt]].copy()

        #-------- ARIMA --------
        df_arima = walk_forward_validation_arbitrary(
            sub_df,
            train_size=train_size,
            train_func=lambda Xtr,ytr: train_arima_optuna(Xtr,ytr,n_trials),
            predict_func=predict_arima_optuna,
            x_cols=sub_cols,
            y_col=tgt,
            min_required=10
        )
        # 성능평가
        mA = df_arima['pred'].notnull()
        if mA.sum() > 0:
            rmseA = sqrt(mean_squared_error(df_arima.loc[mA,'actual'], df_arima.loc[mA,'pred']))
            maeA  = mean_absolute_error(df_arima.loc[mA,'actual'], df_arima.loc[mA,'pred'])
        else:
            rmseA, maeA = np.nan, np.nan
        plot_pred_vs_actual(df_arima, "ARIMA", tgt)

        #-------- XGB --------
        df_xgb = walk_forward_validation_arbitrary(
            sub_df,
            train_size=train_size,
            train_func=lambda Xtr,ytr: train_xgb_optuna(Xtr,ytr,n_trials),
            predict_func=predict_xgb_optuna,
            x_cols=sub_cols,
            y_col=tgt,
            min_required=1
        )
        mX = df_xgb['pred'].notnull()
        if mX.sum() > 0:
            rmseX = sqrt(mean_squared_error(df_xgb.loc[mX,'actual'], df_xgb.loc[mX,'pred']))
            maeX  = mean_absolute_error(df_xgb.loc[mX,'actual'], df_xgb.loc[mX,'pred'])
        else:
            rmseX, maeX = np.nan, np.nan
        plot_pred_vs_actual(df_xgb, "XGB", tgt)

        #-------- Prophet --------
        df_ppt = walk_forward_validation_prophet(
            sub_df, train_size=train_size, y_col=tgt
        )
        mP= df_ppt['pred'].notnull()
        if mP.sum() > 0:
            rmseP= sqrt(mean_squared_error(df_ppt.loc[mP,'actual'], df_ppt.loc[mP,'pred']))
            maeP=  mean_absolute_error(df_ppt.loc[mP,'actual'], df_ppt.loc[mP,'pred'])
        else:
            rmseP, maeP = np.nan, np.nan
        plot_pred_vs_actual(df_ppt, "Prophet", tgt)

        #-------- LSTM --------
        df_lstm = walk_forward_validation_lstm(
            df=sub_df,
            train_size=train_size,
            x_cols=sub_cols,
            y_col=tgt,
            train_func=lambda Xtr,ytr: train_lstm_ktuner(
                Xtr, ytr,
                window_size=window_size,
                max_trials=n_trials,
                epochs=10,
                batch_size=4
            ),
            predict_func=predict_lstm_ktuner,
            window_size=window_size
        )
        ml = df_lstm['pred'].notnull()
        if ml.sum() > 0:
            rmseL= sqrt(mean_squared_error(df_lstm.loc[ml,'actual'], df_lstm.loc[ml,'pred']))
            maeL=  mean_absolute_error(df_lstm.loc[ml,'actual'], df_lstm.loc[ml,'pred'])
        else:
            rmseL, maeL = np.nan, np.nan
        plot_pred_vs_actual(df_lstm, "LSTM", tgt)

        #-------- GRU --------
        df_gru = walk_forward_validation_gru(
            df=sub_df,
            train_size=train_size,
            x_cols=sub_cols,
            y_col=tgt,
            train_func=lambda Xtr,ytr: train_gru_ktuner(
                Xtr, ytr,
                window_size=window_size,
                max_trials=n_trials,
                epochs=10,
                batch_size=4
            ),
            predict_func=predict_gru_ktuner,
            window_size=window_size
        )
        mg = df_gru['pred'].notnull()
        if mg.sum() > 0:
            rmseG= sqrt(mean_squared_error(df_gru.loc[mg,'actual'], df_gru.loc[mg,'pred']))
            maeG=  mean_absolute_error(df_gru.loc[mg,'actual'], df_gru.loc[mg,'pred'])
        else:
            rmseG, maeG = np.nan, np.nan
        plot_pred_vs_actual(df_gru, "GRU", tgt)

        # 결과 저장
        results[tgt] = {
            "ARIMA":(rmseA, maeA),
            "XGB":(rmseX, maeX),
            "Prophet":(rmseP, maeP),
            "LSTM":(rmseL, maeL),
            "GRU":(rmseG, maeG)
        }

    # (4) 결과 요약 출력
    print("\n===== [Summary: RMSE, MAE] =====")
    for t,vals in results.items():
        print(f"\n>>> Target={t}")
        for mod,(r_,m_) in vals.items():
            print(f"{mod:10s} => RMSE={r_}, MAE={m_}")


# __main__부분: 실제로 함수를 호출하여 실행
if __name__=="__main__":
    # 파라미터를 원하는 값으로 조절 가능
    # (참고) train_size=3이면 학습 데이터가 매우 적어 과적합 또는 성능저하 가능
    #       window_size=2이면 LSTM, GRU가 매우 짧은 윈도우로 학습
    run_all_models_infectious(
        file_path="Processed_COVID_Data_Filled.csv",  # CSV 파일 이름
        target_list=["Cases"],                        # 예측하고자 하는 열 이름
        train_size=3,                                 # 초반 최소 훈련용 데이터행
        window_size=2,                                # LSTM, GRU에 사용할 시계열 윈도우
        n_trials=2                                    # Optuna/KerasTuner 탐색 횟수
    )
