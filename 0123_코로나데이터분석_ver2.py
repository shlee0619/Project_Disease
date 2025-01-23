###################################################
# 0. 라이브러리 Import
###################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

import optuna
import keras_tuner as kt
import pmdarima as pm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Prophet
from prophet import Prophet

###################################################
# 1. 데이터 로드 및 전처리
###################################################
def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    CSV 파일을 로드한 뒤,
    1) 'Date' 열 -> datetime 변환 + 인덱스 설정
    2) 결측치는 median으로 채움
    3) shape, head 등 기본 정보 출력
    """
    df = pd.read_csv(filepath, encoding='ANSI')  
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)
    else:
        raise ValueError("CSV에 'Date' 컬럼이 없습니다. 파일을 확인하세요.")

    df.fillna(df.median(numeric_only=True), inplace=True)

    print("[INFO] After load, shape=", df.shape)
    print(df.head(3))
    return df


def create_feature_engineering(
    df: pd.DataFrame,
    lag_features=None,
    roll_features=None,
    dropna=True
):
    """
    lag_features: { '컬럼명': [1,2,3...], ... }
    roll_features: { '컬럼명': [2,3,...], ... }
    """
    df_ = df.copy()

    # Lag
    if lag_features:
        for col, lags in lag_features.items():
            if col not in df_.columns:
                continue
            for lag in lags:
                df_[f"{col}_lag{lag}"] = df_[col].shift(lag)

    # Rolling
    if roll_features:
        for col, wins in roll_features.items():
            if col not in df_.columns:
                continue
            for w in wins:
                df_[f"{col}_rmean{w}"] = df_[col].rolling(w).mean()
                df_[f"{col}_rstd{w}"]  = df_[col].rolling(w).std()

    if dropna:
        df_.dropna(inplace=True)

    return df_


###################################################
# 2. Walk-forward - ARIMA / XGB / Prophet 등
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
    임의의 train/predict 함수를 입력받아 매 스텝 walk-forward 검증.
    min_required: ARIMA 등 최소 데이터 갯수 필요 시 사용.
    """
    n = len(df)
    df_res = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_res['actual'] = df[y_col].values

    X_data = df[x_cols].values
    y_data = df[y_col].values

    for i in range(train_size, n):
        X_train = X_data[:i]
        y_train = y_data[:i]

        # 최소 샘플 수 안되면 fallback
        if len(y_train) < min_required:
            df_res.iloc[i, df_res.columns.get_loc('pred')] = np.mean(y_train)
            continue

        try:
            model = train_func(X_train, y_train)
            X_test = X_data[i:i+1]
            yhat = predict_func(model, X_test)
        except:
            # 예외 발생 시 fallback
            yhat = np.mean(y_train)

        df_res.iloc[i, df_res.columns.get_loc('pred')] = yhat

    return df_res


def walk_forward_validation_prophet(
    df: pd.DataFrame,
    train_size: int,
    y_col: str
):
    """
    Prophet 전용 walk-forward
    """
    n = len(df)
    df_res = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_res['actual'] = df[y_col].values

    idx_ = df.index
    y_   = df[y_col].values

    for i in range(train_size, n):
        date_train = idx_[:i]
        y_train    = y_[:i]

        try:
            model_ = Prophet()
            df_p = pd.DataFrame({'ds': date_train, 'y': y_train})
            model_.fit(df_p)

            next_d = idx_[i]
            df_ = pd.DataFrame({'ds': [next_d]})
            fcst = model_.predict(df_)
            df_res.iloc[i, df_res.columns.get_loc('pred')] = fcst['yhat'].values[0]
        except:
            df_res.iloc[i, df_res.columns.get_loc('pred')] = np.mean(y_train)

    return df_res


###################################################
# 3. ARIMA(Optuna) / XGB(Optuna)
###################################################
def objective_arima(trial, X_train, y_train):
    # ARIMA 파라미터 (p,d,q)를 검색
    p = trial.suggest_int('p', 0, 2)
    d = trial.suggest_int('d', 0, 1)
    q = trial.suggest_int('q', 0, 2)
    try:
        exog = X_train if X_train.shape[1] > 0 else None
        model_ = pm.ARIMA(order=(p,d,q))
        fit_   = model_.fit(y_train, exogenous=exog)
        return fit_.aic()  # Optuna에선 AIC 최소화
    except:
        return 1e9

def train_arima_optuna(X_train, y_train, n_trials=2):
    def _obj(trial):
        return objective_arima(trial, X_train, y_train)
    study = optuna.create_study(direction='minimize')
    study.optimize(_obj, n_trials=n_trials)
    best_ = study.best_trial
    p_ = best_.params['p']
    d_ = best_.params['d']
    q_ = best_.params['q']
    exog = X_train if X_train.shape[1] > 0 else None

    model_ = pm.ARIMA(order=(p_, d_, q_))
    fit_   = model_.fit(y_train, exogenous=exog)
    return fit_

def predict_arima_optuna(model, X_test):
    exog = X_test if X_test.shape[1]>0 else None
    if exog is not None and np.isnan(exog).any():
        exog = np.nan_to_num(exog, nan=0.0)
    fcst = model.predict(n_periods=1, exogenous=exog, return_conf_int=False)
    return fcst[0]


from xgboost import XGBRegressor

def objective_xgb(trial, X_train, y_train):
    # XGB 파라미터 (n_est, lr, max_depth) 검색
    n_  = trial.suggest_int('n_est', 50, 200, step=50)
    lr_ = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
    md_ = trial.suggest_int('md', 2, 6)
    xgb_ = XGBRegressor(n_estimators=n_, learning_rate=lr_, max_depth=md_, random_state=42)
    xgb_.fit(X_train, y_train)
    yhat_ = xgb_.predict(X_train)
    return sqrt(mean_squared_error(y_train, yhat_))

def train_xgb_optuna(X_train, y_train, n_trials=2):
    def _obj(trial):
        return objective_xgb(trial, X_train, y_train)
    study = optuna.create_study(direction='minimize')
    study.optimize(_obj, n_trials=n_trials)
    best_ = study.best_trial
    n_  = best_.params['n_est']
    lr_ = best_.params['lr']
    md_ = best_.params['md']

    xgb_final = XGBRegressor(
        n_estimators=n_,
        learning_rate=lr_,
        max_depth=md_,
        random_state=42
    )
    xgb_final.fit(X_train, y_train)
    return xgb_final

def predict_xgb_optuna(model, X_test):
    if np.isnan(X_test).any():
        X_test = np.nan_to_num(X_test, nan=0.0)
    return model.predict(X_test)[0]


###################################################
# 4. 간단 LSTM / GRU (KerasTuner 미사용)
###################################################
@tf.function(reduce_retracing=True)
def _fit_model(model, X, y, epochs, batch_size):
    """
    실제 모델 fit 과정을 감싸서 @tf.function 적용
    reduce_retracing=True: 동일 shape/타입 입력 시 tracing 최소화
    """
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

def make_lstm_dataset(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X)-window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(input_dim, window_size=4, units=32, dropout=0.2, lr=1e-3):
    """
    LSTM 모델 빌드만 담당 (tf.function 사용 X)
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(window_size, input_dim)))
    model.add(layers.LSTM(units, activation='tanh'))
    if dropout>0:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def train_lstm_simple(
    X_train,
    y_train,
    window_size=4,
    units=32,
    dropout=0.2,
    lr=1e-3,
    epochs=5,
    batch_size=8
):
    """
    Walk-forward에서 반복 호출되는 LSTM 훈련 함수.
    1) build_lstm_model()으로 모델 구조 빌드
    2) make_lstm_dataset()으로 시퀀스화
    3) _fit_model()을 @tf.function으로 감싸 retracing 줄이기
    """
    if len(X_train) <= window_size:
        # fallback
        dummy = build_lstm_model(X_train.shape[1], window_size, 16, 0.0, 1e-3)
        if len(X_train) > 0:
            dummy.fit(X_train[:1].reshape(1,window_size,-1),
                      np.array([y_train[0]]),
                      epochs=1, verbose=0)
        return dummy

    # (1) 시퀀스
    X_seq, y_seq = make_lstm_dataset(X_train, y_train, window_size)
    # (2) 모델 빌드
    model = build_lstm_model(X_train.shape[1], window_size, units, dropout, lr)
    # (3) fit
    _fit_model(model, X_seq, y_seq, epochs, batch_size)
    return model

def predict_lstm_simple(model, X_seq):
    if np.isnan(X_seq).any():
        X_seq = np.nan_to_num(X_seq, nan=0.0)
    p = model.predict(X_seq)
    return p[0,0]


# GRU도 같은 방식
def build_gru_model(input_dim, window_size=4, units=32, dropout=0.2, lr=1e-3):
    model = keras.Sequential()
    model.add(layers.Input(shape=(window_size, input_dim)))
    model.add(layers.GRU(units, activation='tanh'))
    if dropout>0:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def train_gru_simple(
    X_train,
    y_train,
    window_size=4,
    units=32,
    dropout=0.2,
    lr=1e-3,
    epochs=5,
    batch_size=8
):
    if len(X_train) <= window_size:
        dummy = build_gru_model(X_train.shape[1], window_size, 16, 0.0, 1e-3)
        if len(X_train) > 0:
            dummy.fit(X_train[:1].reshape(1,window_size,-1),
                      np.array([y_train[0]]),
                      epochs=1, verbose=0)
        return dummy

    X_seq, y_seq = make_lstm_dataset(X_train, y_train, window_size)
    model = build_gru_model(X_train.shape[1], window_size, units, dropout, lr)
    _fit_model(model, X_seq, y_seq, epochs, batch_size)
    return model

def predict_gru_simple(model, X_seq):
    if np.isnan(X_seq).any():
        X_seq = np.nan_to_num(X_seq, nan=0.0)
    p = model.predict(X_seq)
    return p[0,0]


###################################################
# 5. 결과 시각화
###################################################
def plot_pred_vs_actual(df_res, model_name, y_col):
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
    file_path="your_data.csv",
    target_list=["Cases"],     
    train_size=10,            
    window_size=4,
    n_trials=2
):
    """
    Walk-forward 방식으로 ARIMA/XGB/Prophet/LSTM/GRU를 비교.
    @tf.function(reduce_retracing=True) 적용으로 retracing 경고를 줄임.
    """
    # 1) 데이터 로드 + 전처리
    df = load_and_preprocess_data(file_path)

    # 2) 피처 엔지니어링
    exclude_cols = list(set(target_list))
    x_cols = [c for c in df.columns if c not in exclude_cols]

    lagF = {}
    rollF= {}
    for c in x_cols:
        lagF[c] = [1]  
        rollF[c]= [2]  

    df_fe = create_feature_engineering(df, lagF, rollF, dropna=True)
    print("[INFO] After FE shape:", df_fe.shape)

    results = {}
    for tgt in target_list:
        if tgt not in df_fe.columns:
            print(f"Skip target={tgt}, not in columns")
            continue

        print(f"\n=== Target={tgt} ===")
        sub_cols = [c for c in df_fe.columns if c != tgt]
        sub_df   = df_fe[sub_cols + [tgt]].copy()

        # (A) ARIMA
        df_arima = walk_forward_validation_arbitrary(
            sub_df,
            train_size=train_size,
            train_func=lambda Xtr,ytr: train_arima_optuna(Xtr,ytr,n_trials),
            predict_func=predict_arima_optuna,
            x_cols=sub_cols,
            y_col=tgt,
            min_required=10
        )
        mA = df_arima['pred'].notnull()
        if mA.sum()>0:
            rmseA = sqrt(mean_squared_error(df_arima.loc[mA,'actual'], df_arima.loc[mA,'pred']))
            maeA  = mean_absolute_error(df_arima.loc[mA,'actual'], df_arima.loc[mA,'pred'])
        else:
            rmseA, maeA = np.nan, np.nan
        plot_pred_vs_actual(df_arima, "ARIMA", tgt)

        # (B) XGB
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
        if mX.sum()>0:
            rmseX = sqrt(mean_squared_error(df_xgb.loc[mX,'actual'], df_xgb.loc[mX,'pred']))
            maeX  = mean_absolute_error(df_xgb.loc[mX,'actual'], df_xgb.loc[mX,'pred'])
        else:
            rmseX, maeX = np.nan, np.nan
        plot_pred_vs_actual(df_xgb, "XGB", tgt)

        # (C) Prophet
        df_ppt = walk_forward_validation_prophet(
            sub_df, train_size=train_size, y_col=tgt
        )
        mP = df_ppt['pred'].notnull()
        if mP.sum()>0:
            rmseP = sqrt(mean_squared_error(df_ppt.loc[mP,'actual'], df_ppt.loc[mP,'pred']))
            maeP  = mean_absolute_error(df_ppt.loc[mP,'actual'], df_ppt.loc[mP,'pred'])
        else:
            rmseP, maeP = np.nan, np.nan
        plot_pred_vs_actual(df_ppt, "Prophet", tgt)

        # (D) LSTM (simple) - @tf.function 기반 retracing 감소
        df_lstm = walk_forward_validation_arbitrary(
            sub_df,
            train_size=train_size,
            train_func=lambda Xtr,ytr: train_lstm_simple(
                Xtr, ytr,
                window_size=window_size,
                units=32,
                dropout=0.2,
                lr=1e-3,
                epochs=3,     
                batch_size=8
            ),
            predict_func=lambda mod, Xts: predict_lstm_simple(mod, Xts.reshape(1,window_size,Xts.shape[1])),
            x_cols=sub_cols,
            y_col=tgt,
            min_required=window_size
        )
        ml = df_lstm['pred'].notnull()
        if ml.sum()>0:
            rmseL = sqrt(mean_squared_error(df_lstm.loc[ml,'actual'], df_lstm.loc[ml,'pred']))
            maeL  = mean_absolute_error(df_lstm.loc[ml,'actual'], df_lstm.loc[ml,'pred'])
        else:
            rmseL, maeL = np.nan, np.nan
        plot_pred_vs_actual(df_lstm, "LSTM(simple)", tgt)

        # (E) GRU (simple) - @tf.function 기반 retracing 감소
        df_gru = walk_forward_validation_arbitrary(
            sub_df,
            train_size=train_size,
            train_func=lambda Xtr,ytr: train_gru_simple(
                Xtr, ytr,
                window_size=window_size,
                units=32,
                dropout=0.2,
                lr=1e-3,
                epochs=3,     
                batch_size=8
            ),
            predict_func=lambda mod, Xts: predict_gru_simple(mod, Xts.reshape(1,window_size,Xts.shape[1])),
            x_cols=sub_cols,
            y_col=tgt,
            min_required=window_size
        )
        mg = df_gru['pred'].notnull()
        if mg.sum()>0:
            rmseG = sqrt(mean_squared_error(df_gru.loc[mg,'actual'], df_gru.loc[mg,'pred']))
            maeG  = mean_absolute_error(df_gru.loc[mg,'actual'], df_gru.loc[mg,'pred'])
        else:
            rmseG, maeG = np.nan, np.nan
        plot_pred_vs_actual(df_gru, "GRU(simple)", tgt)

        # 결과 저장
        results[tgt] = {
            "ARIMA": (rmseA, maeA),
            "XGB": (rmseX, maeX),
            "Prophet": (rmseP, maeP),
            "LSTM(simple)": (rmseL, maeL),
            "GRU(simple)": (rmseG, maeG)
        }

    # 요약
    print("\n========== [Summary: RMSE, MAE] ==========")
    for t, vals in results.items():
        print(f"\n>>> Target={t}")
        for mod, (r_, m_) in vals.items():
            print(f"{mod:15s} => RMSE={r_}, MAE={m_}")


###################################################
# 실행 예시
###################################################
if __name__=="__main__":
    # CSV 파일 경로
    csv_path = "Processed_COVID_Data_Filled.csv"

    run_all_models_infectious(
        file_path=csv_path,
        target_list=["Cases"],
        train_size=10,
        window_size=4,
        n_trials=2
    )
