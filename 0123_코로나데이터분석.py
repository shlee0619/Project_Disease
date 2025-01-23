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

# 텐서플로
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
    1) CSV 파일 로드 (인코딩 여부 점검)
    2) 'Date' 열을 datetime으로 변환 후 인덱스 지정
    3) 결측치 (수치형) -> median으로 채움
    """
    # (1) encoding 주의: 실제 CSV 인코딩에 맞춰 변경
    #    사용자 CSV가 ANSI일 수도 있지만, 실제로는 UTF-8일 가능성도 존재.
    df = pd.read_csv(filepath, encoding='ANSI')  

    # (2) Date -> datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)
    else:
        raise ValueError("CSV에 'Date' 컬럼이 없습니다. 파일을 확인하세요.")

    # (3) 결측치 median 처리
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
    - lag_features: { '컬럼명': [1,2,3,...], ... }
    - roll_features: { '컬럼명': [2,3,...], ... }
    """
    df_ = df.copy()

    # Lag
    if lag_features:
        for col, lags in lag_features.items():
            if col not in df_.columns:
                continue
            for lag in lags:
                new_col = f"{col}_lag{lag}"
                df_[new_col] = df_[col].shift(lag)

    # Rolling
    if roll_features:
        for col, wins in roll_features.items():
            if col not in df_.columns:
                continue
            for w in wins:
                mean_col = f"{col}_rmean{w}"
                std_col  = f"{col}_rstd{w}"
                df_[mean_col] = df_[col].rolling(w).mean()
                df_[std_col]  = df_[col].rolling(w).std()

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
    - train_func(X_train, y_train) -> model
    - predict_func(model, X_test) -> 예측값
    - min_required: ARIMA 등 최소 샘플 필요 수
    """
    n = len(df)
    df_res = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_res['actual'] = df[y_col].values

    X_data = df[x_cols].values
    y_data = df[y_col].values

    for i in range(train_size, n):
        X_train = X_data[:i]
        y_train = y_data[:i]

        if len(y_train) < min_required:
            df_res.iloc[i, df_res.columns.get_loc('pred')] = np.mean(y_train)
            continue

        try:
            model = train_func(X_train, y_train)
            X_test = X_data[i:i+1]
            yhat = predict_func(model, X_test)
        except:
            yhat = np.mean(y_train)

        df_res.iloc[i, df_res.columns.get_loc('pred')] = yhat

    return df_res


def walk_forward_validation_prophet(
    df: pd.DataFrame,
    train_size: int,
    y_col: str
) -> pd.DataFrame:
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
    p = trial.suggest_int('p', 0, 2)
    d = trial.suggest_int('d', 0, 1)
    q = trial.suggest_int('q', 0, 2)
    try:
        exog = X_train if X_train.shape[1] > 0 else None
        model_ = pm.ARIMA(order=(p,d,q))
        fit_   = model_.fit(y_train, exogenous=exog)
        return fit_.aic()
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
    xgb_final = XGBRegressor(n_estimators=n_, learning_rate=lr_, max_depth=md_, random_state=42)
    xgb_final.fit(X_train, y_train)
    return xgb_final

def predict_xgb_optuna(model, X_test):
    if np.isnan(X_test).any():
        X_test = np.nan_to_num(X_test, nan=0.0)
    return model.predict(X_test)[0]


###################################################
# 4. 간단 LSTM / GRU (KerasTuner 미사용)
###################################################
def make_lstm_dataset(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X)-window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

#
# (A) 간단 LSTM
#
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
    if len(X_train) <= window_size:
        dummy = keras.Sequential([
            layers.Input(shape=(window_size, X_train.shape[1])),
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

    X_seq, y_seq = make_lstm_dataset(X_train, y_train, window_size)

    model = keras.Sequential()
    model.add(layers.Input(shape=(window_size, X_train.shape[1])))
    model.add(layers.LSTM(units, activation='tanh'))
    if dropout>0:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')

    model.fit(
        X_seq, y_seq,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    return model

def predict_lstm_simple(model, X_seq):
    if np.isnan(X_seq).any():
        X_seq = np.nan_to_num(X_seq, nan=0.0)
    p = model.predict(X_seq)
    return p[0,0]

#
# (B) 간단 GRU
#
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
        dummy = keras.Sequential([
            layers.Input(shape=(window_size, X_train.shape[1])),
            layers.GRU(16),
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

    X_seq, y_seq = make_lstm_dataset(X_train, y_train, window_size)

    model = keras.Sequential()
    model.add(layers.Input(shape=(window_size, X_train.shape[1])))
    model.add(layers.GRU(units, activation='tanh'))
    if dropout>0:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')

    model.fit(
        X_seq, y_seq,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
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
    target_list=["Cases"],     # 반드시 CSV 내 존재해야 함
    train_size=10,             # 너무 작은 값(3)보다는 크게, 예:10,20...
    window_size=4,
    n_trials=2
):
    # 1) 데이터 로드
    df = load_and_preprocess_data(file_path)

    # 2) 피처 엔지니어링
    exclude_cols = list(set(target_list))
    x_cols = [c for c in df.columns if c not in exclude_cols]

    lagF = {}
    rollF= {}
    for c in x_cols:
        lagF[c] = [1]       # lag1 예시
        rollF[c]= [2]       # rolling2 예시

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

        # (D) LSTM (simple)
        df_lstm = walk_forward_validation_arbitrary(
            sub_df,
            train_size=train_size,
            train_func=lambda Xtr,ytr: train_lstm_simple(
                Xtr, ytr,
                window_size=window_size,
                units=32,
                dropout=0.2,
                lr=1e-3,
                epochs=3,        # epoch 줄이기(예:3)
                batch_size=8
            ),
            predict_func=lambda mod, Xts: predict_lstm_simple(mod, Xts.reshape(1,window_size,Xts.shape[1])),
            x_cols=sub_cols,
            y_col=tgt,
            min_required=window_size
        )
        # 위에서 walk_forward_validation_arbitrary를 썼으므로 LSTM 시퀀스 reshape는 predict_func에서 처리
        # 필요 시 기존 walk_forward_validation_lstm를 사용해도 됨
        ml = df_lstm['pred'].notnull()
        if ml.sum()>0:
            rmseL = sqrt(mean_squared_error(df_lstm.loc[ml,'actual'], df_lstm.loc[ml,'pred']))
            maeL  = mean_absolute_error(df_lstm.loc[ml,'actual'], df_lstm.loc[ml,'pred'])
        else:
            rmseL, maeL = np.nan, np.nan
        plot_pred_vs_actual(df_lstm, "LSTM(simple)", tgt)

        # (E) GRU (simple)
        df_gru = walk_forward_validation_arbitrary(
            sub_df,
            train_size=train_size,
            train_func=lambda Xtr,ytr: train_gru_simple(
                Xtr, ytr,
                window_size=window_size,
                units=32,
                dropout=0.2,
                lr=1e-3,
                epochs=3,      # epoch 줄이기(예:3)
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

    # 요약 출력
    print("\n========== [Summary: RMSE, MAE] ==========")
    for t, vals in results.items():
        print(f"\n>>> Target={t}")
        for mod, (r_, m_) in vals.items():
            print(f"{mod:15s} => RMSE={r_}, MAE={m_}")


###################################################
# 실행 예시
###################################################
if __name__=="__main__":
    # 실제 CSV 파일 경로
    csv_path = "Processed_COVID_Data_Filled.csv"

    # 실행
    run_all_models_infectious(
        file_path=csv_path,
        target_list=["Cases"],       # CSV 내 'Cases' 컬럼
        train_size=10,               # 초반 10개 데이터로만 첫 학습
        window_size=4,
        n_trials=2
    )
