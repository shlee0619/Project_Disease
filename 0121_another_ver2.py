###################################################
# 0. 라이브러리 Import
###################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 시계열 + AutoML
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
    df = pd.read_csv(filepath)
    # date -> datetime
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    # 간단 결측치 처리(중간값)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # 필요 시 이상치 clip
    # (원하면 여기에 추가 가능)
    
    return df


def create_feature_engineering(
    df: pd.DataFrame, 
    lag_features=None, 
    roll_features=None, 
    dropna=True
):
    df_ = df.copy()
    if lag_features:
        for col, lags in lag_features.items():
            if col not in df_.columns:
                continue
            for lag in lags:
                df_[f"{col}_lag{lag}"] = df_[col].shift(lag)

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
# 2. Walk-forward - ARIMA / XGB / Prophet / LSTM / GRU
###################################################
def walk_forward_validation_arbitrary(
    df: pd.DataFrame,
    train_size: int,
    train_func,
    predict_func,
    x_cols: list,
    y_col: str
) -> pd.DataFrame:
    n = len(df)
    df_res = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_res['actual'] = df[y_col].values

    X_data = df[x_cols].values
    y_data = df[y_col].values

    for i in range(train_size, n):
        X_train = X_data[:i]
        y_train = y_data[:i]
        model   = train_func(X_train, y_train)

        X_test = X_data[i:i+1]
        yhat = predict_func(model, X_test)
        df_res.iloc[i, df_res.columns.get_loc('pred')] = yhat

    return df_res


def walk_forward_validation_prophet(
    df: pd.DataFrame,
    train_size: int,
    y_col: str
):
    n = len(df)
    df_res = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_res['actual'] = df[y_col].values

    idx_ = df.index
    y_   = df[y_col].values

    for i in range(train_size, n):
        date_train = idx_[:i]
        y_train    = y_[:i]

        # 모델 훈련
        model_ = Prophet()
        df_p = pd.DataFrame({'ds': date_train, 'y': y_train})
        model_.fit(df_p)

        next_d = idx_[i]
        df_ = pd.DataFrame({'ds': [next_d]})
        fcst = model_.predict(df_)
        df_res.iloc[i, df_res.columns.get_loc('pred')] = fcst['yhat'].values[0]

    return df_res


# LSTM / GRU WalkForward
def walk_forward_validation_lstm(
    df: pd.DataFrame,
    train_size: int,
    x_cols: list,
    y_col: str,
    train_func,
    predict_func,
    window_size=4
) -> pd.DataFrame:
    n = len(df)
    df_res = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_res['actual'] = df[y_col].values

    X_all = df[x_cols].values
    y_all = df[y_col].values

    for i in range(train_size, n):
        X_train = X_all[:i]
        y_train = y_all[:i]
        model   = train_func(X_train, y_train)

        if i < window_size:
            df_res.iloc[i, df_res.columns.get_loc('pred')] = np.nan
        else:
            x_seq = X_all[i-window_size:i]
            x_seq = x_seq.reshape(1, window_size, x_seq.shape[1])
            yhat  = predict_func(model, x_seq)
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
    n = len(df)
    df_res = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_res['actual'] = df[y_col].values

    X_all = df[x_cols].values
    y_all = df[y_col].values

    for i in range(train_size, n):
        X_train = X_all[:i]
        y_train = y_all[:i]
        model   = train_func(X_train, y_train)

        if i < window_size:
            df_res.iloc[i, df_res.columns.get_loc('pred')] = np.nan
        else:
            x_seq = X_all[i-window_size:i]
            x_seq = x_seq.reshape(1, window_size, x_seq.shape[1])
            yhat  = predict_func(model, x_seq)
            df_res.iloc[i, df_res.columns.get_loc('pred')] = yhat

    return df_res


###################################################
# 3. ARIMA(Optuna) / XGB(Optuna)
###################################################
def objective_arima(trial, X_train, y_train):
    p = trial.suggest_int('p', 0, 3)
    d = trial.suggest_int('d', 0, 2)
    q = trial.suggest_int('q', 0, 3)
    try:
        exog = X_train if X_train.shape[1] > 0 else None
        model_ = pm.ARIMA(order=(p,d,q))
        fit_   = model_.fit(y_train, exogenous=exog)
        return fit_.aic()
    except:
        return 1e9

def train_arima_optuna(X_train, y_train, n_trials=3):
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

def train_xgb_optuna(X_train, y_train, n_trials=3):
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
    return model.predict(X_test)[0]


###################################################
# 4. LSTM / GRU KerasTuner
###################################################
def make_lstm_dataset(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X)-window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)


# LSTM
import keras_tuner as kt

def train_lstm_ktuner(X_train, y_train,
                      window_size=4,
                      max_trials=2,
                      epochs=10,
                      batch_size=8):
    """
    기존과 동일, + directory 인자를 지정하여 
    'FailedPreconditionError: . is not a directory' 오류 방지
    """
    if len(X_train) <= window_size:
        dummy = keras.Sequential([layers.Input(shape=(window_size,X_train.shape[1])),
                                  layers.LSTM(16),
                                  layers.Dense(1)])
        dummy.compile(optimizer='adam', loss='mse')
        dummy.fit(X_train[:1].reshape(1,window_size,-1), np.array([y_train[0]]), epochs=1, verbose=0)
        return dummy

    X_seq, y_seq = make_lstm_dataset(X_train, y_train, window_size)

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

    tuner = kt.RandomSearch(
        build_lstm_with_hp,
        objective='val_loss',
        max_trials=max_trials,
        overwrite=True,
        directory='temp_ktuner_lstm',       # <<< 디렉토리 지정
        project_name='lstm_wf'
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

def predict_lstm_ktuner(model, X_seq):
    p = model.predict(X_seq)
    return p[0,0]


# GRU
def train_gru_ktuner(X_train, y_train,
                     window_size=4,
                     max_trials=2,
                     epochs=10,
                     batch_size=8):
    if len(X_train) <= window_size:
        dummy = keras.Sequential([layers.Input(shape=(window_size,X_train.shape[1])),
                                  layers.GRU(16),
                                  layers.Dense(1)])
        dummy.compile(optimizer='adam', loss='mse')
        dummy.fit(X_train[:1].reshape(1,window_size,-1), np.array([y_train[0]]), epochs=1, verbose=0)
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
        directory='temp_ktuner_gru',        # <<< 디렉토리 지정
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
    file_path="modified_infectious_disease_data_copy.csv",
    target_list=["호흡기_new","매개성"],
    train_size=3,
    window_size=2,
    n_trials=2
):
    # 1) Load
    df = load_and_preprocess_data(file_path)
    print("[INFO] Data shape:", df.shape)
    print(df.head())

    # 2) x_cols
    exclude_cols = ["호흡기_new","매개성"]
    x_cols = [c for c in df.columns if c not in exclude_cols]

    # (예시) Lag1, Rolling2
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
            print(f"Skip {tgt}")
            continue
        print(f"\n===== Target={tgt} =====")

        sub_cols = [c for c in df_fe.columns if c!=tgt]
        sub_df   = df_fe[sub_cols + [tgt]].copy()

        # ARIMA
        df_arima = walk_forward_validation_arbitrary(
            sub_df, train_size=train_size,
            train_func=lambda Xtr,ytr: train_arima_optuna(Xtr,ytr,n_trials),
            predict_func=predict_arima_optuna,
            x_cols=sub_cols,
            y_col=tgt
        )
        mA = df_arima['pred'].notnull()
        rmseA= sqrt(mean_squared_error(df_arima.loc[mA,'actual'], df_arima.loc[mA,'pred']))
        maeA=  mean_absolute_error(df_arima.loc[mA,'actual'], df_arima.loc[mA,'pred'])
        plot_pred_vs_actual(df_arima, "ARIMA", tgt)

        # XGB
        df_xgb = walk_forward_validation_arbitrary(
            sub_df, train_size=train_size,
            train_func=lambda Xtr,ytr: train_xgb_optuna(Xtr,ytr,n_trials),
            predict_func=predict_xgb_optuna,
            x_cols=sub_cols,
            y_col=tgt
        )
        mX = df_xgb['pred'].notnull()
        rmseX= sqrt(mean_squared_error(df_xgb.loc[mX,'actual'], df_xgb.loc[mX,'pred']))
        maeX=  mean_absolute_error(df_xgb.loc[mX,'actual'], df_xgb.loc[mX,'pred'])
        plot_pred_vs_actual(df_xgb, "XGB", tgt)

        # Prophet
        df_ppt = walk_forward_validation_prophet(
            sub_df, train_size=train_size, y_col=tgt
        )
        mP= df_ppt['pred'].notnull()
        rmseP= sqrt(mean_squared_error(df_ppt.loc[mP,'actual'], df_ppt.loc[mP,'pred']))
        maeP=  mean_absolute_error(df_ppt.loc[mP,'actual'], df_ppt.loc[mP,'pred'])
        plot_pred_vs_actual(df_ppt, "Prophet", tgt)

        # LSTM
        df_lstm = walk_forward_validation_lstm(
            df=sub_df,
            train_size=train_size,
            x_cols=sub_cols,
            y_col=tgt,
            train_func=lambda Xtr,ytr: train_lstm_ktuner(Xtr,ytr,window_size,n_trials,10,4),
            predict_func=predict_lstm_ktuner,
            window_size=window_size
        )
        ml= df_lstm['pred'].notnull()
        rmseL= sqrt(mean_squared_error(df_lstm.loc[ml,'actual'], df_lstm.loc[ml,'pred']))
        maeL=  mean_absolute_error(df_lstm.loc[ml,'actual'], df_lstm.loc[ml,'pred'])
        plot_pred_vs_actual(df_lstm, "LSTM", tgt)

        # GRU
        df_gru = walk_forward_validation_gru(
            df=sub_df,
            train_size=train_size,
            x_cols=sub_cols,
            y_col=tgt,
            train_func=lambda Xtr,ytr: train_gru_ktuner(Xtr,ytr,window_size,n_trials,10,4),
            predict_func=predict_gru_ktuner,
            window_size=window_size
        )
        mg= df_gru['pred'].notnull()
        rmseG= sqrt(mean_squared_error(df_gru.loc[mg,'actual'], df_gru.loc[mg,'pred']))
        maeG=  mean_absolute_error(df_gru.loc[mg,'actual'], df_gru.loc[mg,'pred'])
        plot_pred_vs_actual(df_gru, "GRU", tgt)

        results[tgt] = {
            "ARIMA":(rmseA, maeA),
            "XGB":(rmseX, maeX),
            "Prophet":(rmseP, maeP),
            "LSTM":(rmseL, maeL),
            "GRU":(rmseG, maeG)
        }

    print("\n===== [Summary: RMSE, MAE] =====")
    for t,vals in results.items():
        print(f"\n>>> Target={t}")
        for mod,(r_,m_) in vals.items():
            print(f"{mod:10s} => RMSE={r_:.3f}, MAE={m_:.3f}")


if __name__=="__main__":
    run_all_models_infectious(
        file_path="modified_infectious_disease_data_copy.csv",
        target_list=["호흡기_new","매개성"],
        train_size=3,
        window_size=2,
        n_trials=2
    )
