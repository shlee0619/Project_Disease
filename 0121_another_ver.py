########################################################
# 0. 라이브러리 import
########################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

########################################################
# 1. 고급 전처리 + 피처 엔지니어링
########################################################

def advanced_preprocessing(df, datetime_col='date', y_col='target'):
    """
    1) datetime_col(예: date) -> datetime 변환 & 정렬
    2) 사이클형(월,요일) 등 time-features: sin, cos 생성
    3) target col 이외에 결측치 처리(간단: median)
    4) RobustScaler 적용 (이상치에 좀 더 강건)
    """
    df = df.copy()
    # 가정: date가 object -> datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.sort_values(datetime_col, inplace=True)
    df.set_index(datetime_col, inplace=True)

    # (2) 사이클형 피처 (예: 월, 요일 -> sin/cos)
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['wday_sin']  = np.sin(2 * np.pi * df['weekday'] / 7)
    df['wday_cos']  = np.cos(2 * np.pi * df['weekday'] / 7)

    # 불필요한 컬럼 제거
    df.drop(['month','weekday'], axis=1, inplace=True)

    # (3) 결측치: 수치컬럼에 대해 median
    for c in df.columns:
        if c == y_col:  # 타겟은 건너뜀 (or 따로 처리)
            continue
        if df[c].isnull().any():
            df[c].fillna(df[c].median(), inplace=True)

    # (4) RobustScaler
    #   - y_col 제외, 시간열/문자열 제외
    #   - 한 번에 fit_transform -> test set에선 transform 만
    x_cols = [c for c in df.columns if c != y_col]
    scaler = RobustScaler()
    # fit_transform
    df[x_cols] = scaler.fit_transform(df[x_cols])

    df.dropna(inplace=True)
    return df

def create_features_lag_roll(df, y_col='target', 
                             lags=[1,2], rolling_windows=[3,5]):
    """
    lag, rolling mean/std 등 추가. 
    y_col은 모델 입력에서 제외할 것이므로 상관 없음 (필요시 y에도 lag 가능)
    """
    df = df.copy()
    # 예시로 df의 모든 수치열(col)들에 대해 lag, roll
    numeric_cols = [c for c in df.columns if c!=y_col]

    for c in numeric_cols:
        for lag_ in lags:
            df[f"{c}_lag{lag_}"] = df[c].shift(lag_)
        for rw in rolling_windows:
            df[f"{c}_rmean{rw}"] = df[c].rolling(rw).mean()
            df[f"{c}_rstd{rw}"]  = df[c].rolling(rw).std()
    df.dropna(inplace=True)
    return df

########################################################
# 2. CNN + BiLSTM + Attention (예시) 모델
########################################################

class SimpleAttention(layers.Layer):
    """
    매우 간단한 Self-Attention 레이어 예시
    Q=K=V 로부터 attention score -> weighted sum
    (실무에서는 multi-head나 Transformer 구조 적용)
    """
    def __init__(self, units):
        super().__init__()
        self.Wq = layers.Dense(units)
        self.Wk = layers.Dense(units)
        self.Wv = layers.Dense(units)

    def call(self, inputs):
        # inputs: shape (batch, timesteps, features)
        Q = self.Wq(inputs)
        K = self.Wk(inputs)
        V = self.Wv(inputs)
        # scaled dot-product attention
        attn_scores = tf.matmul(Q, K, transpose_b=True)
        d_k = tf.cast(tf.shape(K)[-1], tf.float32)
        attn_scores = attn_scores / tf.math.sqrt(d_k)
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        output = tf.matmul(attn_weights, V)
        return output

def build_cnn_bilstm_attention_model(seq_len, num_feats, hp):
    """
    1D CNN + Bi-LSTM + SimpleAttention 결합 예시

    Hyperparams (hp) 예:
     - conv_filters: Choice(16,32)
     - lstm_units:  Choice(32,64)
     - dropout: Float(0.0~0.5)
     - lr: 1e-3 or 1e-4 ...
    """
    conv_filters = hp.Choice('conv_filters', [16, 32])
    lstm_units   = hp.Choice('lstm_units', [32, 64])
    drop_        = hp.Float('dropout', 0.0, 0.5, step=0.1)
    lr_          = hp.Choice('lr', [1e-3, 5e-4, 1e-4])

    inputs = keras.Input(shape=(seq_len, num_feats))

    # CNN (kernel_size=3)
    x = layers.Conv1D(filters=conv_filters, 
                      kernel_size=3, 
                      activation='relu', 
                      padding='same')(inputs)
    x = layers.MaxPool1D(pool_size=2)(x)

    # Bi-LSTM
    x = layers.Bidirectional(layers.LSTM(lstm_units, 
                                         return_sequences=True))(x)

    # Attention
    x = SimpleAttention(lstm_units*2)(x)  # 2*lstm_units (Bi-LSTM)
    # x shape: (batch, timesteps, hidden)
    # GlobalAveragePooling
    x = layers.GlobalAveragePooling1D()(x)

    if drop_ > 0:
        x = layers.Dropout(drop_)(x)

    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(lr_), loss='mse')
    return model

def make_seq_dataset(X, y, window_size):
    """
    (window_size, n_features) 슬라이딩
    """
    X_seq, y_seq = [], []
    for i in range(len(X)-window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

########################################################
# 3. KerasTuner로 하이퍼파라미터 탐색 + 학습
########################################################
def train_cnn_bilstm_attention_tuner(X_train, y_train, 
                                     window_size=7, 
                                     max_trials=3,
                                     batch_size=16,
                                     epochs=20):
    """
    CNN + BiLSTM + Attention 모델을 KerasTuner로 하이퍼파라미터 탐색

    - window_size: 시퀀스 길이
    - max_trials: KerasTuner 탐색 횟수
    - batch_size, epochs: 고정값 (가령 더 개선 가능)

    Returns
    -------
    final_model
    """
    if len(X_train) <= window_size:
        # 데이터가 매우 적은 경우, fallback
        # 대충 임의 모델
        fallback = build_cnn_bilstm_attention_model(window_size, X_train.shape[1], 
            hp=kt.engine.hyperparameters.HyperParameters()) # or dummy
        fallback.fit(X_train[:1], y_train[:1], epochs=1, verbose=0)
        return fallback

    # (1) 시퀀스화
    X_seq, y_seq = make_seq_dataset(X_train, y_train, window_size)

    # (2) build_model_with_hp
    def build_model_with_hp(hp):
        return build_cnn_bilstm_attention_model(
            seq_len=window_size,
            num_feats=X_train.shape[1],
            hp=hp
        )

    # (3) Tuner
    tuner = kt.RandomSearch(
        hypermodel=build_model_with_hp,
        objective='val_loss',
        max_trials=max_trials,
        project_name='cnn_bilstm_attn_tune'
    )

    # (4) EarlyStopping
    es = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )

    tuner.search(
        X_seq, y_seq,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("[CNN+BiLSTM+Attention Tuner] best params =>",
          f"conv_filters={best_hp.get('conv_filters')}, ",
          f"lstm_units={best_hp.get('lstm_units')}, ",
          f"dropout={best_hp.get('dropout')}, ",
          f"lr={best_hp.get('lr')}")

    final_model = tuner.hypermodel.build(best_hp)
    final_model.fit(
        X_seq, y_seq,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )
    return final_model

def predict_cnn_bilstm_attention(model, X_window):
    """
    X_window shape: (1, window_size, n_features)
    """
    pred = model.predict(X_window)
    return pred[0,0]


########################################################
# 4. 최종 예시: walk-forward + CNN+BiLSTM+Attention
########################################################
def walk_forward_validation_cnn_bilstm_attn(
    df: pd.DataFrame,
    train_size: int,
    x_cols: list,
    y_col: str,
    window_size=7,
    max_trials=3,
    batch_size=16,
    epochs=20
) -> pd.DataFrame:
    """
    CNN+BiLSTM+Attention + KerasTuner walk-forward
    """
    # 결과 저장
    df_res = pd.DataFrame(index=df.index, columns=['actual','pred'])
    df_res['actual'] = df[y_col].values

    X_all = df[x_cols].values
    y_all = df[y_col].values
    n = len(df)

    for i in range(train_size, n):
        # 0~i-1까지 train
        X_train = X_all[:i]
        y_train = y_all[:i]
        # 튜너 모델 훈련
        model = train_cnn_bilstm_attention_tuner(
            X_train, y_train,
            window_size=window_size,
            max_trials=max_trials,
            batch_size=batch_size,
            epochs=epochs
        )

        # 예측
        if i < window_size:
            df_res.iloc[i, df_res.columns.get_loc('pred')] = np.nan
        else:
            X_seq = X_all[i-window_size:i]
            X_seq = X_seq.reshape(1, window_size, X_seq.shape[1])
            yhat = predict_cnn_bilstm_attention(model, X_seq)
            df_res.iloc[i, df_res.columns.get_loc('pred')] = yhat

    return df_res


########################################################
# 5. 종합 실행 예시
########################################################
def run_advanced_pipeline(df: pd.DataFrame, y_col='target'):
    """
    1) advanced_preprocessing + create_features_lag_roll
    2) walk_forward (CNN+BiLSTM+Attention)
    """
    # (1) 고급 전처리
    #    여기서는 이미 df가 전처리 되었다고 가정 or
    #    df = advanced_preprocessing(df, datetime_col='date', y_col=y_col)

    # (2) 피처 엔지니어링 (lag, roll)
    df = create_features_lag_roll(df, y_col=y_col,
                                  lags=[1,2],
                                  rolling_windows=[3,7])

    # (3) x_cols
    x_cols = [c for c in df.columns if c != y_col]

    # (4) walk-forward
    w_size = 7  # 예시
    df_out = walk_forward_validation_cnn_bilstm_attn(
        df, train_size=30,  # 예시
        x_cols=x_cols,
        y_col=y_col,
        window_size=w_size,
        max_trials=2,
        batch_size=8,
        epochs=10
    )

    # 성능 평가
    mask_ = df_out['pred'].notnull()
    rmse_ = sqrt(mean_squared_error(df_out.loc[mask_,'actual'],
                                    df_out.loc[mask_,'pred']))
    mae_  = mean_absolute_error(df_out.loc[mask_,'actual'],
                                df_out.loc[mask_,'pred'])

    # 시각화
    plt.figure(figsize=(8,4))
    plt.plot(df_out.index, df_out['actual'], label='Actual')
    plt.plot(df_out.index, df_out['pred'],   label='Pred', marker='o')
    plt.title(f"CNN+BiLSTM+Attention (RMSE={rmse_:.3f}, MAE={mae_:.3f})")
    plt.legend()
    plt.show()

    return df_out, rmse_, mae_


########################################################
# 만약 이 코드 자체를 실행하고 싶다면:
########################################################
if __name__=="__main__":
    """
    [데모] 아래 예시 df는 가상의 time(0~99)과 'target'을 랜덤 생성.
    실제로는 CSV 로드 -> advanced_preprocessing 등 적용 후
    run_advanced_pipeline()을 호출
    """
    # 가상의 데이터프레임
    np.random.seed(42)
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    df_demo = pd.DataFrame({
        'date': idx,
        'target': np.linspace(10,50,100) + np.random.normal(0,3,100)
    })
    df_demo['extra_1'] = np.random.rand(100)*5
    df_demo['extra_2'] = np.random.rand(100)*10

    df_demo_out, rmse_val, mae_val = run_advanced_pipeline(df_demo, y_col='target')
    print(f"[Result] RMSE={rmse_val:.3f}, MAE={mae_val:.3f}")
