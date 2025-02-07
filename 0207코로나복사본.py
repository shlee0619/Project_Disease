import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import statsmodels.api as sm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, InputLayer
from sklearn.decomposition import PCA
from tcn import TCN
import optuna
import shap
import gc
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from keras import losses


# 데이터 로드 (df는 이미 로드되어 있다고 가정)

df_path = './Processed_COVID_Data_Filled.csv'
try:
    df = pd.read_csv(df_path, encoding='cp949')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(df_path, encoding='latin-1')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(df_path, encoding='cp1252')
        except UnicodeDecodeError:
            print("[ERROR] Could not decode file with common encodings.")
            raise

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
df = df.sort_values('Date').dropna(subset=['Date']).reset_index(drop=True)
# 'Cases_lag1', 'Cases_lag5', 'Cases_lag7', 'Cases_lag14', 'Cases_ma5', 'Cases_std5','Cases_ma7', 'Cases_std7', 'Cases_ma14', 'Cases_std14', 'Cases_diff', 'Cases_lag1_log', 'Cases_lag7_log', 'Cases_lag1_diff', 'Cases_diff2'
# 1,5,7,14일전 확진자수 생성(현재행의 바로 1,5,7,14 일 전 행 값으로 채워짐
# 이동 평균(Moving Average)과 이동 표준편차 생성 (7일/14일 단위)
for ws in [1,5,7,14]:

    df[f'Cases_lag{ws}'] = df['Cases'].shift(ws)
    # 최근 7일/14일 간의 평균 확진자 수 (데이터의 추세 파악용)
    if ws==5 or ws == 7 or ws == 14:
      # 최근 7일/14일 간의 평균 확진자 수 (데이터의 추세 파악용)
      df[f'Cases_ma{ws}'] = df['Cases'].rolling(ws).mean()
      # 최근 7일/14일 간의 확진자 수 표준편차 (데이터의 변동성 측정)
      df[f'Cases_std{ws}'] = df['Cases'].rolling(ws).std()



# 로그 변환: 확진자 수에 자연로그 적용 (큰 숫자의 스케일을 줄여 패턴 분석 용이)
# np.log1p는 log(1+x)로, 0 값이 있어도 계산 가능
df['Cases_log'] = np.log1p(df['Cases'])

# 1일 차분: 전일 대비 확진자 수 변화량 (어제 vs 오늘 증감 계산)
df['Cases_diff'] = df['Cases'].diff(1)

# 1일 전 확진자 수에 로그 변환 적용 (과거 데이터 스케일 조정)
df['Cases_lag1_log'] = np.log1p(df['Cases_lag1'])

# 7일 전 확진자 수에 로그 변환 적용 (과거 데이터 스케일 조정)
df['Cases_lag7_log'] = np.log1p(df['Cases_lag7'])

# 1일 전 확진자 수의 1일 차분 (이틀 전 vs 어제 증감 계산)
df['Cases_lag1_diff'] = df['Cases_lag1'].diff(1)

# 2차 차분: 1일 차분 데이터의 추가 차분 (변화량의 변화량, 추세 변화율 분석)
df['Cases_diff2'] = df['Cases_diff'].diff(1)


'''
차분(Diff): 전날 데이터와 현재 데이터의 차이 (예: 오늘 확진자 - 어제 확진자)

이동 평균(MA): 최근 N일 평균값으로 단기적 추세 파악

로그 변환: 숫자가 클 때 변동성 완화 (100 → 10,000 변화보다 1 → 100 변화가 더 극적임)

표준편차(Std): 데이터가 평균에서 얼마나 퍼져있는지 나타내는 지표

Lag(시차): 과거 데이터를 현재 시점으로 가져와 패턴 비교에 사용
'''

# 결측치 제거(shift/rolling 후 생긴)

print("\n=== 데이터 상위 5행 ===")
print(df.head())
print("\n=== 데이터 info ===")
print(df.info())
print("\n=== 데이터 describe ===")
print(df.describe())
df.isna().sum()
df.dropna(inplace=True)


########################################
# 1. EDA (상관관계 히트맵) - 필요한 부분만 남기고 정리
########################################

# COVID Cases와의 상관계수 시각화 (필요한 변수만 선택)
corr_with_cases = df.corr()['Cases'].drop('Cases').sort_values()
plt.figure(figsize=(10, 6))
corr_with_cases.plot(kind='barh', color=np.where(corr_with_cases >= 0, 'skyblue', 'salmon'))
plt.title("Correlation with COVID Cases")
plt.xlabel("Correlation Coefficient")
plt.grid(axis='x', linestyle='--')
plt.tight_layout()
plt.show()

def mape(y_true, y_pred):
    """MAPE 계산 함수"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

########################################
# 3. Feature Grouping (명확하게 정의)
########################################

weather = ["SO2", "CO", "O3", "NO2", "PM10", "PM25",
           "평균기온(℃)", "평균최고기온(℃)", "최고기온(℃)", "평균최저기온(℃)", "최저기온(℃)",
           "평균일강수량(mm)", "최다일강수량(mm)", "평균풍속(m/s)", "최대풍속(m/s)", "최대순간풍속(m/s)",
           "평균습도(%rh)", "최저습도(%rh)", "일조합(hr)", "일사합(MJ/m2)"]

caselag = ['Cases_lag1', 'Cases_lag5', 'Cases_lag7', 'Cases_lag14',
           'Cases_ma5', 'Cases_std5', 'Cases_ma7', 'Cases_std7', 'Cases_ma14', 'Cases_std14',
           'Cases_diff', 'Cases_lag1_log', 'Cases_lag7_log', 'Cases_lag1_diff', 'Cases_diff2']

########################################
# 4. PCA (함수로 정의하여 재사용성 높임)
########################################

def perform_pca(df, features, prefix, n_components=0.95):
    """
    주어진 데이터프레임에 PCA를 수행하는 함수

    Args:
        df (pd.DataFrame): PCA를 수행할 데이터프레임
        features (list): PCA를 적용할 feature 이름 리스트
        prefix (str): 생성될 PCA feature의 prefix (e.g., "PCA_Weather")
        n_components (float or int):  설명 분산 비율 (0~1) 또는 주성분 개수
                                       default=0.95 (95% 설명력)

    Returns:
        pd.DataFrame: PCA 결과가 추가된 데이터프레임
    """

    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)  # 비율 또는 개수 지정
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    print(f"{prefix} - Number of components: {pca.n_components_}")
    print(f"{prefix} - Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"{prefix} - Cumulative Explained Variance: {np.cumsum(pca.explained_variance_ratio_)}")

    for i in range(pca.n_components_):
        df[f"{prefix}_{i+1}"] = X_pca[:, i]

    return df

# PCA 적용 (weather와 caselag 그룹에 대해)
df = perform_pca(df, weather, "PCA_Weather")
df = perform_pca(df, caselag, "PCA_Caselag")

# 피처들의 조합에 따라 달라질 수 있으니 비교
# 해석을 위해서 의도적으로 선별
# 다 집어넣은 걸로 set up 
# 바뀌는 부분만 피처들 조합, 바뀌는 부분만 바꿔서 집어넣을 수 있겠금 set up
# 피처들을 바꿔보면서 비교분석



########################################
# Train/Val/Test 분할 (날짜 기준)
########################################
train_end_date = pd.to_datetime('2021-05-01')
val_end_date = pd.to_datetime('2021-07-31')

train_df = df[df['Date'] < train_end_date]
val_df = df[(df['Date'] >= train_end_date) & (df['Date'] < val_end_date)]
test_df = df[df['Date'] >= val_end_date]

print(f"Train={train_df.shape}, Val={val_df.shape}, Test={test_df.shape}")

########################################
# 5. 다중공선성 확인 및 처리 (VIF) - 함수화, PCA 적용 후
########################################

def calculate_vif(data):
    """VIF 계산 함수"""
    vif_df = pd.DataFrame()
    vif_df["Variable"] = data.columns
    vif_df["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_df

def handle_multicollinearity(df, features, threshold=10):
    """
    VIF 기반 다중공선성 처리 함수

    Args:
      df: DataFrame
      features: VIF를 확인할 feature 리스트
      threshold: VIF 임계값

    Returns:
      다중공선성이 처리된 DataFrame과 제거된 features 리스트
    """
    X = df[features].dropna()
    vif_df = calculate_vif(X)
    print("Initial VIF:\n", vif_df)

    high_vif_vars = vif_df[vif_df["VIF"] >= threshold]["Variable"].tolist()
    if high_vif_vars:
        print(f"\nVariables with VIF >= {threshold}: {high_vif_vars}")
        X_adjusted = X.drop(columns=high_vif_vars, errors='ignore')
        vif_adjusted = calculate_vif(X_adjusted)
        print("\nVIF after removing high VIF variables:\n", vif_adjusted)
    else:
        X_adjusted = X
        print(f"\nAll variables have VIF < {threshold}")

    return X_adjusted.columns.tolist()

# PCA 적용 후 생성된 feature들
pca_features = [col for col in df.columns if "PCA_" in col]

# VIF 확인 및 다중공선성 처리 (필요한 경우)
final_features = handle_multicollinearity(df, pca_features)

# 최종 feature set 출력
print("\nFinal Features for Modeling:", final_features)

########################################
# 6. 모델링 (RandomForest with Optuna) - 함수화
########################################

def train_random_forest(train_df, val_df, features, target='Cases'):
    """랜덤 포레스트 모델 학습 및 평가 (Optuna 최적화)"""

    X_train_rf = train_df[features]
    y_train_rf = train_df[target]
    X_val_rf = val_df[features]
    y_val_rf = val_df[target]

    def objective(trial):
        n_estimators = trial.suggest_categorical("n_estimators", [50, 100, 150])
        max_depth = trial.suggest_categorical("max_depth", [3, 5, 7, 9])
        min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [1, 3, 5])

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )

        tscv = TimeSeriesSplit(n_splits=3)
        mape_list = []  
        for train_index, val_index in tscv.split(X_train_rf):
            X_train_cv = X_train_rf.iloc[train_index]
            y_train_cv = y_train_rf.iloc[train_index]
            X_val_cv = X_train_rf.iloc[val_index]
            y_val_cv = y_train_rf.iloc[val_index]

            model.fit(X_train_cv, y_train_cv)
            preds = model.predict(X_val_cv)
            mape_val = mape(y_val_cv, preds)  # MAPE 계산
            mape_list.append(mape_val)

        return np.mean(mape_list)  # 평균 MAPE 반환

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("\n[RandomForest] Best trial:")
    print("  MAPE:", study.best_trial.value)  # MAPE 출력
    print("  Params:", study.best_trial.params)

    best_rf = RandomForestRegressor(**study.best_trial.params, random_state=42, n_jobs=-1)
    best_rf.fit(X_train_rf, y_train_rf)

    val_pred_rf = best_rf.predict(X_val_rf)
    val_mape_rf = mape(y_val_rf, val_pred_rf)  # 검증 데이터에 대한 MAPE
    print("[RF] Val MAPE:", val_mape_rf)

    return best_rf

# 모델 학습 (최종 선택된 feature 사용)
rf_model = train_random_forest(train_df, val_df, final_features)

# Feature Importance 시각화
feature_importances = pd.Series(rf_model.feature_importances_, index=final_features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='barh')
plt.title("RandomForest Feature Importance")
plt.show()


########################################
# 7. SARIMAX (with Optuna for Parameter Tuning)
########################################

def train_sarimax(train_df, test_df, features, target='Cases'):

    y_train_sar = train_df[target]
    y_test_sar = test_df[target]
    X_train_sar = train_df[features]
    X_test_sar = test_df[features]

    def objective(trial):
        p = trial.suggest_int('p', 0, 2)
        d = trial.suggest_int('d', 0, 2)
        q = trial.suggest_int('q', 0, 2)
        seasonal_p = trial.suggest_int('seasonal_p', 0, 1)
        seasonal_d = trial.suggest_int('seasonal_d', 0, 1)
        seasonal_q = trial.suggest_int('seasonal_q', 0, 1)
        seasonal_period = 7  # 주간 계절성

        try:
            sar_model = sm.tsa.statespace.SARIMAX(
                endog=y_train_sar,
                exog=X_train_sar,
                order=(p, d, q),
                seasonal_order=(seasonal_p, seasonal_d, seasonal_q, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            sar_result = sar_model.fit(disp=False)
            return sar_result.aic

        except:
            return np.inf

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)  # 50 trials

    print("\n[SARIMAX] Best trial:")
    print("  AIC:", study.best_trial.value)
    print("  Params:", study.best_trial.params)

    best_order = (study.best_trial.params['p'], study.best_trial.params['d'], study.best_trial.params['q'])
    best_seasonal_order = (study.best_trial.params['seasonal_p'],
                           study.best_trial.params['seasonal_d'],
                           study.best_trial.params['seasonal_q'],
                           7)

    best_sar_model = sm.tsa.statespace.SARIMAX(
        endog=y_train_sar,
        exog=X_train_sar,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    best_sar_result = best_sar_model.fit(disp=False)

    n_test = len(y_test_sar)
    pred_sar = best_sar_result.predict(
        start=len(y_train_sar),
        end=len(y_train_sar) + n_test - 1,
        exog=X_test_sar
    )
    sar_mape = mape(y_test_sar, pred_sar) # MAPE
    sar_rmse = np.sqrt(mean_squared_error(y_test_sar, pred_sar)) # RMSE
    print(f"[SARIMAX] MAPE: {sar_mape:.3f}, RMSE: {sar_rmse:.3f}")

    return best_sar_result, pred_sar

sarimax_result, sarimax_pred = train_sarimax(train_df, test_df, final_features) #최종 선택된 feature 사용

# SARIMAX 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(test_df['Date'], test_df['Cases'], label='Actual', marker='o')
plt.plot(test_df['Date'], sarimax_pred, label='Predicted', marker='x')
plt.xlabel("Date")
plt.ylabel("Cases")
plt.title("SARIMAX: Actual vs Predicted")
plt.legend()
plt.show()

########################################
# 8. LSTM (with Optuna for Hyperparameter Tuning)
########################################
def create_sequences(X, y, window_size):
    """시퀀스 데이터 생성 함수"""
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

def train_lstm(train_df, test_df, features, target='Cases', window_size=7):
    """LSTM 모델 학습 및 평가 (Optuna 최적화)"""
    X = train_df[features].values
    y = train_df[target].values

    # Scaling
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

    # Optuna Objective Function
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 2)
        n_units = trial.suggest_int('n_units', 32, 128)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        model = keras.Sequential()
        for i in range(n_layers):
            model.add(LSTM(n_units, return_sequences=(i < n_layers-1)))
            model.add(Dropout(dropout_rate))
        model.add(Dense(1))

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanAbsolutePercentageError())

        tscv = TimeSeriesSplit(n_splits=3)
        val_losses = []

        for train_idx, val_idx in tscv.split(X_seq):
            X_tr, X_val = X_seq[train_idx], X_seq[val_idx]
            y_tr, y_val = y_seq[train_idx], y_seq[val_idx]

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=30, batch_size=32, verbose=0, callbacks=[early_stopping])
            val_loss = history.history['val_loss'][-1]
            val_losses.append(val_loss)

        return np.mean(val_losses)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20) # 20 trials

    print("\n[LSTM] Best trial:")
    print("  Value:", study.best_trial.value)
    print("  Params:", study.best_trial.params)

    best_params = study.best_trial.params
    best_lstm = keras.Sequential()
    for i in range(best_params['n_layers']):
        best_lstm.add(LSTM(best_params['n_units'], return_sequences=(i < best_params['n_layers'] - 1)))
        best_lstm.add(Dropout(best_params['dropout_rate']))
    best_lstm.add(Dense(1))
    best_lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['lr']), loss=tf.keras.losses.MeanAbsolutePercentageError())
    best_lstm.fit(X_seq, y_seq, epochs=50, batch_size=32, verbose=0)  # Train on the full training set


    # Prepare Test Data
    X_test = test_df[features].values
    y_test = test_df[target].values
    X_test_scaled = scaler_X.transform(X_test)  # Use the same scaler
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, window_size)

    # Evaluate on test set
    y_pred_scaled = best_lstm.predict(X_test_seq).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()


    lstm_mape = mape(y_test_orig, y_pred) # MAPE
    lstm_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    print(f"[LSTM] Test MAPE: {lstm_mape:.3f}, Test RMSE: {lstm_rmse:.3f}")

    return best_lstm, y_pred, scaler_y, y_test_orig

lstm_model, lstm_pred, scaler_y_lstm, y_test_lstm = train_lstm(train_df, test_df, final_features) # 최종 feature

# LSTM 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(test_df['Date'].iloc[7:], y_test_lstm, label='Actual', marker='o')
plt.plot(test_df['Date'].iloc[7:], lstm_pred, label='Predicted', marker='x')
plt.xlabel("Date")
plt.ylabel("Cases")
plt.title("LSTM: Actual vs Predicted")
plt.legend()
plt.show()

########################################
# 9. GRU (with Optuna - LSTM과 유사)
########################################
# LSTM의 train_lstm 함수와 거의 동일, LSTM -> GRU 레이어만 변경
def train_gru(train_df, test_df, features, target='Cases', window_size=7):
    """GRU 모델 학습 및 평가 (Optuna 최적화)"""
    X = train_df[features].values
    y = train_df[target].values

    # Scaling
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

    # Optuna Objective Function
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 2)
        n_units = trial.suggest_int('n_units', 32, 128)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        model = keras.Sequential()
        for i in range(n_layers):
            model.add(GRU(n_units, return_sequences=(i < n_layers-1))) # GRU
            model.add(Dropout(dropout_rate))
        model.add(Dense(1))

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanAbsolutePercentageError())

        tscv = TimeSeriesSplit(n_splits=3)
        val_losses = []

        for train_idx, val_idx in tscv.split(X_seq):
            X_tr, X_val = X_seq[train_idx], X_seq[val_idx]
            y_tr, y_val = y_seq[train_idx], y_seq[val_idx]

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=30, batch_size=32, verbose=0, callbacks=[early_stopping])
            val_loss = history.history['val_loss'][-1]
            val_losses.append(val_loss)

        return np.mean(val_losses)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    print("\n[GRU] Best trial:")
    print("  Value:", study.best_trial.value)
    print("  Params:", study.best_trial.params)

    best_params = study.best_trial.params
    best_gru = keras.Sequential()
    for i in range(best_params['n_layers']):
        best_gru.add(GRU(best_params['n_units'], return_sequences=(i < best_params['n_layers'] - 1))) # GRU
        best_gru.add(Dropout(best_params['dropout_rate']))
    best_gru.add(Dense(1))
    best_gru.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['lr']), loss=tf.keras.losses.MeanAbsolutePercentageError())
    best_gru.fit(X_seq, y_seq, epochs=50, batch_size=32, verbose=0)  # Train on full training data

    # Prepare Test Data (using the same scalers)
    X_test = test_df[features].values
    y_test = test_df[target].values
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1,1)).flatten()
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, window_size)


    # Evaluate
    y_pred_scaled = best_gru.predict(X_test_seq).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test_seq.reshape(-1,1)).flatten()

    gru_mape = mape(y_test_orig, y_pred) # MAPE
    gru_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    print(f"[GRU] Test MAPE: {gru_mape:.3f}, Test RMSE: {gru_rmse:.3f}")

    return best_gru, y_pred, scaler_y, y_test_orig

gru_model, gru_pred, scaler_y_gru, y_test_gru = train_gru(train_df, test_df, final_features) # 최종 feature 사용

# GRU 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(test_df['Date'].iloc[7:], y_test_gru, label='Actual', marker='o')
plt.plot(test_df['Date'].iloc[7:], gru_pred, label='Predicted', marker='x')
plt.xlabel("Date")
plt.ylabel("Cases")
plt.title("GRU: Actual vs Predicted")
plt.legend()
plt.show()
########################################
# 10. Prophet (Meta)
########################################
def train_prophet(train_df, test_df, features, target='Cases'):
    prophet_df_train = train_df[['Date', target] + features].copy()
    prophet_df_train.columns = ['ds', 'y'] + features

    prophet_df_test = test_df[['Date', target] + features].copy()
    prophet_df_test.columns = ['ds', 'y'] + features

    model_prophet = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,  
        changepoint_prior_scale=0.05,
        seasonality_mode='additive'
    )

    for col in features:
        model_prophet.add_regressor(col)

    model_prophet.fit(prophet_df_train)

    future = model_prophet.make_future_dataframe(periods=len(prophet_df_test), include_history=False)
    future[features] = prophet_df_test[features].values
    forecast = model_prophet.predict(future)

    prophet_pred = forecast['yhat'].values

    prophet_mape = mape(prophet_df_test['y'], prophet_pred) # MAPE
    prophet_rmse = np.sqrt(mean_squared_error(prophet_df_test['y'], prophet_pred))

    print(f"[Prophet] MAPE: {prophet_mape:.3f}, RMSE: {prophet_rmse:.3f}")


    return model_prophet, prophet_pred

prophet_model, prophet_pred = train_prophet(train_df, test_df, final_features)

# Prophet 결과 시각화
fig1 = prophet_model.plot_components(prophet_model.predict(test_df[['Date'] + final_features].rename(columns={'Date':'ds'})))
fig2 = prophet_model.plot(prophet_model.predict(test_df[['Date'] + final_features].rename(columns={'Date':'ds'})))
plt.show()

# Actual vs Predicted 시각화
plt.figure(figsize=(12,6))
plt.plot(test_df['Date'], test_df['Cases'], label='Actual', marker='o')
plt.plot(test_df['Date'], prophet_pred, label='Predicted', marker='x')
plt.xlabel("Date")
plt.ylabel("Cases")
plt.title("Prophet: Actual vs Predicted")
plt.legend()
plt.show()

########################################
# 11. Ensemble (Stacking with XGBoost)
########################################

# Stacking에 사용할 예측 결과 준비 (시점 일치시켜야 함)
# RandomForest, SARIMAX, LSTM, GRU, Prophet
# (LSTM, GRU는 window_size 때문에 길이가 짧아짐)
min_len = min(len(test_df['Cases']), len(sarimax_pred), len(lstm_pred), len(gru_pred), len(prophet_pred))

# RandomForest 예측 결과 (test set 전체 길이)
rf_pred_test = rf_model.predict(test_df[final_features])

stack_X = np.column_stack([
    rf_pred_test[-min_len:],       # RF
    sarimax_pred[-min_len:],      # SARIMAX
    lstm_pred[-min_len:],         # LSTM
    gru_pred[-min_len:],         # GRU
    prophet_pred[-min_len:]        #Prophet
])
stack_y = test_df['Cases'].values[-min_len:]  # 실제 값

# 결측치 처리 (SimpleImputer)
stack_imputer = SimpleImputer(strategy='mean')
stack_X = stack_imputer.fit_transform(stack_X)

# Stacking 모델 (XGBoost)
stack_final = XGBRegressor(random_state=42)
stack_final.fit(stack_X, stack_y)
stack_pred = stack_final.predict(stack_X)

stack_mape = mape(stack_y, stack_pred)
stack_rmse = np.sqrt(mean_squared_error(stack_y, stack_pred))
print(f"[Stacking Ensemble] MAPE: {stack_mape:.3f}, RMSE: {stack_rmse:.3f}")

# Stacking 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(test_df['Date'][-min_len:], stack_y, label='Actual', marker='o')
plt.plot(test_df['Date'][-min_len:], stack_pred, label='Predicted', marker='x')
plt.xlabel("Date")
plt.ylabel("Cases")
plt.title("Stacking Ensemble: Actual vs Predicted")
plt.legend()
plt.show()
########################################
# 12. TCN (with Optuna)
########################################
tf.config.run_functions_eagerly(True)

def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

def train_tcn(train_df, test_df, features, target='Cases', window_size=14):
    X = train_df[features].values
    y = train_df[target].values

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

    def objective(trial):
        nb_filters = trial.suggest_int("nb_filters", 32, 128, step=32)
        kernel_size = trial.suggest_int("kernel_size", 2, 5)
        dilations = trial.suggest_categorical("dilations", [[1, 2], [1, 2, 4], [1, 2, 4, 8]])
        dropout_rate = trial.suggest_float("dropout", 0.0, 0.3, step=0.1)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        num_layers = trial.suggest_int("num_layers", 1, 2)

        base_model = keras.Sequential()  # 기본 모델 구조
        base_model.add(InputLayer(input_shape=(X_seq.shape[1], X_seq.shape[2])))
        for i in range(num_layers):
            base_model.add(
                TCN(
                    nb_filters=nb_filters,
                    kernel_size=kernel_size,
                    dilations=dilations,
                    dropout_rate=dropout_rate,
                    return_sequences=(i < num_layers - 1)
                )
            )
        base_model.add(Dense(1))


        tscv = TimeSeriesSplit(n_splits=3)
        cv_losses = []

        for train_idx, val_idx in tscv.split(X_seq):
            X_train_cv, X_val_cv = X_seq[train_idx], X_seq[val_idx]
            y_train_cv, y_val_cv = y_seq[train_idx], y_seq[val_idx]

            model_cv = keras.models.clone_model(base_model) # 모델 구조 복제

            # 매 루프마다 *새로운* optimizer 생성
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            model_cv.compile(optimizer=optimizer, loss=tf.keras.losses.MeanAbsolutePercentageError()) # 컴파일

            model_cv.fit(X_train_cv, y_train_cv, epochs=20, batch_size=32, verbose=0, shuffle=False)
            val_pred = model_cv.predict(X_val_cv)
            mse = mean_squared_error(y_val_cv, val_pred)
            cv_losses.append(mse)

        return np.mean(cv_losses)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, timeout=600)

    best_trial = study.best_trial
    best_params = best_trial.params

    print("\n[TCN] Best trial:")
    print("  Value:", best_trial.value)
    print("  Params:", best_params)

    # 최적 모델
    final_model = keras.Sequential()
    final_model.add(InputLayer(input_shape=(X_seq.shape[1], X_seq.shape[2])))
    for i in range(best_params['num_layers']):
        final_model.add(
            TCN(
                nb_filters=best_params['nb_filters'],
                kernel_size=best_params['kernel_size'],
                dilations=best_params['dilations'],
                dropout_rate=best_params['dropout'],
                return_sequences=(i < best_params['num_layers'] - 1)
            )
        )
    final_model.add(Dense(1))
    final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['lr']), loss=tf.keras.losses.MeanAbsolutePercentageError())
    final_model.fit(X_seq, y_seq, epochs=50, batch_size=32, verbose=0, shuffle=False)

    # Test 데이터 준비 (동일한 scaler 사용)
    X_test = test_df[features].values
    y_test = test_df[target].values
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1,1)).flatten()
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, window_size)

    # 평가
    y_pred_scaled = final_model.predict(X_test_seq).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test_seq.reshape(-1,1)).flatten()

    tcn_mape = mape(y_test_orig, y_pred)
    tcn_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))

    print(f"[TCN] Test MAPE: {tcn_mape:.3f}, Test RMSE: {tcn_rmse:.3f}")
    return final_model, y_pred, y_test_orig

tcn_model, tcn_pred, y_test_tcn = train_tcn(train_df, test_df, final_features)

# TCN 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(test_df['Date'].iloc[14:], y_test_tcn, label='Actual', marker='o')
plt.plot(test_df['Date'].iloc[14:], tcn_pred, label='Predicted', marker='x')
plt.xlabel("Date")
plt.ylabel("Cases")
plt.title("TCN: Actual vs Predicted")
plt.legend()
plt.show()

########################################
# 13. SHAP (RandomForest 예시 - 다른 모델은 함수 참고)
########################################

def plot_shap_values(model, X_test, model_type='RandomForest'):
    """SHAP 값을 계산하고 시각화하는 함수"""

    if model_type == 'RandomForest':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title(f"{model_type} SHAP Feature Importance (Bar)")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f"{model_type} SHAP Feature Importance (Summary)")
        plt.tight_layout()
        plt.show()

    # 다른 모델에 대한 SHAP 분석... 시간에 여유가 있다면 추가 예정.
    # elif model_type == 'LSTM':
    #   ...
    # elif model_type == 'GRU':
    #    ...
    # elif model_type == "TCN":
    #   ...
    else:
        print("Unsupported model type for SHAP analysis.")

# RandomForest SHAP values
plot_shap_values(rf_model, test_df[final_features], model_type='RandomForest')

########################################
# 14. 최종 결과 비교 (MSE, RMSE, MAE, MAPE)
########################################
# Stacking 결과 (min_len 다시 계산)

min_len = min(len(test_df['Cases']), len(sarimax_pred), len(lstm_pred), len(gru_pred), len(prophet_pred), len(tcn_pred))
rf_pred_test = rf_model.predict(test_df[final_features]) # RF는 전체 test set에 대해 예측

stack_X = np.column_stack([
    rf_pred_test[-min_len:],        # RF
    sarimax_pred[-min_len:],       # SARIMAX
    lstm_pred[-min_len:],          # LSTM
    gru_pred[-min_len:],          # GRU
    prophet_pred[-min_len:],      # Prophet
    tcn_pred[-min_len:]            # TCN
])

# 스태킹은 해석이 어렵다. 
# 해석이 중요하기 때문에 해석이 되는 모델이 중요한. 동시에 성능이 가장 좋은 모델을 선택
# 어떻게 해석할 것이냐를 봐야 하기때문에. 
# 범위를 좁힌다. 


stack_y = test_df['Cases'].values[-min_len:]

stack_imputer = SimpleImputer(strategy='mean')
stack_X = stack_imputer.fit_transform(stack_X)

stack_final = XGBRegressor(random_state=42) # Stacking model
stack_final.fit(stack_X, stack_y)
stack_pred = stack_final.predict(stack_X)

stack_mape = mape(stack_y, stack_pred)
stack_rmse = np.sqrt(mean_squared_error(stack_y, stack_pred))


# 모델별 성능 비교
model_names = ['RF', 'SARIMAX', 'LSTM', 'GRU', 'Prophet', 'TCN', 'Stacking']
mape_list = [mape(test_df['Cases'], rf_pred_test),
            mape(test_df['Cases'].iloc[:len(sarimax_pred)], sarimax_pred),
            mape(test_df['Cases'].iloc[7:], lstm_pred), # LSTM, GRU는 window_size=7
            mape(test_df['Cases'].iloc[7:], gru_pred),
            mape(test_df['Cases'], prophet_pred),
            mape(test_df['Cases'].iloc[14:], tcn_pred), # TCN window_size=14
            stack_mape]

rmse_list = [np.sqrt(mean_squared_error(test_df['Cases'], rf_pred_test)),
             np.sqrt(mean_squared_error(test_df['Cases'].iloc[:len(sarimax_pred)], sarimax_pred)),
             np.sqrt(mean_squared_error(test_df['Cases'].iloc[7:], lstm_pred)),
             np.sqrt(mean_squared_error(test_df['Cases'].iloc[7:], gru_pred)),
             np.sqrt(mean_squared_error(test_df['Cases'], prophet_pred)),
             np.sqrt(mean_squared_error(test_df['Cases'].iloc[14:], tcn_pred)),
             stack_rmse]

# MAE 비교 시각화
plt.figure(figsize=(10, 6))
plt.bar(model_names, mape_list, color='skyblue')
plt.title("Model Comparison (MAPE)")
plt.ylabel("MAE")
for i, v in enumerate(mape_list):
    plt.text(i, v + 0.1, f"{v:.2f}", ha='center', va='bottom')
plt.tight_layout()
plt.show()

# RMSE 비교 시각화
plt.figure(figsize=(10, 6))
plt.bar(model_names, rmse_list, color='salmon')
plt.title("Model Comparison (RMSE)")
plt.ylabel("RMSE")
for i, v in enumerate(rmse_list):
    plt.text(i, v + 0.1, f"{v:.2f}", ha='center', va='bottom')
plt.tight_layout()
plt.show()

print("\n=== ALL DONE ===")