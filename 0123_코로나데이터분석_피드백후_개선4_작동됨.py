# 예시 코드 (핵심만 발췌)

import numpy as np
import pandas as pd
import statsmodels.api as sm



# 1) CSV 로드
df = pd.read_csv("Processed_COVID_Data_Filled.csv", parse_dates=["Date"], index_col="Date", encoding='ANSI')

# 2) 종속변수(log 변환)
df["Cases_log"] = np.log1p(df["Cases"])

# 3) 유의하다고 본 변수를 exog로 선택
#    (SO2, NO2, 평균최저기온(℃), 최저기온(℃))
use_cols = ["SO2", "NO2", "평균최저기온(℃)", "최저기온(℃)"]


df_exog = df[use_cols].copy()

# (선택) 만약 exog도 log 변환하고 싶다면:
df_exog["SO2"] = np.log1p(df_exog["SO2"])
df_exog["NO2"] = np.log1p(df_exog["NO2"])
# df_exog["평균최저기온(℃)"] = np.log1p(df_exog["평균최저기온(℃)"])
# df_exog["최저기온(℃)"] = np.log1p(df_exog["최저기온(℃)"])
# etc...

# 4) 정상성 체크 후, Rolling SARIMAX
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# (a) ADF -> Cases_log
from statsmodels.tsa.stattools import adfuller
res = adfuller(df["Cases_log"].dropna(), autolag="AIC")
print("ADF p-value =", res[1])




# 필요하다면 차분




# (b) Train/Test 분할
test_size = 10
y_all = df["Cases_log"]
X_all = df_exog
train_n = len(y_all) - test_size

y_train, y_test = y_all.iloc[:train_n], y_all.iloc[train_n:]
X_train, X_test = X_all.iloc[:train_n], X_all.iloc[train_n:]

# (c) 초기 모델 피팅
mod = sm.tsa.statespace.SARIMAX(
    y_train,
    exog=X_train,
    order=(1,0,1),  # 예: (1,0,1)
    seasonal_order=(0,0,0,0),
    enforce_stationarity=False,
    enforce_invertibility=False
)
res_fit = mod.fit(disp=False)
print(res_fit.summary())

# (d) rolling forecast
preds = []
acts = []
current_res = res_fit

for i in range(len(y_test)):
    y_true = y_test.iloc[i]
    x_row = X_test.iloc[[i]]

    fc_val = current_res.predict(
        start=current_res.nobs,
        end=current_res.nobs,
        exog=x_row
    )
    y_pred_log = fc_val.iloc[0]

    preds.append(np.expm1(y_pred_log))  # log -> 원래 스케일
    acts.append(np.expm1(y_true))

    # 업데이트
    current_res = current_res.append(endog=[y_true], exog=x_row, refit=False)

df_res = pd.DataFrame({"actual": acts, "pred": preds}, index=y_test.index)

rmse_ = sqrt(mean_squared_error(df_res["actual"], df_res["pred"]))
mae_  = mean_absolute_error(df_res["actual"], df_res["pred"])
print("RMSE=", rmse_, "MAE=", mae_)
print(df_res.head(len(df_res)))
