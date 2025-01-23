###################################################
# 라이브러리 Import
###################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error


###################################################
# 1. 데이터 로드 & 간단 전처리
###################################################
def load_and_preprocess_data(filepath: str, encoding='ANSI') -> pd.DataFrame:
    """
    [역할]
      - CSV 파일에서 데이터를 불러온 뒤,
      - 'Date' 열을 datetime으로 변환,
      - 결측치(빈 칸)을 median으로 대체,
      - 'Date' 열을 인덱스로 설정.
    """
    df = pd.read_csv(filepath, encoding=encoding)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    # 결측치는 중간값(median)으로 대체
    df.fillna(df.median(numeric_only=True), inplace=True)

    print(f"[Load] CSV loaded: shape={df.shape}")
    return df


###################################################
# 2. ADF(정상성) 검정 + (옵션) 차분 + ACF/PACF
###################################################
def check_stationarity_and_plots(ts, max_lag=30, alpha=0.05, do_diff=True):
    """
    [역할]
      - ADF(정상성) 테스트
      - p-value < alpha 이면 정상성 만족.
      - 아니면 1차 차분(옵션)에 재도전.
      - ACF/PACF 플롯 시각화

    [리턴]
      (시계열(Series), 차분실행여부(bool))
    """
    # 1) ADF 테스트
    print("\n=== ADF Test ===")
    adf_res = adfuller(ts.dropna(), autolag='AIC')
    adf_stat, pval, usedlag, nobs, crit_vals, icbest = adf_res
    print(f"ADF Statistic = {adf_stat:.4f}, p-value = {pval:.6f}")
    print("Used lag:", usedlag, ", Nobs:", nobs)
    print("Critical Values:", crit_vals)

    # 2) p-value < alpha → 정상성
    if pval < alpha:
        print(f"=> 유의수준 alpha={alpha}에서 정상성으로 판정!\n")
        diff_needed = False
        ts_out = ts
    else:
        print(f"=> 비정상. p={pval:.3f} >= alpha={alpha}")
        diff_needed = True

    # 3) 필요 시 1차 차분
    did_diff = False
    if diff_needed and do_diff:
        print("[Info] 1차 차분 시도..")
        ts_diff = ts.diff(1).dropna()
        adf_res2 = adfuller(ts_diff.dropna(), autolag='AIC')
        pval2 = adf_res2[1]
        if pval2 < alpha:
            print(f"=> 1차 차분 후 정상성 획득(p={pval2:.6f})\n")
            ts_out = ts_diff
            did_diff = True
        else:
            print(f"=> 1차 차분 후에도 비정상(p={pval2:.6f}), 추가 차분 고려필요.\n")
            ts_out = ts_diff
            did_diff = True
    else:
        ts_out = ts

    # 4) ACF/PACF Plot
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    plot_acf(ts_out.dropna(), lags=max_lag, ax=axes[0], title="ACF")
    plot_pacf(ts_out.dropna(), lags=max_lag, ax=axes[1], title="PACF")
    plt.show()

    return ts_out, did_diff


###################################################
# 3. SARIMAX(ARIMAX) 모델 - Rolling Forecast
###################################################
def rolling_sarimax_forecast(
    endog: pd.Series,
    exog: pd.DataFrame = None,
    order=(1,0,1),
    seasonal_order=(0,0,0,0),
    test_size=10
):
    """
    Statsmodels SARIMAX 롤링(rolling) 예측 (ARIMAX).
    test_size만큼 한 시점씩 예측 후, 실제값을 .append()로 업데이트.

    [핵심 수정 사항]
      - exog를 get_forecast(steps=1, exog=...)에 넘길 때,
        예측할 날짜를 인덱스로 하는 (1, n_features)짜리 DataFrame으로 만들어줌.
    """
    # 1) train/test 분할
    n = len(endog)
    train_n = n - test_size
    train_endog = endog.iloc[:train_n]
    test_endog  = endog.iloc[train_n:]
    print(train_endog)
    print(test_endog)


    if exog is not None:
        train_exog = exog.iloc[:train_n, :]
        test_exog  = exog.iloc[train_n:, :]
    else:
        train_exog = None
        test_exog  = None

    print(f"\n[Rolling SARIMAX] train_n={train_n}, test_n={len(test_endog)}")
    print(f"order={order}, seasonal_order={seasonal_order}")

    # 2) 초기 모델 적합
    model = sm.tsa.statespace.SARIMAX(
        endog=train_endog,
        exog=train_exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    print("\n[SARIMAX] Initial train fit done.")
    print(results.summary())

    preds = []
    actuals = []
    current_results = results


    # 3) 테스트 구간 롤링 예측
    for i in range(len(test_endog)):
        # 이번 시점에 해당하는 날짜와 실제값
        this_date = test_endog.index[i]
        y_true = test_endog.iloc[i]

        if test_exog is not None:
            # ※ exog는 (1 x n_features) DataFrame 형태여야 하며,
            #    인덱스가 예측할 시점(this_date)과 동일해야 함!
            x_row = test_exog.iloc[i:i+1]            # shape (1, n_features)
            x_row.index = [this_date]                # 인덱스를 예측할 날짜로
        else:
            x_row = None

        # (A) i 시점 예측 (1-step)
        #     out_of_sample → exog= x_row
        forecast_res = current_results.get_forecast(steps=1, exogenous=x_row)
        y_pred = forecast_res.predicted_mean.iloc[0]

        # (B) 결과 저장
        preds.append(y_pred)
        actuals.append(y_true)

        # (C) 실제 관측값 + exog 로 모델 상태 업데이트
        #     endog=[y_true], exog=x_row
        current_results = current_results.append(
            endog=[y_true],
            exog=x_row,
            refit=False
        )

    # 4) 결과 취합
    df_res = pd.DataFrame({
        'actual': actuals,
        'pred': preds
    }, index=test_endog.index)

    rmse_ = sqrt(mean_squared_error(df_res['actual'], df_res['pred']))
    mae_  = mean_absolute_error(df_res['actual'], df_res['pred'])

    print("\n=== [Test Result: Rolling Forecast] ===")
    print(df_res)
    print(f"Test RMSE={rmse_:.4f}, MAE={mae_:.4f}")

    # 5) 시각화
    plt.figure(figsize=(8,4))
    plt.plot(train_endog.index, train_endog, marker='o', label='Train')
    plt.plot(test_endog.index, test_endog, marker='o', label='Test Actual')
    plt.plot(df_res.index, df_res['pred'], marker='x', label='Test Pred')
    plt.title(f"SARIMAX Rolling - RMSE={rmse_:.4f}, MAE={mae_:.4f}")
    plt.legend()
    plt.grid()
    plt.show()

    # 6) 잔차 분석 (초기 훈련 구간 기준)
    residuals = results.resid
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    sm.qqplot(residuals, line='45', ax=axes[0])
    axes[0].set_title("Q-Q Plot of Residuals(Train)")

    sm.graphics.tsa.plot_acf(residuals.dropna(), lags=20, ax=axes[1], title="ACF(Train Residuals)")
    plt.show()

    return df_res, current_results

###################################################
# 4. 최종 실행 예시
###################################################
if __name__=="__main__":
    # (A) CSV 로드
    csv_file = "Processed_COVID_Data_Filled.csv"
    df = load_and_preprocess_data(csv_file)

    # (B) 타겟 시계열, 예: 'Cases'
    y_col = 'Cases'
    ts = df[y_col].astype(float)

    # (C) 외생변수(exog) 목록 (예: 기상 + 대기오염 등)
    exog_cols = [
        'SO2','CO','O3','NO2','PM10','PM25',
        '평균기온(℃)','평균습도(%rh)'
        # ... 필요하면 더
    ]
    exog = df[exog_cols].astype(float)

    # (D) 정상성 확인 + (필요시 차분)
    ts_sta, did_diff = check_stationarity_and_plots(ts, max_lag=20, alpha=0.05, do_diff=True)
    # 참고: exog도 필요하면 차분이나 변환 가능

    # (E) rolling 사리맥스
    #     예: order=(1,1,1) or (p,d,q) 보고 결정
    #     (교수님 피드백: ACF/PACF 보고 p,q 정)
    #     ex) p=1, d=1(이미 차분했으면 0), q=1
    order_ = (1, 0 if did_diff else 1, 1)
    # (seasonal_order_=...) 계절성 있으면 설정

    df_res, final_model = rolling_sarimax_forecast(
        endog=ts_sta,
        exog=exog if exog is not None else None,
        order=order_,
        seasonal_order=(0,0,0,0),  # 계절성 없다고 가정
        test_size=10
    )

    print("\n[Done] Rolling SARIMAX ARIMAX example complete.\n")
