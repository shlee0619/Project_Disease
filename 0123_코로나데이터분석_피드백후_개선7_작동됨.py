import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # 불필요한 경고 숨기기(optional)

def check_stationarity_and_diff(series, alpha=0.05, max_diff=2):
    """
    [역할]
      - ADF 검정을 통해 시계열의 정상성 여부 파악
      - p-value >= alpha 이면 1차 차분을 수행 (최대 max_diff번)
      - 최종적으로 정상성이 확보된 시계열(series) 반환, 그리고 차분 횟수(diffs) 반환
    """
    ts = series.copy()
    diffs = 0

    for d in range(max_diff + 1):
        test_res = adfuller(ts.dropna(), autolag="AIC")
        pval = test_res[1]
        if pval < alpha:
            print(f"[ADF] p={pval:.5f}, d={diffs} -> 정상성 확보")
            return ts, diffs
        else:
            if diffs < max_diff:
                print(f"[ADF] p={pval:.5f} (>= {alpha}), 차분 시도 -> d={diffs+1}")
                ts = ts.diff().dropna()
                diffs += 1
            else:
                print(f"[ADF] p={pval:.5f}, max_diff={max_diff} 도달 -> 비정상 가능성")
                return ts, diffs

    return ts, diffs


def remove_outliers_robust(series, z_threshold=3.0):
    """
    [역할]
      - 단순한 outlier 제거(혹은 clip) 예시
      - z_threshold 이상이면 이상치로 간주해 중간값(median) 또는 np.nan 처리 가능
    [입력]
      - series: pd.Series
      - z_threshold: 몇 시그마 밖을 이상치로 볼지
    [출력]
      - outlier 제거 or 대체된 series
    """
    s = series.copy()
    mean_ = s.mean()
    std_ = s.std(ddof=1)
    zscore = (s - mean_) / (std_ if std_ != 0 else 1e-6)
    
    # 예: 이상치를 중간값으로 치환
    median_ = s.median()
    outlier_mask = abs(zscore) > z_threshold
    
    s[outlier_mask] = median_  # 중간값으로 대체
    return s


def run_rolling_sarimax(
    csv_path="Processed_COVID_Data_Filled.csv",
    target_col="Cases",
    exog_cols=("SO2", "NO2", "평균최저기온(℃)", "최저기온(℃)"),
    log_transform=True,
    do_diff=True,
    exog_diff=True,           # (1) 외생변수도 차분할지 여부 추가
    max_diff=2,
    order=(1,0,1),
    # (2) 계절성 고려: m=7 (1주), seasonal_order=(P,D,Q,m)
    # 필요 시 (0,0,0,0) -> (0,0,0,7) 등으로 변경
    seasonal_order=(0,0,0,0),
    test_size=10,
    remove_outlier=False      # (4) 잔차 정규성 개선을 위한 이상치 처리
):
    """
    [역할]
      - CSV 로드 -> 필요 시 이상치 처리 -> 종속변수 로그 + 차분
      - 외생변수도 (옵션) 차분
      - 계절성 파라미터(seasonal_order) 설정 가능
      - Rolling SARIMAX 예측
      - 잔차에 대해 간단한 이분산(ARCH) 테스트 예시

    [출력]
      - df_res: (actual, pred)
      - SARIMAX 마지막 모델
    """

    # 1) CSV 로드
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date", encoding='ANSI')
    df.sort_index(inplace=True)

    # 2) 종속변수 y
    y_raw = df[target_col].astype(float)

    # (opt) 이상치 처리
    if remove_outlier:
        print("[Info] 종속변수 outlier 제거(로버스트)")
        y_raw = remove_outliers_robust(y_raw)

    # (a) 로그 변환
    if log_transform:
        y_proc = np.log1p(y_raw)
    else:
        y_proc = y_raw.copy()

    # 3) exog
    df_exog = df[list(exog_cols)].copy()
    
    # (opt) 이상치 처리(외생변수 쪽도)
    if remove_outlier:
        for c in df_exog.columns:
            df_exog[c] = remove_outliers_robust(df_exog[c])
    
    # 예: SO2, NO2 로그 변환
    df_exog["SO2"] = np.log1p(df_exog["SO2"])
    df_exog["NO2"] = np.log1p(df_exog["NO2"])
    # 필요 시 기온 등도 변환 가능 (여기선 생략)

    # 4) ADF & 차분
    if do_diff:
        print("\n=== [ADF & 차분] ===")
        y_sta, d_used = check_stationarity_and_diff(y_proc, alpha=0.05, max_diff=max_diff)
    else:
        y_sta = y_proc
        d_used = 0

    # (1) 외생변수도 동일 차분(옵션)
    if exog_diff and d_used > 0:
        # d_used번 만큼 차분
        for _ in range(d_used):
            df_exog = df_exog.diff().dropna()

    # y_sta, exog 인덱스 동기화
    #  -> 차분 후 인덱스가 달라졌을 수 있으므로 reindex 교차
    common_idx = y_sta.index.intersection(df_exog.index)
    y_sta = y_sta.loc[common_idx].dropna()
    df_exog = df_exog.loc[common_idx].dropna()

    # 5) Train/Test 분할
    n = len(y_sta)
    train_n = n - test_size
    y_train = y_sta.iloc[:train_n]
    y_test  = y_sta.iloc[train_n:]

    X_train = df_exog.iloc[:train_n]
    X_test  = df_exog.iloc[train_n:]

    print(f"\nTrain_n={len(y_train)}, Test_n={len(y_test)}")
    if len(y_train) > 0:
        print(f"y_train.index[-1]: {y_train.index[-1]}")
    if len(y_test) > 0:
        print(f"y_test.index[0]: {y_test.index[0]}")

    # 6) SARIMAX 초기 피팅
    print(f"\n=== [Initial SARIMAX Fit] order={order}, seas={seasonal_order}")
    mod = sm.tsa.statespace.SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res_fit = mod.fit(disp=False)
    print(res_fit.summary())

    # 7) Rolling 예측
    preds = []
    actuals = []
    current_res = res_fit

    for i in range(len(y_test)):
        y_true_val = y_test.iloc[i]
        this_date  = y_test.index[i]

        # exog 한 행
        x_row = X_test.iloc[[i]].copy()
        x_row.index = [this_date]

        fc_val = current_res.predict(
            start=current_res.nobs,
            end=current_res.nobs,
            exog=x_row
        )
        y_pred_val = fc_val.iloc[0]

        # (b) 역변환
        if log_transform:
            pred_ = np.expm1(y_pred_val)
            actual_ = np.expm1(y_true_val)
        else:
            pred_ = y_pred_val
            actual_ = y_true_val

        preds.append(pred_)
        actuals.append(actual_)

        # (c) 모델 업데이트
        current_res = current_res.append(endog=[y_true_val], exog=x_row, refit=False)

    df_res = pd.DataFrame({"actual": actuals, "pred": preds}, index=y_test.index)

    # 8) RMSE, MAE
    rmse_ = sqrt(mean_squared_error(df_res["actual"], df_res["pred"]))
    mae_  = mean_absolute_error(df_res["actual"], df_res["pred"])
    print("\n=== [Result] Rolling SARIMAX ===")
    print(df_res)
    print(f"RMSE={rmse_:.4f}, MAE={mae_:.4f}")

    # (3) 간단한 ARCH Test (이분산성) 예시
    #     잔차(=resid)에서 ARCH 효과가 있는지 Lagrange multiplier test
    resid = current_res.resid
    # 첫 몇 개는 diff로 인해 NaN이 있을 수 있으니 dropna
    resid = resid.dropna()
    arch_test = sm.stats.diagnostic.het_arch(resid)
    # arch_test => (LM stat, LM p-value, F stat, F p-value)
    print("\n=== [ARCH test] ===")
    print(f"LM stat={arch_test[0]:.4f}, p={arch_test[1]:.4f}")

    # 필요 시 plot_acf(resid**2) 등으로 추가 시각화 가능
    # plot_acf((resid**2).dropna(), title="ACF of squared residuals")
    # plt.show()

    return df_res, current_res

if __name__=="__main__":
    """
    6. 요약 및 개선 방안 반영:
      1) 외생변수도 차분 (exog_diff=True)
      2) 계절성 -> seasonal_order=(0,0,0,7) 등 가능
      3) 이분산(arch) test 예시
      4) 잔차 정규성 개선 위해 로버스트 outlier 처리 (remove_outlier=True)
      ...
    """
    df_result, final_model = run_rolling_sarimax(
        csv_path="Processed_COVID_Data_Filled.csv",
        target_col="Cases",
        exog_cols=["SO2","NO2","평균최저기온(℃)","최저기온(℃)"],
        log_transform=True,
        do_diff=True,
        exog_diff=True,       # 외생변수도 같이 차분
        remove_outlier=False, # 이상치 제거 예시
        max_diff=2,
        # (2) 간단히 계절성 weekly => seasonal_order=(0,0,0,7)
        seasonal_order=(0,0,0,0),  
        order=(1,0,1),
        test_size=10
    )


'''
=== [ADF & 차분] ===
[ADF] p=0.36244 (>= 0.05), 차분 시도 -> d=1
[ADF] p=0.00000, d=1 -> 정상성 확보

Train_n=689, Test_n=10
y_train.index[-1]: 2021-12-21 00:00:00
y_test.index[0]: 2021-12-22 00:00:00

=== [Initial SARIMAX Fit] order=(1, 0, 1), seas=(0, 0, 0, 0)
                               SARIMAX Results
==============================================================================
Dep. Variable:                  Cases   No. Observations:                  689
Model:               SARIMAX(1, 0, 1)   Log Likelihood                -430.224
Date:                Thu, 23 Jan 2025   AIC                            874.449
Time:                        15:41:45   BIC                            906.175
Sample:                    02-02-2020   HQIC                           886.723
                         - 12-21-2021
Covariance Type:                  opg
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
SO2          -66.8402    114.275     -0.585      0.559    -290.815     157.135
NO2            8.6356      6.075      1.421      0.155      -3.271      20.543
평균최저기온(℃)      0.0064      0.017      0.367      0.714      -0.028       0.041
최저기온(℃)       -0.0142      0.013     -1.106      0.269      -0.039       0.011
ar.L1          0.5549      0.033     16.770      0.000       0.490       0.620
ma.L1         -0.8804      0.023    -38.874      0.000      -0.925      -0.836
sigma2         0.2044      0.006     33.237      0.000       0.192       0.216
===================================================================================
Ljung-Box (L1) (Q):                   1.49   Jarque-Bera (JB):               759.58
Prob(Q):                              0.22   Prob(JB):                         0.00
Heteroskedasticity (H):               2.69   Skew:                             0.33
Prob(H) (two-sided):                  0.00   Kurtosis:                         8.11
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).

=== [Result] Rolling SARIMAX ===
              actual      pred
Date
2021-12-22  0.151515 -0.030413
2021-12-23 -0.236842 -0.060736
2021-12-24  0.034483 -0.019301
2021-12-25  0.000000 -0.046172
2021-12-26 -0.033333  0.011988
2021-12-27 -0.068966  0.038771
2021-12-28 -0.037037  0.107841
2021-12-29 -0.038462 -0.057176
2021-12-30  0.000000 -0.000830
2021-12-31  0.120000  0.034531
RMSE=0.1056, MAE=0.0861

=== [ARCH test] ===
LM stat=56.5425, p=0.0000
'''