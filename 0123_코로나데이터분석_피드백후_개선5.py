import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def check_stationarity_and_diff(series, alpha=0.05, max_diff=2):
    """
    [역할]
      - ADF 검정을 통해 시계열의 정상성 여부 파악
      - p-value >= alpha 이면 1차 차분을 수행 (최대 max_diff번)
      - 최종적으로 정상성이 확보된 시계열(series) 반환, 그리고 차분 횟수(diffs) 반환

    [입력]
      - series: 시계열 (pd.Series)
      - alpha: 유의수준(기본 0.05)
      - max_diff: 최대 몇 번 차분할지

    [출력]
      - (시계열, 실제 차분 횟수)
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
                ts = ts.diff(1).dropna()
                diffs += 1
            else:
                # 더 이상 차분 불가
                print(f"[ADF] p={pval:.5f}, max_diff={max_diff} 도달 -> 비정상 가능성")
                return ts, diffs

    return ts, diffs

def run_rolling_sarimax(
    csv_path="Processed_COVID_Data_Filled.csv",
    target_col="Cases",
    exog_cols=("SO2", "NO2", "평균최저기온(℃)", "최저기온(℃)"),
    log_transform=True, 
    do_diff=True,
    max_diff=2,
    order=(1,0,1),
    seasonal_order=(0,0,0,0),
    test_size=10
):
    """
    [역할]
      - CSV 로드 -> exog 변환(선택) -> 종속변수 log 변환(선택)
      - ADF 검정 & 필요 시 차분(differencing)
      - (train_n, test_n) 분할 후 Rolling SARIMAX 예측
      - 예측결과(RMSE, MAE) 출력

    [파라미터]
      - csv_path: CSV 파일 경로
      - target_col: 종속변수(예: "Cases")
      - exog_cols: 사용할 독립변수 컬럼들
      - log_transform: True면 y에 log1p 적용 -> 모델 학습 후 expm1으로 역변환
      - do_diff: True면 ADF p>=0.05 시 차분(d=1) 수행
      - max_diff: 최대 몇 번 차분
      - order, seasonal_order: SARIMAX 모수
      - test_size: 뒤쪽 test_size개 예측(rolling)

    [출력]
      - df_res( "actual", "pred" )
      - RMSE, MAE
    """

    # 1) CSV 로드
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date", encoding='ANSI')
    df.sort_index(inplace=True)

    # 2) 종속변수 y
    y_raw = df[target_col].astype(float)

    # (선택) 로그 변환
    if log_transform:
        # log( x+1 )
        y_proc = np.log1p(y_raw)
    else:
        y_proc = y_raw.copy()

    # 3) exog
    df_exog = df[list(exog_cols)].copy()

    # (선택) 예: SO2, NO2 로그 변환
    #   필요 시 다른 변수도 변환 가능
    df_exog["SO2"] = np.log1p(df_exog["SO2"])
    df_exog["NO2"] = np.log1p(df_exog["NO2"])
    # ... 최저기온 등은 여기선 그대로 사용

    # 4) (옵션) ADF & 차분
    if do_diff:
        print("\n=== [ADF & 차분] ===")
        y_sta, d_used = check_stationarity_and_diff(y_proc, alpha=0.05, max_diff=max_diff)
    else:
        y_sta = y_proc
        d_used = 0

    # exog도 동일 차분(선택)
    #  -> 보통 target을 d번 차분했다면, exog도 동일 d번 차분
    #  -> ex) df_exog = df_exog.diff(d_used).dropna()
    # 여기서는 예시로 차분하지 않음.

    # 5) Train/Test 분할
    n = len(y_sta)
    train_n = n - test_size
    y_train, y_test = y_sta.iloc[:train_n], y_sta.iloc[train_n:]
    X_train, X_test = df_exog.iloc[:train_n], df_exog.iloc[train_n:]

    

    # 6) SARIMAX 초기 피팅
    print(f"\n=== [Initial SARIMAX Fit] order={order}, seas={seasonal_order}")
    print(X_train.head())
    print(y_train.head())
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
    # current_res = res_fit
    x_row = X_test.iloc[[i]]
    x_row.index = [this_date] # exog 인덱스를 예측 시점과 동일하게
    current_res = current_res.append(endog=[y_true_val], exog=x_row, refit=False)

    for i in range(len(y_test)):
        y_true_val = y_test.iloc[i]
        this_date  = y_test.index[i]

        x_row = X_test.iloc[[i]]
        x_row.index = [this_date]  # exog 인덱스를 예측 시점과 동일하게

        fc_val = current_res.predict(
            start=current_res.nobs,
            end=current_res.nobs,
            exog=x_row
        )
        y_pred_val = fc_val.iloc[0]

        # log -> 원본스케일 역변환
        if log_transform:
            pred_ = np.expm1(y_pred_val)  # exp(x)-1
            actual_= np.expm1(y_true_val)
        else:
            pred_ = y_pred_val
            actual_= y_true_val

        preds.append(pred_)
        actuals.append(actual_)

        # 상태 업데이트(append)
        current_res = current_res.append(endog=[y_true_val], exog=x_row, refit=False)

    df_res = pd.DataFrame({"actual": actuals, "pred": preds}, index=y_test.index)

    # 8) RMSE, MAE
    rmse_ = sqrt(mean_squared_error(df_res["actual"], df_res["pred"]))
    mae_  = mean_absolute_error(df_res["actual"], df_res["pred"])
    print("\n=== [Result] Rolling SARIMAX ===")
    print(df_res)
    print(f"RMSE={rmse_:.4f}, MAE={mae_:.4f}")

    return df_res, current_res

if __name__=="__main__":
    # 예시 실행
    df_result, final_model = run_rolling_sarimax(
        csv_path="Processed_COVID_Data_Filled.csv",
        target_col="Cases",
        exog_cols=["SO2","NO2","평균최저기온(℃)","최저기온(℃)"],
        log_transform=True,   # 종속변수 log1p 변환
        do_diff=True,         # ADF 비정상이면 차분 시도
        max_diff=2,           # 최대 2번 차분
        order=(1,0,1),
        seasonal_order=(0,0,0,0),
        test_size=10
    )
