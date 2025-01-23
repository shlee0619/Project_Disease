###################################################
# 라이브러리 Import
###################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

import optuna
import keras_tuner as kt
import pmdarima as pm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from prophet import Prophet

###################################################
# 1. 데이터 로드 및 전처리
###################################################
def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    1) CSV 파일에서 데이터 로드(인코딩: ANSI 가정)
    2) 'Date' 열을 datetime으로 변환
    3) 정렬 + 'Date'를 인덱스로
    4) 결측치는 median으로 대체
    """
    df = pd.read_csv(filepath, encoding='ANSI')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    print("\n[Load] CSV loaded. shape=", df.shape)
    return df


###################################################
# 2. ADF(정상성) 검정 + ACF/PACF 분석 함수
###################################################
def check_stationarity_and_plots(ts, max_lag=30, alpha=0.05, diff_enabled=True):
    """
    [역할]
      - ADFuller(ADF) 테스트로 시계열 정상성 검사
      - ACF, PACF 그래프를 시각화
      - 필요하면 차분(differencing) 수행

    [입력]
      - ts: 시계열(Pandas Series)
      - max_lag: ACF/PACF 에 표시할 최대 래그(기본30)
      - alpha: ADF Test 유의수준(기본 0.05)
      - diff_enabled: True면, 정상성이 없으면 1차 차분

    [출력]
      - (ts_out, differ_done) : (정상성 확보된 시계열, 차분 여부)
    """
    print("\n===== [Step] ADF(정상성) 테스트 =====")
    # 1) ADF 테스트
    result = adfuller(ts.dropna(), autolag='AIC')
    adf_stat = result[0]
    p_value  = result[1]
    usedlag  = result[2]
    nobs     = result[3]
    crit_vals= result[4]
    print(f"ADF Statistic: {adf_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print("Used lag:", usedlag, ", Observations:", nobs)
    print("Critical Values:", crit_vals)

    # 2) p-value < alpha → 정상성 가정 가능
    if p_value < alpha:
        print("=> [결론] 유의수준 알파=%.2f 에서, 정상성(Stationary)으로 판정!\n" % alpha)
        differ_needed = False
        ts_out = ts
    else:
        print("=> [결론] 유의수준 알파=%.2f 에서, 비정상(Non-Stationary)으로 판정." % alpha)
        differ_needed = True

    # 3) 만약 비정상이면서 diff_enabled=True이면 1차 차분 수행
    if (differ_needed) and (diff_enabled):
        print("\n[Info] 1차 차분(Differencing) 시도.")
        ts_out = ts.diff(1).dropna()
        # 차분 후, 다시 ADF
        result2 = adfuller(ts_out.dropna(), autolag='AIC')
        p_value2 = result2[1]
        print(f"[재검정] ADF p-value(after 1-diff) = {p_value2:.6f}")
        if p_value2 < alpha:
            print("=> 차분 후 정상성 획득!\n")
        else:
            print("=> 차분 후에도 비정상... 2차 차분 등 추가 고려 필요.\n")
        differ_done = True
    else:
        ts_out = ts
        differ_done = False

    # 4) ACF/PACF 시각화
    print("\n===== [Step] ACF/PACF Plot =====")
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    sm.graphics.tsa.plot_acf(ts_out.dropna(), lags=max_lag, ax=axes[0], title="ACF")
    sm.graphics.tsa.plot_pacf(ts_out.dropna(), lags=max_lag, ax=axes[1], title="PACF")
    plt.show()

    return ts_out, differ_done


###################################################
# 3. ARIMAX (pmdarima) - 외생변수 사용 예시
###################################################
def train_arimax_auto(ts, exog=None, d=0, max_p=3, max_q=3, seasonal=False):
    """
    [역할]
      - pmdarima의 AutoARIMA로 ARIMAX 모델 적합
      - d: 차분 차수
      - seasonal: 계절모델 여부
    """
    import pmdarima as pm

    print("\n===== [ARIMAX Fit] =====")
    model = pm.AutoARIMA(
        start_p=0, start_q=0,
        max_p=max_p, max_q=max_q,
        d=d,
        seasonal=seasonal,
        stepwise=True,
        suppress_warnings=True
    )
    # fit
    model.fit(y=ts.values, exogenous=exog.values if exog is not None else None)

    # 수정된 부분:
    # AutoARIMA 객체에서 실제 ARIMA 모델은 model.model_ 에 들어있는 경우가 많음
    arima_order = model.model_.order          # (p, d, q)
    seasonal_order = model.model_.seasonal_order   # (P, D, Q, m) if seasonal=True

    print("=> Best ARIMAX order:", arima_order)
    if seasonal:
        print("=> Best ARIMAX seasonal_order:", seasonal_order)

    # summary()로도 정보 확인 가능
    print(model.summary())

    return model


def forecast_arimax(model, steps=1, exog_future=None):
    """
    ARIMAX 모델로 forecast(미래 예측)
    구버전 pmdarima 사용 시엔 'exogenous' 대신 X 인자 필요
    """
    # 버전 확인
    import pmdarima
    print("[INFO] pmdarima version:", pmdarima.__version__)

    # 최신 버전 >=2.0.0이라면
    try:
        fc = model.predict(
            n_periods=steps,
            X=exog_future,  # 구버전에서는 'X=exog_future'로 사용
            return_conf_int=False
        )
    except TypeError:
        # 만약 위 코드에서 TypeError 발생 시, forecast()로 대체
        fc = model.forecast(
            steps=steps,
            X=exog_future,
            return_conf_int=False
        )

    return fc



###################################################
# 4. 최종 예시: ARIMAX 워크플로우
###################################################
def run_arimax_process(
    df,
    y_col='Cases',
    exog_cols=['Temperature'],  # 예: 날씨 등 외생변수
    test_size=10,
    adf_alpha=0.05
):
    """
    [역할]
      - (1) 타겟에 대해 ADF, ACF/PACF 분석
      - (2) 비정상이면 1차 차분
      - (3) ARIMAX( exog ) 모델 학습 + 예측
      - (4) 잔차 분석 등
    """
    print(f"\n===== [ARIMAX 전체 프로세스] Target={y_col} =====")

    # 1) 시계열 추출
    ts = df[y_col].astype(float)
    # 2) exog
    if exog_cols is not None and len(exog_cols)>0:
        exog = df[exog_cols].astype(float)
    else:
        exog = None

    # 3) ADF & ACF/PACF
    ts_sta, diff_done = check_stationarity_and_plots(ts, max_lag=20, alpha=adf_alpha)

    # 4) (option) exog와 길이 맞추기 위해 동일 차분 → 여기서는 간단히 exog는 차분안함
    #     필요하다면 exog도 diff
    #     if diff_done:
    #         exog = exog.diff(1).dropna()

    # 5) train/test 나누기
    n = len(ts_sta)
    train_n = n - test_size
    train_ts = ts_sta.iloc[:train_n]
    test_ts  = ts_sta.iloc[train_n:]
    if exog is not None:
        train_exog = exog.iloc[:train_n, :]
        test_exog  = exog.iloc[train_n:, :]
    else:
        train_exog = None
        test_exog  = None

    print("\n=== Train/Test 분할 ===")
    print(f"train_n={train_n}, test_n={len(test_ts)}")

    # 6) ARIMAX 모델 학습
    #    (d=0 if diff_done=True, else d=0 or 1)
    #    여기서는 diff_done이면 d=0, 아니면 d=0 or 1로 시도
    d_ = 0 if diff_done else 0  # or 1
    model_arimax = train_arimax_auto(train_ts, train_exog, d=d_)

    # 7) 예측(테스트 구간)
    preds = []
    actuals = []
    for i in range(len(test_ts)):
        # i 시점 예측
        # - exog_future = test_exog.iloc[i] shape( exog_features, )
        #   reshape(1, -1) → (1, exog_features)
        if test_exog is not None:
            exog_f = test_exog.iloc[i].values.reshape(1, -1)
        else:
            exog_f = None

        fc = forecast_arimax(model_arimax, steps=1, exog_future=exog_f)
        preds.append(fc)
        # 실제값
        actuals.append(test_ts.iloc[i])

        # 한 스텝 업데이트
        #   pmdarima 모델은 update 기능이 제한적이므로, partial_fit이 없음
        #   AutoARIMA 일괄 학습 -> 이후에는 predict만
        #   필요 시 Statsmodels의 ARIMAResults.append() 등을 고려

    # 8) 예측 결과 시각화
    df_res = pd.DataFrame({
        'actual': actuals,
        'pred': preds
    }, index=test_ts.index)

    print("\n=== [결과: 예측 vs 실제] (Test 구간) ===")
    print(df_res.head(len(df_res)))

    # RMSE, MAE
    rmse_val = sqrt(mean_squared_error(df_res['actual'], df_res['pred']))
    mae_val  = mean_absolute_error(df_res['actual'], df_res['pred'])
    print(f"\nTest RMSE={rmse_val:.4f}, MAE={mae_val:.4f}")

    # 그래프
    plt.figure(figsize=(8,4))
    plt.plot(train_ts.index, train_ts, label='Train', marker='o')
    plt.plot(test_ts.index, test_ts, label='Test-Actual', marker='o')
    plt.plot(df_res.index, df_res['pred'], label='Test-Pred', marker='x')
    plt.title(f"ARIMAX - {y_col} (RMSE={rmse_val:.4f}, MAE={mae_val:.4f})")
    plt.legend()
    plt.grid()
    plt.show()

    # 9) 잔차 분석 (Q-Q plot 등)
    #    ARIMAX는 pmdarima 모델 -> model_arimax.arima_res_ (statsmodels객체) 에서 잔차
    try:
        print("\n=== [잔차 분석] ===")
        residuals = model_arimax.arima_res_.resid
        fig, axes = plt.subplots(1,2, figsize=(12,4))
        sm.qqplot(residuals, line='45', ax=axes[0])
        axes[0].set_title("Q-Q Plot of Residuals")
        sm.graphics.tsa.plot_acf(residuals.dropna(), lags=20, ax=axes[1], title="ACF of Residuals")
        plt.show()

        # 잔차에 대한 ADF Test 등
        res_adf = adfuller(residuals.dropna(), autolag='AIC')
        print(f"Residual ADF p-value={res_adf[1]:.6f}")
    except:
        print("[Warn] 잔차 분석 실패(모델 내부에 statsmodels결과 없음?).")


###################################################
# 5. 예시 실행 (주피터노트북 스타일)
###################################################

if __name__=="__main__":
    """
    교수님 피드백 반영:
      1) ARIMAX (외생변수 고려) 사용
      2) ADF, ACF/PACF, 잔차분석 등 통계 검증
      3) Jupyter처럼 단계별 출력
      4) 데이터 불규칙 변동 고려(차분)
    """

    # (1) CSV 불러오기
    csv_path = "Processed_COVID_Data_Filled.csv"  # 예시
    df_all = load_and_preprocess_data(csv_path)
    print(df_all.head())

    # (2) 대상 타겟: 예) 'Cases'
    #     외생변수: 예) ['평균기온(℃)','평균습도(%rh)'] 등으로 지정 가능
    target_col = "Cases"
    exog_list = ['SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25', '평균기온(℃)', '평균최고기온(℃)', '최고기온(℃)', '평균최저기온(℃)', '최저기온(℃)', 
    '평균일강수량(mm)', '최다일강수량(mm)', '평균풍속(m/s)', '최대풍속(m/s)', '최대순간풍속(m/s)', '평균습도(%rh)', '최저습도(%rh)', '일조합(hr)', 
    '일사합(MJ/m2)'] 
    # (3) ARIMAX 실행 (train/test 분리)
    #     - adf_alpha=0.05
    #     - test_size=10 (뒤쪽 10개를 Test로)
    run_arimax_process(
        df_all,
        y_col=target_col,
        exog_cols=exog_list,
        test_size=10,
        adf_alpha=0.05
    )

    print("\n[Done] ARIMAX process complete.\n")
