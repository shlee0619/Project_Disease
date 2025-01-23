import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

###############################################################################
# 1. ADF 검정 + 차분 (필요 시)
###############################################################################
def check_stationarity_and_diff(series, alpha=0.05, max_diff=2):
    """
    - series: 1차원 시계열(pd.Series)
    - alpha: ADF 테스트 유의수준(기본 0.05)
    - max_diff: 최대 몇 번 차분 시도할지
    - return: (시계열, 최종 차분 횟수)
    """
    # 0) 복사
    ts = series.copy()

    diff_count = 0
    for d in range(max_diff+1):
        # ADF
        test_res = adfuller(ts.dropna(), autolag='AIC')
        pval = test_res[1]
        if pval < alpha:
            print(f"  -> ADF p={pval:.4f}, d={diff_count} -> 정상성 OK")
            return ts, diff_count
        else:
            if diff_count < max_diff:
                print(f"  -> ADF p={pval:.4f} (>= {alpha}), 차분(d={diff_count+1}) 시도")
                ts = ts.diff(1).dropna()
                diff_count += 1
            else:
                print(f"  -> ADF p={pval:.4f}, 더 이상 차분 불가(max_diff={max_diff}). 비정상 가능성")
                return ts, diff_count

    return ts, diff_count


###############################################################################
# 2. SARIMAX Rolling Forecast
###############################################################################
def rolling_sarimax_forecast(endog, exog=None,
                             order=(1,0,1),
                             seasonal_order=(0,0,0,0),
                             test_size=10,
                             freq='D'):
    """
    [역할]
      - (endog, exog) 시계열을 주고, SARIMAX로 rolling forecast(1-step ahead) 하는 예시
      - test_size: 뒤쪽 test_size개를 한 스텝씩 예측

    [핵심 포인트: exog 데이터 처리]
      - get_forecast(steps=1, exogenous=…) 할 때,
        예측할 날짜 인덱스와 같은 (1 x n_features) DataFrame을 만들어야 Out-of-Sample 오류 안 남

    [리턴] (df_res, final_model)
      - df_res: (actual, pred)
      - final_model: 마지막 시점의 fitted model 결과
    """
    # 0) train/test 분할
    n = len(endog)
    train_n = n - test_size
    train_endog = endog.iloc[:train_n]
    test_endog  = endog.iloc[train_n:]

    if exog is not None:
        train_exog = exog.iloc[:train_n, :]
        test_exog  = exog.iloc[train_n:, :]
    else:
        train_exog = None
        test_exog  = None

    print("\n=== Rolling SARIMAX Forecast ===")
    print(f" train_n={train_n}, test_n={test_size}")
    print(f" order={order}, seasonal_order={seasonal_order}")

    # 1) 초기 모델 피팅
    model = sm.tsa.statespace.SARIMAX(
        train_endog,
        exog=train_exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    print("\n[Initial SARIMAX fit]")
    print(results.summary())

    # 2) 롤링 예측
    preds = []
    actuals = []
    current_results = results

    for i in range(len(test_endog)):
        this_date = test_endog.index[i]
        y_true = test_endog.iloc[i]

        # exog 준비
        if test_exog is not None:
            # shape=(1, n_features)
            x_row = test_exog.iloc[[i]]       # i:i+1
            # 인덱스를 예측할 날짜로!
            x_row.index = [this_date]
        else:
            x_row = None

        # (A) 1-step 예측
        #     get_forecast(1, exogenous= x_row)
        # fc = current_results.get_forecast(steps=1, exogenous=x_row)
        # y_pred = fc.predicted_mean.iloc[0]
        fc_val = current_results.predict(
            start=current_results.nobs, 
            end=current_results.nobs, 
            exog=x_row
        )
        y_pred = fc_val.iloc[0]

        preds.append(y_pred)
        actuals.append(y_true)

        # (B) 모델 상태 업데이트(append)
        current_results = current_results.append(
            endog=[y_true],
            exog=x_row,
            refit=False
        )

    df_res = pd.DataFrame({'actual': actuals, 'pred': preds}, index=test_endog.index)

    # RMSE, MAE
    rmse_ = sqrt(mean_squared_error(df_res['actual'], df_res['pred']))
    mae_  = mean_absolute_error(df_res['actual'], df_res['pred'])

    print("\n=== [Rolling Forecast Result] ===")
    print(df_res)
    print(f"RMSE={rmse_:.3f}, MAE={mae_:.3f}")

    # 시각화
    plt.figure(figsize=(10,4))
    plt.plot(train_endog.index, train_endog, label='Train')
    plt.plot(test_endog.index, test_endog, label='Test Actual')
    plt.plot(df_res.index, df_res['pred'], label='Test Pred', marker='x')
    plt.title(f"SARIMAX Rolling - RMSE={rmse_:.3f}, MAE={mae_:.3f}")
    plt.legend()
    plt.grid()
    plt.show()

    return df_res, current_results


###############################################################################
# 3. 전체 실행 예시
###############################################################################
if __name__ == "__main__":

    # (A) 데이터 불러오기
    #  - 이미 질문에 주신 csv 내용 그대로라면, pandas로 읽어온 뒤,
    #    'Date'로 set_index 하시면 됩니다.
    #  - 여기서는 가정: df = pd.read_csv('your_data.csv', parse_dates=['Date'], index_col='Date')
    #    ... 사용자 환경에 맞춰 조정!
    print("\n[Load CSV to df ...]")
    df = pd.read_csv('Processed_COVID_Data_Filled.csv', parse_dates=['Date'], index_col='Date', encoding='ANSI')
    df.sort_index(inplace=True)
    print(df.head())
    print(df.tail())

    # (B) 타겟 / exog 정의
    #  - 'Cases'를 예측 (endog)
    #  - 나머지 컬럼(SO2, CO, O3, ...)을 exog로
    target_col = 'Cases'
    # 가능한 exog 목록에서 'Cases' 제외
    exog_cols = [c for c in df.columns if c != target_col]

    # (C) ADF & 차분
    # endog
    print("\n[ADF & diff] for Cases:")
    y_raw = df[target_col].astype(float)
    y_sta, dcount = check_stationarity_and_diff(y_raw, alpha=0.05, max_diff=2)

    # exog: 여기서는 따로 차분하지 않음(필요하다면 diff)
    X_exog = df[exog_cols].astype(float)
    if dcount>0:
        # 만약 endog를 dcount번 차분했다면,
        # exog도 동일 차분하는 게 보통이지만, 일단 여기서는 생략/혹은 선택
        # ex) X_exog = X_exog.diff(dcount).dropna()
        #     그리고 y_sta.index와 맞춰주기
        X_exog = X_exog.loc[y_sta.index]

    # (D) train/test 분할 크기
    #  - 끝에서 10일을 test
    #  - 질문에 따라 조절
    test_size = 10

    # (E) Rolling SARIMAX 실행
    #  - order, seasonal_order를 ACF/PACF로 추정하거나, AutoARIMA 등으로 찾아도 됨
    #  - 여기서는 (1, dcount, 1) 시도
    order_ = (1, dcount, 1)
    seas_  = (0,0,0,0)

    df_out, final_model = rolling_sarimax_forecast(
        endog=y_sta,
        exog=X_exog,
        order=order_,
        seasonal_order=seas_,
        test_size=test_size
    )

    # (F) 잔차 분석
    resid = final_model.resid
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    sm.qqplot(resid, line='45', ax=plt.gca())
    plt.title("Q-Q Plot (Resid)")
    plt.subplot(1,2,2)
    plot_acf(resid.dropna(), lags=20, ax=plt.gca())
    plt.show()

    # (G) 최종 요약
    rmse_ = sqrt(mean_squared_error(df_out['actual'], df_out['pred']))
    mae_  = mean_absolute_error(df_out['actual'], df_out['pred'])
    print(f"\n[Done] Rolling SARIMAX => RMSE={rmse_:.4f}, MAE={mae_:.4f}")

'''
[Load CSV to df ...]
            Cases       SO2        CO        O3       NO2       PM10       PM25  ...  평균풍속(m/s)  최대풍속(m/s)  최대순간풍속(m/s)  평균습도(%rh)  최저습도(%rh)  일조
합(hr)  일사합(MJ/m2)
Date                                                                             ...
2020-02-01      0  0.003274  0.699089  0.026038  0.022698  63.185218  47.416719  ...        1.5       12.5         17.8         73         31      6.2        9.67        
2020-02-02      0  0.003502  0.748826  0.028493  0.020737  69.092334  52.897434  ...        1.3        9.5         14.6         67         20      7.7       10.86        
2020-02-03      0  0.003140  0.463362  0.029590  0.015894  32.560316  21.600817  ...        2.3       13.1         16.1         56         15      8.5       13.18        
2020-02-04      3  0.003329  0.465114  0.023674  0.019638  29.410079  18.842018  ...        1.8       13.4         19.7         57         10      8.9       14.04        
2020-02-05      3  0.003002  0.374520  0.029291  0.011450  22.960480  11.594881  ...        3.0       14.7         19.6         43          9      9.6       15.46        

[5 rows x 21 columns]
            Cases       SO2        CO        O3       NO2       PM10       PM25  ...  평균풍속(m/s)  최대풍속(m/s)  최대순간풍속(m/s)  평균습도(%rh)  최저습도(%rh)  일조 합(hr)  일사합(MJ/m2)
Date                                                                             ...
2021-12-27     26  0.002879  0.456997  0.019923  0.020431  24.400479  13.778102  ...        2.0       18.7         23.8         59         17      6.7        9.32        
2021-12-28     25  0.003177  0.632841  0.011658  0.029640  45.191351  30.295361  ...        1.3       10.7         14.4         69         14      4.3        6.67        
2021-12-29     24  0.003343  0.602558  0.018986  0.024798  51.443247  35.615763  ...        2.1       18.1         20.9         67         30      3.5        6.22        
2021-12-30     24  0.002769  0.352773  0.030610  0.012173  25.829670  12.368506  ...        3.3       21.9         25.5         51         15      7.3        9.33        
2021-12-31     27  0.002618  0.338690  0.031247  0.010955  20.258406   7.552790  ...        2.9       21.7         26.8         42         10      8.5       11.23        

[5 rows x 21 columns]

[ADF & diff] for Cases:
  -> ADF p=0.0247, d=0 -> 정상성 OK

=== Rolling SARIMAX Forecast ===
 train_n=690, test_n=10
 order=(1, 0, 1), seasonal_order=(0, 0, 0, 0)
C:\Users\user\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
  self._init_dates(dates, freq)
C:\Users\user\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
  self._init_dates(dates, freq)
C:\Users\user\anaconda3\Lib\site-packages\statsmodels\base\model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "

[Initial SARIMAX fit]
                               SARIMAX Results
==============================================================================
Dep. Variable:                  Cases   No. Observations:                  690
Model:               SARIMAX(1, 0, 1)   Log Likelihood               -2297.310
Date:                Thu, 23 Jan 2025   AIC                           4640.621
Time:                        14:40:49   BIC                           4744.898
Sample:                    02-01-2020   HQIC                          4680.962
                         - 12-21-2021
Covariance Type:                  opg
===============================================================================
                  coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------
SO2         -9031.8253   1868.681     -4.833      0.000   -1.27e+04   -5369.277
CO             -5.0109     18.232     -0.275      0.783     -40.744      30.722
O3           -170.8054     99.439     -1.718      0.086    -365.702      24.091
NO2           596.7744    197.162      3.027      0.002     210.345     983.204
PM10           -0.0044      0.043     -0.103      0.918      -0.089       0.080
PM25            0.0786      0.153      0.515      0.607      -0.220       0.378
평균기온(℃)        -1.1734      0.709     -1.654      0.098      -2.564       0.217
평균최고기온(℃)       0.7086      0.472      1.500      0.134      -0.217       1.634
최고기온(℃)         0.3082      0.241      1.277      0.202      -0.165       0.781
평균최저기온(℃)       0.7861      0.357      2.200      0.028       0.086       1.487
최저기온(℃)        -0.4756      0.195     -2.435      0.015      -0.858      -0.093
평균일강수량(mm)     -0.1048      0.062     -1.693      0.090      -0.226       0.017
최다일강수량(mm)      0.0170      0.015      1.164      0.244      -0.012       0.046
평균풍속(m/s)       1.4719      0.922      1.597      0.110      -0.334       3.278
최대풍속(m/s)       0.1878      0.180      1.044      0.296      -0.165       0.540
최대순간풍속(m/s)    -0.0137      0.144     -0.096      0.924      -0.295       0.268
평균습도(%rh)       0.1049      0.075      1.396      0.163      -0.042       0.252
최저습도(%rh)       0.0363      0.046      0.781      0.435      -0.055       0.127
일조합(hr)        -0.1248      0.347     -0.360      0.719      -0.804       0.555
일사합(MJ/m2)      0.3106      0.242      1.284      0.199      -0.163       0.785
ar.L1           0.5402      0.015     35.931      0.000       0.511       0.570
ma.L1           0.6460      0.025     25.949      0.000       0.597       0.695
sigma2         45.0962      1.088     41.461      0.000      42.964      47.228
===================================================================================
Ljung-Box (L1) (Q):                   0.57   Jarque-Bera (JB):            326535.54
Prob(Q):                              0.45   Prob(JB):                         0.00
Heteroskedasticity (H):               9.79   Skew:                             7.29
Prob(H) (two-sided):                  0.00   Kurtosis:                       108.73
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).

=== [Rolling Forecast Result] ===
            actual       pred
Date
2021-12-22    37.0  25.764482
2021-12-23    28.0  31.180092
2021-12-24    29.0  19.072482
2021-12-25    29.0  19.541598
2021-12-26    28.0  20.745304
2021-12-27    26.0  23.529464
2021-12-28    25.0  22.640240
2021-12-29    24.0  15.177126
2021-12-30    24.0  17.357822
2021-12-31    27.0  16.931765
RMSE=7.823, MAE=7.142

[Done] Rolling SARIMAX => RMSE=7.8230, MAE=7.1420

유의미한 변수 (p-value < 0.05):

SO2: 계수가 -9031.8253으로 음수이고, p-값이 매우 작으므로, 아황산가스 농도가 증가하면 Cases가 감소하는 음의 영향을 줍니다 (유의미).

NO2: 계수가 596.7744으로 양수이고, p-값이 0.002이므로, 이산화질소 농도가 증가하면 Cases가 증가하는 양의 영향을 줍니다 (유의미).

평균최저기온(℃): 계수가 0.7861로 양수이고, p-값이 0.028이므로 평균 최저 기온이 증가하면 Cases가 증가하는 양의 영향을 줍니다(유의미).

최저기온(℃): 계수가 -0.4756로 음수이고, p-값이 0.015이므로 최저기온이 증가하면 Cases가 감소하는 음의 영향을 줍니다(유의미).

AR/MA 파라미터:

ar.L1: 자기회귀(AR) 항의 계수로, L1은 시차 1을 의미합니다. 값이 0.5402로 양수이며, p-value 가 매우 작아 유의미합니다.

ma.L1: 이동평균(MA) 항의 계수로, L1은 시차 1을 의미합니다. 값이 0.6460 로 양수이며, p-value가 매우 작아 유의미합니다. 이 둘의 계수가 유의미하다는 것은 해당 모델이 시계열 데이터의 자기상관을 잘 반영한다는 것을 의미합니다.

잔차분석:

Ljung-Box 검정(Prob(Q): 0.45)으로 보아 잔차에 자기상관이 없다는 귀무가설을 기각할 수 없습니다. 즉 잔차에 자기상관이 없으므로 모델이 데이터의 시계열적 패턴을 잘 포착하고 있다고 해석할 수 있습니다.

Jarque-Bera 검정(Prob(JB): 0.00)으로 보아 잔차가 정규분포를 따른다는 귀무가설을 기각해야 합니다. 이는 잔차에 이상치 또는 왜도가 존재한다는 것을 의미하며, 모델의 개선이 필요할 수 있음을 시사합니다.

Heteroskedasticity 검정(Prob(H): 0.00)으로 보아 잔차의 분산이 일정하지 않다는 귀무가설을 기각해야 합니다. 이는 잔차의 분산이 일정하지 않아 모델의 예측 성능이 특정 구간에서 불안정할 수 있음을 의미합니다.


의미 있는 변수: 아황산가스 농도, 이산화질소 농도, 평균최저기온, 최저기온이 Cases에 유의미한 영향을 미치고 있습니다. 특히, 아황산가스 농도 증가는 Cases를 감소시키는 반면, 이산화질소 농도 증가는 Cases를 증가시키는 경향을 보입니다.

잔차 분석 결과: 모델이 데이터의 시계열적 패턴을 어느 정도 잘 포착하고 있지만, 잔차의 정규성 가정이 위반되었고, 잔차의 분산이 일정하지 않으므로, 모델을 추가로 개선할 필요가 있습니다.

모델 개선 방향:

비선형 관계 고려: 현재 모델은 선형 관계를 가정하고 있으므로, 비선형 관계를 고려하는 모델을 시도해볼 수 있습니다.

데이터 변환: 로그 변환 등의 데이터 변환을 통해 모델의 성능을 개선할 수 있습니다.

모델 복잡도 조절: 계절성을 고려하거나 다른 차수의 AR/MA 모델을 시도해볼 수 있습니다.

이상치 처리: 이상치(outlier)가 모델의 성능을 저해할 수 있으므로, 이상치를 확인하고 처리하는 방법을 고려해야 합니다.

'''