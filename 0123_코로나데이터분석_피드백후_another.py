# 1) 라이브러리 불러오기
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

# 2) 데이터 불러오기
df = pd.read_csv('Processed_COVID_Data_Filled.csv', parse_dates=['Date'], encoding='ANSI')  # 가정
df = df.dropna()  # 간단히 결측 제거(혹은 다른 처리)

# 3) 독립변수(X), 종속변수(y) 분리
#    Cases: 타겟
#    나머지(기상,대기오염 등) -> X
y = df['Cases']
X = df.drop(columns=['Cases','Date'])  # Date는 시계열로 따로 처리하거나 드롭

# 4) (옵션) 시차(Lag) 피처 추가 예시
df['Cases_lag1'] = df['Cases'].shift(1)
df['Cases_lag7'] = df['Cases'].shift(7)
#   ==> NaN이 생기므로, 해당 row 삭제 or 적절히 전처리
df = df.dropna()

# 5) Train/Test 분리 (단순 시점 분할 or train_test_split)
train_df = df[df['Date'] < '2020-12-01']
test_df  = df[df['Date'] >= '2020-12-01']

X_train = train_df.drop(columns=['Cases','Date'])
y_train = train_df['Cases']
X_test = test_df.drop(columns=['Cases','Date'])
y_test = test_df['Cases']

# 6) 모델 학습
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 7) 예측 및 평가
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae, "R^2:", r2)

# 8) 피처 중요도 시각화
import matplotlib.pyplot as plt
import seaborn as sns

fi = pd.Series(rf.feature_importances_, index=X_train.columns)
fi.sort_values().plot(kind='barh')
plt.show()
