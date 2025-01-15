import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
# from catboost import CatBoostRegressor

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def load_data(file_path):
    """CSV 파일에서 데이터를 로드하고 DataFrame을 반환합니다."""
    try:
        df = pd.read_csv(file_path)
        print("데이터 로드 완료")
        print(df.head())
        return df
    except FileNotFoundError:
        print("오류: 파일을 찾을 수 없습니다.")
        exit()


def preprocess_data(df):
    """데이터 전처리 함수입니다."""
    df = df.drop('연도', axis=1)
    independent_variables = [
        '면허미보유_medsol', '간호_medsol', '약_medsol', '방사선_medsol', '임상병리_medsol',
        '치과위생_medsol', '물리치료_medsol', '치과가공_medsol', '응급구조_medsol', '간호조무_medsol',
        '통합_medsol', '간호조무사_mednco', '임상병리사_mednco', '방사선사_mednco',
        '물리치료사_mednco', '치위생사_mednco', '응급구조사_mednco', '간호사_medciv',
        '간호조무사_medciv', '약사_medciv', '임상병리사_medciv', '방사선사_medciv',
        '물리치료사_medciv', '치위생사_medciv', '치기공사_medciv', '응급구조사_medciv'
    ]
    dependent_variables = [
        'A형 간염', '결 핵', '렙토스피라증', '말라리아', '매 독', '세균성 이질', '수 두',
        '수막구균성 수막염', '신증후군 출혈열', '유행성 이하선염', '장티푸스', '쯔쯔가무시증',
        '파상풍', '폐 렴', '홍 역'
    ]
    print("\n결측치 확인:")
    print(df.isnull().sum())
    df = df.dropna()  # 결측치가 있는 행 제거
    return df, independent_variables, dependent_variables


def split_and_scale_data(df, independent_variables, dependent_variables):
    """데이터를 분할하고 스케일링합니다."""
    X = df[independent_variables]
    y = df[dependent_variables]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def define_models():
    """사용할 모델을 정의합니다."""
    return {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, 
            learning_rate=0.1, 
            random_state=42,
            min_child_samples=30,
            min_split_gain=0.01,
            num_leaves=31,
            max_depth=5),
    }


def evaluate_models(models, X_train, y_train, X_test, y_test, dependent_variables):
    """모델을 학습하고 평가합니다."""
    model_results = {}
    for model_name, model in models.items():
        print(f"\n------ {model_name} 모델 학습 ------")
        
        mse_list = []
        r2_list = []
        mae_list = []
        y_pred_list = [] # 모델별 모든 질병 예측 값을 저장할 리스트
    
        for disease in dependent_variables:
            # 각 질병에 대한 y 데이터 추출
            y_train_disease = y_train[disease]
            y_test_disease = y_test[disease]
            
            # 모델 학습
            model.fit(X_train, y_train_disease)
            y_pred_disease = model.predict(X_test)
            
            # 평가
            mse = mean_squared_error(y_test_disease, y_pred_disease)
            r2 = r2_score(y_test_disease, y_pred_disease)
            mae = mean_absolute_error(y_test_disease, y_pred_disease)
            
            mse_list.append(mse)
            r2_list.append(r2)
            mae_list.append(mae)
            y_pred_list.append(y_pred_disease)
            print(f"  {disease} - MSE: {mse:.3f}, R2: {r2:.3f}, MAE: {mae:.3f}")
            
        model_results[model_name] = {
            'mse': mse_list,
            'r2': r2_list,
            'mae': mae_list,
            'pred': y_pred_list, # 리스트 형태로 저장
        }
    return model_results
    

def visualize_model_performance(model_results, dependent_variables):
    """모델 성능을 시각화합니다."""
    x = np.arange(len(dependent_variables))
    width = 0.15
    
    metrics = ['mse', 'r2', 'mae']
    titles = ['모델별 MSE 비교', '모델별 R2 비교', '모델별 MAE 비교']
    ylabels = ['MSE', 'R2', 'MAE']
    
    for metric, title, ylabel in zip(metrics, titles, ylabels):
      plt.figure(figsize=(20, 8))
      for i, model_name in enumerate(model_results.keys()):
          plt.bar(x + width * (i - 1.5), model_results[model_name][metric], width, label=f'{model_name} {metric.upper()}')
      plt.xticks(x, dependent_variables, rotation=45, ha='right')
      plt.xlabel('감염병')
      plt.ylabel(ylabel)
      plt.title(title)
      plt.legend()
      plt.tight_layout()
      plt.show()


def visualize_prediction_results(model_results, y_test, dependent_variables):
    """각 질병별 예측 결과를 시각화합니다."""
    num_diseases = len(dependent_variables)
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightpink']
    
    for i, disease in enumerate(dependent_variables):
        ax = axes[i]
        
        for j, model_name in enumerate(model_results.keys()):
          ax.scatter(y_test[disease], model_results[model_name]['pred'][i], color=colors[j], label=model_name, s=20)
          
        ax.plot(y_test[disease], y_test[disease], color='red', linestyle='-', linewidth=1)
        ax.set_title(f"{disease} 예측 결과 비교")
        ax.set_xlabel("실제 환자 수")
        ax.set_ylabel("예측 환자 수")
        ax.legend()
    
    for j in range(num_diseases, len(axes)):
      fig.delaxes(axes[j])
      
    plt.tight_layout()
    plt.show()

# 메인 함수
if __name__ == "__main__":
    file_path = '병합된_연도별_감염병_및_의료인력_현황.csv'
    df = load_data(file_path)
    df, independent_variables, dependent_variables = preprocess_data(df)
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(df, independent_variables, dependent_variables)
    models = define_models()
    model_results = evaluate_models(models, X_train_scaled, y_train, X_test_scaled, y_test, dependent_variables)
    visualize_model_performance(model_results, dependent_variables)
    visualize_prediction_results(model_results, y_test, dependent_variables)