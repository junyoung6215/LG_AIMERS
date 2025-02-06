import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import joblib

# 1️⃣ 학습 데이터 로드
train_file = "feature_engineered_data.csv"
df_train = pd.read_csv(train_file, encoding='utf-8')

# 2️⃣ 테스트 데이터 로드
test_file = "test.csv"
df_test = pd.read_csv(test_file, encoding='utf-8')

# 3️⃣ X, y 데이터 분할
target = '임신 성공 여부'  # 예측할 목표 변수
drop_cols = ['ID', '시술 시기 코드']  # 학습에 필요 없는 변수

# 학습 데이터 (Train)
X_train = df_train.drop(columns=[target] + drop_cols, errors='ignore')
y_train = df_train[target]

# 테스트 데이터 (Test)
X_test = df_test.drop(columns=drop_cols, errors='ignore')  # 예측을 위한 입력 데이터
y_test = df_test[target] if target in df_test.columns else None  # 테스트 데이터에 정답이 있는 경우

# ★★ 추가: 범주형 변수 처리 (문자형 변수가 있을 경우 원-핫 인코딩)
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# 학습, 테스트 데이터의 컬럼 정렬 (누락된 컬럼이 있을 경우 정렬 및 NaN 채우기)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 4️⃣ 여러 모델 학습 및 평가
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

results = {}
for model_name, model in models.items():
    print(f"\n🚀 {model_name} 모델 학습 중...")
    model.fit(X_train, y_train)  # 모델 학습

    # 예측 확률 (Test 데이터) - 소수점 둘째 자리까지 반올림
    y_pred_proba = np.round(model.predict_proba(X_test)[:, 1], 2)

    # 🔹 제출 파일 양식에 맞게 컬럼명 수정: "Prediction" → "probability"
    submission_data = {"ID": df_test["ID"], "probability": y_pred_proba}
    submission_df = pd.DataFrame(submission_data)

    file_name = f"model_predictions_{model_name}.csv"
    submission_df.to_csv(file_name, index=False, encoding="utf-8-sig")
    print(f"📌 {model_name} 예측 확률 저장 완료: {file_name}")

    # 성능 평가 (테스트 데이터에 정답이 있을 경우)
    if y_test is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            results[model_name] = {"ROC-AUC": roc_auc}
            print(f"✅ {model_name} 테스트 ROC-AUC: {roc_auc:.4f}")
        except Exception as e:
            print(f"⚠️ {model_name} ROC-AUC 계산 중 오류 발생: {e}")

    # 모델 저장
    model_path = f"{model_name}_model.pkl"
    joblib.dump(model, model_path)
    print(f"📌 {model_name} 모델 저장 완료: {model_path}")

# 5️⃣ 전체 모델 성능 결과 저장
if results:
    results_df = pd.DataFrame(results).T
    results_df.to_csv("model_performance.csv", encoding="utf-8-sig")
    print("\n📌 모델 성능 비교 결과가 'model_performance.csv'에 저장되었습니다.")
