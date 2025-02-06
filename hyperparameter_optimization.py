import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from config import MODEL_CONFIGS, FEATURE_CONFIGS

# 데이터 로드
train_file = MODEL_CONFIGS['feature_engineered_file']
df_train = pd.read_csv(train_file, encoding='utf-8')

# X, y 데이터 분할
target = FEATURE_CONFIGS['target']
drop_cols = FEATURE_CONFIGS['drop_cols']
X = df_train.drop(columns=[target] + drop_cols, errors='ignore')
y = df_train[target]

# 범주형 변수 처리
X = pd.get_dummies(X, drop_first=True)

def optimize_random_forest(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 42
    }
    model = RandomForestClassifier(**params)
    return np.mean(cross_val_score(model, X, y, cv=5, scoring='roc_auc'))

def optimize_xgboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }
    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    return np.mean(cross_val_score(model, X, y, cv=5, scoring='roc_auc'))

def optimize_lightgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'random_state': 42
    }
    model = LGBMClassifier(**params)
    return np.mean(cross_val_score(model, X, y, cv=5, scoring='roc_auc'))

def optimize_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
        'random_state': 42,
        'verbose': 0
    }
    model = CatBoostClassifier(**params)
    return np.mean(cross_val_score(model, X, y, cv=5, scoring='roc_auc'))

# 각 모델별 최적화 실행
models = {
    'RandomForest': optimize_random_forest,
    'XGBoost': optimize_xgboost,
    'LightGBM': optimize_lightgbm,
    'CatBoost': optimize_catboost
}

best_params = {}
best_scores = {}

n_trials = 50  # 최적화 시도 횟수

for model_name, optimize_func in models.items():
    print(f"\n🔍 {model_name} 최적화 중...")
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_func, n_trials=n_trials)
    
    best_params[model_name] = study.best_params
    best_scores[model_name] = study.best_value
    
    print(f"✅ 최적 하이퍼파라미터: {study.best_params}")
    print(f"🎯 최고 ROC-AUC 점수: {study.best_value:.4f}")

# 결과 저장
results_df = pd.DataFrame({
    'Model': best_params.keys(),
    'Best Score': best_scores.values(),
    'Best Parameters': best_params.values()
})

results_df.to_csv(MODEL_CONFIGS['optimization_results'], index=False, encoding='utf-8-sig')
print("\n📊 최적화 결과가 'hyperparameter_optimization_results.csv'에 저장되었습니다.")
