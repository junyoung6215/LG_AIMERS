MODEL_CONFIGS = {
    'input_file': 'train.csv',
    'preprocessed_file': 'preprocessed_data1.csv',
    'feature_engineered_file': 'feature_engineered_data.csv',
    'optimization_results': 'hyperparameter_optimization_results.csv',
    'model_output_dir': 'trained_models/',
    'submission_dir': 'submissions/'
}

FEATURE_CONFIGS = {
    'target': '임신 성공 여부',
    'drop_cols': ['ID', '시술 시기 코드'],
    'text_cols': ['특정 시술 유형', '배아 생성 주요 이유'],
    'num_cols': ['시술 당시 나이', '총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수']
}
