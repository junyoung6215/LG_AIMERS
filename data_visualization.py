import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from config import MODEL_CONFIGS, FEATURE_CONFIGS

def load_data(file_path):
    """UTF-8과 CP949 인코딩을 모두 고려하여 데이터 로드"""
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:  # UTF-8 실패시 CP949 시도
        return pd.read_csv(file_path, encoding='cp949')

def preprocess_data(df):
    """데이터 전처리 파이프라인 실행"""
    # 원본 데이터 백업 (전처리 실패시 복구용)
    df_original = df.copy()
    
    # 1️⃣ 데이터 타입 변환
    # 문자열로 되어있는 숫자 데이터(예: '만30세', '2회' 등)를 숫자로 변환
    num_cols = FEATURE_CONFIGS['num_cols']
    for col in num_cols:
        try:
            # 정규표현식으로 숫자만 추출 후 float로 변환
            df[col] = df[col].astype(str).str.extract('(\d+)').astype(float)
        except Exception as e:
            print(f"⚠️ {col} 컬럼 변환 실패: {str(e)}")
            df[col] = df_original[col]  # 변환 실패시 원본값 유지

    # 2️⃣ 결측치 처리
    # 전체 결측치 비율 계산
    missing_ratio = df.isnull().sum() / len(df)
    
    # 낮은 결측치(3% 미만) 처리
    # 수치형: 중앙값으로 대체
    # 범주형: 최빈값으로 대체
    low_missing_cols = missing_ratio[(missing_ratio > 0) & (missing_ratio < 0.03)].index
    for col in low_missing_cols:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)  # 수치형
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)  # 범주형

    # 높은 결측치(90% 이상) 처리 - 해당 컬럼 삭제
    high_missing_cols = missing_ratio[missing_ratio > 0.90].index
    print(f"\n⚠️ 삭제될 컬럼: {list(high_missing_cols)}")
    df.drop(columns=high_missing_cols, inplace=True)

    # 3️⃣ 이상치 처리, 나중에 Winsoring 적용 고려
    # 극단치(상하위 2%) 제거 후 최빈값으로 대체
    cont_cols = ['총 생성 배아 수', '미세주입된 난자 수', '이식된 배아 수']
    for col in cont_cols:
        if col in df.columns:
            try:
                # 상하위 2% 기준으로 이상치 판단
                lower_bound = df[col].quantile(0.02)
                upper_bound = df[col].quantile(0.98)
                # 이상치를 NaN으로 변환 후 최빈값 대체
                df[col] = df[col].apply(lambda x: np.nan if (x < lower_bound or x > upper_bound) else x)
                df[col].fillna(df[col].mode()[0], inplace=True)
            except Exception as e:
                print(f"⚠️ {col} 이상치 처리 실패: {str(e)}")

    # 4️⃣ 범주형 변수 처리
    # 문자열 범주를 숫자로 변환 (0부터 시작하는 정수로 매핑)
    cat_cols = ['시술 유형', '난자 출처']
    for col in cat_cols:
        if col in df.columns:
            try:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            except Exception as e:
                print(f"⚠️ {col} 라벨 인코딩 실패: {str(e)}")

    # 5️⃣ 텍스트 데이터 처리
    # 텍스트를 TF-IDF 벡터로 변환 (상위 10개 특성만 추출)
    text_cols = ['특정 시술 유형', '배아 생성 주요 이유']
    for text_col in text_cols:
        if text_col in df.columns:
            try:
                vectorizer = TfidfVectorizer(max_features=10)  # 상위 10개 특성만 추출
                tfidf_matrix = vectorizer.fit_transform(df[text_col].fillna(''))
                # TF-IDF 결과를 새로운 컬럼으로 추가
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(), 
                    columns=[f"{text_col}_TFIDF{i}" for i in range(tfidf_matrix.shape[1])]
                )
                df = pd.concat([df, tfidf_df], axis=1)  # 기존 데이터프레임에 TF-IDF 컬럼 추가
            except Exception as e:
                print(f"⚠️ {text_col} TF-IDF 변환 실패: {str(e)}")

    # 6️⃣ 전처리된 데이터 저장
    output_path = MODEL_CONFIGS['preprocessed_file']
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 전처리 완료! 데이터가 '{output_path}'로 저장되었습니다.")
    
    return df

"""
✅ 주요 기능 및 변경 사항
✅ 적용 단계	
✅ 변경 내용
 1️⃣ 데이터 타입 변환	"만XX-XX세", "X회" → 숫자로 변환
 2️⃣ 결측치 처리	2~3% 미만 결측치 → 최빈값 대체
    90% 이상 결측치 → 컬럼 삭제
 3️⃣ 이상치 처리	상위 98% & 하위 2% 제거 후 최빈값 대체
 4️⃣ 범주형 처리	라벨 인코딩 적용 (Label Encoding)
 5️⃣ 텍스트 데이터 처리	TF-IDF 변환 후 기존 컬럼을 대체
 6️⃣ CSV 저장	전처리된 데이터를 기존 내용과 함께 저장 (preprocessed_data.csv)
"""