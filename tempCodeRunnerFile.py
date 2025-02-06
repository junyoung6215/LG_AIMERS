import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def feature_engineering(df):
    """상관관계 분석 및 변수 조합 + 텍스트 데이터 처리"""
    
    # 1️⃣ 상관관계 분석 및 변수 조합
    corr_matrix = df.corr().abs()
    high_corr_pairs = [(i, j) for i in corr_matrix.columns for j in corr_matrix.columns
                       if i != j and corr_matrix.loc[i, j] > 0.85]
    
    # 중복 변수를 찾고 하나만 유지 (예: '남성 주 불임 원인', '부부 주 불임 원인')
    redundant_vars = set()
    for var1, var2 in high_corr_pairs:
        redundant_vars.add(var2)  # 두 변수 중 하나를 제거
    
    print(f"\n⚠️ 높은 상관관계로 제거될 변수: {list(redundant_vars)}")
    df.drop(columns=list(redundant_vars), inplace=True)

    # 배아 사용 여부 변수 조합
    if {'동결 배아 사용 여부', '신선 배아 사용 여부'}.issubset(df.columns):
        df['배아 사용 유형'] = df['동결 배아 사용 여부'] - df['신선 배아 사용 여부']

    # 비율 및 조합 변수 추가
    if {'총 생성 배아 수', '수집된 신선 난자 수'}.issubset(df.columns):
        df['배아 생성 효율'] = df['총 생성 배아 수'] / (df['수집된 신선 난자 수'] + 1)  # 0 방지
    
    # 2️⃣ 텍스트 데이터 처리 (TF-IDF + SVD)
    text_cols = ['특정 시술 유형', '배아 생성 주요 이유']
    vectorizer = TfidfVectorizer(max_features=100)
    
    for text_col in text_cols:
        tfidf_matrix = vectorizer.fit_transform(df[text_col].fillna(''))
        
        # 차원 축소 (SVD 적용, 주요 5개 차원 유지)
        svd = TruncatedSVD(n_components=5)
        tfidf_reduced = svd.fit_transform(tfidf_matrix)

        # 변환된 데이터를 데이터프레임에 추가
        tfidf_df = pd.DataFrame(tfidf_reduced, columns=[f"{text_col}_SVD{i}" for i in range(5)])
        df.drop(columns=[text_col], inplace=True)  # 원본 텍스트 컬럼 삭제
        df = pd.concat([df, tfidf_df], axis=1)

    # 3️⃣ 최종 전처리된 데이터 저장
    output_path = "feature_engineered_data.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 피처 엔지니어링 완료! 데이터가 '{output_path}'로 저장되었습니다.")

    return df

# CSV 파일 불러오기 (전처리된 데이터 사용)
file_path = "preprocessed_data1.csv"
df = pd.read_csv(file_path, encoding='utf-8')

# 피처 엔지니어링 실행
df_engineered = feature_engineering(df)

# 처리 결과 출력
print("\n✅ 피처 엔지니어링 전후 비교:")
print(f"처리 전 크기: {df.shape}")
print(f"처리 후 크기: {df_engineered.shape}")
