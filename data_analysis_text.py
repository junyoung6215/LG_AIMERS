import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_path):
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='cp949')

def analyze_keywords(text_series, col_name):
    # 텍스트가 아닌 경우(ID 등) 건너뛰기
    if col_name.lower() == 'id' or text_series.dtype != 'object':
        return
    
    try:
        vectorizer = TfidfVectorizer(max_features=10, 
                                   min_df=1,       # 최소 1번 이상 등장
                                   max_df=1.0)     # 전체 문서의 100%까지 허용
        
        tfidf_matrix = vectorizer.fit_transform(text_series.fillna(''))
        mean_tfidf = tfidf_matrix.mean(axis=0).A1
        keywords = vectorizer.get_feature_names_out()
        word_score = dict(zip(keywords, mean_tfidf))
        sorted_words = sorted(word_score.items(), key=lambda x: x[1], reverse=True)
        
        print("\n상위 10개 키워드와 TF-IDF 점수:")
        for word, score in sorted_words:
            print(f"{word}: {score:.4f}")
    except ValueError as e:
        print(f"\n{col_name} 컬럼은 텍스트 분석에 적합하지 않습니다.")
        print(f"에러 메시지: {str(e)}")

file_path = "train.csv"
df = load_data(file_path)

# 1. 데이터 기본 정보
print("\n✅ 1. 데이터 기본 정보")
print(f"총 행 수: {len(df)}")
print(f"총 열 수: {len(df.columns)}")
print("\n컬럼 정보:")
print(df.info())

# 2. 결측치 분석
print("\n✅ 2. 결측치 분석")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'결측치 수': missing, '결측치 비율(%)': missing_pct})
print(missing_df[missing_df['결측치 수'] > 0])

# 3. 수치형 변수 분석
num_cols = df.select_dtypes(include=['number']).columns
print("\n✅ 3. 수치형 변수 분석")
for col in num_cols:
    stats = df[col].describe()
    print(f"\n{col} 통계:")
    print(f"평균: {stats['mean']:.2f}")
    print(f"표준편차: {stats['std']:.2f}")
    print(f"최소값: {stats['min']:.2f}")
    print(f"최대값: {stats['max']:.2f}")

# 4. 상관관계 분석
print("\n✅ 4. 상관관계 분석")
corr_matrix = df[num_cols].corr()

print("\n강한 양의 상관관계 (> 0.75):")
high_corr = np.where((corr_matrix > 0.75) & (corr_matrix < 1.0))
for i, j in zip(*high_corr):
    if i < j:
        print(f"{num_cols[i]} - {num_cols[j]}: {corr_matrix.iloc[i, j]:.3f}")

print("\n강한 음의 상관관계 (< -0.75):")
neg_corr = np.where(corr_matrix < -0.75)
for i, j in zip(*neg_corr):
    if i < j:
        print(f"{num_cols[i]} - {num_cols[j]}: {corr_matrix.iloc[i, j]:.3f}")

# 5. 이상치 분석
print("\n✅ 5. 이상치 분석")
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
    print(f"\n{col} 이상치:")
    print(f"이상치 개수: {len(outliers)}")
    print(f"이상치 비율: {(len(outliers)/len(df))*100:.2f}%")

# 6. 텍스트 데이터 분석
print("\n✅ 6. 텍스트데이터 분석")
text_cols = df.select_dtypes(include=['object']).columns
for col in text_cols:
    if col.lower() != 'id' and df[col].nunique() > 10:  # ID 제외, 고유값 10개 이상
        print(f"\n{col} 텍스트 분석:")
        print(f"고유 값 수: {df[col].nunique()}")
        print(f"평균 텍스트 길이: {df[col].str.len().mean():.2f}")
        analyze_keywords(df[col], col)