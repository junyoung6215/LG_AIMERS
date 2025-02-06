import pandas as pd
import numpy as np

# 파일 읽기
file = "train.csv"
df = pd.read_csv(file)

# 데이터 크기 확인
print("전체 데이터 크기:", df.shape)

# 데이터를 10개로 분할
splitted_dfs = np.array_split(df, 10)

# 분할된 데이터를 각각 CSV 파일로 저장
for idx, sub_df in enumerate(splitted_dfs):
    output_file = f"train_{idx}.csv"
    sub_df.to_csv(output_file, index=False, encoding='utf-8-sig')  # UTF-8-SIG로 저장 (한글 호환)
    print(f"저장 완료: {output_file} ({sub_df.shape})")
