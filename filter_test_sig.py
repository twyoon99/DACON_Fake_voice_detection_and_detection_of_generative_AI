import pandas as pd
import numpy as np
# CSV 파일 경로
file_path = 'baseline_submit_mai_115.csv'

#0.5
# CSV 파일 읽기
df = pd.read_csv(file_path)
def sigmoid(z,k):
   return 1/(1 + np.exp(-z*k))
#'fake' 열에 대해 0.5 기준으로 이진 분류 적용
#df['fake'] = (df['fake'] >= 0.5).astype(int)

# 'real' 열에 대해 0.5 기준으로 이진 분류 적용
#df['real'] = (df['real'] >= 0.5).astype(int)

df['fake'] = sigmoid(df['fake'],0.5)
df['real'] = sigmoid(df['real'],0.5)

#base
# CSV 파일 읽기
df = pd.read_csv(file_path)
def sigmoid(z):
   return 1/(1 + np.exp(-z))
#'fake' 열에 대해 0.5 기준으로 이진 분류 적용
#df['fake'] = (df['fake'] >= 0.5).astype(int)

# 'real' 열에 대해 0.5 기준으로 이진 분류 적용
#df['real'] = (df['real'] >= 0.5).astype(int)

df['fake'] = sigmoid(df['fake'])
df['real'] = sigmoid(df['real'])


# 새로운 CSV 파일로 저장 (기존 'id' 열을 유지하고, 'fake'와 'real' 열이 수정됨)
output_file_path = 'baseline_submit_mai_115_2_2.csv'
df.to_csv(output_file_path, index=False)

print(f"Thresholded predictions saved to '{output_file_path}'")
