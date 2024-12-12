import pandas as pd

df1 = pd.read_csv('baseline_submit_mai_115_1.csv')
df2 = pd.read_csv('baseline_submit_mai_114_1.csv')
df3 =  pd.read_csv('baseline_submit_mai_112_1.csv')
df4 =  pd.read_csv('baseline_submit_mai_112_2_sig.csv')
df5 =  pd.read_csv('baseline_submit_mai_112_2_1.csv')
df6 =  pd.read_csv('baseline_submit_mai_112_2_2.csv')
df7 =  pd.read_csv('baseline_submit_mai_114_2_2.csv')
df8 =  pd.read_csv('baseline_submit_mai_115_2_2.csv')

df_selected = (df1[['real', 'fake']] + df2[['real', 'fake']] + df3[['real', 'fake']] + df4[['real', 'fake']] + df5[['real', 'fake']] + df6[['real', 'fake']] + df7[['real', 'fake']] + df8[['real', 'fake']]) / 8

df_new = df1.copy()
df_new[['real', 'fake']] = df_selected

print(df1.head())
print(df_new.head())

df_new.to_csv('ensemble.csv', index=False)
