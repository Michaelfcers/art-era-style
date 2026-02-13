import pandas as pd
import ast

df = pd.read_csv('classes_clean.csv')
df['estilo'] = df['genre'].apply(lambda x: ast.literal_eval(x)[0])
clases = sorted(df['estilo'].unique().tolist())
print(clases)