import pandas as pd

url = 'https://github.com/Accord-Project/accord-nlp/blob/main/sentence-classification/data/Single-Clauses-Data_Binary-Classification.csv?raw=true'

try:
    df = pd.read_csv(url, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(url, encoding='latin1')

print(df.head())

print(df.shape)

print(df.columns)

# print(df['label'].value_counts())