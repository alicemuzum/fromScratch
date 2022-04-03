from sklearn.preprocessing import OneHotEncoder

from seaborn import load_dataset
import pandas as pd

df = load_dataset('penguins')
for col in df.columns:
    print(col)
print(df.head())


ohe = OneHotEncoder(df['sex'])
transformed = ohe.fit_transform(df)

print(transformed)