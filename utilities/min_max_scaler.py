import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('../data/obesity/obesity_num.csv')

print(df)
categorical_cols = ['Gender', 'family_history_with_overweight',
            'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']

scaler = MinMaxScaler()

# Scale all values except categorical value
df[df.columns.difference(categorical_cols)] = scaler.fit_transform(df[df.columns.difference(categorical_cols)])

df.to_csv('data/obesity/obesity_scaled.csv', index=False)
