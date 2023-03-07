import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/lower_back_pain.csv')
df['Class_att'].replace(['Normal', 'Abnormal'],
                        [0, 1], inplace=True)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
score = accuracy_score(y_test,y_pred)

print("SCORE: ", score)
