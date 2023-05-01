import pandas as pd
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from config import DataConfig
"""
Simple script that trains on fake data and test on real
"""

config = DataConfig(dataset_name='lower_back_pain', model_name='copulagan', epochs=500, batch_size=100)

real_path, fake_path, mixed_path, result_path = config.real_path, config.fake_path, config.mixed_path, config.result_path

# clf = RandomForestClassifier(max_depth=12, min_samples_leaf=4, min_samples_split=10, n_estimators=150)
clf = LogisticRegression(C=100, penalty='l2', solver='newton-cg')
scaler = MinMaxScaler()

real = pd.read_csv(real_path)
fake = pd.read_csv(fake_path)


x_real = real.iloc[:, :-1].values
y_real = real.iloc[:, -1].values

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(x_real, y_real, test_size=0.25, random_state=16)
X_train_r = scaler.fit_transform(X_train_r)
X_test_r = scaler.transform(X_test_r)

x_fake = fake.iloc[:, :-1].values
y_fake = fake.iloc[:, -1].values

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(x_fake, y_fake, test_size=0.25, random_state=16)
X_train_f = scaler.transform(X_train_f)
X_test_f = scaler.transform(X_test_f)


#trained on fake
clf.fit(X_train_f, y_train_f)

# tested on real
y_pred_r = clf.predict(X_test_r)
y_pred_proba_r = clf.predict_proba(X_test_r)[:, 1]
ns_fpr_r, ns_tpr_r, _ = roc_curve(y_test_r, y_pred_proba_r)

f1_r = f1_score(y_test_r, y_pred_r)
print("trained on fake, tested on real: ", f1_r)

# tested on fake
y_pred_f = clf.predict(X_test_f)
y_pred_proba_f = clf.predict_proba(X_test_f)[:, 1]
ns_fpr_f, ns_tpr_f, _ = roc_curve(y_test_f, y_pred_proba_f)

f1_f = f1_score(y_test_f, y_pred_f)
print("trained on fake, tested on fake: ", f1_f)

pyplot.plot(ns_fpr_r, ns_tpr_r, label='real')
pyplot.plot(ns_fpr_f, ns_tpr_f, label='fake')
pyplot.plot([0, 1], [0, 1], color='r', linestyle='--', lw=2, label='Random', alpha=.8)

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()
# show the plot
pyplot.show()







