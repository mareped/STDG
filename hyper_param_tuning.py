from sklearn.model_selection import GridSearchCV
from sdv.tabular import CTGAN
import pandas as pd

df = pd.read_csv('data/lower_back_pain/lower_back_pain_scaled.csv')

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

model = CTGAN()

batch_size = [10]
epochs = [100]

param_grid = dict(batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))