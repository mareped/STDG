from sklearn.model_selection import GridSearchCV
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.metrics import roc_auc_score, make_scorer

import pandas as pd

df = pd.read_csv('../data/lower_back_pain/lower_back_pain_scaled.csv')

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

METADATA_FILENAME = "../data/lower_back_pain/metadata.json"
metadata = SingleTableMetadata.load_from_json(METADATA_FILENAME)
model = CTGANSynthesizer(metadata)
model.get_parameters()


batch_size = [10, 30]
epochs = [10, 30]

param_grid = dict(batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = -1, cv = 3, scoring = 'roc_auc')

grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))