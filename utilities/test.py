import pandas as pd

from utilities.plots import column_correlation_plot
from utilities.encode_data import merge_datasets


one = merge_datasets(pd.read_csv("../data/cardio/cardio.csv"), pd.read_csv("../data/cardio/copulagan_250_epochs_400_batch.csv"))

two= merge_datasets(pd.read_csv("../data/cardio/cardio.csv"), pd.read_csv("../data/cardio/ctgan_300_epochs_400_batch.csv"))

one.to_csv("../data/cardio/copulagan_mix.csv", index=False)
two.to_csv("../data/cardio/ctgan_mix.csv", index=False)