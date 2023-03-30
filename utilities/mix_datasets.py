import pandas as pd

real = pd.read_csv("../data/lower_back_pain/lower_back_pain_scaled.csv")
print(real)
fake = pd.read_csv("../data/lower_back_pain/ctgan_800_epochs_100_batch.csv")

mixed = pd.merge(real, fake)

print(mixed.head(10))

