import pandas as pd
import matplotlib.pyplot as plt

path = "../data/cardio/cardio.csv"
dataset = pd.read_csv(path, delimiter=";")

dataset.rename(columns = {
    "cardio": "class"}, inplace=True)

print(dataset.isna().sum())
dataset["class"].value_counts().sort_index().plot.bar()

plt.xlabel("Class distribution")
plt.title("Cardiovascular Disease")
plt.show()