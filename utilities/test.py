import pandas as pd

"""# Creates a mixed dataset of two dataframes
def merge_datasets(df1, df2):
    # Concatenate the two DataFrames
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df = merged_df.sample(frac=1).reset_index(drop=True)

    return merged_df

two = pd.read_csv("../data/lower_back_pain/copulagan_500_epochs_100_batch.csv")
one = pd.read_csv("../data/lower_back_pain/lower_back_pain.csv")
print(len(one))

print(len(merge_datasets(one, two)))"""

one = pd.read_csv("../data/diabetes.csv")
print(len(one))
