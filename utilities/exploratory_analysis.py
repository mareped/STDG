import pandas as pd
import matplotlib.pyplot as plt


def class_distribution(data, plot_title, label_col):
    # Rename the class column
    data.rename(columns={label_col: "class"}, inplace=True)
    # Check for missing values
    print(data.isna().sum())
    # Calculate class distribution
    class_counts = data["class"].value_counts().sort_index()
    # Plot class distribution
    plt.bar(class_counts.index, class_counts.values)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(plot_title)
    plt.show()


path = "../data/obesity/obesity.csv"
# Load dataset from file
dataset = pd.read_csv(path)
title = "Cardivacoular Disease"
class_distribution(dataset, title, "class")
