import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class BasicStatEvaluation:

    def __init__(self, real_path, synthetic_path, result_path):
        self.real_data = pd.read_csv(real_path)
        self.synthetic_data = pd.read_csv(synthetic_path)
        self.real_corr = self.real_data.corr()
        self.synthetic_corr = self.synthetic_data.corr()
        self.result_path = result_path + "/basic_stats"

        # Create the directory if it doesn't exist
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    def column_corr_plot(self, save=False):
        f, ax = plt.subplots(figsize=(13, 8))
        sns.heatmap(self.synthetic_corr,
                    cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    vmin=-1.0, vmax=1.0,
                    square=True, ax=ax)

        plt.title("Column Correlation")

        if save:
            plt.savefig(f'{self.result_path}/corrmatrix.png')

        plt.show()

    def subtracted_corr_matrix(self, save=False):

        correlation_matrix_difference = self.real_corr - self.synthetic_corr

        f, ax = plt.subplots(figsize=(13, 8))
        sns.heatmap(correlation_matrix_difference,
                    cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    vmin=-1.0, vmax=1.0,
                    square=True, ax=ax)
        plt.title("Correlation Matrix Real - Synthetic")

        if save:
            plt.savefig(f'{self.result_path}/corrmatrix_difference.png')

        plt.show()

    def abs_difference_corr_matrix(self, save=False):

        abs_correlation_matrix_difference = np.abs(self.real_corr - self.synthetic_corr)

        sns.heatmap(abs_correlation_matrix_difference, cmap="coolwarm", center=0,  square=True)
        plt.title("Abosulte Correlation difference Real - Synthetic")

        if save:
            plt.savefig(f'{self.result_path}/corrmatrix_difference.png')

        plt.show()

    def corr_scatter_plot(self, save=False):
        # Flatten the correlation matrices and remove the diagonal elements
        flat_corr_matrix1 = self.real_corr.where(~np.eye(self.real_corr.shape[0], dtype=bool)).stack()
        flat_corr_matrix2 = self.synthetic_corr.where(~np.eye(self.synthetic_corr.shape[0], dtype=bool)).stack()

        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Draw the scatter plot
        ax.scatter(flat_corr_matrix1, flat_corr_matrix2)
        ax.set_xlabel("Correlations in Real")
        ax.set_ylabel("Correlations in Fake")

        # Add a 45-degree line to visualize perfect agreement between the two datasets
        ax.plot([-1, 1], [-1, 1], 'k--', linewidth=1)

        plt.title("Scatter Plot of Correlation Coefficients")

        if save:
            plt.savefig(f'{self.result_path}/corr_coeff_scatter.png')

        plt.show()

    def count_identical_rows(self):
        # Create a new dataframe to hold the identical rows
        identical_rows = pd.DataFrame(columns=self.real_data.columns)

        # Loop through the rows of the synthetic data
        for index, row in self.synthetic_data.iterrows():
            # Check if the row is identical to any row in the real data
            identical = self.real_data.eq(row).all(axis=1)
            if identical.any():
                # Add the identical row to the new dataframe
                identical_rows = identical_rows.append(self.synthetic_data.iloc[index])

        # Get the number of identical rows
        num_identical_rows = len(identical_rows)

        # If there are identical rows, print them out
        if num_identical_rows > 0:
            print(f"There are {num_identical_rows} identical rows:")
            for index, row in identical_rows.iterrows():
                row_str = ", ".join([f"{val:.2f}" if isinstance(val, float) else str(val) for val in row.values])
                print(f"Identical row between fake and real data: {row_str}")
                print("=" * 30)

        print("There are no identical rows")

        return num_identical_rows







