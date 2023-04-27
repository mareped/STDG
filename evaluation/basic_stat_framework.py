import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


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

    """
    Positive values in the heatmap indicate that the correlation is stronger in Real, 
    while negative values indicate that the correlation is stronger in Fake.
    """

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

    def corr_scatter_plot(self, save=False):
        """
        If the correlations between variables are similar across the two datasets,
        the points will be close to the 45-degree line, while points far from the line represent variables with different
        correlation structures in the two datasets
        """
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
