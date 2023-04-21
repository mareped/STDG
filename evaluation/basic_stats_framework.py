import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utilities.plots import column_correlation_plot
from config import DataConfig

config = DataConfig(dataset_name='obesity', model_name='copulagan', epochs=400, batch_size=100)

real_path, fake_path, result_path = config.real_path, config.fake_path, config.result_path

# TODO: make function that prints all the correlations next to each other
column_correlation_plot(fake_path, save_plot=False, save_path=f'{result_path}/corrmatrix.png')
