import pandas as pd
import json
import sys
from io import StringIO
import os

import plotly.graph_objects as go

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CopulaGANSynthesizer

from config import DataConfig
"""
Simple script generating data from CopulaGAN
"""

# create a new instance of DataConfig, with the dataset (lower_back_pain, obesity)
config = DataConfig(dataset_name='lower_back_pain', model_name='ctgan', epochs=800, batch_size=20)

real_path, fake_path, result_path, meta_data_path, model_path = \
    config.real_path, config.fake_path, config.result_path, config.meta_data, config.model_path
dataset_name, model_name = config.dataset_name, config.model_name
epochs, batch_size = config.epochs, config.batch_size

# Load the metadata
metadata = SingleTableMetadata.load_from_json(meta_data_path)
real_data = pd.read_csv(dataset_name)


model = CopulaGANSynthesizer(
    metadata,
    verbose=True,
    epochs=epochs,
    batch_size=batch_size,
    enforce_rounding=False)

# Used to save the output of training to a file
tmp = sys.stdout
output = StringIO()
sys.stdout = output

# Fits the data to the model
model.fit(real_data)
# Ends the saving of output
sys.stdout = tmp

model.save(model_path)

# Load saved model
loaded = CopulaGANSynthesizer.load(model_path)

# Create synthetic data
synthetic_data = loaded.sample(num_rows=len(real_data),
                               output_file_path=fake_path)

# Read the synthetic data file
synthetic_df = pd.read_csv(fake_path)


def save_loss_values():
    # CTGAN prints out a new line for each epoch
    epochs_output = str(output.getvalue()).split('\n')

    # CTGAN separates the values with commas
    raw_values = [line.split(',') for line in epochs_output]
    loss_values = pd.DataFrame(raw_values)[:-1]  # convert to df and delete last row (empty)

    # Rename columns
    loss_values.columns = ['Epoch', 'Generator Loss', 'Discriminator Loss']

    # Extract the numbers from each column
    loss_values['Epoch'] = loss_values['Epoch'].str.extract('(\d+)').astype(int)
    loss_values['Generator Loss'] = loss_values['Generator Loss'].str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
    loss_values['Discriminator Loss'] = loss_values['Discriminator Loss'].str.extract('([-+]?\d*\.\d+|\d+)').astype(
        float)

    loss_values_path = result_path
    # Check whether the specified path exists or not
    if not os.path.exists(loss_values_path):
        os.makedirs(loss_values_path)

    loss_values.to_csv(result_path + f'/{model_name}_loss.csv', index=False)


def plot_loss(save=False):
    path = result_path + f'/{model_name}_loss.csv'
    loss_values = pd.read_csv(path)
    # Plot loss function
    fig = go.Figure(data=[go.Scatter(x=loss_values['Epoch'], y=loss_values['Generator Loss'], name='Generator Loss'),
                          go.Scatter(x=loss_values['Epoch'], y=loss_values['Discriminator Loss'],
                                     name='Discriminator Loss')])

    # Update the layout for best viewing
    fig.update_layout(template='plotly_white',
                      legend_orientation="h",
                      legend=dict(x=0, y=1.1))

    title = 'CTGAN loss function for dataset: ' + dataset_name
    fig.update_layout(title=title, xaxis_title='Epoch', yaxis_title='Loss')
    fig.show()

    if save:
        fig.write_image(result_path + f'/{model_name}_loss_graph.png')


# save_loss_values()
plot_loss(True)

# Save the model parameters to results
parameters = model.get_parameters()
with open(f'{result_path}/hyperparameters_training.txt', 'w') as fp:
    json.dump(parameters, fp)

