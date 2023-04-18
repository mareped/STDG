###################################################
###################### CTGAN ######################
###################################################
import pandas as pd
import json
import sys
from io import StringIO
import os

import plotly.graph_objects as go

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

DATASET_NAME = 'obesity'
MODEL_NAME = 'ctgan'

# Hyperparameters
EPOCHS = 300
BATCH_SIZE = 50


DATA_FILENAME = f'../data/{DATASET_NAME}/{DATASET_NAME}_scaled.csv'
METADATA_FILENAME = f"../data/{DATASET_NAME}/metadata.json"

file_ending = f'{DATASET_NAME}/{MODEL_NAME}_{EPOCHS}_epochs_{BATCH_SIZE}_batch'
real_data = pd.read_csv(DATA_FILENAME)

# Load the metadata
metadata = SingleTableMetadata.load_from_json(METADATA_FILENAME)

RESULT_PATH = f'../evaluation/results/' + file_ending

model = CTGANSynthesizer(
    metadata,
    verbose=True,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    enforce_rounding=False)

# Used to save the output of training to a file
tmp = sys.stdout
output = StringIO()
sys.stdout = output

# Fits the data to the model
model.fit(real_data)
# Ends the saving of output
sys.stdout = tmp

model_path = f'../saved_models/' + file_ending + '.pkl'
model.save(model_path)

# Load saved model
loaded = CTGANSynthesizer.load(model_path)

synthetic_data_path = f'../data/' + file_ending + '.csv'

# Create synthetic data
synthetic_data = loaded.sample(num_rows=len(real_data),
                               output_file_path=synthetic_data_path)

# Read the synthetic data file
synthetic_df = pd.read_csv(synthetic_data_path)


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

    loss_values_path = RESULT_PATH
    # Check whether the specified path exists or not
    if not os.path.exists(loss_values_path):
        os.makedirs(loss_values_path)

    loss_values.to_csv(RESULT_PATH + f'/{MODEL_NAME}_loss.csv', index=False)


def plot_loss(save=False):
    path = RESULT_PATH + f'/{MODEL_NAME}_loss.csv'
    loss_values = pd.read_csv(path)
    # Plot loss function
    fig = go.Figure(data=[go.Scatter(x=loss_values['Epoch'], y=loss_values['Generator Loss'], name='Generator Loss'),
                          go.Scatter(x=loss_values['Epoch'], y=loss_values['Discriminator Loss'],
                                     name='Discriminator Loss')])

    # Update the layout for best viewing
    fig.update_layout(template='plotly_white',
                      legend_orientation="h",
                      legend=dict(x=0, y=1.1))

    title = 'CTGAN loss function for dataset: ' + DATASET_NAME
    fig.update_layout(title=title, xaxis_title='Epoch', yaxis_title='Loss')
    fig.show()

    if save:
        fig.write_image(RESULT_PATH + f'/{MODEL_NAME}_loss_graph.png')


save_loss_values()
plot_loss(True)

# Save the model parameters to results
parameters = model.get_parameters()
with open(f'{RESULT_PATH}/hyperparameters_training.txt', 'w') as fp:
    json.dump(parameters, fp)

