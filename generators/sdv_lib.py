###################################################
# CopulaGAN and CTGAN from the sdv library
###################################################
import pandas as pd
import json
import sys
from io import StringIO
import os

import plotly.graph_objects as go

from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

DATASET_NAME = 'lower_back_pain'
MODEL_NAME = 'ctgan'
DATA_FILENAME = '../data/lower_back_pain/lower_back_pain_scaled.csv'
METADATA_FILENAME = "../data/lower_back_pain/metadata.json"

# HYPERPARAMETERS
EPOCHS = 20
BATCH_SIZE = 20

real_data = pd.read_csv(DATA_FILENAME)
metadata = SingleTableMetadata.load_from_json(METADATA_FILENAME)

RESULT_PATH = f'../evaluation/results/{DATASET_NAME}/{MODEL_NAME}_{EPOCHS}_epochs'

"""
# Used to save the output of training to a file
tmp = sys.stdout
output = StringIO()
sys.stdout = output

model = CTGANSynthesizer(
    metadata,
    verbose=True,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE)

model.fit(real_data)
sys.stdout = tmp

model_path = f'../saved_models/{DATASET_NAME}/{MODEL_NAME}_{EPOCHS}_epochs.pkl'
model.save(model_path)

loaded = CTGANSynthesizer.load(model_path)
new_data = loaded.sample(num_rows=len(real_data),
                         output_file_path=f"../data/{DATASET_NAME}/{MODEL_NAME}_{EPOCHS}_epochs.csv")

new = pd.read_csv(f"../data/{DATASET_NAME}/{MODEL_NAME}_{EPOCHS}_epochs.csv")


def create_metadata(real_data):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data)
    python_dict = metadata.to_dict()
    python_dict['columns']['Class_att'] = {'sdtype': 'categorical'}
    json_object = json.dumps(python_dict, indent=4)
    print(json_object)




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

"""


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


def evaluate_data(original_data, synthetic_data, metadata_dict):
    rep = QualityReport()

    return rep.generate(original_data, synthetic_data, metadata_dict)
