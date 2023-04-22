import pandas as pd
import os
import json

import io
import contextlib

import plotly.graph_objects as go

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, CopulaGANSynthesizer


def fit_model(model_type, metadata, epochs, batch_size, real_data):
    if model_type == 'ctgan':
        model_class = CTGANSynthesizer
    elif model_type == 'copulagan':
        model_class = CopulaGANSynthesizer
    else:
        raise ValueError('Invalid model type')

    # Fit new model
    model = model_class(
        metadata,
        verbose=True,
        epochs=epochs,
        batch_size=batch_size,
        enforce_rounding=False)

    model.fit(real_data)

    return model


def load_saved_model(model_type, model_path):
    if model_type == 'ctgan':
        model = CTGANSynthesizer.load(model_path)
    elif model_type == 'copulagan':
        model = CopulaGANSynthesizer.load(model_path)
    else:
        raise ValueError('Invalid model type')

    return model


def save_loss_values_and_plot(output, result_path, model_type, dataset_name):
    # Saves the loss values output to a CSV file
    epochs_output = str(output.getvalue()).split('\n')
    raw_values = [line.split(',') for line in epochs_output]
    loss_values = pd.DataFrame(raw_values)[:-1]
    loss_values.columns = ['Epoch', 'Generator Loss', 'Discriminator Loss']
    loss_values['Epoch'] = loss_values['Epoch'].str.extract('(\d+)').astype(int)
    loss_values['Generator Loss'] = loss_values['Generator Loss'].str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
    loss_values['Discriminator Loss'] = loss_values['Discriminator Loss'].str.extract('([-+]?\d*\.\d+|\d+)').astype(
        float)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    loss_values_path = result_path + f'/{model_type}_loss.csv'
    loss_values.to_csv(loss_values_path)

    # Plots graph based on loss_values file
    loss_values = pd.read_csv(loss_values_path)
    fig = go.Figure(data=[go.Scatter(x=loss_values['Epoch'], y=loss_values['Generator Loss'], name='Generator Loss'),
                          go.Scatter(x=loss_values['Epoch'], y=loss_values['Discriminator Loss'],
                                     name='Discriminator Loss')])
    fig.update_layout(template='plotly_white', legend_orientation="h", legend=dict(x=0, y=1.1))
    title = f'{model_type} loss function for dataset: ' + dataset_name
    fig.update_layout(title=title, xaxis_title='Epoch', yaxis_title='Loss')
    fig.show()
    plot_path = result_path + f'/{model_type}_loss_graph.png'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    fig.write_image(plot_path)


def save_model_parameters(result_path, model):
    parameters = model.get_parameters()
    with open(f'{result_path}/hyperparameters_training.txt', 'w') as fp:
        json.dump(parameters, fp)


def main(meta_data_path, real_path, model_path, result_path, fake_path, model_type, dataset_name,
         model_epochs, model_batch_size,
         plot=False, save_parameters=False):
    metadata = SingleTableMetadata.load_from_json(meta_data_path)
    real_data = pd.read_csv(real_path)

    # Used to save the output of training to a file
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        model = fit_model(model_type, metadata, model_epochs, model_batch_size, real_data)
        if plot:
            try:
                save_loss_values_and_plot(buf, result_path, model_type, dataset_name)
            except ValueError as e:
                print(str(e))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    if save_parameters:
        loaded_model = load_saved_model(model_type, model_path)
        os.makedirs(result_path, exist_ok=True)
        save_model_parameters(result_path, loaded_model)

    loaded = load_saved_model(model_type, model_path)
    loaded.sample(num_rows=len(real_data),
                  output_file_path=fake_path)



