"""
This class defines configuration for generating synthetic data using GAN models on the available datasets
The allowed values for datasets and models are listed in the class constants.
The configuration includes the number of epochs, batch size, file name, and file paths for the real, synthetic, and mixed data

Use this file to define where to read and write the data used for easier configuration.
"""


class DataConfig:
    ALLOWED_DATASETS = ['lower_back_pain', 'obesity', 'cardio']
    ALLOWED_MODELS = ['ctgan', 'copulagan']

    def __init__(self, dataset_name, model_name, epochs, batch_size):
        if dataset_name not in self.ALLOWED_DATASETS:
            raise ValueError(
                f"Invalid dataset name '{dataset_name}'. Allowed values are: {', '.join(self.ALLOWED_DATASETS)}")

        if model_name not in self.ALLOWED_MODELS:
            raise ValueError(f"Invalid model name '{model_name}'. Allowed values are: {', '.join(self.ALLOWED_MODELS)}")

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.file_ending = f'{model_name}_{epochs}_epochs_{batch_size}_batch'
        self.real_path = f'../data/{dataset_name}/{dataset_name}' + '.csv'
        self.fake_path = f'../data/{dataset_name}/' + self.file_ending  + '.csv'
        self.result_path = f'..//results/{dataset_name}/' + self.file_ending
        self.meta_data = f'../data/{dataset_name}/metadata.json'
        self.model_path = f'../saved_models/{dataset_name}/' + self.file_ending + '.pkl'
