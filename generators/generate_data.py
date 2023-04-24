from config import DataConfig
from generators.copulagan_ctgan_models import main

# New instance of DataConfig, with dataset (lower_back_pain, obesity), model_name(ctgan, copulagan), epochs, batch size
config = DataConfig(dataset_name='cardio', model_name='copulagan', epochs=800, batch_size=400)

real_path, fake_path, result_path, meta_data_path, model_path = \
    config.real_path, config.fake_path, config.result_path, config.meta_data, config.model_path
dataset_name, model_name = config.dataset_name, config.model_name
epochs, batch_size = config.epochs, config.batch_size

main(
    meta_data_path, real_path, model_path, result_path, fake_path,
    model_name, dataset_name,
    epochs, batch_size,
    plot=True, save_parameters=True)

