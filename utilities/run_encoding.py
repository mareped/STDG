from config import DataConfig
from utilities.encode_data import *
from utilities.pre_processing import merge_datasets

# Script that used encode_data.py to encode dataset and creates a mixed dataset with the same encoding
config = DataConfig(dataset_name='lower_back_pain', model_name='copulagan', epochs=800, batch_size=100)
real_path, fake_path, mixed_path = config.real_path, config.fake_path, config.mixed_path

# Load the JSON-encoded encoding map if it exist
with open(real_path + '_encoding_map.json', 'r') as f:
    encoding_map_json = f.read()

# Convert the JSON-encoded encoding map to a Python dictionary
existing_encoding_map = json.loads(encoding_map_json)

categorical_cols = ['Class_att']

real_data_enc, fake_data_enc, encoding_map = encode_data_consistent(real_path, fake_path, categorical_cols,
                                                                    existing_encoding_map)
# real_data_enc.to_csv(real_path + '_encoded.csv', index=False)
fake_data_enc.to_csv(fake_path + '_encoded.csv', index=False)

# Concatenate the real and fake datasets to make mixed data
mixed_data = merge_datasets(real_data_enc, fake_data_enc)
mixed_data.to_csv(mixed_path + '_encoded.csv', index=False)

print(encoding_map)


