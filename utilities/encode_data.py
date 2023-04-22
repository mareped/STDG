import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import DataConfig
import json


# Assign specific columns to categorical values
def assign_categorical(df, cols):
    for col in cols:
        df[col] = pd.Categorical(df[col])
    return df


def print_column_types(df):
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")


# Creates a mixed dataset of two dataframes
def merge_datasets(df1, df2):
    # Concatenate the two DataFrames
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df = merged_df.sample(frac=1).reset_index(drop=True)

    return merged_df


def save_encoding_map(encoding_map, file_path):
    with open(file_path, 'w') as f:
        json.dump(encoding_map, f)


def label_encode_categories(df, encoding_map=None):
    df_encoded = df.copy()

    if encoding_map is None:
        encoding_map = {}

    le = LabelEncoder()

    for col in df_encoded.columns:
        if df_encoded[col].dtype.name == 'category':
            if col in encoding_map:
                # Use the encoding map if available
                df_encoded[col] = df_encoded[col].map(encoding_map[col]).fillna(-1)
            else:
                df_encoded[col] = le.fit_transform(df_encoded[col])
                encoding_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    return df_encoded, encoding_map


#  Encodes real and fake data consistently
def encode_data_consistent(real_path, fake_path, categorical_columns, encoding_map=None):
    # Load real dataset
    real_data = pd.read_csv(real_path + '.csv')

    real_data = assign_categorical(real_data, categorical_columns)

    # Encode the real dataset
    real_data_encoded, encoding_map = label_encode_categories(real_data, encoding_map)

    # Load the fake dataset
    fake_data = pd.read_csv(fake_path + '.csv')
    fake_data = assign_categorical(fake_data, categorical_columns)

    # Encode the categorical columns of the fake dataset using the same encoding map
    fake_data_encoded, _ = label_encode_categories(fake_data, encoding_map)

    return real_data_encoded, fake_data_encoded, encoding_map


config = DataConfig(dataset_name='lower_back_pain', model_name='copulagan', epochs=800, batch_size=100)
real_path, fake_path, mixed_path = config.real_path, config.fake_path, config.mixed_path

# Load the JSON-encoded encoding map if it exist
with open(real_path + '_encoding_map.json', 'r') as f:
    encoding_map_json = f.read()

# Convert the JSON-encoded encoding map to a Python dictionary
existing_encoding_map = json.loads(encoding_map_json)

categorical_cols = ['Class_att']

real_data_enc, fake_data_enc, encoding_map = encode_data_consistent(real_path, fake_path, categorical_cols, existing_encoding_map)
# real_data_enc.to_csv(real_path + '_encoded.csv', index=False)
fake_data_enc.to_csv(fake_path + '_encoded.csv', index=False)

# Concatenate the real and fake datasets to make mixed data
mixed_data = merge_datasets(real_data_enc, fake_data_enc)
mixed_data.to_csv(mixed_path + '_encoded.csv', index=False)

print(encoding_map)
"""
# Save encoding map as JSON file
with open(real_path + '_encoding_map.json', 'w') as f:
    encoding_map_json = {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in encoding_map.items()}
    json.dump(encoding_map_json, f, indent=2)"""
