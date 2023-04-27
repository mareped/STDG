import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

from utilities.pre_processing import merge_datasets


# print the type of columns in a dataframe
def print_column_types(df):
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")


# Assign specific columns to categorical values
def assign_categorical(df, cols):
    for col in cols:
        df[col] = pd.Categorical(df[col])
    return df


# label encodes categorical values in a dataframe. Note: the categorical values needs to be of type "category"
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


# Save encoding map as JSON file
def save_encoding_map(encoding_map, file_path):
    with open(file_path, 'w') as f:
        encoding_map_json = {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in encoding_map.items()}
        json.dump(encoding_map_json, f, indent=2)


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


# decodes the data to original values based on saved encoding map
def decode_data(data_path, encoding_map_path):
    with open(encoding_map_path, 'r') as f:
        encoding_map = json.load(f)

    df = pd.read_csv(data_path)

    # Generate a dictionary for each encoded column with encoded values as keys and original values as values
    decode_maps = {}
    for column, encoding in encoding_map.items():
        if column in df.columns:
            decode_maps[column] = {v: k for k, v in encoding.items()}

    # Replace encoded values in each column of the DataFrame
    for column in df.columns:
        if column in decode_maps:
            df[column].replace(decode_maps[column], inplace=True)

    return df
