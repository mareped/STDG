import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# Scale all the numeric values, except the categorical columns and the label
def min_max_scale_df(df, categorical_columns, label):
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Fit and transform the numerical columns using the scaler object
    numerical_columns = df.columns.difference(categorical_columns + [label])
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Return the original DataFrame with categorical columns unchanged
    return df


# encodes one specific column
def label_encode_one_col(df, col):
    df[col] = pd.factorize(df[col])[0].astype(int)
    return df


# encodes one all cat cols
def encode_all_cat_cols(df, categorical_columns):
    for col in categorical_columns:
        df[col] = pd.factorize(df[col])[0].astype(int)
    return df


# One-Hot encodes the categorical values, except label. Label is numerical-encoded.
def one_hot_encode(df, categorical_columns, label):
    # Make a copy of the original DataFrame
    df_copy = df.copy()

    # Iterate over each categorical column
    for col in categorical_columns:
        # Exclude the label column from encoding
        if col != label:
            # Use pandas.get_dummies to one-hot encode the column
            dummies = pd.get_dummies(df_copy[col], prefix=col)

            # Add the one-hot encoded columns to the DataFrame
            df_copy = pd.concat([df_copy, dummies], axis=1)

            # Drop the original categorical column from the DataFrame
            df_copy.drop(col, axis=1, inplace=True)

    # Move the label column to the last position in the DataFrame
    cols = df_copy.columns.tolist()
    cols.remove(label)
    cols.append(label)
    df_copy = df_copy[cols]

    label_encode_one_col(df_copy, label)

    return df_copy





# For obesity
"""data = pd.read_csv('../data/obesity/ctgan_300_epochs_50_batch.csv')

label = 'NObeyesdad'
categorical_cols = ['Gender', 'family_history_with_overweight',
                    'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']

# data = min_max_scale_df(data, categorical_cols, label)
data = encode_all_cat_cols(data, categorical_cols)

#data = one_hot_encode(data, categorical_cols, label)
data.to_csv('../data/obesity/ctgan_300_epochs_50_batch_encoded.csv', index=False)"""

# For lbp
"""
data = pd.read_csv('../data/lower_back_pain/copulagan_800_epochs_100_batch.csv')
label = 'Class_att'
data = label_encode_one_col(data, label)
categorical_cols = []
data = min_max_scale_df(data, categorical_cols, label)
data.to_csv('../data/lower_back_pain/copulagan_800_epochs_100_batch_prep.csv', index=False)"""
