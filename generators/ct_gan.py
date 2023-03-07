import pandas as pd
import json
from sdmetrics.reports.single_table import QualityReport
from sdv.tabular import CTGAN

real_data = pd.read_csv('../data/real_data/real_lower_back_pain_num.csv')

f = open('../data/metadata.json')

metadata = json.load(f)

def train_model_ctgan(data, save_model=True, path='11.pkl'):
    model = CTGAN()
    model.fit(data)

    if save_model:
        model.save(path)

    return model


def evaluate_data(original_data, synthetic_data, metadata_dict):
    rep = QualityReport()

    return rep.generate(original_data, synthetic_data, metadata_dict)


# model = train_model_ctgan(real_data)

# loaded = CTGAN.load('../saved_models/ctgan/001.pkl')
# new_data = loaded.sample(num_rows=300, output_file_path="../data/ctgan/001.csv")

fake_data = pd.read_csv('../data/fake_ctgan.csv')

# report = evaluate_data(real_data, fake_data, metadata)

# print(report)
