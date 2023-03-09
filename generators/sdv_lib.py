import pandas as pd
import json
from sdmetrics.reports.single_table import QualityReport
from sdv.tabular import CTGAN
from sdv.tabular import CopulaGAN

real_data = pd.read_csv('../data/lower_back_pain/lower_back_pain_num.csv')

f = open('../data/lower_back_pain/metadata.json')

metadata = json.load(f)


def train_model_ctgan(data, save_model=True, path='../saved_models/copulagan/001.pkl', model_type='ctgan'):

    if model_type == 'ctgan':
        model = CTGAN()
    elif model_type == 'copulagan':
        model = CopulaGAN()
    else:
        raise Exception("Need to put valid model type")

    model.fit(data)

    if save_model:
        model.save(path)

    return model


def evaluate_data(original_data, synthetic_data, metadata_dict):
    rep = QualityReport()

    return rep.generate(original_data, synthetic_data, metadata_dict)


# model = train_model_ctgan(real_data, model_type='copulagan')

loaded = CTGAN.load('../saved_models/lower_back_pain/copulagan.pkl')
new_data = loaded.sample(num_rows=300, output_file_path="../data/lower_back_pain/copulagan.csv")

#fake_data = pd.read_csv('../data/ctgan/001.csv')

#report = evaluate_data(real_data, fake_data, metadata)

#print(report)
