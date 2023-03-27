from table_evaluator import load_data, TableEvaluator
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
import datetime

# IMPORTANT: DEFINE ALL GLOBAL ARGUMENTS BEFORE RUNNING SCRIPT

# define dataset name you are using (lower_back_pain, obesity)
DATASET_NAME = 'lower_back_pain'

# define dataset model you are using (ctgan, copulagan)
MODEL_NAME = 'copulagan'

# Hyperparameters
EPOCHS = 800
BATCH_SIZE = 100

file_ending = f'{MODEL_NAME}_{EPOCHS}_epochs_{BATCH_SIZE}_batch'

# Define where the real and fake data path is. IMPORTANT: change real file name
real = f'../data/{DATASET_NAME}/lower_back_pain_scaled.csv'
fake = f'../data/{DATASET_NAME}/' + file_ending + '.csv'

RESULT_PATH = f'../evaluation/results/{DATASET_NAME}/' + file_ending


def get_cat_columns_from_datasets(data=DATASET_NAME):
    cat_cols = []
    target = ""
    if data == "lower_back_pain":
        # Define all the categorical columns involved
        cat_cols = ['Class_att']

        # Define the target column
        target = 'Class_att'

    if data == "obesity":
        cat_cols = ['Gender', 'family_history_with_overweight',
                    'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
        target = 'NObeyesdad'

    return cat_cols, target


def table_ev(real_path, fake_path, data=DATASET_NAME, visual=True, plot_path=RESULT_PATH):
    cat_cols, target = get_cat_columns_from_datasets(data)

    r_data, f_data = load_data(real_path, fake_path)

    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    e = datetime.datetime.now()
    original_stdout = sys.stdout

    # Logs the results to file
    with open(RESULT_PATH + "/results.txt", 'w') as f:
        sys.stdout = f
        print("Date and time = %s" % e, "\n\n\n")
        table_evaluator = TableEvaluator(r_data, f_data, cat_cols=cat_cols)
        table_evaluator.evaluate(target_col=target)

        # Reset the standard output
        sys.stdout = original_stdout

    if visual:
        table_evaluator.visual_evaluation(plot_path)


def column_correlation(path, save=False, save_path='plots/xgan_matrix.png'):
    data = pd.read_csv(path)
    f, ax = plt.subplots(figsize=(13, 8))
    corr = data.corr()
    sns.heatmap(corr,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                vmin=-1.0, vmax=1.0,
                square=True, ax=ax)
    plt.title("Column Correlation")

    if save:
        plt.savefig(save_path)

    plt.show()


#table_ev(real, fake, visual=True, plot_path=RESULT_PATH)

column_correlation(fake, save=True, save_path=f'{RESULT_PATH}/corrmatrix.png')

# column_correlation(real, save=True, save_path='results/obesity_real_corrmatrix.png')
