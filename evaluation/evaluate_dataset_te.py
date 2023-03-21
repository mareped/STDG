from table_evaluator import load_data, TableEvaluator
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import sys
import datetime

# IMPORTANT: DEFINE ALL GLOBAL ARGUMENTS BEFORE RUNNING SCRIPT

# define dataset name you are using (lower_back_pain, obesity)
dataset = 'obesity'

# Define where the real and fake data path is
real = '../data/' + dataset + '/obesity_num.csv'
fake = '../data/' + dataset + '/copulagan.csv'

# what folder to save results
result_path = "results/copulagan/obesity"


def get_cat_columns_from_datasets(data=dataset):
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


def table_ev(real_path, fake_path, data=dataset, visual=True, plot_path="results/test"):
    cat_cols, target = get_cat_columns_from_datasets(data)

    real, fake = load_data(real_path, fake_path)

    table_evaluator = TableEvaluator(real, fake, cat_cols=cat_cols)

    table_evaluator.evaluate(target_col=target)

    if visual:
        table_evaluator.visual_evaluation(plot_path)


e = datetime.datetime.now()
original_stdout = sys.stdout

# TODO: move this into table_ev method

# Logs the results to file
with open(result_path + "/results.txt", 'w') as f:
    sys.stdout = f
    print("Date and time = %s" % e, "\n\n\n")
    table_ev(real, fake, visual=True, plot_path=result_path)
    # Reset the standard output
    sys.stdout = original_stdout


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

# column_correlation(real, save=True, save_path='results/obesity_real_corrmatrix.png')
# column_correlation(fake, save=True, save_path='results/ctgan/obesity_2/fake_corrmatrix.png')
