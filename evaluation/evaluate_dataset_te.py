from table_evaluator import load_data, TableEvaluator
import sys
import os
import datetime

from config import DataConfig
from utilities.plots import column_correlation_plot

config = DataConfig(dataset_name='obesity', model_name='copulagan', epochs=400, batch_size=100)

real_path, fake_path, result_path = config.real_path, config.fake_path, config.result_path


def get_cat_columns_from_datasets(data=config.dataset_name):
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


def table_ev(real_path, fake_path, data=config.dataset_name, visual=True, plot_path=result_path):
    cat_cols, target = get_cat_columns_from_datasets(data)

    r_data, f_data = load_data(real_path, fake_path)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    e = datetime.datetime.now()
    original_stdout = sys.stdout

    # Logs the results to file
    with open(result_path + "/results.txt", 'w') as f:
        sys.stdout = f
        print("Date and time = %s" % e, "\n\n\n")
        table_evaluator = TableEvaluator(r_data, f_data, cat_cols=cat_cols)
        table_evaluator.evaluate(target_col=target)

        # Reset the standard output
        sys.stdout = original_stdout

    if visual:
        table_evaluator.visual_evaluation(plot_path)



# table_ev(real, fake, visual=True, plot_path=RESULT_PATH)

column_correlation_plot(fake_path, save_plot=False, save_path=f'{result_path}/corrmatrix.png')

# column_correlation(real, save=True, save_path='results/obesity_real_corrmatrix.png')
