from table_evaluator import load_data, TableEvaluator
import sys
import os
import datetime

from utilities.plots import column_correlation_plot


class TableEvaluatorEvaluation:
    def __init__(self, real_path, synthetic_path, result_path, dataset_name):
        self.real_path = real_path
        self.synthetic_path = synthetic_path
        self.result_path = result_path + "/table_evaluator"
        self.dataset_name = dataset_name

    def get_cat_columns_from_datasets(self):
        cat_cols = []
        target = ""
        if self.dataset_name == "lower_back_pain":
            # Define all the categorical columns involved
            cat_cols = ['Class_att']

            # Define the target column
            target = 'Class_att'

        if self.dataset_name == "obesity":
            cat_cols = ['Gender', 'family_history_with_overweight',
                        'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
            target = 'NObeyesdad'

        return cat_cols, target

    def get_evaluation(self, save_plot=True):
        cat_cols, target = self.get_cat_columns_from_datasets()

        r_data, f_data = load_data(self.real_path, self.synthetic_path)

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        table_evaluator = TableEvaluator(r_data, f_data, cat_cols=cat_cols)

        if save_plot:
            table_evaluator.visual_evaluation(self.result_path)

        e = datetime.datetime.now()
        original_stdout = sys.stdout

        # Logs the results to file
        with open(self.result_path + "/table_ev_results.txt", 'w') as f:
            sys.stdout = f
            print("Date and time = %s" % e, "\n\n\n")
            table_evaluator = TableEvaluator(r_data, f_data, cat_cols=cat_cols)
            table_evaluator.evaluate(target_col=target)

            # Reset the standard output
            sys.stdout = original_stdout

"""    def column_corr_plot(self, save=False):
        column_correlation_plot(self.synthetic_path, save_plot=save, save_path=f'{self.result_path}/corrmatrix.png')

"""