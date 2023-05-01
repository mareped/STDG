from table_evaluator import load_data, TableEvaluator
import sys
import os
import datetime


class TableEvaluatorEvaluation:
    def __init__(self, real_path, synthetic_path, result_path, dataset_name):
        self.real_path = real_path
        self.synthetic_path = synthetic_path
        self.result_path = result_path + "/table_evaluator"
        self.dataset_name = dataset_name

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    # this is hard coded for the master thesis, to make it easier. Only datasets from the project is accepted
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

        if self.dataset_name == "cardio":
            cat_cols = ['gender', 'cholesterol',
                        'gluc', 'smoke', 'alco', 'active', 'cardio']
            target = 'cardio'

        return cat_cols, target

    def plot_results(self, save_plot=True):
        cat_cols, target = self.get_cat_columns_from_datasets()

        r_data, f_data = load_data(self.real_path, self.synthetic_path)

        table_evaluator = TableEvaluator(r_data, f_data, cat_cols=cat_cols)

        if save_plot:
            save_dir = self.result_path
        else:
            save_dir = None

        table_evaluator.visual_evaluation(save_dir=save_dir)

    def get_copies(self):
        cat_cols, target = self.get_cat_columns_from_datasets()

        r_data, f_data = load_data(self.real_path, self.synthetic_path)

        table_evaluator = TableEvaluator(r_data, f_data, cat_cols=cat_cols)

        return table_evaluator.get_copies()

    def get_results_report(self, cat_cols=None, target_col=None):

        """if the user decides to want to try the function with another dataset,
         they need to give the categorical columns and target column as a parameter"""
        if cat_cols is None or target_col is None:
            default_cat_cols, default_target_col = self.get_cat_columns_from_datasets()
            cat_cols = cat_cols if cat_cols is not None else default_cat_cols
            target_col = target_col if target_col is not None else default_target_col
        else:
            cat_cols = cat_cols
            target_col = target_col

        r_data, f_data = load_data(self.real_path, self.synthetic_path)

        e = datetime.datetime.now()
        original_stdout = sys.stdout

        # Logs the results to file
        with open(self.result_path + "/table_ev_results.txt", 'w') as f:
            sys.stdout = f
            print("Date and time = %s" % e, "\n\n\n")

            table_evaluator = TableEvaluator(r_data, f_data, cat_cols=cat_cols)
            table_evaluator.evaluate(target_col=target_col)

            # Reset the standard output
            sys.stdout = original_stdout
