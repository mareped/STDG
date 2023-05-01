import pandas as pd

from config import DataConfig
from evaluation.evaluate_dataset_sdv import SDVEvaluation
from evaluation.evaluate_dataset_te import TableEvaluatorEvaluation
from evaluation.basic_stat_framework import BasicStatEvaluation

config = DataConfig(dataset_name='obesity', model_name='copulagan', epochs=350, batch_size=100)

real_path, fake_path, meta_data_path, result_path, data_name, mixed_path = \
    config.real_path, config.fake_path, config.meta_data, config.result_path, config.dataset_name, config.mixed_path

def run_sdv():
    # for fake
    sdv_evaluation = SDVEvaluation(real_path, fake_path, meta_data_path, result_path)
    # for mixed
    # sdv_evaluation = SDVEvaluation(real_path, mixed_path, meta_data_path, result_path)

    sdv_evaluation.write_reports_to_file()
    columns = pd.read_csv(real_path).columns.tolist()
    sdv_evaluation.plot_all_columns_ranges(columns)


def run_te():
    # for fake
    te_evaluation = TableEvaluatorEvaluation(real_path, fake_path, result_path, data_name)

    # for mixed
    #te_evaluation = TableEvaluatorEvaluation(real_path, mixed_path, result_path, data_name)
    te_evaluation.plot_results(save_plot=True)
    te_evaluation.get_results_report()


def run_bs():
    bs_evaluation = BasicStatEvaluation(real_path, fake_path, result_path)
    bs_evaluation.column_corr_plot(save=True)
    bs_evaluation.subtracted_corr_matrix(save=True)
    bs_evaluation.corr_scatter_plot(save=True)


# run_bs()
# run_sdv()
# run_te()


"""sdv_evaluation = SDVEvaluation(real_path, fake_path, meta_data_path, result_path)
print(sdv_evaluation.row_synhesis())
"""

te_evaluation = TableEvaluatorEvaluation(real_path, fake_path, result_path, data_name)

# for mixed
# te_evaluation = TableEvaluatorEvaluation(real_path, mixed_path, result_path, data_name)
te_evaluation.get_results_report()
