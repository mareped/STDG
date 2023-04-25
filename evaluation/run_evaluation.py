import pandas as pd

from config import DataConfig
from evaluation.evaluate_dataset_sdv import SDVEvaluation
from evaluation.evaluate_dataset_te import TableEvaluatorEvaluation
config = DataConfig(dataset_name='cardio', model_name='copulagan', epochs=250, batch_size=400)

real_path, fake_path, meta_data_path, result_path, data_name, mixed_path = \
    config.real_path, config.fake_path, config.meta_data, config.result_path, config.dataset_name, config.mixed_path


def run_sdv():
    # for fake
    #sdv_evaluation = SDVEvaluation(real_path, fake_path, meta_data_path, result_path)

    # for mixed
    sdv_evaluation = SDVEvaluation(real_path, mixed_path, meta_data_path, result_path)

    sdv_evaluation.write_reports_to_file()
    columns = pd.read_csv(real_path).columns.tolist()
    sdv_evaluation.plot_all_columns_boundaries(columns)


def run_te():
    # for fake
    #te_evaluation = TableEvaluatorEvaluation(real_path, fake_path, result_path, data_name)

    # for mixed
    te_evaluation = TableEvaluatorEvaluation(real_path, mixed_path, result_path, data_name)
    te_evaluation.column_corr_plot(save=True)
    te_evaluation.get_evaluation(save_plot=True)


run_sdv()
#run_te()
