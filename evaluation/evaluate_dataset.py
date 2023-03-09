from table_evaluator import load_data, TableEvaluator


real, fake = load_data('../data/lower_back_pain/lower_back_pain_num.csv', '../data/lower_back_pain/copulagan.csv')

cat_cols = ['Class_att']

table_evaluator = TableEvaluator(real, fake, cat_cols=cat_cols)

#table_evaluator.visual_evaluation('plots/ctgan.png')

table_evaluator.evaluate(target_col='Class_att')
