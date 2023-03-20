from sklearn.model_selection import GridSearchCV
from sdv.tabular import CTGAN
from sdv.evaluation import evaluate
import pandas as pd

df = pd.read_csv('data/lower_back_pain/lower_back_pain_scaled.csv')

model_1 = CTGAN(
   epochs=800,
   batch_size=100,
   generator_lr=0.001,
   discriminator_lr=0.001,
   verbose= False
)

print(model_1)
model_1.fit(df)
new_data_1 = model_1.sample(len(df))


print("Model 1: ", evaluate(new_data_1, df))
