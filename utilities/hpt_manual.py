from sdv.single_table import CTGANSynthesizer, CopulaGANSynthesizer

from sdv.metadata import SingleTableMetadata

from sdv.evaluation.single_table import evaluate_quality
import pandas as pd

METADATA_FILENAME = "../data/lower_back_pain/metadata.json"

df = pd.read_csv('../data/lower_back_pain/lower_back_pain_scaled.csv')

metadata = SingleTableMetadata.load_from_json(METADATA_FILENAME)

model = CopulaGANSynthesizer(
   metadata= metadata,
   epochs=800,
   batch_size=40,
   verbose= False
)
model.fit(df)
synthetic_data = model.sample(len(df))

quality_report = evaluate_quality(
    real_data=df,
    synthetic_data=synthetic_data,
    metadata=metadata)

print("Model 1: ", quality_report.get_score())
