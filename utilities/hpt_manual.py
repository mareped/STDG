from sdv.single_table import CTGANSynthesizer, CopulaGANSynthesizer

from sdv.metadata import SingleTableMetadata

from sdv.evaluation.single_table import evaluate_quality
import pandas as pd

METADATA_FILENAME = "../data/obesity/metadata.json"

df = pd.read_csv('../data/obesity/obesity_scaled.csv')

metadata = SingleTableMetadata.load_from_json(METADATA_FILENAME)

model = CopulaGANSynthesizer(
    metadata=metadata,
    epochs= 400,
    batch_size=100,
    verbose=False,
    enforce_rounding=False
)
model.fit(df)
synthetic_data = model.sample(len(df))

quality_report = evaluate_quality(
    real_data=df,
    synthetic_data=synthetic_data,
    metadata=metadata)

print("Model 1: ", quality_report.get_score())
