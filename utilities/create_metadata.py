# Script that is used to create metadata for the generators
import json
import pandas as pd

from sdv.metadata import SingleTableMetadata

DATA_FILENAME = '../data/obesity/obesity_scaled.csv'

real_data = pd.read_csv(DATA_FILENAME)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=real_data)
print(metadata)


python_dict = metadata.to_dict()

# Change the dict if needed
# python_dict['columns']['Class_att'] = {'sdtype': 'categorical'}

json_object = json.dumps(python_dict, indent=4)

with open("metadata.json", "w") as outfile:
    outfile.write(json_object)
