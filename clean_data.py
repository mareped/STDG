import pandas as pd

df = pd.read_csv('data/lower_back_pain.csv')

df= df.rename(columns = {'Col1': 'pelvic_incidence', 'Col2': 'pelvic_tilt',
                                    'Col3': 'lumbar_lordosis_angle', 'Col4': 'sacral_slope',
                                    'Col5': 'pelvic_radius', 'Col6': 'degree_spondylolisthesis',
                                    'Col7': 'pelvic_slope', 'Col8': 'direct_tilt',
                                    'Col9': 'thoracic_slope', 'Col10': 'cervical_tilt',
                                    'Col11': 'sacrum_angle', 'Col12': 'scoliosis_slope',
                                    }, inplace = False)

df['Class_att'].replace(['Normal', 'Abnormal'],
                        [0, 1], inplace=True)

df['Class_att'] = df["Class_att"].astype("category")

df.to_csv('data/lower_back_pain_numeric.csv', index=False)
