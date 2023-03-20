import pandas as pd

df = pd.read_csv('data/obesity/obesity.csv')


categorical_cols = ['Gender', 'family_history_with_overweight',
            'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']


df[categorical_cols] = df[categorical_cols].astype(object)

cat_columns = df.select_dtypes(['object']).columns

df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])


df.to_csv('data/obesity/obesity_num_2.csv', index=False)

"""df= df.rename(columns = {'Col1': 'pelvic_incidence', 'Col2': 'pelvic_tilt',
                                    'Col3': 'lumbar_lordosis_angle', 'Col4': 'sacral_slope',
                                    'Col5': 'pelvic_radius', 'Col6': 'degree_spondylolisthesis',
                                    'Col7': 'pelvic_slope', 'Col8': 'direct_tilt',
                                    'Col9': 'thoracic_slope', 'Col10': 'cervical_tilt',
                                    'Col11': 'sacrum_angle', 'Col12': 'scoliosis_slope',
                                    }, inplace = False)

df['Class_att'].replace(['Normal', 'Abnormal'],
                        [0, 1], inplace=True)

df['Class_att'] = df["Class_att"].astype("category")

"""
